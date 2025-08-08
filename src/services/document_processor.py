"""
Document Processing Service

Handles PDF download, text extraction, chunking, and embedding generation
Target: 5 seconds for complete document processing
"""

import asyncio
import hashlib
import time
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
import io
import aiohttp
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
import logging
from concurrent.futures import ThreadPoolExecutor

from ..core.config import settings
from ..models.api_models import DocumentData, DocumentChunk
from .vector_store import VectorStoreService

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing with optimized performance"""
    
    def __init__(self, vector_store: VectorStoreService):
        self.vector_store = vector_store
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.document_cache: Dict[str, DocumentData] = {}
        # Initialize HTTP session for downloads
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self):
        """Initialize the document processor"""
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=5,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "User-Agent": "HackRX-DocumentProcessor/1.0"
            }
        )
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        self.executor.shutdown(wait=True)
    
    def _generate_document_id(self, url: str) -> str:
        """Generate a unique document ID from URL"""
        return hashlib.md5(url.encode()).hexdigest()
    
    async def process_document(self, document_url: str, request_id: str) -> Dict[str, Any]:
        """
        Process document from URL with parallel processing
        Target: 5 seconds total
        """
        start_time = time.time()
        document_id = self._generate_document_id(document_url)
        
        logger.info(f"Processing document {document_id}", extra={"request_id": request_id})
        
        # Check cache first
        if settings.ENABLE_DOCUMENT_CACHE and document_id in self.document_cache:
            cached_doc = self.document_cache[document_id]
            logger.info(f"Using cached document {document_id}")
            return {
                "document_id": document_id,
                "chunks": [chunk.dict() for chunk in cached_doc.chunks],
                "chunks_count": len(cached_doc.chunks),
                "processing_time": time.time() - start_time,
                "from_cache": True
            }
        
        try:
            # THREAD 1: Document Download (Target: 1s)
            download_start = time.time()
            pdf_content = await self._download_document(document_url)
            download_time = time.time() - download_start
            
            logger.info(f"Document downloaded in {download_time:.2f}s",
                       extra={"request_id": request_id, "size_mb": len(pdf_content) / 1024 / 1024})
            
            # THREAD 2 & 3: Parallel Text Extraction and Chunking (Target: 2s)
            extraction_start = time.time()
            # Run text extraction in thread pool
            loop = asyncio.get_event_loop()
            text_content, page_metadata = await loop.run_in_executor(
                self.executor,
                self._extract_text_from_pdf,
                pdf_content
            )
            
            extraction_time = time.time() - extraction_start
            logger.info(f"Text extracted in {extraction_time:.2f}s",
                       extra={"request_id": request_id, "pages": len(page_metadata)})
            
            # THREAD 3: Chunking (overlapped with extraction)
            chunking_start = time.time()
            chunks = await loop.run_in_executor(
                self.executor,
                self._create_chunks,
                text_content,
                page_metadata
            )
            
            chunking_time = time.time() - chunking_start
            logger.info(f"Text chunked in {chunking_time:.2f}s",
                       extra={"request_id": request_id, "chunks": len(chunks)})
            
            # THREAD 4: Embedding Generation + Storage (Target: 2s) - CRITICAL FIX
            embedding_start = time.time()
            # Generate embeddings and store them in vector store
            await self._generate_embeddings_and_store(chunks, document_id, request_id)
            embedding_time = time.time() - embedding_start
            
            logger.info(f"Embeddings generated and stored in {embedding_time:.2f}s",
                       extra={"request_id": request_id})
            
            # Create document data
            document_data = DocumentData(
                document_id=document_id,
                chunks=chunks,
                total_pages=len(page_metadata),
                total_chunks=len(chunks),
                processing_time=time.time() - start_time,
                metadata={
                    "url": document_url,
                    "size_bytes": len(pdf_content),
                    "download_time": download_time,
                    "extraction_time": extraction_time,
                    "chunking_time": chunking_time,
                    "embedding_time": embedding_time
                }
            )
            
            # Cache the document
            if settings.ENABLE_DOCUMENT_CACHE:
                self.document_cache[document_id] = document_data
            
            total_time = time.time() - start_time
            logger.info(f"Document processing completed in {total_time:.2f}s",
                       extra={"request_id": request_id})
            
            return {
                "document_id": document_id,
                "chunks": [chunk.dict() for chunk in chunks],
                "chunks_count": len(chunks),
                "processing_time": total_time,
                "from_cache": False
            }
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}",
                        extra={"request_id": request_id, "url": document_url})
            raise
    
    async def _download_document(self, url: str) -> bytes:
        """Download document from URL with streaming"""
        if not self.session:
            await self.initialize()
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download document: HTTP {response.status}")
                
                content_length = response.headers.get('content-length')
                if content_length:
                    size_mb = int(content_length) / 1024 / 1024
                    if size_mb > settings.MAX_DOCUMENT_SIZE_MB:
                        raise Exception(f"Document too large: {size_mb:.1f}MB > {settings.MAX_DOCUMENT_SIZE_MB}MB")
                
                # Stream download with chunks
                chunks = []
                async for chunk in response.content.iter_chunked(8192):
                    chunks.append(chunk)
                
                return b''.join(chunks)
                
        except aiohttp.ClientError as e:
            raise Exception(f"Network error downloading document: {str(e)}")
    
    def _extract_text_from_pdf(self, pdf_content: bytes) -> tuple[str, List[Dict]]:
        """Extract text from PDF using PyMuPDF for speed"""
        try:
            # Try PyMuPDF first (faster)
            pdf_stream = io.BytesIO(pdf_content)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            text_content = ""
            page_metadata = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                page_metadata.append({
                    "page_number": page_num + 1,
                    "text_length": len(page_text),
                    "start_char": len(text_content),
                    "end_char": len(text_content) + len(page_text)
                })
                text_content += page_text + "\n\n"
            
            doc.close()
            return text_content, page_metadata
            
        except Exception as e:
            logger.warning(f"PyMuPDF failed, falling back to PyPDF2: {str(e)}")
            # Fallback to PyPDF2
            try:
                pdf_stream = io.BytesIO(pdf_content)
                reader = PdfReader(pdf_stream)
                text_content = ""
                page_metadata = []
                
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    page_metadata.append({
                        "page_number": page_num + 1,
                        "text_length": len(page_text),
                        "start_char": len(text_content),
                        "end_char": len(text_content) + len(page_text)
                    })
                    text_content += page_text + "\n\n"
                
                return text_content, page_metadata
                
            except Exception as e2:
                raise Exception(f"Failed to extract text from PDF: {str(e2)}")
    
    def _create_chunks(self, text: str, page_metadata: List[Dict]) -> List[DocumentChunk]:
        """Create text chunks with overlap and metadata"""
        chunks = []
        chunk_size = settings.CHUNK_SIZE
        overlap = settings.CHUNK_OVERLAP
        
        # Simple word-based chunking
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_text.strip()) < 50:  # Skip very short chunks
                continue
            
            # Find which page(s) this chunk belongs to
            chunk_start = len(' '.join(words[:i]))
            chunk_end = chunk_start + len(chunk_text)
            
            relevant_pages = []
            for page_meta in page_metadata:
                if (chunk_start < page_meta["end_char"] and 
                    chunk_end > page_meta["start_char"]):
                    relevant_pages.append(page_meta["page_number"])
            
            chunk = DocumentChunk(
                content=chunk_text,
                page_number=relevant_pages[0] if relevant_pages else None,
                chunk_index=len(chunks),
                metadata={
                    "word_count": len(chunk_words),
                    "char_start": chunk_start,
                    "char_end": chunk_end,
                    "pages": relevant_pages
                }
            )
            
            chunks.append(chunk)
            
            if len(chunks) >= settings.MAX_CHUNKS_PER_DOCUMENT:
                logger.warning(f"Reached maximum chunks limit: {settings.MAX_CHUNKS_PER_DOCUMENT}")
                break
        
        return chunks
    
    async def _generate_embeddings_and_store(self, chunks: List[DocumentChunk], document_id: str, request_id: str):
        """Generate embeddings and store them in vector store - CRITICAL FIX"""
        if not chunks:
            return

        # Extract text for embedding
        texts = [chunk.content for chunk in chunks]

        # Generate embeddings using vector store
        embeddings = await self.vector_store.generate_embeddings(texts)

        # Assign embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        # CRITICAL FIX: Store embeddings in request-specific vector store
        await self.vector_store.store_document_chunks_for_request(
            request_id=request_id,
            document_id=document_id, 
            chunks=chunks
        )
        
        logger.info(f"Generated and stored {len(chunks)} chunks with embeddings for request {request_id}")
