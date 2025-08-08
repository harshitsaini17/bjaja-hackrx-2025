"""
Fixed Vector Store Service - Using single persistent Pinecone index
Based on successful test results
"""

import asyncio
import hashlib
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor

# Vector store backends
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.config import settings
from src.models.api_models import DocumentChunk

logger = logging.getLogger(__name__)

class VectorStoreService:
    """Fixed Vector store service using single persistent Pinecone index"""
    
    def __init__(self):
        self.embedding_model = None
        self.dimension = 1536  # OpenAI embedding dimension
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.use_pinecone = PINECONE_AVAILABLE and settings.PINECONE_API_KEY
        
        # Single persistent index
        self.pc = None
        self.index = None
        self.openai_client = None
        self.index_name = "hackrx-main-index"  # Single persistent index
    
    async def initialize(self):
        """Initialize the vector store service"""
        logger.info("Initializing Fixed Vector Store Service")
        
        # Initialize embedding model
        await self._initialize_embedding_model()
        
        # Initialize Pinecone connection and index
        if self.use_pinecone:
            await self._initialize_pinecone()
        else:
            raise Exception("This fixed version requires Pinecone")
        
        logger.info(f"Fixed vector store initialized with Pinecone index: {self.index_name}")
    
    async def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
    
    async def _initialize_embedding_model(self):
        """Initialize the embedding model"""
        if OPENAI_AVAILABLE and settings.AZURE_OPENAI_API_KEY:
            self.embedding_model = "azure_openai"
            self.openai_client = AzureOpenAI(
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
            )
            logger.info("Configured Azure OpenAI embeddings with 1536 dimensions")
        else:
            raise Exception("Azure OpenAI API key not available")
    
    async def _initialize_pinecone(self):
        """Initialize Pinecone with single persistent index"""
        try:
            # Initialize Pinecone
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            logger.info("Connected to Pinecone")
            
            # Check if main index exists
            existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating main Pinecone index: {self.index_name}")
                
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                
                # Wait for index to be ready
                time.sleep(15)
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            
            # Get index stats
            stats = self.index.describe_index_stats()
            logger.info(f"Connected to Pinecone index '{self.index_name}' with {stats['total_vector_count']} vectors")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if not texts:
            return []
        
        try:
            # Use Azure OpenAI embeddings
            embeddings = await self._generate_openai_embeddings(texts)
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise
    
    async def _generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Azure OpenAI"""
        try:
            # Batch process texts
            batch_size = 16
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                response = self.openai_client.embeddings.create(
                    model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                    input=batch_texts
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {str(e)}")
            raise
    
    async def store_document_chunks_for_request(self, request_id: str, document_id: str, chunks: List[DocumentChunk]) -> int:
        """Store document chunks in the main Pinecone index"""
        try:
            logger.info(f"Storing {len(chunks)} chunks for request {request_id}, document {document_id}")
            
            # Prepare vectors for batch upload
            vectors_to_upsert = []
            
            # Generate embeddings for all chunk contents
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = await self.generate_embeddings(chunk_texts)
            
            for chunk, embedding in zip(chunks, embeddings):
                # Create unique vector ID with request prefix
                vector_id = f"{request_id}_{chunk.chunk_index}"
                
                # Prepare metadata
                metadata = {
                    "request_id": request_id,
                    "document_id": document_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content[:1000],  # Limit for metadata
                    "page_number": chunk.page_number or 0,
                    "upload_time": time.time()
                }
                
                vectors_to_upsert.append((vector_id, embedding, metadata))
            
            # Batch upload to Pinecone
            if vectors_to_upsert:
                self.index.upsert(vectors=vectors_to_upsert)
                logger.info(f"Successfully stored {len(vectors_to_upsert)} chunks for request {request_id}, document {document_id}")
            
            return len(vectors_to_upsert)
            
        except Exception as e:
            logger.error(f"Failed to store chunks for request {request_id}, document {document_id}: {str(e)}")
            raise
    
    async def search_similar_chunks_for_request(self, request_id: str, query: str, document_id: str = None, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks within a specific request"""
        try:
            logger.info(f"Searching for '{query}' in request {request_id} (top_k={top_k})")
            
            # Generate query embedding
            query_embeddings = await self.generate_embeddings([query])
            query_embedding = query_embeddings[0]
            
            # Search with request filter
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k * 2,  # Get more results to filter
                include_metadata=True,
                filter={"request_id": request_id}
            )
            
            # Process results and filter by similarity threshold
            results = []
            for match in search_results['matches']:
                score = match['score']
                
                # Apply similarity threshold
                if score >= settings.SIMILARITY_THRESHOLD:
                    metadata = match['metadata']
                    
                    # Reconstruct DocumentChunk
                    chunk = DocumentChunk(
                        content=metadata.get('content', ''),
                        page_number=metadata.get('page_number', 0),
                        chunk_index=metadata.get('chunk_index', 0),
                        metadata={}
                    )
                    
                    results.append((chunk, score))
                    logger.debug(f"Found chunk {chunk.chunk_index} with score {score:.3f}")
            
            # Sort by score and limit results
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:top_k]
            
            logger.info(f"Found {len(results)} relevant chunks (threshold: {settings.SIMILARITY_THRESHOLD})")
            return results
            
        except Exception as e:
            logger.error(f"Search failed for request {request_id}: {str(e)}")
            return []
    
    async def delete_request_data(self, request_id: str) -> int:
        """Delete all data for a specific request"""
        try:
            # Query all vectors for this request
            search_results = self.index.query(
                vector=[0.0] * self.dimension,  # Dummy vector
                top_k=10000,  # Large number to get all
                include_metadata=True,
                filter={"request_id": request_id}
            )
            
            # Extract IDs and delete
            ids_to_delete = [match['id'] for match in search_results['matches']]
            
            if ids_to_delete:
                self.index.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks for request {request_id}")
                return len(ids_to_delete)
            else:
                logger.info(f"No chunks found for request {request_id}")
                return 0
                
        except Exception as e:
            logger.error(f"Failed to delete data for request {request_id}: {str(e)}")
            return 0
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get current index statistics"""
        try:
            stats = self.index.describe_index_stats()
            
            # Convert to serializable format
            return {
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness,
                'metric': stats.metric,
                'total_vector_count': stats.total_vector_count,
                'namespaces': dict(stats.namespaces) if stats.namespaces else {}
            }
            
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {}
