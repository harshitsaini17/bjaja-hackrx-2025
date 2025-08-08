"""
Enhanced Vector Store Service with Hybrid Retrieval
Combines semantic search + keyword matching + reranking
OPTIMIZED VERSION - Targets 80-90% accuracy improvement
"""

import asyncio
import hashlib
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
from concurrent.futures import ThreadPoolExecutor
import re
from collections import Counter

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

# For BM25 keyword search
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.config import settings
from src.models.api_models import DocumentChunk

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Advanced hybrid retrieval combining semantic + keyword search"""
    
    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.bm25 = None
        self._setup_bm25_index()
    
    def _setup_bm25_index(self):
        """Setup BM25 index for keyword search"""
        if not BM25_AVAILABLE:
            logger.warning("BM25 not available, using simplified keyword matching")
            return
        
        try:
            # Tokenize documents for BM25
            tokenized_docs = []
            for chunk in self.chunks:
                # Simple tokenization
                tokens = re.findall(r'\b\w+\b', chunk.content.lower())
                tokenized_docs.append(tokens)
            
            self.bm25 = BM25Okapi(tokenized_docs)
            logger.info(f"BM25 index created with {len(tokenized_docs)} documents")
        except Exception as e:
            logger.error(f"Failed to create BM25 index: {e}")
    
    def keyword_search(self, query: str, top_k: int = 20) -> List[Tuple[DocumentChunk, float]]:
        """Perform keyword-based search using BM25"""
        if not self.bm25:
            # Fallback to simple keyword matching
            return self._simple_keyword_search(query, top_k)
        
        try:
            # Tokenize query
            query_tokens = re.findall(r'\b\w+\b', query.lower())
            
            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)
            
            # Combine with chunks and sort
            chunk_scores = [(chunk, score) for chunk, score in zip(self.chunks, scores)]
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            
            return chunk_scores[:top_k]
        
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return self._simple_keyword_search(query, top_k)
    
    def _simple_keyword_search(self, query: str, top_k: int) -> List[Tuple[DocumentChunk, float]]:
        """Fallback simple keyword matching"""
        query_words = set(query.lower().split())
        chunk_scores = []
        
        for chunk in self.chunks:
            chunk_words = set(chunk.content.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            score = overlap / len(query_words) if query_words else 0
            chunk_scores.append((chunk, score))
        
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        return chunk_scores[:top_k]
    
    @staticmethod
    def reciprocal_rank_fusion(
        semantic_results: List[Tuple[DocumentChunk, float]], 
        keyword_results: List[Tuple[DocumentChunk, float]], 
        k: int = 60
    ) -> List[Tuple[DocumentChunk, float]]:
        """Combine results using Reciprocal Rank Fusion"""
        
        # Create dictionaries for efficient lookup
        semantic_dict = {id(chunk): (chunk, score) for chunk, score in semantic_results}
        keyword_dict = {id(chunk): (chunk, score) for chunk, score in keyword_results}
        
        # Get all unique chunks
        all_chunk_ids = set(semantic_dict.keys()) | set(keyword_dict.keys())
        
        # Calculate RRF scores
        fused_scores = {}
        
        for chunk_id in all_chunk_ids:
            rrf_score = 0.0
            
            # Add semantic rank contribution
            if chunk_id in semantic_dict:
                semantic_rank = next(
                    (i for i, (c, _) in enumerate(semantic_results) if id(c) == chunk_id), 
                    len(semantic_results)
                )
                rrf_score += settings.SEMANTIC_WEIGHT / (k + semantic_rank + 1)
            
            # Add keyword rank contribution
            if chunk_id in keyword_dict:
                keyword_rank = next(
                    (i for i, (c, _) in enumerate(keyword_results) if id(c) == chunk_id), 
                    len(keyword_results)
                )
                rrf_score += settings.BM25_WEIGHT / (k + keyword_rank + 1)
            
            # Store the chunk and score
            chunk = semantic_dict.get(chunk_id, keyword_dict.get(chunk_id))[0]
            fused_scores[chunk_id] = (chunk, rrf_score)
        
        # Sort by RRF score
        sorted_results = sorted(fused_scores.values(), key=lambda x: x[1], reverse=True)
        return sorted_results

class QueryExpander:
    """Enhanced query expansion using domain knowledge"""
    
    def __init__(self):
        self.domain_synonyms = self._build_domain_synonyms()
    
    def _build_domain_synonyms(self) -> Dict[str, Set[str]]:
        """Build domain-specific synonym mappings"""
        return {
            "insurance": {
                "premium": {"payment", "cost", "fee", "charge"},
                "coverage": {"benefit", "protection", "insurance", "plan"},
                "claim": {"reimbursement", "payment", "settlement"},
                "waiting period": {"waiting time", "delay", "deferment"},
                "grace period": {"grace time", "extension", "buffer"},
                "deductible": {"excess", "copay", "out-of-pocket"},
                "exclusion": {"exception", "limitation", "restriction"},
                "pre-existing": {"prior condition", "existing condition"},
                "maternity": {"pregnancy", "childbirth", "delivery"},
                "surgery": {"operation", "procedure", "treatment"}
            },
            "legal": {
                "contract": {"agreement", "document", "terms"},
                "clause": {"provision", "section", "term"},
                "liability": {"responsibility", "obligation", "duty"},
                "breach": {"violation", "default", "non-compliance"},
                "termination": {"ending", "cancellation", "closure"}
            },
            "hr": {
                "employee": {"worker", "staff", "personnel"},
                "salary": {"wage", "compensation", "pay"},
                "benefits": {"perks", "allowances", "extras"},
                "leave": {"time off", "absence", "vacation"},
                "performance": {"evaluation", "review", "assessment"}
            },
            "compliance": {
                "regulation": {"rule", "requirement", "standard"},
                "audit": {"review", "inspection", "examination"},
                "violation": {"breach", "infringement", "non-compliance"},
                "penalty": {"fine", "punishment", "sanction"}
            }
        }
    
    def expand_query(self, query: str, domain: str = "general") -> str:
        """Expand query with domain-specific synonyms"""
        if not settings.ENABLE_QUERY_EXPANSION:
            return query
        
        expanded_terms = set()
        query_lower = query.lower()
        
        # Add domain-specific synonyms
        if domain in self.domain_synonyms:
            for term, synonyms in self.domain_synonyms[domain].items():
                if term in query_lower:
                    expanded_terms.update(list(synonyms)[:2])  # Add top 2 synonyms
        
        # Add domain keywords if relevant
        domain_keywords = {
            "insurance": settings.INSURANCE_KEYWORDS,
            "legal": settings.LEGAL_KEYWORDS,
            "hr": settings.HR_KEYWORDS,
            "compliance": settings.COMPLIANCE_KEYWORDS
        }.get(domain, [])
        
        for keyword in domain_keywords:
            if keyword.lower() in query_lower:
                expanded_terms.add(keyword)
        
        # Combine original query with expanded terms
        if expanded_terms:
            additional_terms = " ".join(list(expanded_terms)[:settings.QUERY_EXPANSION_TERMS])
            return f"{query} {additional_terms}"
        
        return query

class VectorStoreService:
    """Enhanced Vector store service with hybrid retrieval capabilities"""
    
    def __init__(self):
        self.embedding_model = None
        self.dimension = 1536  # OpenAI embedding dimension
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.use_pinecone = PINECONE_AVAILABLE and settings.PINECONE_API_KEY
        
        # Single persistent index
        self.pc = None
        self.index = None
        self.openai_client = None
        self.index_name = "hackrx-main-index"
        
        # Enhanced retrieval components
        self.query_expander = QueryExpander()
        self.request_retrievers: Dict[str, HybridRetriever] = {}
    
    async def initialize(self):
        """Initialize the enhanced vector store service"""
        logger.info("Initializing Enhanced Vector Store Service")
        
        # Initialize embedding model
        await self._initialize_embedding_model()
        
        # Initialize Pinecone connection and index
        if self.use_pinecone:
            await self._initialize_pinecone()
        else:
            raise Exception("This enhanced version requires Pinecone")
        
        logger.info(f"Enhanced vector store initialized with Pinecone index: {self.index_name}")
    
    async def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        self.request_retrievers.clear()
    
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
    
    async def store_document_chunks_for_request(
        self, 
        request_id: str, 
        document_id: str, 
        chunks: List[DocumentChunk]
    ) -> int:
        """Store document chunks and setup hybrid retriever"""
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
                logger.info(f"Successfully stored {len(vectors_to_upsert)} chunks for request {request_id}")
            
            # Setup hybrid retriever for this request
            if settings.ENABLE_HYBRID_RETRIEVAL and BM25_AVAILABLE:
                self.request_retrievers[request_id] = HybridRetriever(chunks)
                logger.info(f"Hybrid retriever setup complete for request {request_id}")
            
            return len(vectors_to_upsert)
            
        except Exception as e:
            logger.error(f"Failed to store chunks for request {request_id}: {str(e)}")
            raise
    
    def _detect_query_domain(self, query: str) -> str:
        """Detect the domain of a query based on keywords"""
        query_lower = query.lower()
        domain_scores = {}
        
        domains = {
            "insurance": settings.INSURANCE_KEYWORDS,
            "legal": settings.LEGAL_KEYWORDS,
            "hr": settings.HR_KEYWORDS,
            "compliance": settings.COMPLIANCE_KEYWORDS
        }
        
        for domain, keywords in domains.items():
            score = sum(1 for keyword in keywords if keyword.lower() in query_lower)
            domain_scores[domain] = score
        
        # Return domain with highest score, default to 'general'
        best_domain = max(domain_scores.items(), key=lambda x: x[1])
        return best_domain[0] if best_domain[1] > 0 else "general"
    
    async def search_similar_chunks_for_request(
        self, 
        request_id: str, 
        query: str, 
        document_id: str = None, 
        top_k: int = 5
    ) -> List[Tuple[DocumentChunk, float]]:
        """Enhanced hybrid search with query expansion and reranking"""
        try:
            logger.info(f"Enhanced search for '{query}' in request {request_id} (top_k={top_k})")
            
            # Step 1: Query expansion
            domain = self._detect_query_domain(query)
            expanded_query = self.query_expander.expand_query(query, domain)
            
            if expanded_query != query:
                logger.debug(f"Query expanded from '{query}' to '{expanded_query}'")
            
            # Step 2: Multi-stage retrieval
            if settings.ENABLE_HYBRID_RETRIEVAL and request_id in self.request_retrievers:
                results = await self._hybrid_search(
                    request_id, expanded_query, top_k * 2
                )
            else:
                results = await self._semantic_search_only(
                    request_id, expanded_query, top_k * 2
                )
            
            # Step 3: Apply adaptive threshold
            filtered_results = self._apply_adaptive_threshold(results, query, domain)
            
            # Step 4: Limit to requested number
            final_results = filtered_results[:top_k]
            
            logger.info(f"Found {len(final_results)} relevant chunks (threshold: {settings.SIMILARITY_THRESHOLD})")
            return final_results
            
        except Exception as e:
            logger.error(f"Enhanced search failed for request {request_id}: {str(e)}")
            return []
    
    async def _hybrid_search(
        self, 
        request_id: str, 
        query: str, 
        top_k: int
    ) -> List[Tuple[DocumentChunk, float]]:
        """Perform hybrid semantic + keyword search"""
        
        # Semantic search
        semantic_results = await self._semantic_search_only(request_id, query, top_k)
        
        # Keyword search using hybrid retriever
        hybrid_retriever = self.request_retrievers[request_id]
        keyword_results = hybrid_retriever.keyword_search(query, top_k)
        
        # Fusion using Reciprocal Rank Fusion
        fused_results = HybridRetriever.reciprocal_rank_fusion(
            semantic_results, keyword_results
        )
        
        logger.debug(f"Hybrid search: {len(semantic_results)} semantic + {len(keyword_results)} keyword → {len(fused_results)} fused")
        
        return fused_results[:top_k]
    
    async def _semantic_search_only(
        self, 
        request_id: str, 
        query: str, 
        top_k: int
    ) -> List[Tuple[DocumentChunk, float]]:
        """Perform semantic search only"""
        
        # Generate query embedding
        query_embeddings = await self.generate_embeddings([query])
        query_embedding = query_embeddings[0]
        
        # Search with request filter
        search_results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter={"request_id": request_id}
        )
        
        # Process results
        results = []
        for match in search_results['matches']:
            score = match['score']
            metadata = match['metadata']
            
            # Reconstruct DocumentChunk
            chunk = DocumentChunk(
                content=metadata.get('content', ''),
                page_number=metadata.get('page_number', 0),
                chunk_index=metadata.get('chunk_index', 0),
                metadata={}
            )
            
            results.append((chunk, score))
        
        return results
    
    def _apply_adaptive_threshold(
        self, 
        results: List[Tuple[DocumentChunk, float]], 
        query: str, 
        domain: str
    ) -> List[Tuple[DocumentChunk, float]]:
        """Apply adaptive similarity threshold based on query and domain"""
        
        if not results:
            return []
        
        # Start with configured threshold
        threshold = settings.SIMILARITY_THRESHOLD
        
        # Adjust threshold based on domain
        domain_adjustments = {
            "insurance": 0.05,  # Slightly higher for insurance
            "legal": 0.03,      # Slightly higher for legal
            "hr": 0.02,         # Normal for HR
            "compliance": 0.04  # Higher for compliance
        }
        
        threshold += domain_adjustments.get(domain, 0.0)
        
        # Adjust based on query complexity
        query_words = len(query.split())
        if query_words > 10:  # Complex queries
            threshold -= 0.02
        elif query_words < 4:  # Simple queries
            threshold += 0.02
        
        # Apply fallback if no results above threshold
        filtered_results = [(chunk, score) for chunk, score in results if score >= threshold]
        
        if not filtered_results and settings.ENABLE_FALLBACK_THRESHOLD:
            logger.warning(f"No results above adaptive threshold {threshold:.3f}, using fallback")
            threshold = settings.FALLBACK_SIMILARITY_THRESHOLD
            filtered_results = [(chunk, score) for chunk, score in results if score >= threshold]
        
        # Always return at least the top result if available
        if not filtered_results and results:
            filtered_results = [results[0]]
            logger.info(f"Using top result with score {results[0][1]:.3f} below threshold")
        
        return filtered_results
    
    async def delete_request_data(self, request_id: str) -> int:
        """Delete all data for a specific request"""
        try:
            # Query all vectors for this request
            search_results = self.index.query(
                vector=[0.0] * self.dimension,
                top_k=10000,
                include_metadata=True,
                filter={"request_id": request_id}
            )
            
            # Extract IDs and delete
            ids_to_delete = [match['id'] for match in search_results['matches']]
            if ids_to_delete:
                self.index.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks for request {request_id}")
            
            # Clean up hybrid retriever
            if request_id in self.request_retrievers:
                del self.request_retrievers[request_id]
            
            return len(ids_to_delete)
            
        except Exception as e:
            logger.error(f"Failed to delete data for request {request_id}: {str(e)}")
            return 0
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get current index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness,
                'metric': stats.metric,
                'total_vector_count': stats.total_vector_count,
                'namespaces': dict(stats.namespaces) if stats.namespaces else {},
                'active_hybrid_retrievers': len(self.request_retrievers)
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {}
