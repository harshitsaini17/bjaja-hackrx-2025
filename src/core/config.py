"""
Core configuration settings for the LLM Document Retrieval System
OPTIMIZED VERSION - Improved for better retrieval accuracy
"""

import os
from typing import Optional

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "LLM Document Retrieval System"
    VERSION: str = "1.0.0"
    
    # Authentication
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Valid Bearer Token (from specs)
    VALID_BEARER_TOKEN: str = "45373a6a634dff62b774382b7d01e401e8d1c711debf1d1d6c465b7a898e7620"
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
    AZURE_OPENAI_API_DEPLOYMENT_NAME: str = os.getenv("AZURE_OPENAI_API_DEPLOYMENT_NAME", "gpt-4o-mini")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small")
    
    # Vector Store Configuration
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp-free")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "hackrx-documents")
    
    # FAISS Configuration (fallback)
    USE_FAISS: bool = os.getenv("USE_FAISS", "false").lower() == "true"
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
    
    # Document Processing Configuration - OPTIMIZED FOR SPEED
    MAX_DOCUMENT_SIZE_MB: int = 50
    CHUNK_SIZE: int = 400  # Increased from 256 for fewer chunks
    CHUNK_OVERLAP: int = 50  # Reduced from 64 for faster processing  
    MAX_CHUNKS_PER_DOCUMENT: int = 150  # Limit total chunks for speed
    
    # Advanced Chunking Configuration
    ENABLE_SEMANTIC_CHUNKING: bool = True
    SENTENCE_OVERLAP: int = 1  # Reduced overlap for speed
    MIN_CHUNK_SIZE: int = 200  # Increased minimum for fewer small chunks
    MAX_CHUNK_SIZE: int = 1000  # Increased maximum for larger chunks
    
    # LLM Configuration - OPTIMIZED
    LLM_MAX_TOKENS: int = 800  # Increased for better responses
    LLM_TEMPERATURE: float = 0.1
    LLM_TIMEOUT_SECONDS: int = 15
    MAX_RETRIES: int = 3  # Increased retries
    
    # Performance Configuration - OPTIMIZED
    MAX_CONCURRENT_QUESTIONS: int = 10
    REQUEST_TIMEOUT_SECONDS: int = 30
    TARGET_RESPONSE_TIME_SECONDS: int = 25
    
    # Embedding Configuration - OPTIMIZED FOR SPEED
    EMBEDDING_MODEL: str = "azure_openai"
    EMBEDDING_DIMENSION: int = 1536  # Fixed for OpenAI
    SIMILARITY_THRESHOLD: float = 0.15  # More permissive threshold
    TOP_K_RESULTS: int = 12  # Increased for better retrieval
    RERANK_TOP_K: int = 25  # For multi-stage retrieval
    
    # Parallel Processing Configuration
    MAX_EMBEDDING_BATCH_SIZE: int = 32  # Larger batches for parallel processing
    MAX_CONCURRENT_EMBEDDINGS: int = 4  # Parallel embedding generation
    EMBEDDING_TIMEOUT_SECONDS: int = 10  # Reduced timeout
    
    # Context Window Configuration - OPTIMIZED
    MAX_CONTEXT_TOKENS: int = 3000  # CRITICAL FIX: Larger context window
    CONTEXT_BUFFER_TOKENS: int = 500  # Reserve tokens for question/response
    ENABLE_CONTEXT_COMPRESSION: bool = True
    
    # Multi-Stage Retrieval Configuration - OPTIMIZED
    ENABLE_HYBRID_RETRIEVAL: bool = True
    ENABLE_QUERY_EXPANSION: bool = True
    ENABLE_RERANKING: bool = True
    BM25_WEIGHT: float = 0.4  # Increased weight for keyword search
    SEMANTIC_WEIGHT: float = 0.6  # Balanced semantic search
    HYBRID_TOP_K: int = 20  # Initial retrieval candidates
    FINAL_TOP_K: int = 5  # Final reranked results
    
    # MongoDB Configuration (optional for caching)
    MONGODB_URL: Optional[str] = os.getenv("MONGODB_URL")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "hackrx_cache")
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "json"
    
    # Cache Configuration
    ENABLE_DOCUMENT_CACHE: bool = True
    CACHE_TTL_HOURS: int = 24
    MAX_CACHE_SIZE_MB: int = 1000
    
    # Azure Blob Storage Configuration
    AZURE_STORAGE_CONNECTION_STRING: Optional[str] = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 10
    
    # Error Handling - OPTIMIZED
    MAX_RETRY_ATTEMPTS: int = 3
    RETRY_DELAY_SECONDS: float = 1.0
    ENABLE_FALLBACK_THRESHOLD: bool = True
    FALLBACK_SIMILARITY_THRESHOLD: float = 0.05  # Emergency fallback
    
    # Domain-specific Configuration - EXPANDED
    INSURANCE_KEYWORDS: list = [
        "premium", "coverage", "deductible", "claim", "policy", "benefit",
        "exclusion", "waiting period", "grace period", "maternity", "pre-existing",
        "copayment", "coinsurance", "network", "provider", "preauthorization",
        "rider", "endorsement", "renewal", "lapse", "reinstatement", "hospitalization",
        "surgery", "treatment", "diagnosis", "medication", "therapy", "rehabilitation"
    ]
    
    LEGAL_KEYWORDS: list = [
        "contract", "agreement", "clause", "liability", "indemnity", "warranty",
        "breach", "termination", "dispute", "jurisdiction", "governing law",
        "arbitration", "mediation", "damages", "remedy", "obligation", "right",
        "duty", "performance", "default", "force majeure", "assignment", "novation"
    ]
    
    HR_KEYWORDS: list = [
        "employee", "salary", "benefits", "leave", "termination", "performance",
        "promotion", "disciplinary", "grievance", "policy", "procedure",
        "compensation", "bonus", "incentive", "appraisal", "training", "development",
        "harassment", "discrimination", "safety", "wellness", "retirement", "health"
    ]
    
    COMPLIANCE_KEYWORDS: list = [
        "regulation", "compliance", "audit", "reporting", "disclosure",
        "penalty", "violation", "requirement", "standard", "certification",
        "governance", "oversight", "monitoring", "assessment", "risk", "control",
        "documentation", "record", "evidence", "investigation", "enforcement"
    ]
    
    # Query Enhancement Configuration
    QUERY_EXPANSION_TERMS: int = 5  # Number of terms to add per query
    ENABLE_SYNONYM_EXPANSION: bool = True
    ENABLE_DOMAIN_EXPANSION: bool = True
    
    # Answer Quality Configuration
    ENABLE_ANSWER_VALIDATION: bool = True
    MIN_ANSWER_CONFIDENCE: float = 0.3
    ENABLE_SOURCE_VERIFICATION: bool = True
    MAX_SOURCE_DISTANCE: float = 0.8  # Maximum semantic distance for source attribution

    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()
