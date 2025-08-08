"""
Core configuration settings for the LLM Document Retrieval System
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
    
    # Document Processing Configuration
    MAX_DOCUMENT_SIZE_MB: int = 50
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MAX_CHUNKS_PER_DOCUMENT: int = 500
    
    # LLM Configuration
    LLM_MAX_TOKENS: int = 500  # Increased for better responses
    LLM_TEMPERATURE: float = 0.1
    LLM_TIMEOUT_SECONDS: int = 15  # Increased timeout for LangChain reliability
    MAX_RETRIES: int = 2
    
    # Performance Configuration
    MAX_CONCURRENT_QUESTIONS: int = 10
    REQUEST_TIMEOUT_SECONDS: int = 30
    TARGET_RESPONSE_TIME_SECONDS: int = 25
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    SIMILARITY_THRESHOLD: float = 0.05  # Lower threshold for better results
    TOP_K_RESULTS: int = 5
    
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
    
    # Error Handling
    MAX_RETRY_ATTEMPTS: int = 3
    RETRY_DELAY_SECONDS: float = 1.0
    
    # Domain-specific Configuration
    INSURANCE_KEYWORDS: list = [
        "premium", "coverage", "deductible", "claim", "policy", "benefit",
        "exclusion", "waiting period", "grace period", "maternity", "pre-existing"
    ]
    
    LEGAL_KEYWORDS: list = [
        "contract", "agreement", "clause", "liability", "indemnity", "warranty",
        "breach", "termination", "dispute", "jurisdiction", "governing law"
    ]
    
    HR_KEYWORDS: list = [
        "employee", "salary", "benefits", "leave", "termination", "performance",
        "promotion", "disciplinary", "grievance", "policy", "procedure"
    ]
    
    COMPLIANCE_KEYWORDS: list = [
        "regulation", "compliance", "audit", "reporting", "disclosure",
        "penalty", "violation", "requirement", "standard", "certification"
    ]

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
