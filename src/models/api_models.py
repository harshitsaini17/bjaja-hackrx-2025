"""
API Models for request/response handling
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl, validator
import time


class HackRXRequest(BaseModel):
    """Request model for the /hackrx/run endpoint"""
    
    documents: HttpUrl = Field(
        ...,
        description="URL to the document blob (Azure Storage)",
        example="https://hackrx.blob.core.windows.net/documents/insurance_policy.pdf"
    )
    
    questions: List[str] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="Array of questions to ask about the document",
        example=[
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?",
            "Does this policy cover maternity expenses?"
        ]
    )
    
    @validator('questions')
    def validate_questions(cls, v):
        """Validate questions array"""
        if not v:
            raise ValueError("At least one question is required")
        
        if len(v) > 10:
            raise ValueError("Maximum 10 questions allowed")
        
        for i, question in enumerate(v):
            if not question.strip():
                raise ValueError(f"Question {i+1} cannot be empty")
            
            if len(question) > 500:
                raise ValueError(f"Question {i+1} exceeds 500 character limit")
        
        return v


class AnswerMetadata(BaseModel):
    """Metadata for the response"""
    
    processing_time: float = Field(
        ...,
        description="Total processing time in seconds",
        example=23.4
    )
    
    document_processed: bool = Field(
        ...,
        description="Whether the document was successfully processed",
        example=True
    )
    
    questions_count: int = Field(
        ...,
        description="Number of questions processed",
        example=10
    )
    
    request_id: Optional[str] = Field(
        None,
        description="Unique request identifier",
        example="hackrx_1640995200"
    )
    
    document_chunks: Optional[int] = Field(
        None,
        description="Number of document chunks created",
        example=45
    )
    
    average_confidence: Optional[float] = Field(
        None,
        description="Average confidence score for answers",
        example=0.85
    )
    
    timestamp: float = Field(
        default_factory=time.time,
        description="Response timestamp"
    )


class HackRXResponse(BaseModel):
    """Response model for the /hackrx/run endpoint"""
    
    answers: List[str] = Field(
        ...,
        description="Array of answers corresponding to the input questions",
        example=[
            "Grace period of thirty days is provided for premium payment...",
            "Waiting period of thirty-six months applies for pre-existing diseases...",
            "Yes, this policy covers maternity expenses after waiting period..."
        ]
    )
    
    metadata: Optional[AnswerMetadata] = Field(
        None,
        description="Additional metadata about the processing"
    )


class DocumentChunk(BaseModel):
    """Model for document chunks"""
    
    content: str = Field(..., description="Text content of the chunk")
    page_number: Optional[int] = Field(None, description="Page number in source document")
    chunk_index: int = Field(..., description="Index of this chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")


class DocumentData(BaseModel):
    """Model for processed document data"""
    
    document_id: str = Field(..., description="Unique document identifier")
    chunks: List[DocumentChunk] = Field(..., description="Document chunks")
    total_pages: Optional[int] = Field(None, description="Total pages in document")
    total_chunks: int = Field(..., description="Total number of chunks")
    processing_time: float = Field(..., description="Document processing time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class QuestionContext(BaseModel):
    """Model for question processing context"""
    
    question: str = Field(..., description="The question being processed")
    relevant_chunks: List[DocumentChunk] = Field(..., description="Relevant document chunks")
    confidence_score: float = Field(..., description="Confidence score for the answer")
    sources: List[str] = Field(default_factory=list, description="Source references")


class ProcessingMetrics(BaseModel):
    """Model for processing performance metrics"""
    
    request_id: str
    total_time: float
    validation_time: float
    document_processing_time: float
    question_processing_time: float
    response_compilation_time: float
    questions_count: int
    success: bool
    error: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)


class HealthCheckResponse(BaseModel):
    """Model for health check response"""
    
    status: str = Field(..., description="Overall system status", example="healthy")
    timestamp: float = Field(default_factory=time.time, description="Check timestamp")
    services: Dict[str, bool] = Field(..., description="Individual service status")
    version: str = Field(..., description="System version")


class ErrorResponse(BaseModel):
    """Model for error responses"""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    request_id: Optional[str] = Field(None, description="Request identifier")
    timestamp: float = Field(default_factory=time.time, description="Error timestamp")
    status_code: int = Field(..., description="HTTP status code")
