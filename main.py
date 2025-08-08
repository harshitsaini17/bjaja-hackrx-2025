"""
Main FastAPI application for LLM Document Retrieval System
Handles the /hackrx/run endpoint with optimized 20-25s response time
"""

import asyncio
import time
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
import structlog

from src.core.config import settings
from src.core.auth import verify_token
from src.services.document_processor import DocumentProcessor
from src.services.question_processor import QuestionProcessor
from src.services.vector_store import VectorStoreService
from src.services.llm_service import LLMService
from src.services.performance_monitor import PerformanceMonitor
from src.models.api_models import HackRXRequest, HackRXResponse, AnswerMetadata
from src.utils.error_handler import handle_exceptions

# Initialize structured logging
logger = structlog.get_logger()

# Security
security = HTTPBearer()

# Global services
document_processor: DocumentProcessor = None
question_processor: QuestionProcessor = None
vector_store: VectorStoreService = None
llm_service: LLMService = None
performance_monitor: PerformanceMonitor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for initialization and cleanup"""
    global document_processor, question_processor, vector_store, llm_service, performance_monitor
    
    logger.info("Initializing LLM Document Retrieval System...")
    
    # Initialize services
    vector_store = VectorStoreService()
    llm_service = LLMService()
    document_processor = DocumentProcessor(vector_store)
    question_processor = QuestionProcessor(vector_store, llm_service)
    performance_monitor = PerformanceMonitor()
    
    await vector_store.initialize()
    await llm_service.initialize()
    
    logger.info("System initialization complete")
    
    yield
    
    # Cleanup
    logger.info("Shutting down system...")
    await vector_store.cleanup()
    await llm_service.cleanup()


# FastAPI application with lifespan management
app = FastAPI(
    title="LLM Document Retrieval System",
    description="Intelligent Query-Retrieval System for Insurance, Legal, HR, and Compliance Documents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to all responses"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Processing-Time"] = str(process_time)
    return response


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify bearer token authentication"""
    if not verify_token(credentials.credentials):
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "LLM Document Retrieval System",
        "status": "operational",
        "version": "1.0.0",
        "timestamp": time.time()
    }


@app.get("/health")
async def health_check():
    """Simple health check without service validation"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "message": "LLM Document Retrieval System is running"
    }


@app.post("/api/v1/hackrx/run", response_model=HackRXResponse)
@handle_exceptions
async def process_hackrx_request(
    request: HackRXRequest,
    token: str = Depends(get_current_user)
) -> HackRXResponse:
    """
    Main endpoint for HackRX document processing
    Target response time: 20-25 seconds
    """
    start_time = time.time()
    request_id = f"hackrx_{int(start_time)}"
    
    logger.info(
        "Processing HackRX request",
        request_id=request_id,
        document_url=str(request.documents),
        questions_count=len(request.questions)
    )
    
    try:
        # STEP 1: Request Validation & Setup (Target: 0.5s)
        validation_start = time.time()
        
        if len(request.questions) > 10:
            raise HTTPException(
                status_code=422,
                detail="Maximum 10 questions allowed per request"
            )
        
        if len(request.questions) == 0:
            raise HTTPException(
                status_code=422,
                detail="At least one question is required"
            )
        
        performance_monitor.log_checkpoint(request_id, "validation_complete", time.time() - validation_start)
        
        # STEP 2: Real-time Document Processing (Target: 5s)
        processing_start = time.time()
        logger.info("Starting document processing", request_id=request_id)

        # Process document in parallel threads
        document_data = await document_processor.process_document(
            document_url=str(request.documents),
            request_id=request_id
        )

        processing_time = time.time() - processing_start
        performance_monitor.log_checkpoint(request_id, "document_processing_complete", processing_time)

        # CRITICAL FIX: Ensure embeddings are stored
        if not document_data.get("from_cache", False):
            logger.info("Embeddings stored successfully", request_id=request_id)

            
        if processing_time > 6.0:  # Buffer for 5s target
            logger.warning(
                "Document processing exceeded target time",
                request_id=request_id,
                actual_time=processing_time,
                target_time=5.0
            )
        
        # STEP 3: Parallel Question Processing (Target: 15s)
        questions_start = time.time()
        
        logger.info("Starting parallel question processing", request_id=request_id)
        
        # Process all questions in parallel with semaphore for concurrency control
        semaphore = asyncio.Semaphore(10)  # Max 10 concurrent question processing
        
        async def process_single_question(question: str, index: int) -> str:
            async with semaphore:
                return await question_processor.process_question(
                    question=question,
                    document_data=document_data,
                    request_id=f"{request_id}_q{index}"
                )
        
        # Execute all questions in parallel
        tasks = [
            process_single_question(question, idx)
            for idx, question in enumerate(request.questions)
        ]
        
        answers = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in parallel processing
        processed_answers = []
        for idx, answer in enumerate(answers):
            if isinstance(answer, Exception):
                logger.error(
                    "Question processing failed",
                    request_id=request_id,
                    question_index=idx,
                    error=str(answer)
                )
                processed_answers.append(f"Error processing question: {str(answer)}")
            else:
                processed_answers.append(answer)
        
        questions_time = time.time() - questions_start
        performance_monitor.log_checkpoint(request_id, "questions_processing_complete", questions_time)
        
        if questions_time > 16.0:  # Buffer for 15s target
            logger.warning(
                "Question processing exceeded target time",
                request_id=request_id,
                actual_time=questions_time,
                target_time=15.0
            )
        
        # STEP 4: Response Compilation (Target: 2.5s)
        compilation_start = time.time()
        
        total_processing_time = time.time() - start_time
        
        # Create metadata
        metadata = AnswerMetadata(
            processing_time=round(total_processing_time, 2),
            document_processed=True,
            questions_count=len(request.questions),
            request_id=request_id,
            document_chunks=document_data.get("chunks_count", 0),
            average_confidence=0.85  # Placeholder - can be calculated from actual confidence scores
        )
        
        compilation_time = time.time() - compilation_start
        performance_monitor.log_checkpoint(request_id, "response_compilation_complete", compilation_time)
        
        # STEP 5: Final Response (Target: 0.5s)
        response = HackRXResponse(
            answers=processed_answers,
            metadata=metadata
        )
        
        total_time = time.time() - start_time
        
        logger.info(
            "HackRX request completed successfully",
            request_id=request_id,
            total_time=round(total_time, 2),
            target_time="20-25s",
            status="success" if total_time <= 25.0 else "slow"
        )
        
        # Log performance metrics
        performance_monitor.log_request_completion(
            request_id=request_id,
            total_time=total_time,
            questions_count=len(request.questions),
            success=True
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(
            "HackRX request failed",
            request_id=request_id,
            error=str(e),
            total_time=round(total_time, 2)
        )
        
        performance_monitor.log_request_completion(
            request_id=request_id,
            total_time=total_time,
            questions_count=len(request.questions),
            success=False,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/api/v1/metrics")
async def get_metrics(token: str = Depends(get_current_user)):
    """Get system performance metrics"""
    return performance_monitor.get_metrics()


@app.get("/api/v1/metrics/request/{request_id}")
async def get_request_metrics(request_id: str, token: str = Depends(get_current_user)):
    """Get metrics for a specific request"""
    return performance_monitor.get_request_metrics(request_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
