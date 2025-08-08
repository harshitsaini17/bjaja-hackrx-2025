"""
Question Processing Service

Handles parallel question processing with vector search and LLM inference
Target: 3 seconds per question in parallel execution
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
import logging

from ..core.config import settings
from ..models.api_models import DocumentChunk, QuestionContext
from .vector_store import VectorStoreService
from .llm_service import LLMService

logger = logging.getLogger(__name__)

class QuestionProcessor:
    """Handles question processing pipeline with optimization"""
    
    def __init__(self, vector_store: VectorStoreService, llm_service: LLMService):
        self.vector_store = vector_store
        self.llm_service = llm_service
        
        # Performance tracking
        self.processing_stats = {
            "total_questions": 0,
            "successful_questions": 0,
            "failed_questions": 0,
            "average_processing_time": 0.0
        }
    
    async def initialize(self):
        """Initialize the question processor"""
        logger.info("Question Processor initialized")
        pass
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Question Processor cleanup complete")
        pass
    
    async def process_question(
        self,
        question: str,
        document_data: Dict[str, Any],
        request_id: str
    ) -> str:
        """
        Process a single question through the complete pipeline
        Target: 3 seconds total (0.1s enhance + 0.2s search + 0.1s prep + 2.5s LLM + 0.1s buffer)
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing question: {question[:50]}...",
                       extra={"request_id": request_id})
            
            # STEP 1: Query Enhancement (Target: 0.1s)
            enhancement_start = time.time()
            # Extract intent and enhance query
            intent_data = await self.llm_service.extract_intent(question)
            enhanced_query = await self.llm_service.enhance_query(question)
            enhancement_time = time.time() - enhancement_start
            
            # STEP 2: Vector Search in request-specific index (Target: 0.2s)
            search_start = time.time()
            
            # Extract main request ID (remove question suffix if exists)
            main_request_id = request_id.split('_q')[0] if '_q' in request_id else request_id
            
            # Search for relevant chunks in request-specific index - CRITICAL FIX
            similar_chunks = await self.vector_store.search_similar_chunks_for_request(
                request_id=main_request_id,
                query=enhanced_query,
                document_id=document_data.get("document_id"),
                top_k=settings.TOP_K_RESULTS
            )
            
            search_time = time.time() - search_start
            
            # STEP 3: Context Preparation (Target: 0.1s)
            prep_start = time.time()
            # Prepare context from search results
            context_chunks = self._prepare_context(
                question=question,
                search_results=similar_chunks,
                intent_data=intent_data
            )
            prep_time = time.time() - prep_start
            
            # STEP 4: LLM Inference (Target: 2.5s)
            llm_start = time.time()
            # Generate answer using LLM
            answer = await self.llm_service.generate_answer(
                question=question,
                context_chunks=context_chunks,
                request_id=request_id
            )
            llm_time = time.time() - llm_start
            
            # STEP 5: Post-processing and Validation (Target: 0.1s)
            post_start = time.time()
            # Validate and enhance answer
            final_answer = self._post_process_answer(
                answer=answer,
                question=question,
                context_chunks=context_chunks,
                intent_data=intent_data
            )
            post_time = time.time() - post_start
            
            total_time = time.time() - start_time
            
            # Update statistics
            self._update_processing_stats(total_time, True)
            
            logger.info(
                f"Question processed successfully in {total_time:.2f}s",
                extra={
                    "request_id": request_id,
                    "enhancement_time": enhancement_time,
                    "search_time": search_time,
                    "prep_time": prep_time,
                    "llm_time": llm_time,
                    "post_time": post_time,
                    "chunks_found": len(similar_chunks),
                    "chunks_used": len(context_chunks)
                }
            )
            
            return final_answer
            
        except Exception as e:
            total_time = time.time() - start_time
            self._update_processing_stats(total_time, False)
            
            logger.error(
                f"Question processing failed: {str(e)}",
                extra={"request_id": request_id, "processing_time": total_time}
            )
            
            return f"Error processing question: {str(e)}"
    
    def _prepare_context(
        self,
        question: str,
        search_results: List[tuple],
        intent_data: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Prepare and optimize context from search results
        """
        if not search_results:
            return []
        
        # Sort by relevance score
        sorted_results = sorted(search_results, key=lambda x: x[1], reverse=True)
        
        # Select best chunks based on intent and domain
        selected_chunks = []
        total_tokens = 0
        max_tokens = 2000  # Leave room for question and system prompt
        
        domain = intent_data.get("domain", "general")
        intent = intent_data.get("intent", "query")
        
        for chunk, score in sorted_results:
            # Estimate token count (rough approximation: 1 token ≈ 4 characters)
            chunk_tokens = len(chunk.content) // 4
            
            if total_tokens + chunk_tokens > max_tokens:
                break
            
            # Apply domain-specific filtering
            if self._is_chunk_relevant(chunk, question, domain, intent):
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
                
                if len(selected_chunks) >= settings.TOP_K_RESULTS:
                    break
        
        # Deduplicate similar chunks
        selected_chunks = self._deduplicate_chunks(selected_chunks)
        
        # Sort by page number for better context flow
        selected_chunks.sort(key=lambda x: (x.page_number or 0, x.chunk_index))
        
        return selected_chunks
    
    def _is_chunk_relevant(
        self,
        chunk: DocumentChunk,
        question: str,
        domain: str,
        intent: str
    ) -> bool:
        """
        Check if a chunk is relevant based on domain and intent
        """
        chunk_content_lower = chunk.content.lower()
        question_lower = question.lower()
        
        # Basic relevance check
        if len(chunk.content.strip()) < 20:
            return False
        
        # Domain-specific relevance
        domain_keywords = self.llm_service.domain_keywords.get(domain, [])
        domain_score = sum(1 for keyword in domain_keywords if keyword.lower() in chunk_content_lower)
        
        # Intent-specific relevance
        intent_bonus = 0
        if intent == "amount" and any(char.isdigit() for char in chunk.content):
            intent_bonus = 1
        elif intent == "temporal" and any(word in chunk_content_lower for word in ["period", "days", "months", "years", "time"]):
            intent_bonus = 1
        elif intent == "boolean" and any(word in chunk_content_lower for word in ["yes", "no", "covered", "excluded", "allowed", "prohibited"]):
            intent_bonus = 1
        
        # Question keyword overlap
        question_words = set(question_lower.split())
        chunk_words = set(chunk_content_lower.split())
        overlap_score = len(question_words.intersection(chunk_words))
        
        # Calculate relevance score
        relevance_score = domain_score + intent_bonus + (overlap_score * 0.1)
        
        return relevance_score > 0.5
    
    def _deduplicate_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Remove duplicate or highly similar chunks
        """
        if len(chunks) <= 1:
            return chunks
        
        unique_chunks = []
        seen_content = set()
        
        for chunk in chunks:
            # Create a signature for similarity detection
            content_words = set(chunk.content.lower().split())
            
            # Check for significant overlap with existing chunks
            is_duplicate = False
            for seen_words in seen_content:
                overlap = len(content_words.intersection(seen_words))
                overlap_ratio = overlap / max(len(content_words), len(seen_words))
                
                if overlap_ratio > 0.8:  # 80% overlap threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_chunks.append(chunk)
                seen_content.add(frozenset(content_words))
        
        return unique_chunks
    
    def _post_process_answer(
        self,
        answer: str,
        question: str,
        context_chunks: List[DocumentChunk],
        intent_data: Dict[str, Any]
    ) -> str:
        """
        Post-process and validate the generated answer
        """
        if not answer or answer.strip() == "":
            return "No answer could be generated for this question."
        
        # Clean up answer
        answer = answer.strip()
        
        # Add source attribution if available
        if context_chunks:
            pages = set()
            for chunk in context_chunks:
                if chunk.page_number:
                    pages.add(chunk.page_number)
            
            if pages:
                page_list = sorted(list(pages))
                if len(page_list) == 1:
                    answer += f" (Source: Page {page_list[0]})"
                else:
                    answer += f" (Sources: Pages {', '.join(map(str, page_list))})"
        
        # Domain-specific post-processing
        domain = intent_data.get("domain", "general")
        if domain == "insurance":
            # Ensure insurance-specific formatting
            answer = self._format_insurance_answer(answer, question)
        elif domain == "legal":
            # Ensure legal-specific formatting
            answer = self._format_legal_answer(answer, question)
        
        return answer
    
    def _format_insurance_answer(self, answer: str, question: str) -> str:
        """Format insurance-specific answers"""
        question_lower = question.lower()
        
        # Ensure time periods are clearly stated
        if "period" in question_lower:
            # Make sure periods are highlighted
            import re
            period_pattern = r'(\d+)\s*(days?|months?|years?)'
            matches = re.findall(period_pattern, answer.lower())
            if matches:
                for number, unit in matches:
                    old_text = f"{number} {unit}"
                    new_text = f"{number} {unit}"
                    answer = re.sub(old_text, new_text, answer, flags=re.IGNORECASE)
        
        return answer
    
    def _format_legal_answer(self, answer: str, question: str) -> str:
        """Format legal-specific answers"""
        # Add legal disclaimers if needed
        if "law" in question.lower() or "legal" in question.lower():
            if not any(phrase in answer.lower() for phrase in ["according to", "as stated", "per the document"]):
                answer = "According to the document, " + answer[0].lower() + answer[1:]
        
        return answer
    
    def _update_processing_stats(self, processing_time: float, success: bool):
        """Update processing statistics"""
        self.processing_stats["total_questions"] += 1
        
        if success:
            self.processing_stats["successful_questions"] += 1
        else:
            self.processing_stats["failed_questions"] += 1
        
        # Update average processing time
        total = self.processing_stats["total_questions"]
        current_avg = self.processing_stats["average_processing_time"]
        self.processing_stats["average_processing_time"] = (current_avg * (total - 1) + processing_time) / total
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return self.processing_stats.copy()
    
    async def process_multiple_questions(
        self,
        questions: List[str],
        document_data: Dict[str, Any],
        request_id: str
    ) -> List[str]:
        """
        Process multiple questions in parallel using LangChain batch processing
        """
        start_time = time.time()
        
        logger.info(f"Processing {len(questions)} questions with LangChain batch processing",
                   extra={"request_id": request_id})
        
        try:
            # Prepare context chunks for each question
            context_chunks_list = []
            
            for idx, question in enumerate(questions):
                # STEP 1: Query Enhancement
                intent_data = await self.llm_service.extract_intent(question)
                enhanced_query = await self.llm_service.enhance_query(question)
                
                # STEP 2: Vector Search
                main_request_id = request_id.split('_q')[0] if '_q' in request_id else request_id
                similar_chunks = await self.vector_store.search_similar_chunks_for_request(
                    request_id=main_request_id,
                    query=enhanced_query,
                    document_id=document_data.get("document_id"),
                    top_k=settings.TOP_K_RESULTS
                )
                
                # STEP 3: Context Preparation
                context_chunks = self._prepare_context(
                    question=question,
                    search_results=similar_chunks,
                    intent_data=intent_data
                )
                
                context_chunks_list.append(context_chunks)
            
            # STEP 4: Batch LLM Processing using LangChain (simplified for reliability)
            try:
                answers = await self.llm_service.process_questions_parallel(
                    questions=questions,
                    context_chunks_list=context_chunks_list,
                    request_id=request_id
                )
            except Exception as e:
                logger.warning(f"LangChain batch processing failed: {e}, using individual processing")
                # Fallback to individual processing
                answers = []
                for idx, (question, chunks) in enumerate(zip(questions, context_chunks_list)):
                    question_request_id = f"{request_id}_q{idx}"
                    answer = await self.process_question(question, document_data, question_request_id)
                    answers.append(answer)
            
            # STEP 5: Post-processing
            processed_answers = []
            for idx, (answer, question) in enumerate(zip(answers, questions)):
                if isinstance(answer, Exception):
                    logger.error(f"Question {idx} failed: {str(answer)}",
                               extra={"request_id": request_id})
                    processed_answers.append(f"Error processing question: {str(answer)}")
                else:
                    # Apply domain-specific post-processing
                    intent_data = await self.llm_service.extract_intent(question)
                    final_answer = self._post_process_answer(
                        answer=answer,
                        question=question,
                        context_chunks=context_chunks_list[idx],
                        intent_data=intent_data
                    )
                    processed_answers.append(final_answer)
            
            total_time = time.time() - start_time
            
            logger.info(f"Completed LangChain batch processing {len(questions)} questions in {total_time:.2f}s",
                       extra={"request_id": request_id})
            
            return processed_answers
            
        except Exception as e:
            logger.error(f"Batch processing failed, falling back to parallel: {str(e)}",
                        extra={"request_id": request_id})
            
            # Fallback to original parallel processing
            return await self._process_multiple_questions_fallback(questions, document_data, request_id)
    
    async def _process_multiple_questions_fallback(
        self,
        questions: List[str],
        document_data: Dict[str, Any],
        request_id: str
    ) -> List[str]:
        """
        Fallback parallel processing method
        """
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_QUESTIONS)
        
        async def process_single_question_with_semaphore(question: str, index: int) -> str:
            async with semaphore:
                question_request_id = f"{request_id}_q{index}"
                return await self.process_question(question, document_data, question_request_id)
        
        # Execute all questions in parallel
        tasks = [
            process_single_question_with_semaphore(question, idx)
            for idx, question in enumerate(questions)
        ]
        
        answers = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_answers = []
        for idx, answer in enumerate(answers):
            if isinstance(answer, Exception):
                logger.error(f"Question {idx} failed: {str(answer)}",
                           extra={"request_id": request_id})
                processed_answers.append(f"Error processing question: {str(answer)}")
            else:
                processed_answers.append(answer)
        
        return processed_answers

    async def process_question_with_structured_output(
        self,
        question: str,
        document_data: Dict[str, Any],
        request_id: str
    ) -> Dict[str, Any]:
        """
        Process a question and return structured output with confidence and sources
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing question with structured output: {question[:50]}...",
                       extra={"request_id": request_id})
            
            # STEP 1: Query Enhancement and Intent Extraction
            intent_data = await self.llm_service.extract_intent(question)
            enhanced_query = await self.llm_service.enhance_query(question)
            
            # STEP 2: Vector Search
            main_request_id = request_id.split('_q')[0] if '_q' in request_id else request_id
            similar_chunks = await self.vector_store.search_similar_chunks_for_request(
                request_id=main_request_id,
                query=enhanced_query,
                document_id=document_data.get("document_id"),
                top_k=settings.TOP_K_RESULTS
            )
            
            # STEP 3: Context Preparation
            context_chunks = self._prepare_context(
                question=question,
                search_results=similar_chunks,
                intent_data=intent_data
            )
            
            # STEP 4: Generate Structured Answer
            structured_result = await self.llm_service.generate_structured_answer(
                question=question,
                context_chunks=context_chunks,
                request_id=request_id
            )
            
            processing_time = time.time() - start_time
            
            # Add metadata
            structured_result.update({
                "processing_time": processing_time,
                "intent_data": intent_data,
                "chunks_found": len(similar_chunks),
                "chunks_used": len(context_chunks),
                "request_id": request_id
            })
            
            logger.info(f"Structured question processed in {processing_time:.2f}s",
                       extra={"request_id": request_id})
            
            return structured_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Structured question processing failed: {str(e)}",
                        extra={"request_id": request_id})
            
            return {
                "answer": f"Error processing question: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "processing_time": processing_time,
                "error": str(e),
                "request_id": request_id
            }
