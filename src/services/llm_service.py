"""
LLM Service with LangChain Integration

Handles interactions with Azure OpenAI GPT models using LangChain
Features: JSON parsing, parallel processing, structured outputs
Optimized for fast inference with timeout handling
"""

import asyncio
import time
import json
from typing import List, Dict, Any, Optional, Union
import logging

# LangChain imports
try:
    from langchain_openai import AzureChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from pydantic import BaseModel, Field  # Use pydantic v2 directly
    from langchain.callbacks.base import AsyncCallbackHandler
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Fallback to OpenAI
try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..core.config import settings
from ..models.api_models import QuestionContext, DocumentChunk

logger = logging.getLogger(__name__)

# Pydantic models for structured outputs
class AnswerWithConfidence(BaseModel):
    """Structured answer with confidence score"""
    answer: str = Field(description="The answer to the question")
    confidence: float = Field(description="Confidence score between 0 and 1")
    sources: List[str] = Field(description="List of source references used")

class EnhancedQuery(BaseModel):
    """Enhanced query with additional terms"""
    original_query: str = Field(description="The original query")
    enhanced_terms: List[str] = Field(description="Additional search terms")
    domain: str = Field(description="Detected domain of the query")

class IntentExtraction(BaseModel):
    """Intent and entity extraction result"""
    intent: str = Field(description="The primary intent of the question")
    domain: str = Field(description="The domain category")
    entities: List[Dict[str, Any]] = Field(description="Extracted entities")
    confidence: float = Field(description="Overall confidence score")

class TimeoutCallbackHandler(AsyncCallbackHandler):
    """Custom callback handler for timeout management"""
    def __init__(self, timeout_seconds: int = 15):  # Increased timeout
        self.timeout_seconds = timeout_seconds
        self.start_time = None
    
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        self.start_time = time.time()
    
    async def on_llm_end(self, response, **kwargs):
        if self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                logger.warning(f"LLM call took {elapsed:.2f}s, exceeding timeout of {self.timeout_seconds}s")

class LLMService:
    """Service for LLM interactions with LangChain and Azure OpenAI"""
    
    def __init__(self):
        self.client = None
        self.langchain_llm = None
        self.model_name = settings.AZURE_OPENAI_API_DEPLOYMENT_NAME
        self.max_tokens = settings.LLM_MAX_TOKENS
        self.temperature = settings.LLM_TEMPERATURE
        self.timeout = 15  # Force 15 second timeout for production reliability
        
        # LangChain components
        self.answer_parser = StrOutputParser()
        self.json_parser = JsonOutputParser()
        self.structured_answer_parser = JsonOutputParser(pydantic_object=AnswerWithConfidence)
        self.query_enhancer_parser = JsonOutputParser(pydantic_object=EnhancedQuery)
        self.intent_parser = JsonOutputParser(pydantic_object=IntentExtraction)
        
        # Callback handler for timeout management
        self.timeout_callback = TimeoutCallbackHandler(self.timeout)
        
        # Domain-specific prompts
        self.domain_keywords = {
            "insurance": settings.INSURANCE_KEYWORDS,
            "legal": settings.LEGAL_KEYWORDS,
            "hr": settings.HR_KEYWORDS,
            "compliance": settings.COMPLIANCE_KEYWORDS
        }
    
    async def initialize(self):
        """Initialize the LLM service with LangChain"""
        if not LANGCHAIN_AVAILABLE and not OPENAI_AVAILABLE:
            raise Exception("Neither LangChain nor OpenAI library available")
        
        if not settings.AZURE_OPENAI_API_KEY:
            raise Exception("Azure OpenAI API key not configured")
        
        # Initialize LangChain Azure OpenAI
        if LANGCHAIN_AVAILABLE:
            try:
                self.langchain_llm = AzureChatOpenAI(
                    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                    api_key=settings.AZURE_OPENAI_API_KEY,
                    api_version=settings.AZURE_OPENAI_API_VERSION,
                    deployment_name=self.model_name,
                    # Use default temperature as 0.1 might not be supported
                    temperature=1.0,
                    # For newer models, only use model_kwargs
                    model_kwargs={
                        "max_completion_tokens": 1000
                    }
                )
                logger.info(f"LLM Service initialized with LangChain and model: {self.model_name}")
            except Exception as e:
                logger.warning(f"LangChain initialization failed: {e}, falling back to OpenAI client")
                self.langchain_llm = None
        
        # Fallback to direct OpenAI client (always initialize for fallback)
        if OPENAI_AVAILABLE:
            self.client = AzureOpenAI(
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
            )
            if not self.langchain_llm:
                logger.info(f"LLM Service initialized with OpenAI client only and model: {self.model_name}")
            else:
                logger.info(f"OpenAI client also available as fallback")
    
    async def cleanup(self):
        """Cleanup resources"""
        pass
    
    def _detect_domain(self, question: str) -> str:
        """Detect the domain of a question based on keywords"""
        question_lower = question.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in question_lower)
            domain_scores[domain] = score
        
        # Return domain with highest score, default to 'general'
        best_domain = max(domain_scores.items(), key=lambda x: x[1])
        return best_domain[0] if best_domain[1] > 0 else "general"
    
    def _create_system_prompt(self, domain: str) -> str:
        """Create domain-specific system prompt"""
        base_prompt = """You are an expert AI assistant specialized in document analysis and question answering.
Your task is to provide accurate, concise, and helpful answers based on the provided document context.

Instructions:
1. Answer based ONLY on the provided context
2. Be specific and cite relevant details from the document
3. If the information is not in the context, say "The information is not available in the provided document"
4. Keep answers concise but complete (aim for 1-3 sentences)
5. Use clear, professional language
6. Include specific details like numbers, dates, and conditions when available"""

        domain_specific = {
            "insurance": """
Focus on insurance terms, coverage details, premiums, deductibles, exclusions, waiting periods,
grace periods, benefits, claims procedures, and policy conditions.""",
            "legal": """
Focus on legal terms, contract clauses, obligations, rights, liabilities, jurisdiction,
governing law, termination conditions, and compliance requirements.""",
            "hr": """
Focus on employee policies, benefits, procedures, performance management, leave policies,
compensation, disciplinary actions, and workplace guidelines.""",
            "compliance": """
Focus on regulatory requirements, compliance standards, audit procedures, reporting obligations,
penalties, certifications, and risk management."""
        }
        
        if domain in domain_specific:
            return base_prompt + "\n\nDomain Focus:\n" + domain_specific[domain]
        
        return base_prompt
    
    def _create_user_prompt(self, question: str, context_chunks: List[DocumentChunk]) -> str:
        """Create user prompt with question and context"""
        # Combine relevant chunks into context
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            page_ref = f" (Page {chunk.page_number})" if chunk.page_number else ""
            context_parts.append(f"[Context {i+1}]{page_ref}: {chunk.content}")
        
        context_text = "\n\n".join(context_parts)
        
        prompt = f"""Document Context:
{context_text}

Question: {question}

Please provide a clear and accurate answer based on the document context above."""
        
        return prompt
    
    async def generate_answer(
        self,
        question: str,
        context_chunks: List[DocumentChunk],
        request_id: str = None
    ) -> str:
        """
        Generate answer using LangChain with context
        Target: 2.5 seconds per question
        """
        start_time = time.time()
        
        if not context_chunks:
            return "No relevant information found in the document to answer this question."
        
        try:
            # Detect domain for specialized prompting
            domain = self._detect_domain(question)
            
            # Create prompts using LangChain
            system_prompt = self._create_system_prompt(domain)
            context_text = self._create_context_text(context_chunks)
            
            logger.info(f"Generating answer for domain: {domain}",
                       extra={"request_id": request_id, "context_chunks": len(context_chunks)})
            
            # Use LangChain chain if available, but with shorter prompts for reliability
            if self.langchain_llm:
                try:
                    answer = await self._generate_with_langchain_simple(
                        system_prompt, question, context_text, request_id
                    )
                except Exception as e:
                    logger.warning(f"LangChain failed, using fallback: {str(e)}")
                    answer = await self._generate_with_openai(
                        system_prompt, question, context_text, request_id
                    )
            else:
                # Fallback to OpenAI client
                answer = await self._generate_with_openai(
                    system_prompt, question, context_text, request_id
                )
            
            # Post-process answer
            answer = self._post_process_answer(answer, question, domain)
            
            generation_time = time.time() - start_time
            logger.info(f"Answer generated in {generation_time:.2f}s",
                       extra={"request_id": request_id})
            
            return answer
            
        except Exception as e:
            logger.error(f"LLM answer generation failed: {str(e)}",
                        extra={"request_id": request_id})
            return f"Error generating answer: {str(e)}"
    
    async def _generate_with_langchain_simple(
        self,
        system_prompt: str,
        question: str,
        context_text: str,
        request_id: str = None
    ) -> str:
        """Generate answer using simplified LangChain chain for better reliability"""
        try:
            # Create simplified prompt - shorter and more direct
            simple_prompt = f"{system_prompt}\n\nContext: {context_text[:2000]}\n\nQuestion: {question}\n\nAnswer:"
            
            # Use direct message invocation for better reliability
            messages = [HumanMessage(content=simple_prompt)]
            
            # Execute with timeout
            response = await asyncio.wait_for(
                self.langchain_llm.ainvoke(messages),
                timeout=self.timeout  # Use the configured timeout
            )
            
            return response.content.strip()
            
        except asyncio.TimeoutError:
            logger.warning(f"LangChain simple request timed out after {self.timeout}s",
                          extra={"request_id": request_id})
            raise
        except Exception as e:
            logger.warning(f"LangChain simple generation failed: {str(e)}")
            raise

    async def _generate_with_langchain(
        self,
        system_prompt: str,
        question: str,
        context_text: str,
        request_id: str = None
    ) -> str:
        """Generate answer using LangChain chain"""
        try:
            # Create prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_prompt),
                HumanMessagePromptTemplate.from_template(
                    "Document Context:\n{context}\n\nQuestion: {question}\n\n"
                    "Please provide a clear and accurate answer based on the document context above."
                )
            ])
            
            # Create chain
            chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | prompt_template
                | self.langchain_llm
                | self.answer_parser
            )
            
            # Execute with timeout - increased timeout for better reliability
            answer = await asyncio.wait_for(
                chain.ainvoke({"context": context_text, "question": question}),
                timeout=self.timeout * 2  # Double the timeout for LangChain
            )
            
            return answer.strip()
            
        except asyncio.TimeoutError:
            logger.warning(f"LangChain request timed out after {self.timeout * 2}s",
                          extra={"request_id": request_id})
            return "Answer generation timed out. Please try again."
        except Exception as e:
            logger.warning(f"LangChain answer generation failed: {str(e)}, using fallback",
                          extra={"request_id": request_id})
            # Try fallback to OpenAI
            return await self._generate_with_openai(system_prompt, question, context_text, request_id)
    
    async def _generate_with_openai(
        self,
        system_prompt: str,
        question: str,
        context_text: str,
        request_id: str = None
    ) -> str:
        """Fallback generation using OpenAI client"""
        user_prompt = f"""Document Context:
{context_text}

Question: {question}

Please provide a clear and accurate answer based on the document context above."""
        
        try:
            response = await asyncio.wait_for(
                self._call_openai_async(system_prompt, user_prompt),
                timeout=self.timeout
            )
            
            return response.choices[0].message.content.strip()
            
        except asyncio.TimeoutError:
            logger.warning(f"OpenAI request timed out after {self.timeout}s",
                          extra={"request_id": request_id})
            return "Answer generation timed out. Please try again."
    
    def _create_context_text(self, context_chunks: List[DocumentChunk]) -> str:
        """Create formatted context text from chunks"""
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            page_ref = f" (Page {chunk.page_number})" if chunk.page_number else ""
            context_parts.append(f"[Context {i+1}]{page_ref}: {chunk.content}")
        
        return "\n\n".join(context_parts)
    
    async def _call_openai_async(self, system_prompt: str, user_prompt: str):
        """Make async call to OpenAI API with proper temperature handling"""
        loop = asyncio.get_event_loop()
        
        def call_openai():
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=self.max_tokens,  # Fixed for gpt-4o-mini
                temperature=1.0  # Use default temperature for compatibility
            )
        
        return await loop.run_in_executor(None, call_openai)
    
    def _post_process_answer(self, answer: str, question: str, domain: str) -> str:
        """Post-process the generated answer"""
        # Remove common unwanted phrases
        unwanted_phrases = [
            "Based on the provided context,",
            "According to the document,",
            "The document states that",
            "From the context provided,"
        ]
        
        for phrase in unwanted_phrases:
            answer = answer.replace(phrase, "").strip()
        
        # Ensure answer starts with capital letter
        if answer and not answer[0].isupper():
            answer = answer[0].upper() + answer[1:]
        
        return answer
    
    async def enhance_query(self, question: str) -> str:
        """
        Enhance query for better retrieval - using fallback for reliability
        """
        # For production reliability, use the fallback method directly
        return await self._enhance_query_fallback(question)
    
    async def _enhance_query_with_langchain(self, question: str) -> str:
        """Enhance query using LangChain with structured output"""
        try:
            # Create prompt for query enhancement
            enhance_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "You are an expert at enhancing search queries. "
                    "Analyze the question and provide additional search terms that would help find relevant information. "
                    "Detect the domain and return a JSON response with the original query, enhanced terms, and domain.\n"
                    "Respond with ONLY a valid JSON object in this exact format:\n"
                    '{{"original_query": "the question", "enhanced_terms": ["term1", "term2"], "domain": "domain_name"}}'
                ),
                HumanMessagePromptTemplate.from_template("Question: {question}")
            ])
            
            # Create chain with string output parser
            chain = (
                {"question": RunnablePassthrough()}
                | enhance_prompt
                | self.langchain_llm
                | self.answer_parser
            )
            
            # Execute with timeout - increased timeout
            result_str = await asyncio.wait_for(
                chain.ainvoke({"question": question}),
                timeout=8.0  # Increased timeout for query enhancement
            )
            
            # Try to parse JSON manually
            try:
                result = json.loads(result_str.strip())
                enhanced_terms = result.get("enhanced_terms", [])
                if enhanced_terms:
                    return f"{question} {' '.join(enhanced_terms)}"
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON response: {result_str}")
            
            return question
            
        except Exception as e:
            logger.warning(f"LangChain query enhancement failed: {e}, using fallback")
            return await self._enhance_query_fallback(question)
    
    async def _enhance_query_fallback(self, question: str) -> str:
        """Fallback query enhancement"""
        domain = self._detect_domain(question)
        
        # Add domain-specific keywords to improve retrieval
        domain_keywords = self.domain_keywords.get(domain, [])
        
        # Simple keyword expansion
        enhanced_terms = []
        question_lower = question.lower()
        
        for keyword in domain_keywords:
            if keyword.lower() in question_lower:
                # Add related terms
                if keyword == "premium" and domain == "insurance":
                    enhanced_terms.extend(["payment", "cost", "fee"])
                elif keyword == "waiting period" and domain == "insurance":
                    enhanced_terms.extend(["delay", "waiting", "period"])
                elif keyword == "coverage" and domain == "insurance":
                    enhanced_terms.extend(["benefit", "covered", "include"])
        
        if enhanced_terms:
            return f"{question} {' '.join(enhanced_terms)}"
        
        return question
    
    async def extract_intent(self, question: str) -> Dict[str, Any]:
        """Extract intent and entities from question - using fallback for reliability"""
        # For production reliability, use the fallback method directly
        return await self._extract_intent_fallback(question)
    
    async def _extract_intent_with_langchain(self, question: str) -> Dict[str, Any]:
        """Extract intent using LangChain with structured JSON output"""
        try:
            # Create prompt for intent extraction
            intent_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "You are an expert at analyzing questions to extract intent, domain, and entities. "
                    "Analyze the question and classify the intent (definition, amount, temporal, boolean, process, query), "
                    "detect the domain (insurance, legal, hr, compliance, general), and extract relevant entities.\n"
                    "Intent types:\n"
                    "- definition: asking for explanation or definition\n"
                    "- amount: asking for costs, fees, amounts\n" 
                    "- temporal: asking about time periods, dates\n"
                    "- boolean: yes/no questions\n"
                    "- process: asking about procedures\n"
                    "- query: general information seeking\n\n"
                    "You must respond with ONLY a valid JSON object in this exact format:\n"
                    '{{"intent": "intent_type", "domain": "domain_name", "entities": [list_of_entities], "confidence": 0.8}}'
                ),
                HumanMessagePromptTemplate.from_template("Question: {question}")
            ])
            
            # Create chain with simple string output parser (we'll parse JSON manually)
            chain = (
                {"question": RunnablePassthrough()}
                | intent_prompt
                | self.langchain_llm
                | self.answer_parser
            )
            
            # Execute with timeout - increased timeout
            result_str = await asyncio.wait_for(
                chain.ainvoke({"question": question}),
                timeout=8.0  # Increased timeout for intent extraction
            )
            
            # Try to parse JSON manually
            try:
                result = json.loads(result_str.strip())
                return result
            except json.JSONDecodeError:
                # If JSON parsing fails, extract what we can from the response
                logger.warning(f"Could not parse JSON response: {result_str}")
                raise Exception("Invalid JSON response")
            
        except Exception as e:
            logger.warning(f"LangChain intent extraction failed: {e}, using fallback")
            return await self._extract_intent_fallback(question)
    
    async def _extract_intent_fallback(self, question: str) -> Dict[str, Any]:
        """Fallback intent extraction using simple classification"""
        domain = self._detect_domain(question)
        
        # Simple intent classification
        question_lower = question.lower()
        intent = "query"
        
        if any(word in question_lower for word in ["what is", "define", "explain"]):
            intent = "definition"
        elif any(word in question_lower for word in ["how much", "cost", "fee", "amount"]):
            intent = "amount"
        elif any(word in question_lower for word in ["when", "period", "time", "date"]):
            intent = "temporal"
        elif any(word in question_lower for word in ["does", "is", "can", "will"]):
            intent = "boolean"
        elif any(word in question_lower for word in ["how", "procedure", "process"]):
            intent = "process"
        
        # Extract entities (simplified)
        entities = []
        for domain_name, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword.lower() in question_lower:
                    entities.append({
                        "text": keyword,
                        "type": domain_name,
                        "confidence": 0.9
                    })
        
        return {
            "intent": intent,
            "domain": domain,
            "entities": entities,
            "confidence": 0.8
        }
    
    async def generate_structured_answer(
        self,
        question: str,
        context_chunks: List[DocumentChunk],
        request_id: str = None
    ) -> Dict[str, Any]:
        """
        Generate structured answer with confidence score and sources using LangChain
        """
        if not self.langchain_llm:
            # Fallback to regular answer generation
            answer = await self.generate_answer(question, context_chunks, request_id)
            return {
                "answer": answer,
                "confidence": 0.8,
                "sources": [f"Page {chunk.page_number}" for chunk in context_chunks if chunk.page_number]
            }
        
        try:
            domain = self._detect_domain(question)
            system_prompt = self._create_system_prompt(domain)
            context_text = self._create_context_text(context_chunks)
            
            # Create prompt for structured answer
            structured_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    system_prompt + "\n\n"
                    "Provide your response in JSON format with the answer, confidence score (0-1), "
                    "and list of sources used.\n"
                    "{format_instructions}"
                ),
                HumanMessagePromptTemplate.from_template(
                    "Document Context:\n{context}\n\nQuestion: {question}\n\n"
                    "Please provide a structured answer with confidence and sources."
                )
            ])
            
            # Create chain with structured output
            chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | structured_prompt.partial(format_instructions=self.structured_answer_parser.get_format_instructions())
                | self.langchain_llm
                | self.structured_answer_parser
            )
            
            # Execute with timeout
            result = await asyncio.wait_for(
                chain.ainvoke({"context": context_text, "question": question}),
                timeout=self.timeout
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Structured answer generation failed: {e}", extra={"request_id": request_id})
            # Fallback to regular answer
            answer = await self.generate_answer(question, context_chunks, request_id)
            return {
                "answer": answer,
                "confidence": 0.7,
                "sources": [f"Page {chunk.page_number}" for chunk in context_chunks if chunk.page_number]
            }
    
    async def process_questions_parallel(
        self,
        questions: List[str],
        context_chunks_list: List[List[DocumentChunk]],
        request_id: str = None
    ) -> List[str]:
        """
        Process multiple questions in parallel using LangChain
        """
        if len(questions) != len(context_chunks_list):
            raise ValueError("Number of questions must match number of context chunk lists")
        
        logger.info(f"Processing {len(questions)} questions in parallel with LangChain",
                   extra={"request_id": request_id})
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_QUESTIONS)
        
        async def process_single_question_with_semaphore(
            question: str, 
            chunks: List[DocumentChunk], 
            index: int
        ) -> str:
            async with semaphore:
                individual_request_id = f"{request_id}_q{index}" if request_id else f"parallel_q{index}"
                return await self.generate_answer(question, chunks, individual_request_id)
        
        # Execute all questions in parallel
        tasks = [
            process_single_question_with_semaphore(question, chunks, idx)
            for idx, (question, chunks) in enumerate(zip(questions, context_chunks_list))
        ]
        
        start_time = time.time()
        answers = await asyncio.gather(*tasks, return_exceptions=True)
        processing_time = time.time() - start_time
        
        # Handle any exceptions
        processed_answers = []
        for idx, answer in enumerate(answers):
            if isinstance(answer, Exception):
                logger.error(f"Question {idx} failed: {str(answer)}",
                           extra={"request_id": request_id})
                processed_answers.append(f"Error processing question: {str(answer)}")
            else:
                processed_answers.append(answer)
        
        logger.info(f"Parallel processing completed in {processing_time:.2f}s",
                   extra={"request_id": request_id, "questions_count": len(questions)})
        
        return processed_answers
    
    async def batch_process_with_langchain_runnable(
        self,
        questions: List[str],
        context_chunks_list: List[List[DocumentChunk]],
        request_id: str = None
    ) -> List[str]:
        """
        Advanced batch processing using LangChain's RunnableLambda for optimization
        """
        if not self.langchain_llm:
            return await self.process_questions_parallel(questions, context_chunks_list, request_id)
        
        try:
            # Create a batch processing function
            def prepare_batch_inputs(inputs):
                questions, context_lists = inputs
                batch_inputs = []
                for question, chunks in zip(questions, context_lists):
                    domain = self._detect_domain(question)
                    system_prompt = self._create_system_prompt(domain)
                    context_text = self._create_context_text(chunks)
                    batch_inputs.append({
                        "system_prompt": system_prompt,
                        "question": question,
                        "context": context_text
                    })
                return batch_inputs
            
            # Create batch prompt
            batch_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template("{system_prompt}"),
                HumanMessagePromptTemplate.from_template(
                    "Document Context:\n{context}\n\nQuestion: {question}\n\n"
                    "Please provide a clear and accurate answer based on the document context above."
                )
            ])
            
            # Create batch processing chain
            batch_chain = (
                RunnableLambda(prepare_batch_inputs)
                | batch_prompt.batch
                | self.langchain_llm.batch
                | RunnableLambda(lambda responses: [resp.content.strip() for resp in responses])
            )
            
            # Execute batch processing
            start_time = time.time()
            answers = await asyncio.wait_for(
                batch_chain.ainvoke((questions, context_chunks_list)),
                timeout=self.timeout * len(questions)  # Adjust timeout for batch
            )
            processing_time = time.time() - start_time
            
            logger.info(f"Batch processing completed in {processing_time:.2f}s",
                       extra={"request_id": request_id, "questions_count": len(questions)})
            
            return answers
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}, falling back to parallel processing",
                        extra={"request_id": request_id})
            return await self.process_questions_parallel(questions, context_chunks_list, request_id)
    
    async def health_check(self) -> bool:
        """Check if LLM service is healthy"""
        try:
            if self.langchain_llm:
                # Test LangChain LLM
                response = await asyncio.wait_for(
                    self.langchain_llm.ainvoke([HumanMessage(content="Say 'OK'")]),
                    timeout=5.0
                )
                return "OK" in response.content
            elif self.client:
                # Test OpenAI client
                response = await asyncio.wait_for(
                    self._call_openai_async("You are a helpful assistant.", "Say 'OK'"),
                    timeout=5.0
                )
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
