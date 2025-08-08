#!/usr/bin/env python3
"""
Performance test script to validate optimizations for 20-25s response time target
Tests the optimized chunking, parallel embeddings, and hybrid retrieval
"""

import asyncio
import requests
import json
import time
from datetime import datetime

# Test configuration
TEST_DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

# Optimized test questions targeting different aspects
TEST_QUESTIONS = [
    "What is the premium payment schedule?",
    "What are the coverage limitations?", 
    "How do I file a claim?",
    "What is the waiting period for maternity benefits?",
    "What are the exclusions in this policy?"
]

PERFORMANCE_TARGETS = {
    "total_time": 25.0,  # Target: 20-25s
    "document_processing": 8.0,  # Target: optimized chunking
    "embedding_generation": 5.0,  # Target: parallel processing
    "question_processing": 12.0,  # Target: hybrid retrieval
}

def test_performance_optimizations():
    """Test all performance optimizations"""
    
    print("=" * 80)
    print("PERFORMANCE OPTIMIZATION TEST")
    print("=" * 80)
    print(f"Target: 20-25 seconds total response time")
    print(f"Previous: 49-55 seconds")
    print(f"Expected improvement: 50-60% reduction")
    print()
    
    # Prepare request
    url = "http://localhost:8000/api/v1/hackrx/run"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer hackrx_test_token_12345"
    }
    
    payload = {
        "document_url": TEST_DOCUMENT_URL,
        "questions": TEST_QUESTIONS
    }
    
    print(f"📄 Document URL: {TEST_DOCUMENT_URL}")
    print(f"❓ Questions: {len(TEST_QUESTIONS)}")
    print(f"🎯 Target Response Time: {PERFORMANCE_TARGETS['total_time']}s")
    print()
    
    # Execute test
    start_time = time.time()
    print(f"🚀 Starting performance test at {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        
        total_time = time.time() - start_time
        
        print(f"\n✅ Request completed in {total_time:.2f} seconds")
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Performance analysis
            print(f"\n🎯 PERFORMANCE ANALYSIS")
            print(f"{'Metric':<25} {'Actual':<10} {'Target':<10} {'Status':<10}")
            print(f"{'-'*55}")
            
            # Total time
            target_met = total_time <= PERFORMANCE_TARGETS['total_time']
            status = "✅ PASS" if target_met else "❌ FAIL"
            print(f"{'Total Response Time':<25} {total_time:<10.2f} {PERFORMANCE_TARGETS['total_time']:<10.1f} {status:<10}")
            
            # Document processing time
            doc_time = result.get('processing_time', 0)
            target_met = doc_time <= PERFORMANCE_TARGETS['document_processing']
            status = "✅ PASS" if target_met else "❌ FAIL"
            print(f"{'Document Processing':<25} {doc_time:<10.2f} {PERFORMANCE_TARGETS['document_processing']:<10.1f} {status:<10}")
            
            # Calculate improvement
            baseline_time = 52.0  # Average of 49-55s
            improvement = ((baseline_time - total_time) / baseline_time) * 100
            print(f"\n📈 OPTIMIZATION RESULTS")
            print(f"Baseline Time: {baseline_time:.1f}s")
            print(f"Optimized Time: {total_time:.2f}s")
            print(f"Improvement: {improvement:.1f}% faster")
            
            if total_time <= 25.0:
                print(f"🎉 SUCCESS: Target achieved!")
            elif total_time <= 35.0:
                print(f"⚠️  PARTIAL: Good improvement, but target missed")
            else:
                print(f"❌ FAILURE: Insufficient optimization")
            
            # Detailed breakdown
            print(f"\n📋 DETAILED RESULTS")
            print(f"Request ID: {result.get('request_id', 'N/A')}")
            print(f"Total Answers: {len(result.get('answers', []))}")
            
            # Show chunk count reduction
            if 'processing_details' in result:
                details = result['processing_details']
                chunks_count = details.get('chunks_count', 'N/A')
                print(f"Chunks Generated: {chunks_count} (target: ≤150)")
                
                # Show timing breakdown
                print(f"\n⏱️  TIMING BREAKDOWN")
                for key, value in details.items():
                    if isinstance(value, (int, float)) and 'time' in key.lower():
                        print(f"   {key}: {value:.3f}s")
            
            # Sample answers
            print(f"\n📝 SAMPLE ANSWERS (Quality Check)")
            for i, answer in enumerate(result.get('answers', [])[:2]):
                print(f"\nQ{i+1}: {answer.get('question', 'N/A')}")
                print(f"A{i+1}: {answer.get('answer', 'N/A')[:200]}...")
                confidence = answer.get('confidence_score', 0)
                print(f"Confidence: {confidence:.2f}")
                
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print(f"⏰ Request timed out after 180 seconds")
        print(f"❌ FAILURE: Optimization target not met")
    except requests.exceptions.ConnectionError:
        print(f"🔌 Connection error - is the server running on http://localhost:8000?")
    except Exception as e:
        print(f"💥 Unexpected error: {str(e)}")
    
    print("\n" + "=" * 80)
    print("PERFORMANCE TEST COMPLETED")
    print("=" * 80)

def test_individual_optimizations():
    """Test individual optimization components"""
    
    print("\n" + "=" * 80)
    print("INDIVIDUAL OPTIMIZATION TESTS")
    print("=" * 80)
    
    # Test 1: Configuration validation
    print("\n1. 📋 CONFIGURATION VALIDATION")
    try:
        from src.core.config import settings
        
        config_tests = [
            ("CHUNK_SIZE", settings.CHUNK_SIZE, 400, "Larger chunks for fewer total chunks"),
            ("CHUNK_OVERLAP", settings.CHUNK_OVERLAP, 50, "Reduced overlap for speed"),
            ("MAX_CHUNKS_PER_DOCUMENT", settings.MAX_CHUNKS_PER_DOCUMENT, 150, "Chunk limit for performance"),
            ("MAX_EMBEDDING_BATCH_SIZE", settings.MAX_EMBEDDING_BATCH_SIZE, 32, "Larger batches for parallel processing"),
            ("ENABLE_HYBRID_RETRIEVAL", settings.ENABLE_HYBRID_RETRIEVAL, True, "Hybrid retrieval enabled"),
            ("FINAL_TOP_K", settings.FINAL_TOP_K, 5, "Fewer final results")
        ]
        
        for name, actual, expected, description in config_tests:
            status = "✅" if actual == expected else "❌"
            print(f"   {status} {name}: {actual} (expected: {expected}) - {description}")
            
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
    
    # Test 2: Module availability
    print("\n2. 📦 MODULE AVAILABILITY")
    modules = [
        ("rank_bm25", "BM25 keyword search"),
        ("faiss", "Vector similarity search"),
        ("openai", "Azure OpenAI embeddings"),
        ("nltk", "Text processing")
    ]
    
    for module, description in modules:
        try:
            __import__(module)
            print(f"   ✅ {module}: Available - {description}")
        except ImportError:
            print(f"   ❌ {module}: Missing - {description}")
    
    print(f"\n3. 🎯 OPTIMIZATION SUMMARY")
    print(f"   • Chunking: Larger chunks (400 vs 256), reduced overlap (50 vs 64)")
    print(f"   • Parallel Processing: 32-item batches, 4 concurrent embedding requests")
    print(f"   • Hybrid Retrieval: Semantic + BM25 keyword search with reranking")
    print(f"   • Chunk Limiting: Maximum 150 chunks per document")
    print(f"   • Expected Result: 50-60% reduction in response time (20-25s target)")

if __name__ == "__main__":
    test_individual_optimizations()
    test_performance_optimizations()
