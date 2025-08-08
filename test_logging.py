#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced logging and timing in document processing
"""

import asyncio
import requests
import json
import time

# Test URL for document processing
TEST_DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

# Test questions
TEST_QUESTIONS = [
    "What is the company's privacy policy?",
    "What are the terms of service?",
    "How is user data handled?",
    "What are the refund policies?",
    "Who can I contact for support?"
]

def test_hackrx_endpoint():
    """Test the HackRX endpoint with enhanced logging"""
    
    print("=" * 60)
    print("TESTING ENHANCED LOGGING IN DOCUMENT PROCESSING")
    print("=" * 60)
    
    # Prepare the request
    url = "http://localhost:8000/api/v1/hackrx/run"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer hackrx_test_token_12345"
    }
    
    payload = {
        "document_url": TEST_DOCUMENT_URL,
        "questions": TEST_QUESTIONS
    }
    
    print(f"\n📄 Document URL: {TEST_DOCUMENT_URL}")
    print(f"❓ Questions: {len(TEST_QUESTIONS)}")
    print(f"🌐 Endpoint: {url}")
    print("\n🚀 Starting request...")
    
    start_time = time.time()
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        
        request_time = time.time() - start_time
        
        print(f"\n✅ Request completed in {request_time:.2f} seconds")
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"📋 Response received:")
            print(f"   - Request ID: {result.get('request_id', 'N/A')}")
            print(f"   - Processing Time: {result.get('processing_time', 'N/A')} seconds")
            print(f"   - Total Answers: {len(result.get('answers', []))}")
            
            # Show timing breakdown if available
            if 'processing_details' in result:
                details = result['processing_details']
                print(f"\n⏱️  Processing Breakdown:")
                for key, value in details.items():
                    if isinstance(value, (int, float)) and 'time' in key.lower():
                        print(f"   - {key}: {value:.3f}s")
            
            print(f"\n📝 Sample answers:")
            for i, answer in enumerate(result.get('answers', [])[:3]):
                print(f"   Q{i+1}: {answer.get('question', 'N/A')[:50]}...")
                print(f"   A{i+1}: {answer.get('answer', 'N/A')[:100]}...")
                print()
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"📄 Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out after 120 seconds")
    except requests.exceptions.ConnectionError:
        print("🔌 Connection error - is the server running?")
    except Exception as e:
        print(f"💥 Unexpected error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    test_hackrx_endpoint()
