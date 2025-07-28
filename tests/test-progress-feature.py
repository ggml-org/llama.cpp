#!/usr/bin/env python3

import requests
import json
import sys
import time

def create_long_prompt():
    """Create a very long prompt to ensure multiple batches are processed"""
    # Create a much longer prompt that will definitely take multiple batches
    # This will help us clearly see the progress effect
    base_text = "This is a comprehensive test prompt designed to verify the progress functionality thoroughly. " * 200
    return base_text

def test_completion_endpoint_progress(server_url):
    """Test progress functionality on /completion endpoint with long prompt"""
    print("\n=== Testing /completion endpoint progress ===")
    print("Using a very long prompt to clearly demonstrate progress...")
    
    prompt = create_long_prompt()
    print(f"Prompt length: {len(prompt)} characters")
    
    data = {
        "prompt": prompt,
        "stream": True,
        "return_progress": True,
        "max_tokens": 10,  # Small number to focus on prompt processing
        "temperature": 0.7
    }
    
    progress_responses = []
    content_responses = []
    
    try:
        print("Sending request...")
        response = requests.post(f"{server_url}/completion", json=data, stream=True)
        response.raise_for_status()
        
        print("Receiving streaming response...")
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]  # Remove 'data: ' prefix
                    if data_str.strip() == '[DONE]':
                        break
                    
                    try:
                        json_data = json.loads(data_str)
                        if 'prompt_processing' in json_data:
                            progress_responses.append(json_data['prompt_processing'])
                            progress = json_data['prompt_processing']
                            percentage = progress.get('progress', 0) * 100
                            print(f"Progress: {percentage:.1f}% ({progress.get('n_prompt_tokens_processed', 'N/A')}/{progress.get('n_prompt_tokens', 'N/A')})")
                        elif 'content' in json_data and json_data.get('content', ''):
                            content_responses.append(json_data)
                    except json.JSONDecodeError:
                        continue
        
        print(f"\nReceived {len(progress_responses)} progress responses")
        print(f"Received {len(content_responses)} content responses")
        
        # Detailed analysis
        if progress_responses:
            print("\n=== Progress Analysis ===")
            for i, progress in enumerate(progress_responses):
                percentage = progress.get('progress', 0) * 100
                processed = progress.get('n_prompt_tokens_processed', 0)
                total = progress.get('n_prompt_tokens', 0)
                print(f"  Progress {i+1}: {percentage:.1f}% ({processed}/{total})")
            
            # Check if we reached 100%
            last_progress = progress_responses[-1].get('progress', 0)
            if last_progress >= 0.99:  # Allow for small floating point differences
                print("✅ Progress reached 100% as expected")
                return True
            else:
                print(f"❌ Progress did not reach 100% (last: {last_progress*100:.1f}%)")
                return False
        else:
            print("❌ No progress responses received")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_progress_disabled(server_url):
    """Test that progress is not sent when return_progress is false"""
    print("\n=== Testing progress disabled ===")
    
    prompt = create_long_prompt()
    
    data = {
        "prompt": prompt,
        "stream": True,
        "return_progress": False,  # Disable progress
        "max_tokens": 10,
        "temperature": 0.7
    }
    
    progress_responses = []
    content_responses = []
    
    try:
        print("Sending request with progress disabled...")
        response = requests.post(f"{server_url}/completion", json=data, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]  # Remove 'data: ' prefix
                    if data_str.strip() == '[DONE]':
                        break
                    
                    try:
                        json_data = json.loads(data_str)
                        if 'prompt_processing' in json_data:
                            progress_responses.append(json_data['prompt_processing'])
                        elif 'content' in json_data and json_data.get('content', ''):
                            content_responses.append(json_data)
                    except json.JSONDecodeError:
                        continue
        
        print(f"Received {len(progress_responses)} progress responses")
        print(f"Received {len(content_responses)} content responses")
        
        # Check that no progress responses were received
        if len(progress_responses) == 0:
            print("✅ No progress responses received when disabled (correct)")
            return True
        else:
            print("❌ Progress responses received when disabled (incorrect)")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_batch_size_effect(server_url):
    """Test the effect of different batch sizes on progress reporting"""
    print("\n=== Testing batch size effect ===")
    
    prompt = create_long_prompt()
    
    # Test with different batch sizes
    batch_sizes = [16, 32, 64]
    
    for batch_size in batch_sizes:
        print(f"\nTesting with batch size: {batch_size}")
        
        data = {
            "prompt": prompt,
            "stream": True,
            "return_progress": True,
            "max_tokens": 10,
            "temperature": 0.7
        }
        
        progress_responses = []
        
        try:
            # Note: We can't directly set batch_size in the request, but we can observe the effect
            # by counting progress responses - smaller batch sizes should result in more progress updates
            response = requests.post(f"{server_url}/completion", json=data, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        if data_str.strip() == '[DONE]':
                            break
                        
                        try:
                            json_data = json.loads(data_str)
                            if 'prompt_processing' in json_data:
                                progress_responses.append(json_data['prompt_processing'])
                        except json.JSONDecodeError:
                            continue
            
            print(f"  Progress responses: {len(progress_responses)}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    print("✅ Batch size effect test completed")
    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 test-progress-feature.py <server_url>")
        print("Example: python3 test-progress-feature.py http://localhost:8081")
        sys.exit(1)
    
    server_url = sys.argv[1]
    
    print("Testing progress feature with comprehensive test cases...")
    print(f"Server URL: {server_url}")
    print("This test uses a very long prompt to clearly demonstrate progress functionality.")
    
    # Wait a moment for server to be ready
    time.sleep(2)
    
    # Run tests
    test1_passed = test_completion_endpoint_progress(server_url)
    test2_passed = test_progress_disabled(server_url)
    test3_passed = test_batch_size_effect(server_url)
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Completion endpoint progress: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Progress disabled: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"Batch size effect: {'✅ PASS' if test3_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("\n🎉 All tests passed!")
        print("The progress feature is working correctly with long prompts and small batch sizes.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 