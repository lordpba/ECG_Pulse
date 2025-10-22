"""
Test script for ECG PULSE API
Usage: python test_api.py <path_to_ecg_image>
"""

import requests
import sys
import os

# API Configuration
API_URL = "http://localhost:8000"
API_KEY = os.environ.get("API_KEY", None)  # Optional API key

def test_health():
    """Test the health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_analyze_ecg(image_path, custom_prompt=None):
    """Test the analyze endpoint"""
    print("\n=== Testing Analyze Endpoint ===")
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return False
    
    # Prepare headers
    headers = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    
    # Prepare files and data
    with open(image_path, 'rb') as f:
        files = {'image': (os.path.basename(image_path), f, 'image/png')}
        
        data = {}
        if custom_prompt:
            data['prompt'] = custom_prompt
        else:
            data['prompt'] = "Please provide a detailed analysis of this ECG image. Focus on rhythm, rate, intervals, and any abnormalities you can identify."
        
        print(f"Uploading image: {image_path}")
        print(f"Prompt: {data['prompt']}")
        
        response = requests.post(
            f"{API_URL}/analyze",
            files=files,
            data=data,
            headers=headers
        )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n--- Analysis Results ---")
        print(f"Success: {result.get('success')}")
        print(f"Filename: {result.get('filename')}")
        print(f"Model: {result.get('model')}")
        print(f"\nAnalysis:\n{result.get('analysis')}")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def test_batch_analyze(image_paths, custom_prompt=None):
    """Test the batch analyze endpoint"""
    print("\n=== Testing Batch Analyze Endpoint ===")
    
    # Check all files exist
    for path in image_paths:
        if not os.path.exists(path):
            print(f"Error: Image file not found: {path}")
            return False
    
    # Prepare headers
    headers = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    
    # Prepare files
    files = []
    for path in image_paths:
        files.append(('images', (os.path.basename(path), open(path, 'rb'), 'image/png')))
    
    data = {}
    if custom_prompt:
        data['prompt'] = custom_prompt
    else:
        data['prompt'] = "Please provide a detailed analysis of this ECG image."
    
    print(f"Uploading {len(image_paths)} images")
    print(f"Prompt: {data['prompt']}")
    
    response = requests.post(
        f"{API_URL}/analyze-batch",
        files=files,
        data=data,
        headers=headers
    )
    
    # Close file handles
    for _, (_, file_obj, _) in files:
        file_obj.close()
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n--- Batch Results ---")
        print(f"Total: {result.get('total')}")
        for idx, res in enumerate(result.get('results', [])):
            print(f"\n[{idx+1}] {res.get('filename')}")
            if res.get('success'):
                print(f"Analysis: {res.get('analysis')[:200]}...")
            else:
                print(f"Error: {res.get('error')}")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <path_to_ecg_image> [additional_images...]")
        print("\nExample:")
        print("  python test_api.py ecg_sample.png")
        print("  python test_api.py ecg1.png ecg2.png ecg3.png")
        sys.exit(1)
    
    image_paths = sys.argv[1:]
    
    # Test health endpoint
    if not test_health():
        print("\n❌ Health check failed. Is the API server running?")
        print("Start the server with: python api_gradio.py")
        sys.exit(1)
    
    print("\n✓ API server is healthy")
    
    # Test single or batch analysis
    if len(image_paths) == 1:
        # Single image analysis
        success = test_analyze_ecg(
            image_paths[0],
            custom_prompt="Analyze this ECG image and identify any abnormalities."
        )
        if success:
            print("\n✓ Analysis completed successfully")
        else:
            print("\n❌ Analysis failed")
            sys.exit(1)
    else:
        # Batch analysis
        success = test_batch_analyze(
            image_paths,
            custom_prompt="Provide a brief analysis of this ECG."
        )
        if success:
            print("\n✓ Batch analysis completed successfully")
        else:
            print("\n❌ Batch analysis failed")
            sys.exit(1)

if __name__ == "__main__":
    main()
