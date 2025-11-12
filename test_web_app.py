"""
Demo script to test the ML/DL Paper to Code web application.
"""

import requests
import time
import os

def test_web_app():
    """Test the web application functionality."""
    base_url = "http://localhost:5000"
    
    print("Testing ML/DL Paper to Code Web Application...")
    print("=" * 50)
    
    # Test 1: Check if the main page loads
    print("1. Testing main page...")
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            print("   [OK] Main page loads successfully")
        else:
            print(f"   [ERROR] Main page returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"   [ERROR] Cannot connect to web app: {e}")
        print("   Make sure the Flask app is running with: python app.py")
        return False
    
    # Test 2: Check if we have a PDF file to test with
    print("\n2. Looking for test PDF file...")
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    if pdf_files:
        test_pdf = pdf_files[0]
        print(f"   [OK] Found test PDF: {test_pdf}")
    else:
        print("   [WARNING] No PDF files found for testing")
        print("   You can test manually by uploading a PDF at http://localhost:5000")
        return True
    
    # Test 3: Test file upload (if PDF available)
    if pdf_files:
        print(f"\n3. Testing file upload with {test_pdf}...")
        try:
            with open(test_pdf, 'rb') as f:
                files = {'file': f}
                data = {'framework': 'pytorch'}
                response = requests.post(f"{base_url}/upload", files=files, data=data)
                
            if response.status_code == 200:
                result = response.json()
                task_id = result.get('task_id')
                print(f"   [OK] File uploaded successfully. Task ID: {task_id}")
                
                # Test 4: Monitor processing status
                print("\n4. Monitoring processing status...")
                max_attempts = 30  # 30 seconds max
                for attempt in range(max_attempts):
                    status_response = requests.get(f"{base_url}/status/{task_id}")
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        progress = status_data.get('progress', 0)
                        message = status_data.get('message', '')
                        print(f"   Progress: {progress}% - {message}")
                        
                        if status_data.get('status') == 'completed':
                            print("   [OK] Processing completed successfully!")
                            
                            # Test 5: Check results
                            print("\n5. Testing results retrieval...")
                            results_response = requests.get(f"{base_url}/results/{task_id}")
                            if results_response.status_code == 200:
                                results_data = results_response.json()
                                algorithms_found = results_data.get('algorithms_found', 0)
                                framework = results_data.get('framework', '')
                                print(f"   [OK] Found {algorithms_found} algorithms for {framework}")
                                
                                # Test 6: Test code preview
                                print("\n6. Testing code preview...")
                                preview_response = requests.get(f"{base_url}/preview/{task_id}")
                                if preview_response.status_code == 200:
                                    preview_data = preview_response.json()
                                    code_length = len(preview_data.get('code', ''))
                                    print(f"   [OK] Code preview available ({code_length} characters)")
                                else:
                                    print(f"   [WARNING] Code preview failed: {preview_response.status_code}")
                                
                                print("\n[SUCCESS] All web application tests passed!")
                                return True
                            else:
                                print(f"   [ERROR] Results retrieval failed: {results_response.status_code}")
                                return False
                        
                        elif status_data.get('status') == 'error':
                            print(f"   [ERROR] Processing failed: {status_data.get('message', 'Unknown error')}")
                            return False
                    
                    time.sleep(1)
                
                print("   [TIMEOUT] Processing took too long")
                return False
                
            else:
                print(f"   [ERROR] Upload failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   [ERROR] Upload test failed: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("ML/DL Paper to Code - Web Application Test")
    print("Make sure the Flask app is running: python app.py")
    print()
    
    success = test_web_app()
    
    if success:
        print("\n" + "=" * 50)
        print("Web application is working correctly!")
        print("You can now:")
        print("1. Open http://localhost:5000 in your browser")
        print("2. Upload a PDF file")
        print("3. Select a framework (PyTorch, TensorFlow, or scikit-learn)")
        print("4. Click 'Generate Code' and watch the magic happen!")
    else:
        print("\n" + "=" * 50)
        print("Some tests failed. Please check the error messages above.")
