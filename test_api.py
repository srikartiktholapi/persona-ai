import requests
import sys

def test_analyze_endpoint(file_path):
    url = "http://127.0.0.1:8000/analyze"
    print(f"Uploading and testing file: {file_path}")
    try:
        with open(file_path, "rb") as f:
            response = requests.post(url, files={"file": f})
        
        print("\n--- Response Status Code ---")
        print(response.status_code)
        
        print("\n--- Response JSON ---")
        try:
            print(response.json())
        except Exception as e:
            print(f"Could not parse JSON. Raw text: {response.text}")
            
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        # Default placeholder if no argument is passed
        target_file = "test/video1.mp4" 
        
    test_analyze_endpoint(target_file)
