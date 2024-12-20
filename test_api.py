import requests
import json

def test_api():
    # API endpoint
    url = "http://localhost:5000/predict"
    
    # Test URLs
    test_urls = [
        "http://example.com",
        "http://bit.ly/suspicious",
        "http://login-paypal-secure.com",
        "https://google.com"
    ]
    
    print("Testing URL Classification API...")
    print("-" * 50)
    
    for test_url in test_urls:
        # Prepare the request data
        data = {"url": test_url}
        
        try:
            # Make the POST request
            response = requests.post(url, json=data)
            
            # Check if request was successful
            if response.status_code == 200:
                result = response.json()
                print(f"\nURL: {test_url}")
                print(f"Prediction: {result['prediction']}")
                print(f"Prediction Code: {result['prediction_code']}")
                print(f"Timestamp: {result['timestamp']}")
                print("Features:", json.dumps(result['features'], indent=2))
                print("-" * 50)
            else:
                print(f"\nError for URL {test_url}:")
                print(f"Status Code: {response.status_code}")
                print(f"Error Message: {response.text}")
                print("-" * 50)
                
                
        except requests.exceptions.RequestException as e:
            print(f"\nError connecting to API: {e}")
            print("Make sure the Flask API is running on http://localhost:5000")
            break

if __name__ == "__main__":
    test_api()
#susanth