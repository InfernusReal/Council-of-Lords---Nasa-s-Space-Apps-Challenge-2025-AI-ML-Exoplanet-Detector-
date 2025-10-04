import requests
import json

# Test the API response to see what's actually being returned
url = 'http://127.0.0.1:8000/analyze'

print('Testing API response...')

try:
    # Test with a simple file
    with open('../COUNCIL_OF_LORDS_NASA_NATIVE/brutal_reality_test/kepler_disaster.csv', 'rb') as f:
        files = {'file': ('kepler_disaster.csv', f, 'text/csv')}
        response = requests.post(url, files=files, timeout=60)
    
    print(f'Response status: {response.status_code}')
    print(f'Response headers: {response.headers}')
    
    if response.status_code == 200:
        try:
            result = response.json()
            print('✅ JSON parsed successfully')
            print(f'Status: {result.get("status")}')
            print(f'Verdict: {result.get("verdict")}')
            print(f'Confidence: {result.get("confidence")}')
            print(f'Message: {result.get("message", "No message")}')
            
            # Check if there are any weird data types
            print('\nDetailed response structure:')
            for key, value in result.items():
                print(f'  {key}: {type(value)} = {value}')
                
        except json.JSONDecodeError as e:
            print(f'❌ JSON decode error: {e}')
            print(f'Raw response: {response.text[:500]}')
    else:
        print(f'❌ HTTP Error: {response.status_code}')
        print(f'Response: {response.text}')

except Exception as e:
    print(f'❌ Request error: {e}')