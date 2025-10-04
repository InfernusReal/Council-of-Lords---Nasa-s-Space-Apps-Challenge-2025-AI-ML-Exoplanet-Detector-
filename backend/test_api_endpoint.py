import requests
import os
from pathlib import Path

# Test the /analyze endpoint with the correct CSV file
data_file = Path('../COUNCIL_OF_LORDS_NASA_NATIVE/brutal_reality_test/tess_nightmare.csv')

print('Testing /analyze endpoint with TESS nightmare CSV...')

url = 'http://localhost:8000/analyze'

try:
    with open(data_file, 'rb') as f:
        files = {'file': ('tess_nightmare.csv', f, 'text/csv')}
        response = requests.post(url, files=files, timeout=60)
    
    if response.status_code == 200:
        result = response.json()
        print('✅ API RESULT:')
        print(f"Status: {result['status']}")
        print(f"Verdict: {result['verdict']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        
        if result['red_flags']:
            print("Red flags:")
            for flag in result['red_flags']:
                print(f"  - {flag}")
        
        print("\nAdvanced detection:")
        for key, value in result['advanced_detection'].items():
            print(f"  {key}: {value}")
    else:
        print(f"❌ API Error: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"❌ Connection error: {e}")