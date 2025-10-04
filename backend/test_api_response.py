import requests
import json
from pathlib import Path

# Test the actual API response
data_file = Path('../COUNCIL_OF_LORDS_NASA_NATIVE/brutal_reality_test/kepler_disaster.csv')

print('🔍 TESTING ACTUAL API RESPONSE...')

url = 'http://localhost:8000/analyze'

try:
    with open(data_file, 'rb') as f:
        files = {'file': ('kepler_disaster.csv', f, 'text/csv')}
        response = requests.post(url, files=files, timeout=60)
    
    if response.status_code == 200:
        result = response.json()
        print('✅ API RESPONSE:')
        print(json.dumps(result, indent=2))
        
        print('\n🔍 KEY FIELDS:')
        print(f"verdict: {result.get('verdict', 'NOT_FOUND')}")
        print(f"confidence: {result.get('confidence', 'NOT_FOUND')}")
        print(f"status: {result.get('status', 'NOT_FOUND')}")
        
    else:
        print(f"❌ API Error: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"❌ Connection error: {e}")