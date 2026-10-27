import urllib.request
import urllib.parse
import json

def test_route(url, name):
    try:
        response = urllib.request.urlopen(url)
        status = response.getcode()
        html = response.read().decode('utf-8')
        if status == 200:
            print(f"[OK] {name} loaded successfully (Status: 200)")
            return True
        else:
            print(f"[FAIL] {name} returned status: {status}")
            return False
    except Exception as e:
        print(f"[FAIL] {name} failed: {e}")
        return False

def test_prediction_api():
    url = "http://127.0.0.1:5000/api/predict"
    # Weather metrics corresponding to flood (Actual: 1)
    data = {
        "Temp": 24.7,
        "Humidity": 93.7,
        "Cloud_Visibility": 5.0,
        "Jan_Feb": 92.3,
        "Mar_May": 181.8,
        "Jun_Sep": 2930.7,
        "Oct_Dec": 402.0,
        "avgjune": 707.2,
        "sub": 3
    }
    
    req_data = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(
        url, 
        data=req_data, 
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        response = urllib.request.urlopen(req)
        status = response.getcode()
        res_body = json.loads(response.read().decode('utf-8'))
        
        if status == 200 and res_body.get('status') == 'success':
            print(f"[OK] Prediction API endpoint `/api/predict` functional (Status: 200)")
            print("\nSimulated Predictions (Flood Case):")
            for model_name, pred in res_body['predictions'].items():
                print(f"  - {model_name}: Risk: {pred['risk_level']} (Prob: {pred['probability']*100:.2f}%)")
            return True
        else:
            print(f"[FAIL] Prediction API failed (Status: {status}, Response: {res_body})")
            return False
    except Exception as e:
        print(f"[FAIL] Prediction API call failed: {e}")
        return False

if __name__ == '__main__':
    print("Testing locally running Flask application...\n")
    all_ok = True
    all_ok &= test_route("http://127.0.0.1:5000/", "Early Warning Page (index)")
    all_ok &= test_route("http://127.0.0.1:5000/dashboard", "Disaster Response Page (dashboard)")
    all_ok &= test_route("http://127.0.0.1:5000/validation", "Model Validation Page (validation)")
    all_ok &= test_prediction_api()
    
    if all_ok:
        print("\nAll backend routes and API simulation tests passed successfully!")
    else:
        print("\n[WARNING] Some validation tests failed. Check server logs.")
