from flask import Flask, request, jsonify, send_from_directory
import os
import requests

import sys

app = Flask(__name__, static_folder='static')

# Global error handler to ensure JSON responses
@app.errorhandler(Exception)
def handle_exception(e):
    print(f"[Flask] Unhandled exception: {str(e)}", flush=True)
    import traceback
    traceback.print_exc()
    return jsonify({
        "content": "",
        "camp": "",
        "userId": "",
        "counterMessage": "",
        "userType": None,
        "error": f"Server error: {str(e)}"
    }), 500

# Backend API URL
BACKEND_URL = os.environ.get('BACKEND_URL', 'http://localhost:8000')

# Real Users from the dataset
USERS = [
    {
        "id": "tw1042226763286294528",
        "name": "GlobalObserver",
        "camp": "RU_CH",
        "avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=GlobalObserver",
        "bio": "Analyzing multipolar world dynamics."
    },
    {
        "id": "tw962651436994658304",
        "name": "EcoWarrior99",
        "camp": "LEFT",
        "avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=EcoWarrior99",
        "bio": "Climate change is real. #GreenNewDeal"
    },
    {
        "id": "tw176348732",
        "name": "PatriotEagle",
        "camp": "RIGHT",
        "avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=PatriotEagle",
        "bio": "Freedom first. Energy independence."
    }
]

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/users', methods=['GET'])
def get_users():
    return jsonify(USERS)

@app.route('/generate', methods=['POST'])
def generate():
    """Proxy to FastAPI backend's /generate_counter_message endpoint."""
    # Log raw request info
    print(f"[Flask] === New /generate request ===", flush=True)
    print(f"[Flask] Content-Type: {request.content_type}", flush=True)
    print(f"[Flask] Content-Length: {request.content_length}", flush=True)
    print(f"[Flask] Raw data: {request.get_data(as_text=True)[:500]}", flush=True)
    
    try:
        data = request.json
        if data is None:
            print(f"[Flask] ERROR: No JSON data received", flush=True)
            return jsonify({
                "content": "",
                "camp": "",
                "userId": "",
                "counterMessage": "",
                "userType": None,
                "error": "No JSON data received"
            }), 400
    except Exception as e:
        print(f"[Flask] ERROR parsing JSON: {str(e)}", flush=True)
        return jsonify({
            "content": "",
            "camp": "",
            "userId": "",
            "counterMessage": "",
            "userType": None,
            "error": f"Invalid JSON: {str(e)}"
        }), 400
    
    print(f"[Flask] Parsed JSON: {data}", flush=True)
    print(f"[Flask] Forwarding to: {BACKEND_URL}/generate_counter_message", flush=True)
    
    try:
        # Forward request to FastAPI backend
        response = requests.post(
            f"{BACKEND_URL}/generate_counter_message",
            json=data,
            headers={'Content-Type': 'application/json'},
            timeout=120  # Increased timeout for LLM generation
        )
        
        # Check if backend returned an error status
        if response.status_code != 200:
            print(f"[Flask] Backend returned status {response.status_code}: {response.text}", flush=True)
            return jsonify({
                "content": data.get('content', ''),
                "camp": data.get('camp', ''),
                "userId": data.get('userId', ''),
                "counterMessage": "",
                "userType": None,
                "error": f"Backend error: {response.status_code}"
            }), response.status_code
        
        result = response.json()
        print(f"[Flask] Backend response received, counterMessage length: {len(result.get('counterMessage', ''))}", flush=True)
        return jsonify(result)
    except requests.exceptions.Timeout:
        print(f"[Flask] Request timed out after 120s", flush=True)
        return jsonify({
            "content": data.get('content', ''),
            "camp": data.get('camp', ''),
            "userId": data.get('userId', ''),
            "counterMessage": "",
            "userType": None,
            "error": "Request timed out. Please try again."
        }), 504
    except requests.exceptions.RequestException as e:
        print(f"[Flask] Backend error: {str(e)}")
        return jsonify({
            "content": data.get('content', ''),
            "camp": data.get('camp', ''),
            "userId": data.get('userId', ''),
            "counterMessage": "",
            "userType": None,
            "error": f"Backend connection error: {str(e)}"
        }), 500

if __name__ == '__main__':
    print(f"Starting Flask UI Server on http://localhost:5005")
    print(f"Backend URL: {BACKEND_URL}")
    # Disable reloader to prevent mid-request restarts on HPC
    app.run(debug=True, port=5005, use_reloader=False)
