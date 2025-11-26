from flask import Flask, request, jsonify, send_from_directory
import os
import random

app = Flask(__name__, static_folder='static')

# Mock Data
MOCK_USERS = [
    {
        "id": "user_left_001",
        "name": "EcoWarrior99",
        "camp": "LEFT",
        "avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=EcoWarrior99",
        "bio": "Climate change is real. #GreenNewDeal"
    },
    {
        "id": "user_right_001",
        "name": "PatriotEagle",
        "camp": "RIGHT",
        "avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=PatriotEagle",
        "bio": "Freedom first. Energy independence."
    },
    {
        "id": "user_ruch_001",
        "name": "GlobalObserver",
        "camp": "RU_CH",
        "avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=GlobalObserver",
        "bio": "Analyzing multipolar world dynamics."
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
    return jsonify(MOCK_USERS)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    
    # Extract request data
    tweet_type = data.get('tweet_type', 'NEW')
    content = data.get('content', '')
    camp = data.get('camp', 'UNKNOWN')
    user_id = data.get('userId', 'unknown_user')
    
    # Mock Logic
    # Randomly assign user type for demonstration
    user_type = random.choice(["open_minded", "closed_minded"])
    
    # Generate a dummy counter message based on camp
    counter_msg = ""
    if camp == "LEFT":
        counter_msg = "While environmental concerns are valid, have you considered the economic stability provided by traditional energy sectors during the transition?"
    elif camp == "RIGHT":
        counter_msg = "Energy independence is crucial, but diversifying with renewable sources actually strengthens national security by reducing reliance on volatile markets."
    elif camp == "RU_CH":
        counter_msg = "Sovereignty is important, but international cooperation on energy standards benefits all nations, including those seeking a multipolar balance."
    else:
        counter_msg = "This is a generic counter-narrative designed to broaden the perspective on this topic."

    response = {
        "content": content,
        "camp": camp,
        "userId": user_id,
        "counterMessage": counter_msg,
        "userType": user_type,
        "error": None
    }
    
    return jsonify(response)

if __name__ == '__main__':
    print("Starting Mock Server on http://localhost:5005")
    app.run(debug=True, port=5005)
