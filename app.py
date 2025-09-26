# local_app.py - Improved version
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json

app = Flask(__name__)
CORS(app)

# Try multiple possible Colab URLs
COLAB_URLS = [
     "https://myaichatbot.loca.lt",  # localtunnel URL
     "http://localhost:5000"  # Fallback for testing
]

CURRENT_COLAB_URL = COLAB_URLS[0]

def test_colab_connection():
     """Test which Colab URL is working"""
     for url in COLAB_URLS:
          try:
               response = requests.get(f"{url}/health", timeout=10)
               if response.status_code == 200:
                    print(f"✅ Connected to Colab: {url}")
                    return url
          except:
               continue
     return None

def ask_colab_server(question):
     """Send question to Colab with robust error handling"""
     global CURRENT_COLAB_URL
     
     # Test connection first
     working_url = test_colab_connection()
     if not working_url:
          return {"error": "Cannot connect to Colab server", "status": "error"}
     
     CURRENT_COLAB_URL = working_url
     
     try:
          response = requests.post(
               f"{CURRENT_COLAB_URL}/ask",
               json={"input": question},
               timeout=30,
               headers={'Content-Type': 'application/json'}
          )
          
          # Check if response is valid JSON
          try:
               result = response.json()
               return result
          except json.JSONDecodeError:
               return {"error": f"Invalid response from server: {response.text}", "status": "error"}
               
     except requests.exceptions.Timeout:
          return {"error": "Request timeout - Colab server may be busy", "status": "error"}
     except requests.exceptions.ConnectionError:
          return {"error": "Connection refused - check Colab URL", "status": "error"}
     except Exception as e:
          return {"error": f"Unexpected error: {str(e)}", "status": "error"}

@app.route('/ask', methods=['POST', 'GET'])
def ask():
     if request.method == 'GET':
          user_input = request.args.get('input', '')
     else:
          data = request.get_json() if request.is_json else {}
          user_input = data.get('input', '')
     
     if not user_input:
          return jsonify({"error": "Input is required", "status": "error"}), 400
     
     print(f"📨 Local app received: {user_input}")
     
     result = ask_colab_server(user_input)
     return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
     colab_status = "connected" if test_colab_connection() else "disconnected"
     return jsonify({
          "local_status": "healthy",
          "colab_server": colab_status,
          "current_url": CURRENT_COLAB_URL
     })

@app.route('/test', methods=['GET'])
def test():
     """Test the connection to Colab"""
     test_result = ask_colab_server("What programming languages do you teach?")
     return jsonify({"test_result": test_result})

if __name__ == '__main__':
     print("🚀 Starting Local Flask App...")
     print("📡 Testing Colab connection...")
     
     if test_colab_connection():
          print("✅ Colab server is connected!")
     else:
          print("❌ Cannot connect to Colab server")
          print("💡 Make sure your Colab notebook is running and accessible")
     
     print("🌐 Local app running on: http://localhost:5000")
     app.run(debug=True, port=5000, host='0.0.0.0')