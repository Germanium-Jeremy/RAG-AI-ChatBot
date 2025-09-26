from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import threading
import time
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize the sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Hugging Face Configuration - CORRECT API URL
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

# CORRECT: Models that actually work with the Inference API
# These are specifically models that support the text-generation pipeline
GUARANTEED_WORKING_MODELS = [
    "gpt2",  # Text generation
    "distilgpt2",  # Text generation
    "microsoft/DialoGPT-small",  # Dialogue
    "microsoft/DialoGPT-medium",  # Dialogue
    "facebook/bart-large-cnn",  # Summarization but can do text generation
    "google/flan-t5-small",  # Text-to-text
]

CURRENT_MODEL = GUARANTEED_WORKING_MODELS[0]  # Start with GPT-2

# CORRECT API URL - This is the key fix!
HF_API_URL = f"https://api-inference.huggingface.co/models/{CURRENT_MODEL}"

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

# Sample knowledge base
knowledge_base = {
    "programming": [
        "Our platform supports Python, JavaScript, and Java programming languages.",
        "You can find code examples in the 'Examples' section of our website.",
        "To get help with debugging, use the 'Help' forum where experts assist with coding problems.",
        "We offer interactive coding exercises for beginners to advanced developers."
    ],
    "english": [
        "We offer grammar checking tools in the 'Tools' section of our platform.",
        "You can practice English with our interactive exercises and quizzes.",
        "Our vocabulary builder helps you learn new words daily with spaced repetition.",
        "We provide writing assistance for essays, emails, and professional documents."
    ],
    "french": [
        "Nous proposons des cours de français pour tous les niveaux, débutant à avancé.",
        "Utilisez notre outil de conjugaison pour pratiquer les verbes français.",
        "Rejoignez notre club de conversation française chaque jeudi à 18h.",
        "Nous avons des exercices interactifs pour améliorer votre compréhension orale."
    ],
    "mathematics": [  # Added math content
        "Calculus is a branch of mathematics focused on limits, functions, derivatives, and integrals.",
        "We offer mathematics courses including algebra, geometry, calculus, and statistics.",
        "Basic calculus covers differentiation and integration concepts.",
        "Our platform provides step-by-step calculus tutorials and practice problems."
    ],
    "navigation": [
        "The main menu is located on the left side of the screen with all major sections.",
        "Use the search bar at the top to find specific content or features.",
        "Your profile can be accessed by clicking on your avatar in the top right corner.",
        "The dashboard provides an overview of your progress and recent activity."
    ]
}

# Convert knowledge base to embeddings
knowledge_embeddings = {}
for category, texts in knowledge_base.items():
    knowledge_embeddings[category] = model.encode(texts)

def test_model_availability(model_name):
    """Test if a model is available through the API"""
    test_url = f"https://api-inference.huggingface.co/models/{model_name}"
    try:
        response = requests.get(test_url, headers=headers, timeout=10)
        return response.status_code == 200
    except:
        return False

def query_hugging_face_model(prompt):
    """Send prompt to Hugging Face API and get response"""
    # CORRECT: Use the proper API URL
    api_url = f"https://api-inference.huggingface.co/models/{CURRENT_MODEL}"
    
    # Simple payload for text generation models
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "return_full_text": False
        },
        "options": {
            "use_cache": True,
            "wait_for_model": True
        }
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 404:
            return f"Error: Model {CURRENT_MODEL} not found or not available via Inference API."
        elif response.status_code == 503:
            return "The model is currently loading. Please try again in 30-60 seconds."
        
        response.raise_for_status()
        result = response.json()
        
        # Handle response format
        if isinstance(result, list) and len(result) > 0:
            if "generated_text" in result[0]:
                return result[0]['generated_text']
            return str(result[0])
        elif isinstance(result, dict):
            if "generated_text" in result:
                return result['generated_text']
        
        return str(result)
            
    except requests.exceptions.RequestException as e:
        return f"API Error: {str(e)}"

def retrieve_relevant_info(query, top_k=2):
    """Retrieve the most relevant information from the knowledge base"""
    query_embedding = model.encode([query])
    best_matches = []
    
    for category, embeddings in knowledge_embeddings.items():
        similarities = cosine_similarity(query_embedding, embeddings)
        best_indices = np.argsort(similarities[0])[-top_k:][::-1]
        
        for idx in best_indices:
            if similarities[0][idx] > 0.3:
                best_matches.append((knowledge_base[category][idx], similarities[0][idx]))
    
    best_matches.sort(key=lambda x: x[1], reverse=True)
    return [match[0] for match in best_matches[:3]]

def format_prompt(query, context):
    """Format the prompt for the language model with RAG context"""
    context_str = "\n".join([f"- {text}" for text in context])
    
    return f"""Based on the following information:
{context_str}

Please answer this question: {query}

Answer:"""

def smart_fallback_response(user_input, context):
    """Generate a smart response without API calls"""
    if context:
        return f"Based on our information: {context[0]}"
    
    user_input_lower = user_input.lower()
    
    if any(word in user_input_lower for word in ['python', 'programming', 'code']):
        return "I can help with programming topics! Our platform supports Python, JavaScript, and Java."
    elif any(word in user_input_lower for word in ['english', 'grammar', 'vocabulary']):
        return "For English learning, we offer grammar tools, vocabulary builders, and writing assistance."
    elif any(word in user_input_lower for word in ['french', 'français']):
        return "Pour le français, nous proposons des cours interactifs."
    elif any(word in user_input_lower for word in ['calculus', 'math', 'mathematics']):
        return "We offer mathematics courses including calculus, algebra, and statistics."
    else:
        return "I'm here to help with educational topics. What would you like to know?"

def model_response(user_input):
    """Enhanced model response with RAG and Hugging Face integration"""
    # Handle greetings without API call
    greetings = ["hi", "hello", "hey", "greetings", "howdy"]
    if user_input.lower() in greetings:
        return "Hello! How can I assist you with programming, English, French, or mathematics today?"
    
    # Retrieve relevant context
    context = retrieve_relevant_info(user_input)
    
    # Try to use Hugging Face API
    if context:
        prompt = format_prompt(user_input, context)
    else:
        prompt = f"Please answer this question: {user_input}"
    
    api_response = query_hugging_face_model(prompt)
    
    # If API returns an error, use fallback
    if "Error" in api_response or "unavailable" in api_response.lower():
        return smart_fallback_response(user_input, context)
    else:
        return api_response

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_input = data.get('input')
    
    if not user_input:
        return jsonify({"error": "Input is required"}), 400
    
    response = model_response(user_input)
    return jsonify({
        "response": response, 
        "model_used": CURRENT_MODEL
    })

@app.route('/test-models', methods=['GET'])
def test_models():
    """Test which models are available"""
    working_models = []
    for model_name in GUARANTEED_WORKING_MODELS:
        if test_model_availability(model_name):
            working_models.append(model_name)
    
    return jsonify({
        "working_models": working_models,
        "current_model": CURRENT_MODEL
    })

@app.route('/models/<model_name>', methods=['POST'])
def switch_model(model_name):
    """Switch to a different model"""
    global CURRENT_MODEL
    if model_name in GUARANTEED_WORKING_MODELS:
        CURRENT_MODEL = model_name
        return jsonify({"message": f"Switched to model: {model_name}"})
    else:
        return jsonify({"error": "Model not available"}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "model": CURRENT_MODEL,
        "api_url": f"https://api-inference.huggingface.co/models/{CURRENT_MODEL}"
    })

def run_flask():
    app.run(debug=True, port=5000, use_reloader=False)

if __name__ == '__main__':
    if not HF_API_TOKEN:
        print("⚠️  Warning: HF_API_TOKEN environment variable not set!")
        print("💡 Create a .env file with: HF_API_TOKEN=your_token_here")
        print("🔗 Get token from: https://huggingface.co/settings/tokens")
    
    print("🔍 Testing model availability...")
    working_models = []
    for model_name in GUARANTEED_WORKING_MODELS:
        if test_model_availability(model_name):
            working_models.append(model_name)
    
    if working_models:
        CURRENT_MODEL = working_models[0]
        print(f"✅ Using model: {CURRENT_MODEL}")
        print(f"📋 Available models: {working_models}")
    else:
        print("❌ No working models found. Using fallback mode only.")
    
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    time.sleep(2)
    
    print(f"\n🤖 RAG Chatbot is running! http://localhost:5000")
    print("💬 Try questions about: programming, english, french, mathematics")
    print("🧪 Test models: GET http://localhost:5000/test-models")
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if user_input.lower() == 'exit':
                print("👋 Exiting...")
                break
            elif user_input.lower() == 'test models':
                with app.test_client() as client:
                    response = client.get('/test-models')
                    data = response.get_json()
                    print(f"✅ Working models: {data['working_models']}")
                continue
            
            if not user_input:
                continue
                
            with app.test_client() as client:
                response = client.post('/ask', json={"input": user_input})
                if response.status_code == 200:
                    data = response.get_json()
                    print(f"🤖 Bot: {data['response']}")
                else:
                    print(f"❌ Error: {response.get_json()}")
                    
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break