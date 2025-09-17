from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pairwise import cosine_similarity
import threading
import time
import requests
import os

app = Flask(__name__)
CORS(app)

# Initialize the sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Hugging Face Configuration - Using guaranteed working models
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

# Models that are guaranteed to work with the inference API
GUARANTEED_WORKING_MODELS = [
     "gpt2",  # Always available, reliable
     "distilgpt2",  # Lightweight version of GPT-2
     "microsoft/DialoGPT-small",  # Smaller but reliable
     "microsoft/DialoGPT-medium",
     "bert-base-uncased",  # For classification tasks
     "t5-small",  # Good for text-to-text generation
     "facebook/bart-base"  # Good for text generation
]

CURRENT_MODEL = GUARANTEED_WORKING_MODELS[0]  # Start with GPT-2
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

def query_hugging_face_model(prompt):
     """Send prompt to Hugging Face API and get response"""
     # Standard payload for most models
     payload = {
          "inputs": prompt,
          "parameters": {
               "max_new_tokens": 50,
               "temperature": 0.7,
               "top_p": 0.9,
               "return_full_text": False
          },
          "options": {
               "use_cache": True,
               "wait_for_model": False  # Don't wait if model is loading
          }
     }
     
     try:
          response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=15)
          
          if response.status_code == 404:
               # Model not found, try the next one in the list
               switch_to_next_model()
               return "Model was unavailable. Switched to a different model. Please try your question again."
          
          response.raise_for_status()
          result = response.json()
          
          # Handle different response formats
          if isinstance(result, list) and len(result) > 0:
               if "generated_text" in result[0]:
                    return result[0]['generated_text']
               return str(result[0])  # Fallback
          elif isinstance(result, dict) and "generated_text" in result:
               return result['generated_text']
          else:
               return "I received an unexpected response from the AI service."
               
     except requests.exceptions.RequestException as e:
          print(f"API Error: {e}")
          if hasattr(e, 'response') and e.response is not None:
               if e.response.status_code == 404:
                    switch_to_next_model()
                    return "Model unavailable. Switched to backup. Please try again."
               elif e.response.status_code == 503:
                    return "The model is currently loading. Please try again in a few moments."
          return "I'm experiencing temporary technical difficulties. Please try again later."

def switch_to_next_model():
     """Switch to the next available model"""
     global CURRENT_MODEL, HF_API_URL
     current_index = GUARANTEED_WORKING_MODELS.index(CURRENT_MODEL)
     next_index = (current_index + 1) % len(GUARANTEED_WORKING_MODELS)
     CURRENT_MODEL = GUARANTEED_WORKING_MODELS[next_index]
     HF_API_URL = f"https://api-inference.huggingface.co/models/{CURRENT_MODEL}"
     print(f"Auto-switched to model: {CURRENT_MODEL}")

def retrieve_relevant_info(query, top_k=3):
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

def model_response(user_input):
     """Enhanced model response with RAG and Hugging Face integration"""
     # Handle greetings without API call
     greetings = ["hi", "hello", "hey", "greetings", "howdy", "hola"]
     if user_input.lower() in greetings:
          return "Hello! I'm here to help with programming, English, and French topics. How can I assist you today?"
     
     # Handle thanks without API call
     thanks = ["thanks", "thank you", "thank", "appreciate it"]
     if user_input.lower() in thanks:
          return "You're welcome! Is there anything else I can help you with?"
     
     # Retrieve relevant context
     context = retrieve_relevant_info(user_input)
     
     if context:
          prompt = format_prompt(user_input, context)
     else:
          prompt = f"Please answer this question helpfully: {user_input}"
     
     response = query_hugging_face_model(prompt)
     return response

@app.route('/ask', methods=['POST'])
def ask():
     data = request.get_json()
     user_input = data.get('input')
     
     if not user_input:
          return jsonify({"error": "Input is required"}), 400
     
     response = model_response(user_input)
     return jsonify({"response": response, "model_used": CURRENT_MODEL})

@app.route('/models', methods=['GET'])
def get_models():
     """Endpoint to get available models and switch between them"""
     return jsonify({
          "available_models": GUARANTEED_WORKING_MODELS,
          "current_model": CURRENT_MODEL
     })

@app.route('/models/<model_name>', methods=['POST'])
def switch_model(model_name):
     """Switch to a different model"""
     global CURRENT_MODEL, HF_API_URL
     if model_name in GUARANTEED_WORKING_MODELS:
          CURRENT_MODEL = model_name
          HF_API_URL = f"https://api-inference.huggingface.co/models/{CURRENT_MODEL}"
          return jsonify({"message": f"Switched to model: {model_name}"})
     else:
          return jsonify({"error": "Model not available. Choose from the available_models list."}), 400

@app.route('/health', methods=['GET'])
def health_check():
     return jsonify({
          "status": "healthy", 
          "model": CURRENT_MODEL,
          "api_url": HF_API_URL
     })

def run_flask():
     app.run(debug=True, port=5000, use_reloader=False)

if __name__ == '__main__':
     if HF_API_TOKEN == os.environ.get("HF_API_TOKEN"):
          print("⚠️  Warning: Please set your HF_API_TOKEN environment variable!")
          print("💡 On Mac/Linux: export HF_API_TOKEN='your_token_here'")
          print("💡 On Windows: set HF_API_TOKEN=your_token_here")
          print("🔗 Get token from: https://huggingface.co/settings/tokens")
          # Exit or continue with fallback mode
          # You could add a fallback mode that doesn't use HF API
     
     print(f"🤖 Using model: {CURRENT_MODEL}")
     print("📋 Available models:", GUARANTEED_WORKING_MODELS)
     
     flask_thread = threading.Thread(target=run_flask)
     flask_thread.daemon = True
     flask_thread.start()
     
     time.sleep(2)
     
     print("\n✅ RAG Chatbot is running! http://localhost:5000")
     print("❌ Type 'exit' to quit")
     print("💬 Try questions about: programming, english, french, navigation")
     print("🔄 Type 'switch model' to change models")
     
     while True:
          try:
               user_input = input("\n👤 You: ").strip()
               if user_input.lower() == 'exit':
                    print("👋 Exiting...")
                    break
               elif user_input.lower() == 'switch model':
                    print("📋 Available models:")
                    for i, model in enumerate(GUARANTEED_WORKING_MODELS):
                         print(f"  {i}: {model}")
                    try:
                         choice = int(input("🔢 Enter model number: "))
                         if 0 <= choice < len(GUARANTEED_WORKING_MODELS):
                              CURRENT_MODEL = GUARANTEED_WORKING_MODELS[choice]
                              HF_API_URL = f"https://api-inference.huggingface.co/models/{CURRENT_MODEL}"
                              print(f"✅ Switched to: {CURRENT_MODEL}")
                         else:
                              print("❌ Invalid choice")
                    except ValueError:
                         print("❌ Please enter a number")
                    continue
               
               if not user_input:
                    continue
                    
               with app.test_client() as client:
                    response = client.post('/ask', json={"input": user_input})
                    if response.status_code == 200:
                         data = response.get_json()
                         print(f"🤖 Bot: {data['response']}")
                         print(f"   (using model: {data['model_used']})")
                    else:
                         print(f"❌ Error: {response.get_json()}")
                         
          except KeyboardInterrupt:
               print("\n👋 Exiting...")
               break