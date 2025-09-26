from flask import Flask, request, jsonify
from flask_cors import CORS  # Add CORS support
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import threading

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load models
print("Loading models...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
print("✅ Models loaded!")

# Your knowledge base
knowledge_base = {
     "programming": [
          "Python is a programming language used for web development, data science, and automation.",
          "JavaScript is used for web development and can run in browsers.",
          "Java is an object-oriented programming language for enterprise applications.",
          "Our platform offers coding exercises and projects for all skill levels."
     ],
     "english": [
          "English grammar includes tenses, articles, and sentence structure rules.",
          "Vocabulary building involves learning new words and their usage contexts.",
          "Our platform provides interactive English lessons and practice exercises.",
          "You can improve pronunciation through our speech recognition tools."
     ],
     "french": [
          "Le français est une langue romane parlée par millions de personnes.",
          "La conjugaison française inclut des verbes réguliers et irréguliers.",
          "Notre plateforme offre des cours de français pour débutants et avancés.",
          "La grammaire française comprend les genres, les articles et les accords."
     ],
     "mathematics": [  # Added math content for your calculus question
          "Calculus is a branch of mathematics focused on limits, functions, derivatives, and integrals.",
          "We offer mathematics courses including algebra, geometry, calculus, and statistics.",
          "Basic calculus covers differentiation and integration concepts.",
          "Our platform provides step-by-step calculus tutorials and practice problems."
     ],
     "navigation": [
          "Use the menu on the left to access different sections of the platform.",
          "The search bar at the top helps you find specific content quickly.",
          "Your profile settings can be adjusted in the top-right corner.",
          "The dashboard shows your progress and recent activities."
     ]
}

# Convert to embeddings
print("Creating knowledge embeddings...")
knowledge_embeddings = {}
for category, texts in knowledge_base.items():
     knowledge_embeddings[category] = embedding_model.encode(texts)
print("✅ Knowledge base ready!")

def retrieve_relevant_info(query):
     query_embedding = embedding_model.encode([query])
     best_matches = []
     
     for category, embeddings in knowledge_embeddings.items():
          similarities = cosine_similarity(query_embedding, embeddings)
          best_indices = np.argsort(similarities[0])[-2:][::-1]
          
          for idx in best_indices:
               if similarities[0][idx] > 0.3:
                    best_matches.append((knowledge_base[category][idx], similarities[0][idx]))
     
     best_matches.sort(key=lambda x: x[1], reverse=True)
     return [match[0] for match in best_matches[:3]]

@app.route('/ask', methods=['POST', 'GET'])  # Allow both POST and GET
def ask():
     try:
          # Handle both JSON and form data
          if request.method == 'GET':
               user_input = request.args.get('input', '')
          else:
               if request.is_json:
                    data = request.get_json()
                    user_input = data.get('input', '')
               else:
                    user_input = request.form.get('input', '')
          
          if not user_input:
               return jsonify({"error": "Input is required"}), 400
          
          print(f"📥 Received: {user_input}")
          
          # Retrieve relevant context
          context_pieces = retrieve_relevant_info(user_input)
          context = " ".join(context_pieces) if context_pieces else "Our educational platform offers courses in programming, English, French, and mathematics."
          
          # Get answer from QA model
          result = qa_model(question=user_input, context=context)
          
          response_data = {
               "response": result['answer'],
               "confidence": float(result['score']),
               "context_used": len(context_pieces) > 0,
               "status": "success"
          }
          
          print(f"📤 Response: {result['answer'][:100]}...")
          return jsonify(response_data)
          
     except Exception as e:
          error_msg = f"Server error: {str(e)}"
          print(f"❌ {error_msg}")
          return jsonify({"error": error_msg, "status": "error"}), 500

@app.route('/health', methods=['GET'])
def health():
     return jsonify({
          "status": "healthy", 
          "service": "AI Chatbot API",
          "endpoints": {
               "POST /ask": "Ask questions",
               "GET /ask?input=question": "Ask via GET",
               "GET /health": "Health check"
          }
     })

@app.route('/test', methods=['GET'])
def test():
     """Simple test endpoint"""
     return jsonify({"message": "Server is working!", "test": "success"})

# Simple way to expose the app
if __name__ == '__main__':
     print("🚀 Starting Colab AI Server...")
     
     # Use localtunnel for easy public access
     !npm install -g localtunnel
     import threading
     import time
     
     def start_flask():
          app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
     
     # Start Flask in background
     flask_thread = threading.Thread(target=start_flask)
     flask_thread.daemon = True
     flask_thread.start()
     
     # Wait for Flask to start
     time.sleep(3)
     
     # Start localtunnel
     print("🌐 Starting localtunnel...")
     !lt --port 5000 --subdomain myaichatbot &
     
     time.sleep(5)
     print("✅ Server should be available at: https://myaichatbot.loca.lt")
     print("💡 Keep this Colab running!")
     
     # Keep the cell alive
     try:
          while True:
               time.sleep(10)
     except KeyboardInterrupt:
          print("Server stopped.")