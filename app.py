from flask import Flask, request, jsonify
import multiprocessing

app = Flask(__name__)

# Dummy model response function for demonstration
def model_response(user_input):
     # Here you would implement the logic to interact with your RAG model
     return f"Model response to: {user_input}"

@app.route('/ask', methods=['POST'])
def ask():
     data = request.get_json()
     user_input = data.get('input')
     
     if not user_input:
          return jsonify({"error": "Input is required"}), 400
     
     # Get the model's response
     response = model_response(user_input)
     
     return jsonify({"response": response})

def run_flask():
     app.run(debug=True, port=5000)

def terminal_input():
     while True:
          user_input = input("Enter your question (or 'exit' to quit): ")
          if user_input.lower() == 'exit':
               print("Exiting...")
               break
          
          # Send request to the Flask app
          response = app.test_client().post('/ask', json={"input": user_input})
          print(f"Response: {response.get_json()['response']}")

if __name__ == '__main__':
     # Start Flask app in a separate process
     flask_process = multiprocessing.Process(target=run_flask)
     flask_process.start()
     
     # Start terminal input loop
     terminal_input()
     
     # Ensure the Flask process is terminated when done
     flask_process.terminate()
