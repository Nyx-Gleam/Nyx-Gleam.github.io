# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot_core import AILocalChatbot
import logging

app = Flask(__name__)
CORS(app)  # Permite solicitudes desde cualquier origen (*)

logging.basicConfig(level=logging.DEBUG)

# Inicializar el chatbot (ajusta la ruta del modelo)
chatbot = AILocalChatbot("../models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "Mensaje vacío"}), 400
        
        response = chatbot.generate_response(user_message)
        return jsonify({"response": response})
    
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": "Error interno del servidor"}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "Servidor en funcionamiento"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    # Para producción, considera usar un servidor WSGI como gunicorn o uWSGI
    # app.run(debug=True)  # Para desarrollo, habilita el modo debug