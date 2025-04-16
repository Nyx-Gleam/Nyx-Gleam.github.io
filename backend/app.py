#backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot_core import AILocalChatbot
import logging
import os
import pathlib
import json

# Configuración única de Flask y CORS
app = Flask(__name__)
CORS(
    app,
    resources={r"/api/*": {
        "origins": ["https://nyx-gleam.github.io", "*"],
        "supports_credentials": True
    }},
    expose_headers=["Content-Type"]
)

logging.basicConfig(level=logging.DEBUG)

# Inicializar el chatbot sin especificar un modelo - usará la selección automática
chatbot = AILocalChatbot()

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def handle_chat():
    try:
        if request.method == 'OPTIONS':
            return _build_cors_preflight_response()
        
        data = request.get_json()
        user_message = data.get('message', '')
        history = data.get('history', [])  # Obtener el historial de la solicitud
        
        if not user_message:
            return jsonify({"error": "Mensaje vacío"}), 400

        # Actualizar historial del chatbot si se proporciona
        if history:
            # Reiniciar el historial del chatbot
            chatbot.reset_history()
            
            # Reconstruir el historial desde los mensajes recibidos
            for msg in history:
                if 'role' in msg and 'content' in msg:
                    if msg['role'] == 'user':
                        # Añadir solo el mensaje del usuario
                        chatbot.conversation_history.append(msg['content'])
                    elif msg['role'] == 'assistant' and len(chatbot.conversation_history) > 0:
                        # Añadir la respuesta del asistente después de un mensaje de usuario
                        chatbot.conversation_history.append(msg['content'])

        logging.info("Procesando mensaje: %s", user_message)
        logging.debug("Historial actual: %s", chatbot.conversation_history)
        
        response = chatbot.generate_response(user_message)
        logging.info("Respuesta generada: %s", response[:50] + "...")  # Log parcial
        
        return _corsify_response(jsonify({"response": response}))

    except Exception as e:
        logging.error("Error crítico:", exc_info=True)
        return _corsify_response(jsonify({"error": str(e)}), 500)

def _build_cors_preflight_response():
    response = jsonify({"status": "preflight"})
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "POST")
    return response

def _corsify_response(response, status_code=200):
    response.headers.add("Access-Control-Allow-Origin", ["https://nyx-gleam.github.io", "*"])
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response

@app.route('/api/set_personality', methods=['POST'])
def set_personality():
    try:
        data = request.get_json()

        ___name__ = data.get('name', '')
        description = data.get('description', '')
        traits = data.get('traits', [])
        speaking_style = data.get('speaking_style', '')
        interests = data.get('interests', [])
        system_prompt = data.get('system_prompt', '')
        long_description = data.get('long_description', '')

        if not ___name__:
            return jsonify({"error": "Nombre vacío"}), 400

        if not description:
            return jsonify({"error": "Descripción vacía"}), 400

        if not traits:
            return jsonify({"error": "Rasgos vacíos"}), 400

        if not speaking_style:
            return jsonify({"error": "Estilo de hablar vacío"}), 400

        if not interests:
            return jsonify({"error": "Intereses vacíos"}), 400

        if not system_prompt:
            return jsonify({"error": "Indicador del sistema vacío"}), 400

        if not long_description:
            return jsonify({"error": "Descripción larga vacía"}), 400

        with open("config/personality.json", "w") as f:
            json.dump({"name": ___name__, "description": description, "traits": traits, "speaking_style": speaking_style, "interests": interests, "system_prompt": system_prompt, "long_description": long_description}, f)
        return _corsify_response(jsonify({"name": ___name__, "description": description, "traits": traits, "speaking_style": speaking_style, "interests": interests, "system_prompt": system_prompt, "long_description": long_description}))

    except Exception as e:
        logging.error("Error crítico:", exc_info=True)
        return _corsify_response(jsonify({"error": str(e)}), 500)

def _corsify_response(response, status_code=200):
    response.headers.add("Access-Control-Allow-Origin", "https://nyx-gleam.github.io")
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response

@app.route('/api/models', methods=['GET'])
def list_models():
    try:
        model_dirs = ['models']
        model_files = []

        for model_dir in model_dirs:
            path = pathlib.Path(model_dir)
            if path.exists():
                model_files += [f.name for f in path.glob("*.gguf") if f.is_file()]

        return _corsify_response(jsonify(model_files))
    except Exception as e:
        logging.error("Error al listar modelos:", exc_info=True)
        return _corsify_response(jsonify({"error": str(e)}), 500)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "code": "200"
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
    # Para producción, considera usar un servidor WSGI como gunicorn o uWSGI
    # app.run(debug=True)  # Para desarrollo, habilita el modo debug