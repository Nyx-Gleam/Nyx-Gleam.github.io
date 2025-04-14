import sys
import os
# Removemos el append del sys.path si usamos rutas absolutas basadas en __file__
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
from pathlib import Path
from continuous_learning import ContinuousLearningSystem
from ctransformers import AutoModelForCausalLM

class AILocalChatbot:
    def __init__(self, model_path, config_dir="config", data_dir="data"):
        self.model_path = model_path
        
        # Obtener la carpeta raíz del proyecto: asumiendo que este archivo está en backend/
        base_dir = Path(__file__).resolve().parent.parent
        self.config_dir = base_dir / config_dir
        self.data_dir = base_dir / data_dir
        
        self.conversation_history = []
        
        # Cargar configuraciones
        self.personality = self._load_config("personality.json")
        self.settings = self._load_config("settings.json")
        
        # Inicializar subsistemas
        self.learning_system = ContinuousLearningSystem(
            data_dir=self.data_dir,
            config_dir=self.config_dir
        )
        
        # Cargar modelo
        self._load_model()
        
    def _load_config(self, filename):
        config_path = self.config_dir / filename
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_model(self):
        model_file = Path(self.model_path).name.lower()
        model_type = self._detect_model_type(model_file)
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            model_type=model_type,
            context_length=2048,
            gpu_layers=0
        )
    
    def _detect_model_type(self, model_file):
        # Lógica de detección de tipo de modelo
        if "llama" in model_file: return "llama"
        if "mistral" in model_file: return "mistral"
        if "gpt2" in model_file: return "gpt2"
        return "llama"

    def format_prompt(self, user_input):
        system_prompt = self.personality.get("system_prompt", "")
        history_formatted = ""
        for i, message in enumerate(self.conversation_history[-self.settings.get("history_length", 5):]):
            role = "User" if i % 2 == 0 else "Assistant"
            history_formatted += f"{role}: {message}\n"
        full_prompt = f"{system_prompt}\n{history_formatted}User: {user_input}\nAssistant:"
        return full_prompt

    def generate_response(self, user_input):
        prompt = self.format_prompt(user_input)

        response = ""
        for token in self.llm(prompt, stream=True):
            response += token

        # Guardar en el historial
        self.conversation_history.append(user_input)
        self.conversation_history.append(response.strip())

        # Aprendizaje continuo
        self.learning_system.save_interaction(user_input, response.strip())

        return response.strip()

    def reset_history(self):
        self.conversation_history = []
