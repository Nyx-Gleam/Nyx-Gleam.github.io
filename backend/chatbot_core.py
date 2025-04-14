import os
import json
import time
from pathlib import Path
from ..continuous_learning import ContinuousLearningSystem
from ctransformers import AutoModelForCausalLM

class AILocalChatbot:
    def __init__(self, model_path, config_dir="config", data_dir="data"):
        self.model_path = model_path
        self.config_dir = Path(config_dir)
        self.data_dir = Path(data_dir)
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
            config_file=str(self.config_dir / 'model_config.yaml'),
            context_length=2048,
            gpu_layers=0
        )
    
    def _detect_model_type(self, model_file):
        # Lógica de detección de tipo de modelo
        if "llama" in model_file: return "llama"
        if "mistral" in model_file: return "mistral"
        if "gpt2" in model_file: return "gpt2"
        return "llama"
    
    # Añadir aquí los demás métodos (generate_response, format_prompt, etc.)