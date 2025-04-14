import sys
import os
import json
import time
from pathlib import Path
from continuous_learning import ContinuousLearningSystem
from ctransformers import AutoModelForCausalLM
from tkinter import Tk, messagebox

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

class AILocalChatbot:
    def __init__(self, model_path, config_dir="config", data_dir="data"):
        self.model_path = model_path
        
        # Usar la carpeta raíz del proyecto para obtener las rutas de configuración y datos
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

    def _check_internet_connection(self, host="1.1.1.1", port=53, timeout=3):
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except socket.error:
            return False

    def _show_error_popup(self, message):
        root = Tk()
        root.withdraw()
        messagebox.showerror("Connection Error", message)
        root.destroy()
    
    def _load_model(self):
        model_file = Path(self.model_path)
        model_name = model_file.name.lower()
        model_type = self._detect_model_type(model_name)

        if model_file.exists():
            print(f"[INFO] Loading model from local path: {model_file}")
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                model_type=model_type,
                context_length=2048,
                gpu_layers=0
            )
        else:
            print("[WARNING] Local model not found.")
            if self._check_internet_connection():
                print("[INFO] Internet connection available. Loading model from Hugging Face...")
                self.llm = AutoModelForCausalLM.from_pretrained(
                    "NyxGleam/mistral-7b-instruct-v0.1.Q4_K_M",
                    model_type=model_type,
                    context_length=2048,
                    gpu_layers=0
                )
            else:
                error_message = (
                    "No local model found and no internet connection is available.\n\n"
                    "Please check your network connection and try again."
                )
                self._show_error_popup(error_message)
                raise RuntimeError(error_message)
    
    def _detect_model_type(self, model_file):
        model_file = model_file.lower()

        if any(keyword in model_file for keyword in ["llama", "alpaca", "vicuna", "guanaco", "wizardlm", "koala"]):
            return "llama"
        if any(keyword in model_file for keyword in ["mistral", "mixtral"]):
            return "mistral"
        if any(keyword in model_file for keyword in ["gpt2", "distilgpt2"]):
            return "gpt2"
        if any(keyword in model_file for keyword in ["gptj"]):
            return "gptj"
        if any(keyword in model_file for keyword in ["gptneox", "neox", "redpajama", "pythia"]):
            return "gptneox"
        if any(keyword in model_file for keyword in ["falcon"]):
            return "falcon"
        if any(keyword in model_file for keyword in ["replit"]):
            return "replit"
        if any(keyword in model_file for keyword in ["bloom"]):
            return "bloom"
        if any(keyword in model_file for keyword in ["starcoder", "codegen", "codellama"]):
            return "starcoder"
        if any(keyword in model_file for keyword in ["xgen"]):
            return "xgen"
        if any(keyword in model_file for keyword in ["openllama"]):
            return "openllama"

        # Si no se reconoce, usar un valor por defecto
        print(f"[WARNING] Model not recognized in name '{model_file}'. Using  'llama' by default.")
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