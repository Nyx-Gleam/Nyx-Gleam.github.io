import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from pathlib import Path
from ..backend.chatbot_core import AILocalChatbot
from ..continuous_learning import ContinuousLearningSystem

class ChatbotGUI:
    def __init__(self, master, config_dir="config", data_dir="data", models_dir="models"):
        self.master = master
        self.config_dir = Path(config_dir)
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        
        self.setup_gui()
        self.load_default_settings()
        
    def setup_gui(self):
        # Configuración inicial de la interfaz
        self.master.title("AI Local ChatBot")
        self.master.geometry("1200x800")
        
        # Crear paneles principales
        self.create_chat_panel()
        self.create_status_bar()
        self.create_control_panel()
    
    def create_chat_panel(self):
        # Panel de conversación
        chat_frame = ttk.Frame(self.master, padding=10)
        chat_frame.pack(fill=tk.BOTH, expand=True)
        
        self.chat_history = scrolledtext.ScrolledText(
            chat_frame, wrap=tk.WORD, state=tk.DISABLED
        )
        self.chat_history.pack(fill=tk.BOTH, expand=True)
        
        # Configurar tags para formato
        self.chat_history.tag_config('user', foreground='blue')
        self.chat_history.tag_config('bot', foreground='green')
        self.chat_history.tag_config('error', foreground='red')
    
    # Añadir aquí los demás métodos de la GUI