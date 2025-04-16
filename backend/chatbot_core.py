import sys
import os
import json
import time
import socket
import logging
import traceback
import re
import hashlib
import random
import requests

import tkinter as tk

from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from continuous_learning import ContinuousLearningSystem
from ctransformers import AutoModelForCausalLM
from tkinter import Tk, messagebox, simpledialog
from duckduckgo_search import DDGS
from datetime import datetime
from PIL import Image, ImageTk
from tkinter import ttk

class AILocalChatbot:
    def __init__(self, model_path=None, config_dir="config", data_dir="data"):
        self.MODEL_CONFIG = {
            "llama": {
                "context_length": 4096,
                "temperature": 0.7,
                "max_new_tokens": 512,
                "prompt_template": "<|system|>\n{system_prompt}</s>\n<|user|>\n{user_input}</s>\n<|assistant|>",
                "stop_sequences": ["</s>", "<|"],
                "recommended_models": ["Llama-2", "Alpaca", "Vicuna"]
            },
            "mistral": {
                "context_length": 8192,
                "temperature": 0.6,
                "max_new_tokens": 512,
                "prompt_template": "<s>[INST] {system_prompt}\n\n{user_input} [/INST]",
                "stop_sequences": ["</s>", "[INST]"],
                "recommended_models": ["Mistral-7B", "Mixtral-8x7B"]
            },
            "gpt2": {
                "context_length": 1024,
                "temperature": 0.9,
                "max_new_tokens": 256,
                "prompt_template": "{system_prompt}\n\nUsuario: {user_input}\nAsistente:",
                "stop_sequences": ["\n"]
            },
            "falcon": {
                "context_length": 2048,
                "temperature": 0.5,
                "max_new_tokens": 400,
                "prompt_template": "System: {system_prompt}\nUser: {user_input}\nFalcon:",
                "stop_sequences": ["\nUser:"]
            },
            "starcoder": {
                "context_length": 4096,
                "temperature": 0.3,
                "max_new_tokens": 512,
                "prompt_template": "// System: {system_prompt}\n// User: {user_input}\n// Assistant:",
                "code_completion": True
            },
            "deepseek": {
                "context_length": 4096,
                "max_new_tokens": 2048,
                "temperature": 0.2,
                "prompt_template": (
                    "### System:\n{system_prompt}\n\n"
                    "### History:\n{history}\n\n"
                    "### User:\n{user_input}\n\n"
                    "### Assistant:\n"
                ),
                "stop_sequences": ["###"],
                "code_specialist": True
            },
            "deepseek-code": {
                    "context_length": 4096,
                    "max_new_tokens": 1024,
                    "temperature": 0.2,
                    "prompt_template": (
                        "### System:\n{system_prompt}\n\n"
                        "### User:\n{user_input}\n\n"
                        "### Assistant:\n"
                    ),
                    "stop_sequences": ["###"],
                    "code_specialist": True
            },
            "default": {
                "context_length": 2048,
                "max_new_tokens": 256,
                "temperature": 0.7,
                "stop_sequences": ["</s>"]
            }
        }

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(base_dir / "chatbot.log"),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger("AILocalChatbot")
        self.logger.info("Iniciando chatbot...")

        # Inicializar modelo_path como None
        self.model_path = None

        # Verificar que los directorios existen, crearlos si no
        self.config_dir = base_dir / config_dir
        self.data_dir = base_dir / data_dir

        for directory in [self.config_dir, self.data_dir]:
            if not directory.exists():
                self.logger.warning(f"El directorio {directory} no existe. Creándolo...")
                directory.mkdir(parents=True, exist_ok=True)

        self.conversation_history = []

        # Cargar configuraciones
        try:
            self.personality = self._load_config("personality.json")
            self.logger.info("Configuración de personalidad cargada correctamente")
        except Exception as e:
            self.logger.error(f"Error al cargar personality.json: {str(e)}")
            self.personality = {"system_prompt": "Eres un asistente virtual útil y amable. Responde de manera concisa y coherente."}

        try:
            self.settings = self._load_config("settings.json")
            self.logger.info("Configuración de ajustes cargada correctamente")
        except Exception as e:
            self.logger.error(f"Error al cargar settings.json: {str(e)}")
            self.settings = {"history_length": 3}

        try:
            self.learning_system = ContinuousLearningSystem(
                data_dir=self.data_dir,
                config_dir=self.config_dir
            )
            self.logger.info("Sistema de aprendizaje continuo inicializado")
        except Exception as e:
            self.logger.error(f"Error al inicializar el sistema de aprendizaje: {str(e)}")
            self._show_error_popup(f"No se pudo inicializar el sistema de aprendizaje: {str(e)}")

        # Inicialización del modelo
        try:
            # Si se proporcionó un model_path específico, úsalo
            if model_path:
                self.model_path = model_path
                self.logger.info(f"Usando modelo proporcionado: {model_path}")
            else:
                # Si no se proporcionó model_path, mostrar selector siempre
                self.logger.info("Mostrando selector de modelos...")
                self.model_path = self._select_model_with_fallback()

            # Intentar cargar el modelo seleccionado
            if self.model_path:
                self._load_model()
            else:
                self.logger.warning("No se pudo seleccionar un modelo local. Intentando cargar modelo de respaldo.")
                self._handle_initialization_error()
        except Exception as e:
            self.logger.error(f"Error al inicializar el modelo: {str(e)}")
            self._handle_initialization_error()

        self.output_dir = Path(__file__).resolve().parent.parent / "output"
        self._init_output_dir()

    def _select_model_with_fallback(self):
        """Muestra un selector de modelos y retorna el modelo seleccionado o None si no hay modelos"""
        model_dir = base_dir / "models"

        # Asegurarse de que el directorio existe
        if not model_dir.exists():
            try:
                model_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Se ha creado el directorio de modelos: {model_dir}")
                self._show_error_popup(f"Se ha creado el directorio de modelos: {model_dir}\nPor favor, coloca tus archivos .gguf allí e intenta de nuevo.")
            except Exception as e:
                self.logger.error(f"No se pudo crear el directorio de modelos: {str(e)}")
                self._show_error_popup(f"No se pudo crear el directorio de modelos: {str(e)}")
            return None

        # Buscar modelos .gguf
        try:
            gguf_files = [f for f in os.listdir(model_dir) if f.endswith(".gguf")]
            self.logger.info(f"Modelos encontrados: {gguf_files}")
        except Exception as e:
            self.logger.error(f"Error al buscar modelos: {str(e)}")
            self._show_error_popup(f"Error al buscar modelos: {str(e)}")
            return None

        # Si no hay modelos, retornar None
        if not gguf_files:
            self.logger.warning("No se encontraron modelos .gguf en la carpeta models")
            self._show_error_popup("No se encontraron modelos .gguf en la carpeta 'models/'\nPor favor, descarga un modelo en formato .gguf y colócalo en la carpeta 'models/' o se usará un modelo de internet.")
            return None

        # Mostrar selector de modelos siempre, incluso si solo hay uno
        try:
            # Crear ventana
            root = tk.Tk()
            root.title("Selección de Modelo")

            # Configurar ventana
            window_width = 500
            window_height = 350  # Aumentado un poco para asegurar espacio para botones
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            x_coordinate = (screen_width - window_width) // 2
            y_coordinate = (screen_height - window_height) // 2
            root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")
            root.resizable(False, False)
            root.attributes('-topmost', True)

            # Establecer icono de la ventana y barra de tareas
            icon_path = self.config_dir / "icon.png"
            if icon_path.exists():
                try:
                    # Cargar el icono para la ventana y barra de tareas
                    img = Image.open(icon_path)

                    
                    # Variable para almacenar el resultado
                    result = [None]

                    # Para la barra de tareas en Windows
                    temp_ico = self.config_dir / "temp_icon.ico"
                    img.save(temp_ico, format="ICO")
                    root.iconbitmap(default=str(temp_ico))

                    # Para sistemas basados en X11/Wayland
                    icon = ImageTk.PhotoImage(img)
                    root.iconphoto(True, icon)

                    self.logger.info(f"Iconos de ventana y barra de tareas establecidos desde: {icon_path}")
                except Exception as e:
                    self.logger.error(f"Error al cargar los iconos: {str(e)}")

            root.lift()
            root.focus_force()

            # Función para manejar la selección
            def on_select():
                try:
                    selected = listbox.get(listbox.curselection())
                    result[0] = selected
                    root.destroy()
                except Exception as e:
                    self.logger.error(f"Error al seleccionar modelo: {str(e)}")

            # Marco principal
            main_frame = ttk.Frame(root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

            # Mostrar título de la aplicación
            title_label = ttk.Label(main_frame, text="NOMBRE POR ELEGIR", font=("Arial", 18, "bold"))
            title_label.pack(pady=(0, 15))

            # Etiqueta de instrucción
            label = ttk.Label(main_frame, text="Selecciona un modelo:", font=("Arial", 12))
            label.pack(pady=(0, 10))

            # Lista de modelos
            listbox_frame = ttk.Frame(main_frame)
            listbox_frame.pack(fill=tk.BOTH, expand=True, pady=10)

            scrollbar = ttk.Scrollbar(listbox_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            listbox = tk.Listbox(listbox_frame, yscrollcommand=scrollbar.set, font=("Arial", 10), height=8)
            listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=listbox.yview)

            # Insertar modelos en la lista
            for model in gguf_files:
                listbox.insert(tk.END, model)

            # Seleccionar el primer elemento por defecto
            if gguf_files:
                listbox.selection_set(0)

            # Frame para botones en la parte inferior
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill=tk.X, pady=(15, 0))

            # Botones con colores y estilos más visibles
            select_button = tk.Button(
                button_frame, 
                text="Seleccionar", 
                command=on_select,
                bg="#4CAF50",  # Verde
                fg="white",
                padx=10,
                pady=5,
                font=("Arial", 10, "bold")
            )
            select_button.pack(side=tk.RIGHT)

            cancel_button = tk.Button(
                button_frame, 
                text="Cancelar", 
                command=root.destroy,
                bg="#f44336",  # Rojo
                fg="white",
                padx=10,
                pady=5,
                font=("Arial", 10, "bold")
            )
            cancel_button.pack(side=tk.RIGHT, padx=10)

            # Debug - verificar que los botones están empaquetados
            self.logger.info("Botones creados y empaquetados en la interfaz")

            # Garantizar que la ventana se muestre completamente antes de entrar en el loop
            root.update_idletasks()

            # Mostrar ventana y esperar
            root.mainloop()

            # Limpiar archivo temporal de icono si se creó
            try:
                if 'temp_ico' in locals() and temp_ico.exists():
                    os.remove(temp_ico)
            except:
                pass

            # Retornar resultado
            if result[0]:
                self.logger.info(f"Modelo seleccionado: {result[0]}")
                return model_dir / result[0]
            else:
                self.logger.warning("No se seleccionó ningún modelo")
                # Si no se seleccionó ninguno pero hay modelos, usar el primero
                if gguf_files:
                    self.logger.info(f"Usando el primer modelo disponible: {gguf_files[0]}")
                    return model_dir / gguf_files[0]
                return None

        except Exception as e:
            self.logger.error(f"Error en el selector de modelos: {str(e)}")
            # En caso de error, intentar usar el primer modelo disponible
            if gguf_files:
                self.logger.info(f"Usando el primer modelo disponible debido a un error: {gguf_files[0]}")
                return model_dir / gguf_files[0]
            return None

    def _init_output_dir(self):
        """Crea el directorio output si no existe"""
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.logger.info(f"Directorio de salida: {self.output_dir}")
    
    def _setup_learning_system(self):
        """Inicializa el sistema de aprendizaje continuo"""
        try:
            return ContinuousLearningSystem()
        except Exception as e:
            self.logger.error(f"Error al configurar sistema de aprendizaje: {str(e)}")
            return None

    def _generate_filename(self, code: str, language: str, user_input: str) -> str:
        """Genera nombre de archivo óptimo usando IA"""
        # Extraer posibles nombres de funciones/clases
        patterns = {
            'python': r'^(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]+)\s*\(|class\s+([A-Z][a-zA-Z0-9_]+)\s*',
            'javascript': r'function\s+([a-zA-Z_$][\w$]+)\s*\(|const\s+([a-zA-Z_$][\w$]+)\s*=\s*\(?',
            'java': r'class\s+([A-Z][a-zA-Z0-9_]+)\s*\{|void\s+([a-z][a-zA-Z0-9_]+)\s*\(',
            'c': r'^\w+\s+\*?([a-zA-Z_][a-zA-Z0-9_]+)\s*\(',
            'cpp': r'^(?:template\s*<.*>)?\s*\w+\s+([A-Z][a-zA-Z0-9_]+)\s*::|\w+\s+([a-z][a-zA-Z0-9_]+)\s*\(',
            'csharp': r'\b(?:public|private|protected)\s+\w+\s+([A-Z][a-zA-Z0-9_]+)\s*\(',
            'go': r'func\s+(?:\([^)]+\)\s+)?([A-Z][a-zA-Z0-9_]+)\s*\(',
            'rust': r'fn\s+([a-z_][a-z0-9_]+)\s*\(',
            'swift': r'func\s+([a-zA-Z_][a-zA-Z0-9_]+)\s*\(',
            'kotlin': r'fun\s+([a-zA-Z_][a-zA-Z0-9_]+)\s*\(',
            'typescript': r'^(?:export\s+)?(?:async\s+)?function\s+([A-Z][a-zA-Z0-9_]+)|const\s+([a-z][a-zA-Z0-9_]+)\s*:\s*\w+',
            'php': r'function\s+([a-z_][a-zA-Z0-9_]+)\s*\(',
            'ruby': r'def\s+(?:self\.)?([a-z_][a-zA-Z0-9_]+)\s*',
            'lua': r'function\s+([a-zA-Z_][a-zA-Z0-9_.]+)\s*\(',
            'perl': r'sub\s+([a-zA-Z_][a-zA-Z0-9_]+)\s*\{',
            'r': r'([a-zA-Z_][a-zA-Z0-9_.]+)\s*<-\s*function\s*\(',
            'scala': r'def\s+([a-zA-Z_][a-zA-Z0-9_]+)\s*[=\(]',
            'dart': r'\b(?:void|dynamic)\s+([a-zA-Z_][a-zA-Z0-9_]+)\s*\(',
            'julia': r'function\s+([a-zA-Z_!][a-zA-Z0-9_!]+)\s*\(',
            'haskell': r'^([a-z][a-zA-Z0-9_]+)\s*::',
            'elixir': r'def(?:p|macro)?\s+([a-zA-Z_][a-zA-Z0-9_!?]+)\s*\(?',
            'clojure': r'\(defn-?\s+([a-zA-Z-]+)\s*\[',
            'matlab': r'function\s+([a-zA-Z_][a-zA-Z0-9_]+)\s*\(',
            'groovy': r'def\s+([a-zA-Z_][a-zA-Z0-9_]+)\s*\(',
            'sql': r'CREATE\s+(?:FUNCTION|PROCEDURE)\s+([a-zA-Z_][a-zA-Z0-9_]+)\s*',
            'bash': r'([a-zA-Z_][a-zA-Z0-9_]+)\s*\(\)\s*\{',
            'powershell': r'function\s+([A-Z][a-zA-Z0-9-]+)\s*\{',
            'assembly': r'^\s*([a-zA-Z_][a-zA-Z0-9_]+)\s*:\s*',
            'objective-c': r'[-+]\s*\([^)]+\)\s*([a-zA-Z_][a-zA-Z0-9_]+)\b',
            'terraform': r'resource\s+"[^"]+"\s+"([^"]+)"\s*\{',
            'solidity': r'function\s+([a-zA-Z_][a-zA-Z0-9_]+)\s*\(',
            'verilog': r'module\s+([a-zA-Z_][a-zA-Z0-9_]+)\s*\(',
            'vhdl': r'entity\s+([a-zA-Z_][a-zA-Z0-9_]+)\s+is',
            'fortran': r'(?:subroutine|function)\s+([a-zA-Z_][a-zA-Z0-9_]+)\s*\(',
            'prolog': r'([a-z][a-zA-Z0-9_]*)\s*\(.*\)\s*:-',
            'racket': r'\(define\s+\(([a-zA-Z-]+)\s*',
            'erlang': r'([a-z][a-zA-Z0-9_]*)\s*\(.*\)\s*->',
            'ocaml': r'let\s+(?:rec\s+)?([a-z_][a-zA-Z0-9_\']*)\s*',
            'delphi': r'\b(?:procedure|function)\s+([a-zA-Z_][a-zA-Z0-9_]+)\b',
            'vimscript': r'function!?\s+([a-zA-Z_#][a-zA-Z0-9_#]*)\s*\('
        }

        # Buscar nombres clave en el código
        matches = re.findall(patterns.get(language, r'\b\w{5,}\b'), code)
        unique_names = list(set(matches))[:3]

        # Construir base del nombre
        if unique_names:
            base_name = '_'.join(unique_names)
        else:
            content_hash = hashlib.md5(code.encode()).hexdigest()[:8]
            base_name = f"code_{content_hash}"

        # Limpiar caracteres especiales
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '', base_name)[:50]

        # Generar timestamp único
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Obtener extensión del lenguaje
        extensions = self.languages.get(language, {}).get('extensions', ['txt'])
        ext = extensions[0]
        
        return f"{clean_name}_{timestamp}.{ext}"

    def save_code_file(self, code: str, language: str, user_input: str) -> str:
        """Guarda el código en un archivo con nombre óptimo"""
        try:
            filename = self._generate_filename(code, language, user_input)
            filepath = self.output_dir / filename
            
            # Prevenir sobreescritura
            counter = 1
            while filepath.exists():
                new_name = f"{filepath.stem}_{counter}{filepath.suffix}"
                filepath = filepath.with_name(new_name)
                counter += 1

            # Escribir archivo
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"// Generado por AI\n")
                f.write(f"// Prompt original: {user_input}\n\n")
                f.write(code)
            
            self.logger.info(f"Código guardado en: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error guardando archivo: {str(e)}")
            return None

    
    def _select_model_file_with_tkinter(self):
        """Selecciona un archivo de modelo usando un diálogo gráfico"""
        model_dir = base_dir / "models"

        # Verificar que el directorio existe
        if not model_dir.exists():
            self.logger.error(f"El directorio de modelos no existe: {model_dir}")
            try:
                # Crear el directorio de modelos si no existe
                model_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Se ha creado el directorio de modelos: {model_dir}")
                self._show_error_popup(f"Se ha creado el directorio de modelos: {model_dir}\nPor favor, coloca tus archivos .gguf allí e intenta de nuevo.")
            except Exception as e:
                self.logger.error(f"No se pudo crear el directorio de modelos: {str(e)}")
                self._show_error_popup(f"No se pudo crear el directorio de modelos: {str(e)}")
            return None

        # Buscar archivos .gguf en el directorio
        try:
            gguf_files = [f for f in os.listdir(model_dir) if f.endswith(".gguf")]
        except Exception as e:
            self.logger.error(f"Error al listar archivos en el directorio de modelos: {str(e)}")
            self._show_error_popup(f"Error al listar archivos en el directorio de modelos: {str(e)}")
            return None

        # Comprobar si hay archivos .gguf
        if not gguf_files:
            self.logger.error("No se encontraron archivos .gguf en la carpeta 'models/'")
            self._show_error_popup("No se encontraron archivos .gguf en la carpeta 'models/'\nPor favor, descarga un modelo en formato .gguf y colócalo en la carpeta 'models/'")
            return None

        # Si solo hay un archivo, usarlo directamente
        elif len(gguf_files) == 1:
            self.logger.info(f"Se encontró un solo modelo: {gguf_files[0]}")
            return model_dir / gguf_files[0]

        # Si hay múltiples archivos, mostrar diálogo de selección
        else:
            self.logger.info(f"Se encontraron múltiples modelos: {gguf_files}")
            try:
                root = Tk()
                root.title("Selección de Modelo")

                # Configurar la ventana para que esté en primer plano
                root.attributes('-topmost', True)
                root.update()

                # Centrar la ventana en la pantalla
                window_width = 400
                window_height = 200
                screen_width = root.winfo_screenwidth()
                screen_height = root.winfo_screenheight()
                x_coordinate = (screen_width - window_width) // 2
                y_coordinate = (screen_height - window_height) // 2
                root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

                # Crear widgets
                label = Label(root, text="Selecciona un modelo:")
                label.pack(pady=10)

                # Variable para almacenar la selección
                selected_model = StringVar(root)
                selected_model.set(gguf_files[0])  # valor por defecto

                # Lista desplegable con los modelos
                dropdown = OptionMenu(root, selected_model, *gguf_files)
                dropdown.pack(pady=10, padx=20, fill='x')

                # Variable para saber si se completó la selección
                selection_complete = False

                # Función para manejar la selección
                def on_select():
                    nonlocal selection_complete
                    selection_complete = True
                    root.destroy()

                # Botón para confirmar la selección
                select_button = Button(root, text="Seleccionar", command=on_select)
                select_button.pack(pady=20)

                # Mostrar la ventana y esperar
                root.mainloop()

                # Verificar si se completó la selección
                if not selection_complete:
                    self.logger.warning("Ventana de selección cerrada sin elegir un modelo")
                    return None

                # Obtener el modelo seleccionado
                model_name = selected_model.get()

                if model_name and model_name in gguf_files:
                    self.logger.info(f"Modelo seleccionado por el usuario: {model_name}")
                    return model_dir / model_name
                else:
                    self.logger.error(f"Modelo seleccionado no válido: {model_name}")
                    return None

            except Exception as e:
                self.logger.error(f"Error durante la selección del modelo: {str(e)}")
                self._show_error_popup(f"Error durante la selección del modelo: {str(e)}")
                return None

    def _load_config(self, filename):
        config_path = self.config_dir / filename
        self.logger.debug(f"Intentando cargar configuración desde: {config_path}")
        
        if not config_path.exists():
            self.logger.warning(f"Archivo de configuración no encontrado: {config_path}")
            
            # Crear configuraciones predeterminadas
            if filename == "personality.json":
                default_config = {"system_prompt": "Eres un asistente virtual útil y amable. Responde de manera concisa y coherente."}
            elif filename == "settings.json":
                default_config = {"history_length": 3}
            else:
                default_config = {}
                
            # Guardar configuración predeterminada
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"Archivo de configuración creado con valores predeterminados: {config_path}")
            return default_config
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            self.logger.debug(f"Configuración cargada: {config}")
            return config

    def _check_internet_connection(self, host="1.1.1.1", port=53, timeout=3):
        try:
            self.logger.debug(f"Verificando conexión a internet: {host}:{port}")
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            self.logger.info("Conexión a internet disponible")
            return True
        except socket.error as e:
            self.logger.warning(f"Sin conexión a internet: {str(e)}")
            return False

    def _show_error_popup(self, message):
        self.logger.error(f"Mostrando mensaje de error: {message}")
        try:
            root = Tk()
            root.withdraw()
            messagebox.showerror("Error", message)
            root.destroy()
        except Exception as e:
            self.logger.error(f"No se pudo mostrar el popup de error: {str(e)}")
            print(f"ERROR: {message}")
    
    def _load_model(self):
        if self.model_path is None:
            self.logger.error("No se especificó una ruta de modelo válida")
            self._show_error_popup("No se especificó una ruta de modelo válida")
            raise FileNotFoundError("No se especificó una ruta de modelo válida")
        
        model_file = Path(self.model_path)
        
        # Verificar que el archivo existe
        if not model_file.exists():
            self.logger.error(f"El archivo del modelo no existe: {model_file}")
            self._show_error_popup(f"El archivo del modelo no existe: {model_file}")
            raise FileNotFoundError(f"El archivo del modelo no existe: {model_file}")

        model_name = model_file.name.lower()
        model_type = self._detect_model_type(model_name)
        self.logger.info(f"Tipo de modelo detectado: {model_type}")

        try:
            self.logger.info(f"Cargando modelo desde ruta local: {model_file}")
            print(f"[INFO] Cargando modelo desde: {model_file}")
            
            # Verificar tamaño del modelo
            file_size_mb = model_file.stat().st_size / (1024 * 1024)
            self.logger.info(f"Tamaño del modelo: {file_size_mb:.2f} MB")
            
            if file_size_mb < 10:
                self.logger.warning(f"El archivo del modelo es sospechosamente pequeño: {file_size_mb:.2f} MB")
                self._show_error_popup(f"Advertencia: El archivo del modelo es muy pequeño ({file_size_mb:.2f} MB). Es posible que esté corrupto o incompleto.")
            
            #self.llm = AutoModelForCausalLM.from_pretrained(
            #    str(model_file),
            #    model_type=model_type,
            #    context_length=4096,
            #    gpu_layers=0
            #)

            try:
                model_config = self.MODEL_CONFIG.get(model_type, self.MODEL_CONFIG["llama"])  # Obtener configuración
        
                self.llm = AutoModelForCausalLM.from_pretrained(
                    str(model_file),
                    model_type=model_type,
                    context_length=model_config["context_length"],
                    max_new_tokens=model_config["max_new_tokens"],
                    temperature=model_config["temperature"],
                    stop=model_config["stop_sequences"],
                    gpu_layers=0
                )
                
                if model_type == "mistral":
                    self.llm.set_cache_size(512)
            
                self._apply_model_specific_settings(model_type)
                self.logger.info("Modelo cargado exitosamente con configuración óptima")
            
                # Prueba básica del modelo
                try:
                    test_tokens = self.llm.tokenize("Hola, prueba")
                    self.logger.info(f"Prueba de tokenización exitosa: {test_tokens}")
                except Exception as e:
                    self.logger.error(f"Error en la prueba de tokenización: {str(e)}")
                    self._show_error_popup(f"Error en la prueba de tokenización. Es posible que el modelo no funcione correctamente: {str(e)}")
                
            except Exception as e:
                self.logger.error(f"Error al cargar el modelo local: {str(e)}")
                error_stack = traceback.format_exc()
                self.logger.error(f"Stack de error:\n{error_stack}")
            
                if "No such file or directory" in str(e):
                    self.logger.error("Archivo no encontrado")
                    self._show_error_popup(f"No se encontró el archivo del modelo: {model_file}")
                elif "not a valid Win32 application" in str(e):
                    self.logger.error("Archivo no válido para esta plataforma")
                    self._show_error_popup(f"El archivo del modelo no es compatible con esta plataforma: {model_file}")
                else:
                    self._show_error_popup(f"Error al cargar el modelo:\n{str(e)}")
                
                    # Intentar cargar modelo alternativo de respaldo
                    try:
                        if self._check_internet_connection():
                            self.logger.info("Intentando cargar modelo de respaldo desde Hugging Face...")
                            self._show_error_popup("Error al cargar el modelo local. Intentando descargar modelo de respaldo...")

                            backup_config = self.MODEL_CONFIG["default"]
                        
                            self.llm = AutoModelForCausalLM.from_pretrained(
                                "NyxGleam/tinyllama-1.1b-chat-v1.0.Q4_K_M",
                                model_type="llama",
                                context_length=backup_config["context_length"],
                                max_new_tokens=backup_config["max_new_tokens"],
                                temperature=backup_config["temperature"],
                                gpu_layers=0
                            )

                            self._apply_model_specific_settings(model_type)
                            self.logger.info("Modelo de respaldo cargado exitosamente")
                    except Exception as e:
                        self.logger.error("No hay conexión a internet para cargar modelo de respaldo")
                        raise RuntimeError("No se pudo cargar el modelo local y no hay conexión a internet para descargar un modelo alternativo.")
        except Exception as inner_e:
            self.logger.error(f"Error al cargar modelo de respaldo: {str(inner_e)}")
            raise RuntimeError(f"No se pudo cargar el modelo local ni el modelo de respaldo: {str(inner_e)}")
    
    def _detect_model_type(self, model_file):
        model_file = model_file.lower()

        is_code_model = "code" in model_file or "coder" in model_file

        if "mistral" in model_file or "mixtral" in model_file:
            return "mistral"
        elif "llama" in model_file or "alpaca" in model_file or "vicuna" in model_file or "tinyllama" in model_file:
            return "llama"
        elif "gpt" in model_file or "neox" in model_file:
            return "gpt2"
        elif "falcon" in model_file:
            return "falcon"
        elif "starcoder" in model_file:
            return "starcoder"
        elif "deepseek" in model_file:
            if is_code_model:
                self.logger.warning("Detectado modelo DeepSeek Coder - optimizado para código, no para chat general")
                # Puedes mostrar una advertencia al usuario aquí
                return "deepseek-code"
            return "deepseek"
        else:
            self.logger.warning(f"Tipo de modelo no reconocido: {model_file}. Usando configuración LLaMA por defecto.")
            return "llama"

    def _apply_model_specific_settings(self, model_type):
        """Aplica ajustes específicos post-carga"""
        if model_type == "starcoder":
            self.llm.set_align_code(True)
            
    def _handle_initialization_error(self):
        """Maneja errores de inicialización e intenta cargar un modelo fallback desde Hugging Face"""
        try:
            if self._check_internet_connection():
                self.logger.info("Intentando cargar modelo de respaldo desde Hugging Face...")
                self._show_error_popup("No se encontró un modelo local válido. Intentando descargar modelo de respaldo...")

                backup_config = self.MODEL_CONFIG["default"]

                self.llm = AutoModelForCausalLM.from_pretrained(
                    "NyxGleam/tinyllama-1.1b-chat-v1.0.Q4_K_M",
                    model_type="llama",
                    context_length=backup_config["context_length"],
                    max_new_tokens=backup_config["max_new_tokens"],
                    temperature=backup_config["temperature"],
                    gpu_layers=0
                )

                self.logger.info("Modelo de respaldo cargado exitosamente")
            else:
                self.logger.error("No hay conexión a internet para cargar modelo de respaldo")
                self._show_error_popup("No se encontró un modelo local válido y no hay conexión a internet. La aplicación no funcionará correctamente.")
        except Exception as e:
            self.logger.error(f"Error al cargar modelo de respaldo: {str(e)}")
            self._show_error_popup(f"No se pudo cargar ningún modelo. La aplicación no funcionará correctamente.\nError: {str(e)}")

    def _check_internet_connection(self):
        """Verifica si hay conexión a internet disponible"""
        try:
            response = requests.get("https://huggingface.co", timeout=5)
            return response.status_code == 200
        except:
            return False

    def _clean_text_for_deepseek_coder(self, text):
        """Limpia el texto para hacerlo compatible con DeepSeek Coder"""
        import re
        # Eliminar emojis y mantener solo ASCII básico
        cleaned = re.sub(r'[^\x00-\x7F]', '', text)
        # Reemplazar caracteres especiales del español
        replacements = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
            'ñ': 'n', 'Ñ': 'N',
            '¡': '!', '¿': '?'
        }
        for k, v in replacements.items():
            cleaned = cleaned.replace(k, v)
        return cleaned

    def format_prompt(self, user_input):
        model_type = getattr(self, "current_model_type", "llama")
        self.logger.debug(f"Formateando prompt para: {user_input}")
        self.logger.debug(f"Modelo detectado como tipo: {model_type}")
    
        # Limpiar el texto si es un modelo DeepSeek Coder
        if model_type == "deepseek-code" or "deepseek" in model_type:
            self.logger.debug("Aplicando limpieza de texto para DeepSeek Coder")
            user_input = self._clean_text_for_deepseek_coder(user_input)
            system_prompt = self._clean_text_for_deepseek_coder(self.personality.get("system_prompt", ""))
        else:
            system_prompt = self.personality.get("system_prompt", "")

        history = self._format_history(model_type)
        if not system_prompt:
            system_prompt = "Eres un asistente virtual útil y amable. Responde de manera concisa y coherente."

        # Limitar la cantidad de pares para evitar exceder el contexto
        pairs = min(self.settings.get("history_length", 3), 3)  # Máximo 3 pares para TinyLlama
        self.logger.debug(f"Usando {pairs} pares de conversación")

        # Solo usar los últimos 'pairs' pares
        relevant_history = []
        history_len = len(self.conversation_history)
        self.logger.debug(f"Historia actual tiene {history_len} mensajes")

        # Asegurar que tomamos pares completos (user+assistant)
        if history_len > 0:
            # Calcular cuántos pares completos tenemos
            max_pairs = history_len // 2
            # Limitar a los últimos 'pairs' o menos
            pairs_to_use = min(pairs, max_pairs)
            # Calcular el índice de inicio
            start_idx = history_len - (pairs_to_use * 2)
            # Obtener el fragmento relevante
            relevant_history = self.conversation_history[start_idx:]
            self.logger.debug(f"Historia relevante: {relevant_history}")

        # Construir el historial formateado
        #formatted_history = ""
        #    for i in range(0, len(relevant_history), 2):
        #        if i+1 < len(relevant_history):  # Asegurar que tenemos el par completo
        #            user_msg = relevant_history[i].strip()
        #            assistant_msg = relevant_history[i+1].strip()
        #            formatted_history += f"<|user|> {user_msg}\n<|assistant|> {assistant_msg}\n"

        
        formatted_history = self._format_history(model_type)

        config = self.MODEL_CONFIG[model_type]

        # Construir el prompt final
        final_prompt = config["prompt_template"].format(
            system_prompt=system_prompt,
            history=formatted_history,
            user_input=user_input.strip()
        )
    
        # Aplicar límites de contexto
        max_context = config["context_length"] - len(self.tokenize(user_input)) - 100
        final_prompt = self._truncate_prompt(final_prompt, max_context)
    
        self.logger.debug(f"Prompt final ({model_type}):\n{final_prompt[:500]}...")
        return final_prompt

    def _format_history(self, model_type):
        """Formatea el historial según las convenciones del modelo"""
        history_config = {
            "llama": {
                "user_prefix": "<|user|>",
                "assistant_prefix": "<|assistant|>",
                "max_pairs": 3
            },
            "mistral": {
                "user_prefix": "[INST]",
                "assistant_prefix": "[/INST]",
                "max_pairs": 5
            },
            "falcon": {
                "user_prefix": "\nUser:",
                "assistant_prefix": "\nFalcon:",
                "max_pairs": 2
            },
            "default": {
                "user_prefix": "\nUser:",
                "assistant_prefix": "\nAssistant:",
                "max_pairs": 3
            }
        }
    
        cfg = history_config.get(model_type, history_config["default"])
        max_pairs = min(self.settings.get("history_length", 3), cfg["max_pairs"])
    
        # Obtener los últimos pares completos
        history = self.conversation_history[-max_pairs*2:]
        formatted = []
    
        for i in range(0, len(history), 2):
            if i+1 < len(history):
                user = history[i].strip()
                assistant = history[i+1].strip()
                formatted.append(
                    f"{cfg['user_prefix']} {user}\n{cfg['assistant_prefix']} {assistant}"
                )
    
        return "\n".join(formatted)

    def _truncate_prompt(self, prompt, max_length):
        """Asegura que el prompt no exceda el contexto del modelo"""
        tokens = self.tokenize(prompt)
        if len(tokens) > max_length:
            self.logger.warning(f"Truncando prompt de {len(tokens)} a {max_length} tokens")
            truncated = self.llm.detokenize(tokens[:max_length])
            return truncated.replace("<|endoftext|>", "").strip()
        return prompt

    def generate_response(self, user_input):
        if not user_input.strip():
            self.logger.warning("Se recibió un mensaje vacío")
            return "Por favor, envía un mensaje para que pueda ayudarte."
            
        self.logger.info(f"Generando respuesta para mensaje: '{user_input}'")
        
        # Verificar que el modelo está cargado
        if self.llm is None:
            self.logger.error("El modelo no está cargado correctamente")
            return "Lo siento, el modelo de lenguaje no está inicializado correctamente. Por favor, reinicia la aplicación."

        if "buscar" in user_input.lower() or "investiga" in user_input.lower():
            search_response = self.handle_search_query(user_input)
        
            self.conversation_history.append(user_input)
            self.conversation_history.append(search_response)
        
            try:
                self.learning_system.save_interaction(user_input, search_response, "search")
            except Exception as e:
                self.logger.error(f"Error guardando búsqueda: {str(e)}")
        
            return search_response
            
        prompt = self.format_prompt(user_input)
        self.logger.debug(f"Prompt completo:\n{prompt}")
        
        # Verificar longitud del prompt
        token_count = len(self.tokenize(prompt))
        self.logger.info(f"Longitud del prompt: {token_count} tokens")
        if token_count > 3500:  # Límite seguro para un contexto de 4096
            self.logger.warning(f"El prompt es muy largo ({token_count} tokens)")
            self.reset_history()
            prompt = self.format_prompt(user_input)
            self.logger.info(f"Se reinició el historial. Nueva longitud: {len(self.tokenize(prompt))} tokens")

        try:
            # Cronometrar generación
            start_time = time.time()
            
            # Antes de generar, hacer diagnóstico del modelo
            try:
                test_prompt = "<|system|> Eres un asistente útil.\n<|user|> Di 'hola'.\n<|assistant|> "
                test_output = self.llm(test_prompt, max_new_tokens=5)
                self.logger.debug(f"Prueba de generación: '{test_output}'")
            except Exception as test_error:
                self.logger.error(f"Error en prueba de generación: {str(test_error)}")
                # Continuar a pesar del error en la prueba
            
            # Generar respuesta real
            self.logger.info("Iniciando generación...")
            raw_response = self.llm(
                prompt,
                max_new_tokens=200,
                temperature=0.7,
                stop=["<|user|>", "<|system|>", "\n\n"]
            )

            # Post-procesamiento para código
            processed_response = self.post_process_code(raw_response.strip())

            elapsed = time.time() - start_time
            self.logger.info(f"Generación completada en {elapsed:.2f}s")

            # Verificar si la respuesta está vacía
            if not raw_response or raw_response.strip() == "":
                self.logger.warning("El modelo devolvió una respuesta vacía")
                return "Lo siento, no pude generar una respuesta. Por favor, intenta reformular tu pregunta."

            # Limpiar respuesta
            response = raw_response.strip()
            self.logger.debug(f"Respuesta cruda: '{raw_response}'")
            
            # Eliminar cualquier texto después de los tokens de parada
            if "<|user|>" in response:
                response = response.split("<|user|>")[0].strip()
            if "<|assistant|>" in response:
                response = response.replace("<|assistant|>", "").strip()
                
            # Diagnóstico de la respuesta
            self.logger.debug(f"Longitud de respuesta: {len(response)} caracteres, {len(response.split())} palabras")
            
            # Detectar y corregir respuestas problemáticas
            is_problematic = False
            reason = ""
            
            if response.startswith("1.") or response.startswith("- "):
                is_problematic = True
                reason = "comenzó como lista"
            elif response.count("?") > 3:
                is_problematic = True
                reason = "demasiadas preguntas"
            elif len(response.split()) < 3:
                is_problematic = True
                reason = "respuesta muy corta"
            elif response.count(".") == 0 and len(response) > 10:
                is_problematic = True
                reason = "sin puntuación"
            elif "no puedo" in response.lower() or "como modelo" in response.lower():
                is_problematic = True
                reason = "respuesta evasiva"
                
            if is_problematic:
                self.logger.warning(f"Respuesta problemática detectada: {reason}")
                self.logger.debug(f"Respuesta problemática: '{response}'")
                
                # Elegir respuesta de fallback
                fallback_responses = [
                    "Hola, ¿en qué puedo ayudarte hoy?",
                    "Soy un asistente virtual. ¿Cómo puedo ayudarte?",
                    "Lo siento, no pude entender bien tu mensaje. ¿Podrías reformularlo?",
                    "Estoy aquí para asistirte. ¿Qué necesitas saber?"
                ]
                
                # Si el mensaje del usuario es un saludo, responder adecuadamente
                if user_input.lower() in ["hola", "hey", "hi", "hello", "saludos", "buenos días", "buenas tardes", "buenas noches"]:
                    fallback_responses = [
                        f"¡Hola! ¿En qué puedo ayudarte hoy?",
                        f"¡Hola! Soy tu asistente virtual. ¿Cómo puedo ayudarte?",
                        f"¡Saludos! ¿En qué puedo asistirte hoy?",
                        f"¡Hola! Estoy aquí para responder tus preguntas."
                    ]
                
                response = random.choice(fallback_responses)
                self.logger.info(f"Usando respuesta de fallback: '{response}'")
            
            self.logger.debug(f"Respuesta procesada: '{response}'")
            self.logger.info(f"Respuesta generada en {elapsed:.2f}s")

            # Guardar en el historial
            self.conversation_history.append(user_input)
            self.conversation_history.append(response)
            self.logger.debug(f"Historia actualizada, ahora tiene {len(self.conversation_history)} mensajes")

            # Aprendizaje continuo
            try:
                self.learning_system.save_interaction(user_input, response, "chat")
                self.logger.debug("Interacción guardada en el sistema de aprendizaje")
            except Exception as learn_error:
                self.logger.error(f"Error al guardar la interacción: {str(learn_error)}")

            # Procesar código
            processed_code, detected_lang = self.post_process_code(raw_response.strip())
        
            # Guardar en archivo
            if detected_lang != 'text':
                saved_path = self.save_code_file(processed_code, detected_lang, user_input)
                if saved_path:
                    response += f"\n\n🔖 Código guardado en: {saved_path}"

            return response

        except Exception as e:
            stack_trace = traceback.format_exc()
            self.logger.error(f"Error en generación: {str(e)}\n{stack_trace}")
            
            # Intentar diagnosticar el error
            error_str = str(e).lower()
            if "memory" in error_str or "allocation" in error_str:
                self.logger.critical("Posible error de memoria")
                return "Lo siento, hubo un problema de memoria al procesar tu mensaje. Intenta con un mensaje más corto o reinicia la aplicación."
            elif "cuda" in error_str or "gpu" in error_str:
                self.logger.critical("Posible error de GPU")
                return "Lo siento, hubo un problema con la GPU. Intenta ejecutar el modelo en CPU modificando la configuración."
            
            return "Lo siento, hubo un error interno al procesar tu mensaje. Por favor, intenta de nuevo con una pregunta diferente."

    def reset_history(self):
        self.logger.info("Reiniciando historial de conversación")
        self.conversation_history = []

    def __call__(self, prompt, stream=False, **kwargs):
        self.logger.debug(f"Llamada directa con prompt (primeros 50 caracteres): {prompt[:50]}...")
        return self.llm.generate(prompt, stream=stream, **kwargs)

    def generate(self, prompt, **kwargs):
        self.logger.debug(f"Generación con prompt (primeros 50 caracteres): {prompt[:50]}...")
        return self.llm.generate(prompt, **kwargs)

    def tokenize(self, text):
        if self.llm is None:
            self.logger.error("Intento de tokenizar con modelo no inicializado")
            return []
            
        try:
            tokens = self.llm.tokenize(text)
            return tokens
        except Exception as e:
            self.logger.error(f"Error al tokenizar texto: {str(e)}")
            return []

    def handle_search_query(self, query): # Añadir a requirements.txt
        try:
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(query, max_results=3)]

                if "canciones de hatsune miku" in query.lower():
                    # Respuesta estructurada para música
                    return "Canciones populares de Hatsune Miku:\n- Tell Your World\n- World is Mine\n- Rolling Girl\n- Melt\n(Powered by Vocaloid Database)"

                return f"Resultados de búsqueda para '{query}':\n" + "\n".join([f"{r['title']}: {r['href']}" for r in results[:3]])

        except Exception as e:
            return f"No pude completar la búsqueda. Error: {str(e)}"

    def post_process_code(self, response):
        """Extrae bloques de código de la respuesta del modelo soportando múltiples lenguajes.
    
        Args:
            response (str): Respuesta cruda del modelo
        
        Returns:
            str: Código limpio o respuesta original si no se detecta código
        """
        languages = {
            'python': {'extensions': ['py', 'python']},
            'javascript': {'extensions': ['js', 'javascript']},
            'java': {'extensions': ['java']},
            'c': {'extensions': ['c', 'h']},
            'cpp': {'extensions': ['cpp', 'cc', 'cxx', 'c++']},
            'csharp': {'extensions': ['cs']},
            'go': {'extensions': ['go']},
            'rust': {'extensions': ['rs']},
            'swift': {'extensions': ['swift']},
            'kotlin': {'extensions': ['kt', 'kts']},
            'typescript': {'extensions': ['ts', 'tsx']},
            'php': {'extensions': ['php', 'phtml']},
            'ruby': {'extensions': ['rb']},
            'lua': {'extensions': ['lua']},
            'perl': {'extensions': ['pl', 'pm']},
            'r': {'extensions': ['r', 'R']},
            'scala': {'extensions': ['scala', 'sc']},
            'dart': {'extensions': ['dart']},
            'julia': {'extensions': ['jl']},
            'haskell': {'extensions': ['hs', 'lhs']},
            'elixir': {'extensions': ['ex', 'exs']},
            'clojure': {'extensions': ['clj', 'cljs', 'cljc']},
            'matlab': {'extensions': ['m']},
            'groovy': {'extensions': ['groovy', 'gy']},
            'sql': {'extensions': ['sql']},
            'html': {'extensions': ['html', 'htm']},
            'css': {'extensions': ['css']},
            'markdown': {'extensions': ['md', 'markdown']},
            'bash': {'extensions': ['sh', 'bash']},
            'powershell': {'extensions': ['ps1']},
            'assembly': {'extensions': ['asm', 's']},
            'objective-c': {'extensions': ['m', 'mm']},
            'docker': {'extensions': ['dockerfile']},
            'terraform': {'extensions': ['tf']},
            'solidity': {'extensions': ['sol']},
            'verilog': {'extensions': ['v', 'vh']},
            'vhdl': {'extensions': ['vhd', 'vhdl']},
            'fortran': {'extensions': ['f90', 'f95']},
            'prolog': {'extensions': ['pl', 'pro']},
            'racket': {'extensions': ['rkt']},
            'erlang': {'extensions': ['erl', 'hrl']},
            'ocaml': {'extensions': ['ml', 'mli']},
            'delphi': {'extensions': ['pas', 'dpr']},
            'vimscript': {'extensions': ['vim']}
        }

        code_block_regex = r"```(?:(\w+)\n)?(.*?)```"
        matches = re.findall(code_block_regex, response, re.DOTALL)

        if matches:
            lang, code = matches[0]
            code = code.strip()
        
            if lang:
                lang = lang.lower()
                for lang_name, data in languages.items():
                    if lang in data['extensions']:
                        return f"// Lenguaje: {lang_name}\n\n{code}"
        
            return f"// Código detectado\n\n{code}"
        return response, 'text'