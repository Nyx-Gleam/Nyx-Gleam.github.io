import sys
import os
import json
import time
import socket
import logging
from pathlib import Path
import traceback

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from continuous_learning import ContinuousLearningSystem
from ctransformers import AutoModelForCausalLM
from tkinter import Tk, messagebox, simpledialog


class AILocalChatbot:
    def __init__(self, model_path, config_dir="config", data_dir="data"):
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
        
        self.model_path = model_path
        self.llm = None  # Inicializar el modelo como None para verificar más tarde

        # Si no se pasa modelo, buscar en carpeta `models/`
        if not self.model_path:
            self.model_path = self._select_model_file_with_tkinter()

        self.logger.info(f"Modelo seleccionado: {self.model_path}")
        
        # Verificar que los directorios existen, crearlos si no
        self.config_dir = base_dir / config_dir
        self.data_dir = base_dir / data_dir
        
        for directory in [self.config_dir, self.data_dir]:
            if not directory.exists():
                self.logger.warning(f"El directorio {directory} no existe. Creándolo...")
                directory.mkdir(parents=True, exist_ok=True)
        
        self.conversation_history = []

        # Cargar configuraciones con manejo de errores
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

        self._load_model()
    
    def _select_model_file_with_tkinter(self):
        model_dir = base_dir / "models"
        
        if not model_dir.exists():
            self.logger.error(f"El directorio de modelos no existe: {model_dir}")
            self._show_error_popup(f"El directorio de modelos no existe: {model_dir}\nPor favor, crea la carpeta 'models' y coloca tus archivos .gguf allí.")
            raise FileNotFoundError(f"El directorio de modelos no existe: {model_dir}")
            
        gguf_files = [f for f in os.listdir(model_dir) if f.endswith(".gguf")]

        if not gguf_files:
            self.logger.error("No se encontraron archivos .gguf en la carpeta 'models/'")
            self._show_error_popup("No se encontraron archivos .gguf en la carpeta 'models/'\nPor favor, descarga un modelo en formato .gguf y colócalo en la carpeta 'models/'")
            raise FileNotFoundError("No .gguf model files found in 'models/' folder.")

        elif len(gguf_files) == 1:
            self.logger.info(f"Se encontró un solo modelo: {gguf_files[0]}")
            return model_dir / gguf_files[0]

        else:
            self.logger.info(f"Se encontraron múltiples modelos: {gguf_files}")
            root = Tk()
            root.withdraw()
            model_name = simpledialog.askstring(
                "Seleccionar modelo",
                "Hay múltiples modelos disponibles:\n\n" + "\n".join(gguf_files) + "\n\nEscribe el nombre exacto del modelo:"
            )
            root.destroy()

            if not model_name or model_name not in gguf_files:
                self.logger.error(f"Modelo seleccionado no válido: {model_name}")
                raise ValueError("Modelo no válido o no seleccionado.")
            
            self.logger.info(f"Modelo seleccionado por el usuario: {model_name}")
            return model_dir / model_name
        
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
            
            self.llm = AutoModelForCausalLM.from_pretrained(
                str(model_file),
                model_type=model_type,
                context_length=4096,
                gpu_layers=0
            )
            
            self.logger.info("Modelo cargado exitosamente")
            
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
                        
                        self.llm = AutoModelForCausalLM.from_pretrained(
                            "NyxGleam/tinyllama-1.1b-chat-v1.0.Q4_K_M",
                            model_type="llama",
                            context_length=4096,
                            gpu_layers=0
                        )
                        self.logger.info("Modelo de respaldo cargado exitosamente")
                    else:
                        self.logger.error("No hay conexión a internet para cargar modelo de respaldo")
                        raise RuntimeError("No se pudo cargar el modelo local y no hay conexión a internet para descargar un modelo alternativo.")
                except Exception as inner_e:
                    self.logger.error(f"Error al cargar modelo de respaldo: {str(inner_e)}")
                    raise RuntimeError(f"No se pudo cargar el modelo local ni el modelo de respaldo: {str(inner_e)}")
    
    def _detect_model_type(self, model_file):
        model_file = model_file.lower()
        self.logger.debug(f"Detectando tipo de modelo para: {model_file}")

        if any(keyword in model_file for keyword in ["llama", "tinyllama", "alpaca", "vicuna", "guanaco", "wizardlm", "koala"]):
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
        self.logger.warning(f"Tipo de modelo no reconocido para: '{model_file}'. Usando 'llama' por defecto.")
        return "llama"

    def format_prompt(self, user_input):
        self.logger.debug(f"Formateando prompt para: {user_input}")
        
        # Obtener y limpiar el mensaje de sistema
        system_prompt = self.personality.get("system_prompt", "").strip()
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
        formatted_history = ""
        for i in range(0, len(relevant_history), 2):
            if i+1 < len(relevant_history):  # Asegurar que tenemos el par completo
                user_msg = relevant_history[i].strip()
                assistant_msg = relevant_history[i+1].strip()
                formatted_history += f"<|user|> {user_msg}\n<|assistant|> {assistant_msg}\n"

        # Construir el prompt final
        final_prompt = f"<|system|> {system_prompt}\n"
        final_prompt += formatted_history
        final_prompt += f"<|user|> {user_input.strip()}\n<|assistant|> "

        self.logger.debug(f"Prompt formateado:\n{final_prompt}")
        return final_prompt

    def generate_response(self, user_input):
        if not user_input.strip():
            self.logger.warning("Se recibió un mensaje vacío")
            return "Por favor, envía un mensaje para que pueda ayudarte."
            
        self.logger.info(f"Generando respuesta para mensaje: '{user_input}'")
        
        # Verificar que el modelo está cargado
        if self.llm is None:
            self.logger.error("El modelo no está cargado correctamente")
            return "Lo siento, el modelo de lenguaje no está inicializado correctamente. Por favor, reinicia la aplicación."
            
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
                import random
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
                self.learning_system.save_interaction(user_input, response)
                self.logger.debug("Interacción guardada en el sistema de aprendizaje")
            except Exception as learn_error:
                self.logger.error(f"Error al guardar la interacción: {str(learn_error)}")

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