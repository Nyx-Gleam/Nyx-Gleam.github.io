import os
import sys
import argparse
import json
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox, simpledialog
import requests
from bs4 import BeautifulSoup
import html2text
import threading
import time
from datetime import datetime
from ..continuous_learning import ContinuousLearningSystem
from ctransformers import AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.DEBUG)

def main():
    try:
        root = tk.Tk()
        gui = ChatbotGUI(root)
        root.mainloop()
    except Exception as e:
        logging.error(f"Error crítico: {str(e)}", exc_info=True)
        input("Presiona Enter para salir...")

class AILocalChatbot:
    def __init__(self, model_path, personality_path="personality.json"):
        """
        Inicializa el chatbot con el modelo LLM y carga la personalidad
        """
        try:
            self.model_path = model_path
            self.personality_path = personality_path
            self.conversation_history = []
            self.learning_system = ContinuousLearningSystem()

            # Cargar personalidad
            self.personality = self.load_personality()
        
            # Inicializar el modelo
            print("Cargando modelo de lenguaje...")
        
            # Detectar el tipo de modelo basado en la extensión del archivo
            model_file = os.path.basename(model_path).lower()
            if "llama" in model_file or "mistral" in model_file:
                model_type = "llama"
            elif "gpt2" in model_file:
                model_type = "gpt2"
            elif "gpt-j" in model_file or "gptj" in model_file:
                model_type = "gpt-j"
            elif "gptneox" in model_file or "neox" in model_file:
                model_type = "gptneox"
            else:
                model_type = "llama"  # Valor por defecto
            
            print(f"Usando tipo de modelo: {model_type}")
        
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type=model_type,
                context_length=2048,    # Ajustar según RAM disponible
                gpu_layers=0            # Usar 0 para solo CPU, aumentar si tienes GPU
            )
        
            print("Modelo cargado correctamente")
        
            # Iniciar sistema de aprendizaje continuo
            self.learning_system.start_learning()
            print("Sistema de aprendizaje iniciado")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            sys.exit(1)
        
    def load_personality(self):
        """Carga la configuración de personalidad desde un archivo"""
        if os.path.exists(self.personality_path):
            with open(self.personality_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        else:
            # Personalidad por defecto
            default_personality = {
                "name": "Asistente IA",
                "description": "Un asistente amigable y servicial",
                "traits": ["amable", "informativo", "paciente"],
                "speaking_style": "formal pero cercano",
                "interests": ["ayudar", "aprender", "compartir conocimiento"],
                "long_description": "Soy un asistente virtual diseñado para ser útil, amable y eficiente. Me encanta proporcionar información precisa y ayudar a resolver problemas. Disfruto aprendiendo cosas nuevas y compartiendo conocimiento de manera clara y accesible."
            }
            
            # Guardar personalidad por defecto
            with open(self.personality_path, 'w', encoding='utf-8') as file:
                json.dump(default_personality, file, ensure_ascii=False, indent=2)
                
            return default_personality
    
    def update_personality(self, new_personality_data):
        """Actualiza la personalidad del chatbot"""
        # Fusionar con la personalidad existente
        self.personality.update(new_personality_data)
        
        # Guardar cambios
        with open(self.personality_path, 'w', encoding='utf-8') as file:
            json.dump(self.personality, file, ensure_ascii=False, indent=2)
            
        return "Personalidad actualizada correctamente"
    
    def format_prompt(self, user_message):
        """
        Formatea el prompt para el modelo integrando la personalidad
        y el historial de conversación
        """
        # Crear un prompt sistema con la personalidad
        system_prompt = f"""Eres {self.personality['name']}, un chatbot con la siguiente personalidad:
        
{self.personality['long_description']}

Tus rasgos principales son: {', '.join(self.personality['traits'])}
Tu estilo de habla es: {self.personality['speaking_style']}
Tus intereses incluyen: {', '.join(self.personality['interests'])}

Responde al usuario de manera coherente con tu personalidad.
"""
        
        # Formatear historial de conversación
        conversation = []
        for entry in self.conversation_history[-5:]:  # Limitamos a las últimas 5 interacciones para no saturar el contexto
            conversation.append(f"Usuario: {entry['user']}")
            conversation.append(f"{self.personality['name']}: {entry['assistant']}")
        
        # Agregar el mensaje actual
        conversation.append(f"Usuario: {user_message}")
        conversation.append(f"{self.personality['name']}: ")
        
        # Combinar todo
        full_prompt = system_prompt + "\n\n" + "\n".join(conversation)
        return full_prompt
    
    def process_response(self, response):
        """
        Procesa la respuesta del modelo para mejorar su calidad
        y adaptar a la personalidad
        """
        # Limpiar la respuesta y eliminar texto adicional después de un posible "Usuario:"
        if "Usuario:" in response:
            response = response.split("Usuario:")[0]
            
        # Eliminar el nombre del asistente si está incluido al principio
        if response.startswith(f"{self.personality['name']}:"):
            response = response[len(f"{self.personality['name']}:"):].strip()
            
        return response.strip()
    
    def generate_response(self, user_message):
        """
        Genera una respuesta usando el modelo LLM
        """
        # Verificar si es una solicitud de búsqueda
        if user_message.lower().startswith(("busca ", "buscar ", "investiga ")):
            search_query = user_message.split(" ", 1)[1]
            search_results = self.learning_system.search_web(search_query)
            
            # Incluir los resultados de la búsqueda en el prompt
            if search_results:
                search_info = "\n\n".join([
                    f"Información sobre '{search_query}':\n"
                    f"Título: {result['title']}\n"
                    f"Resumen: {result['summary']}\n"
                    f"Fuente: {result['url']}"
                    for result in search_results
                ])
                
                user_message = f"{user_message}\n\nResultados de búsqueda:\n{search_info}"
        
        # Formatear el prompt con la personalidad y el historial
        prompt = self.format_prompt(user_message)
        
        # Generar respuesta usando el modelo
        raw_response = self.llm(
            prompt, 
            max_new_tokens=512,
            temperature=0.7,
            repetition_penalty=1.1,
            stop=["Usuario:", f"Usuario: "]
        )
        
        # Procesar la respuesta
        processed_response = self.process_response(raw_response)
        
        # Guardar en el historial
        self.conversation_history.append({
            "user": user_message,
            "assistant": processed_response,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Añadir esta interacción como una frase potencial para aprender
        if len(processed_response.split()) > 5:  # Solo frases relevantes
            self.learning_system.add_phrase(processed_response, context="conversation")
        
        return processed_response
    
    def save_conversation(self, filename="conversation_history.json"):
        """Guarda la conversación actual en un archivo"""
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(self.conversation_history, file, ensure_ascii=False, indent=2)
        return f"Conversación guardada en {filename}"
    
    def load_conversation(self, filename="conversation_history.json"):
        """Carga una conversación desde un archivo"""
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as file:
                self.conversation_history = json.load(file)
            return f"Conversación cargada desde {filename}"
        return f"No se encontró el archivo {filename}"
    
    def get_vocabulary_stats(self):
        """Obtiene estadísticas del vocabulario aprendido"""
        return {
            "total_words": len(self.learning_system.dictionary),
            "total_phrases": len(self.learning_system.phrases),
            "recent_words": [w for w in list(self.learning_system.dictionary.keys())[-10:]]
        }

class ChatbotGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("AI Chatbot Local")
        self.master.geometry("900x700")
        self.master.minsize(800, 600)
        
        # Estado del chatbot
        self.chatbot = None
        self.is_chatbot_loaded = False
        self.is_processing = False
        
        # Crear widgets
        self.create_menu()
        self.create_widgets()
        
        # Archivo de configuración
        self.config_file = "chatbot_config.json"
        self.load_config()

    def create_menu(self):
        """Crea el menú de la aplicación"""
        menubar = tk.Menu(self.master)
        
        # Menú Archivo
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Cargar Modelo", command=self.load_model)
        file_menu.add_command(label="Guardar Conversación", command=self.save_conversation)
        file_menu.add_command(label="Cargar Conversación", command=self.load_conversation)
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self.master.quit)
        menubar.add_cascade(label="Archivo", menu=file_menu)
        
        # Menú Personalidad
        personality_menu = tk.Menu(menubar, tearoff=0)
        personality_menu.add_command(label="Editar Personalidad", command=self.edit_personality)
        personality_menu.add_command(label="Ver Personalidad Actual", command=self.view_personality)
        menubar.add_cascade(label="Personalidad", menu=personality_menu)
        
        # Menú Diccionario
        dict_menu = tk.Menu(menubar, tearoff=0)
        dict_menu.add_command(label="Ver Estadísticas", command=self.view_vocabulary_stats)
        dict_menu.add_command(label="Buscar Palabra", command=self.search_word)
        menubar.add_cascade(label="Diccionario", menu=dict_menu)
        
        # Menú Ayuda
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Acerca de", command=self.show_about)
        help_menu.add_command(label="Ayuda", command=self.show_help)
        menubar.add_cascade(label="Ayuda", menu=help_menu)
        
        self.master.config(menu=menubar)

    def create_widgets(self):
        """Crea los widgets de la interfaz"""
        # Frame principal
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Área de chat (con desplazamiento)
        chat_frame = ttk.LabelFrame(main_frame, text="Conversación", padding="10")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, state="disabled")
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Área de entrada
        input_frame = ttk.Frame(main_frame, padding="5")
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.message_entry = ttk.Entry(input_frame)
        self.message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        self.message_entry.bind("<Return>", self.send_message)
        
        self.send_button = ttk.Button(input_frame, text="Enviar", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT)
        
        # Barra de estado
        self.status_var = tk.StringVar()
        self.status_var.set("Chatbot no iniciado. Cargue un modelo para comenzar.")
        
        status_frame = ttk.Frame(self.master, relief=tk.SUNKEN, padding=(2, 0))
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W)
        status_label.pack(fill=tk.X)
        
        # Deshabilitar entrada hasta que se cargue un modelo
        self.message_entry.config(state="disabled")
        self.send_button.config(state="disabled")

    def load_config(self):
        """Carga la configuración guardada"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    
                # Cargar el último modelo usado si existe
                if "last_model_path" in config and os.path.exists(config["last_model_path"]):
                    self.load_model(model_path=config["last_model_path"])
                    
            except Exception as e:
                messagebox.showwarning("Error de Configuración", 
                                     f"No se pudo cargar la configuración: {str(e)}")
        else:
            # Crear archivo de configuración por defecto
            default_config = {
                "last_model_path": "",
                "window_size": "900x700",
                "theme": "default"
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)

    def save_config(self, **kwargs):
        """Guarda la configuración actual"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
            except:
                config = {}
        else:
            config = {}
            
        # Actualizar configuración
        config.update(kwargs)
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def load_model(self, model_path=None):
        """Carga el modelo del chatbot"""
        if self.is_chatbot_loaded:
            confirm = messagebox.askyesno("Cargar Modelo", 
                                        "Ya hay un modelo cargado. ¿Desea cargar otro modelo?")
            if not confirm:
                return
        
        if not model_path:
            model_path = filedialog.askopenfilename(
                title="Seleccionar Modelo",
                filetypes=[("Modelos", "*.bin *.gguf *.ggml *.pt *.ckpt"), 
                           ("Todos los archivos", "*.*")]
            )
            
        if model_path:
            self.status_var.set("Cargando modelo... Por favor espere")
            self.master.update()
            
            # Cargar modelo en un hilo separado para no bloquear la UI
            threading.Thread(target=self._load_model_thread, args=(model_path,), daemon=True).start()

    def _load_model_thread(self, model_path):
        """Carga el modelo en un hilo separado"""
        try:
            # Inicializar el chatbot
            self.chatbot = AILocalChatbot(model_path)
            
            # Actualizar UI en el hilo principal
            self.master.after(0, self._model_loaded_callback)
            
            # Guardar ruta del modelo en la configuración
            self.save_config(last_model_path=model_path)
            
        except Exception as e:
            error_msg = f"Error al cargar el modelo: {str(e)}"
            self.master.after(0, lambda: self._show_error(error_msg))

    def _model_loaded_callback(self):
        """Callback cuando el modelo termina de cargar"""
        self.is_chatbot_loaded = True
        self.message_entry.config(state="normal")
        self.send_button.config(state="normal")
        
        personality_name = self.chatbot.personality.get('name', 'Asistente')
        self.status_var.set(f"Modelo cargado. Chatbot '{personality_name}' listo.")
        
        # Mensaje de bienvenida
        self.append_to_chat(personality_name, "¡Hola! Estoy listo para conversar contigo.")

    def send_message(self, event=None):
        """Envía un mensaje al chatbot y muestra la respuesta"""
        if not self.is_chatbot_loaded:
            messagebox.showinfo("Chatbot no iniciado", 
                               "Por favor cargue un modelo primero.")
            return
            
        if self.is_processing:
            return
            
        message = self.message_entry.get().strip()
        if not message:
            return
            
        # Limpiar entrada
        self.message_entry.delete(0, tk.END)
        
        # Mostrar mensaje del usuario
        self.append_to_chat("Tú", message)
        
        # Procesar mensaje en un hilo separado
        self.is_processing = True
        self.status_var.set("Procesando mensaje...")
        threading.Thread(target=self._process_message_thread, args=(message,), daemon=True).start()

        self.api_url = "http://localhost:5000/api/chat"  # URL de la API del backend

    def _process_message_thread(self, message):
        """Procesa el mensaje en un hilo separado"""
        try:
            # Enviar solicitud a la API
            response = requests.post(
                self.api_url,
                json={"message": message}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.master.after(0, lambda: self._show_response(data["response"]))
            else:
                raise Exception(f"Error {response.status_code}: {response.text}")
            
        except Exception as e:
            error_msg = f"Error al procesar el mensaje: {str(e)}"
            self.master.after(0, lambda: self._show_error(error_msg))
        finally:
            self.master.after(0, self._end_processing)

    def _show_response(self, response):
        """Muestra la respuesta del chatbot"""
        personality_name = self.chatbot.personality.get('name', 'Asistente')
        self.append_to_chat(personality_name, response)

    def _end_processing(self):
        """Finaliza el estado de procesamiento"""
        self.is_processing = False
        self.status_var.set("Listo")

    def _show_error(self, error_msg):
        """Muestra un mensaje de error"""
        messagebox.showerror("Error", error_msg)
        self.status_var.set("Error: " + error_msg)
        self.is_processing = False

    def append_to_chat(self, sender, message):
        """Añade un mensaje al área de chat"""
        self.chat_display.config(state="normal")
        
        # Formato de tiempo
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Insertar con formato
        self.chat_display.insert(tk.END, f"[{timestamp}] {sender}: ", "sender")
        self.chat_display.insert(tk.END, f"{message}\n\n", "message")
        
        # Desplazar a la última línea
        self.chat_display.see(tk.END)
        self.chat_display.config(state="disabled")

    def save_conversation(self):
        """Guarda la conversación actual"""
        if not self.is_chatbot_loaded:
            messagebox.showinfo("Chatbot no iniciado", 
                               "Por favor cargue un modelo primero.")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Guardar Conversación"
        )
        
        if file_path:
            try:
                result = self.chatbot.save_conversation(file_path)
                messagebox.showinfo("Conversación Guardada", result)
            except Exception as e:
                messagebox.showerror("Error", f"Error al guardar la conversación: {str(e)}")

    def load_conversation(self):
        """Carga una conversación guardada"""
        if not self.is_chatbot_loaded:
            messagebox.showinfo("Chatbot no iniciado", 
                               "Por favor cargue un modelo primero.")
            return
            
        file_path = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Cargar Conversación"
        )
        
        if file_path:
            try:
                result = self.chatbot.load_conversation(file_path)
                messagebox.showinfo("Conversación Cargada", result)
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar la conversación: {str(e)}")

    def edit_personality(self):
        """Abre una ventana para editar la personalidad"""
        if not self.is_chatbot_loaded:
            messagebox.showinfo("Chatbot no iniciado", 
                               "Por favor cargue un modelo primero.")
            return
            
        # Crear ventana de edición
        edit_window = tk.Toplevel(self.master)
        edit_window.title("Editar Personalidad")
        edit_window.geometry("600x500")
        edit_window.minsize(500, 400)
        
        # Cargar personalidad actual
        personality = self.chatbot.personality
        
        # Crear widgets
        main_frame = ttk.Frame(edit_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Nombre
        name_frame = ttk.Frame(main_frame)
        name_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(name_frame, text="Nombre:").pack(side=tk.LEFT)
        name_entry = ttk.Entry(name_frame, width=40)
        name_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        name_entry.insert(0, personality["name"])
        
        # Descripción corta
        desc_frame = ttk.Frame(main_frame)
        desc_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(desc_frame, text="Descripción:").pack(side=tk.LEFT)
        desc_entry = ttk.Entry(desc_frame, width=40)
        desc_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        desc_entry.insert(0, personality["description"])
        
        # Rasgos
        traits_frame = ttk.Frame(main_frame)
        traits_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(traits_frame, text="Rasgos:").pack(side=tk.LEFT)
        traits_entry = ttk.Entry(traits_frame, width=40)
        traits_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        traits_entry.insert(0, ", ".join(personality["traits"]))
        
        # Estilo de habla
        style_frame = ttk.Frame(main_frame)
        style_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(style_frame, text="Estilo de habla:").pack(side=tk.LEFT)
        style_entry = ttk.Entry(style_frame, width=40)
        style_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        style_entry.insert(0, personality["speaking_style"])
        
        # Intereses
        interests_frame = ttk.Frame(main_frame)
        interests_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(interests_frame, text="Intereses:").pack(side=tk.LEFT)
        interests_entry = ttk.Entry(interests_frame, width=40)
        interests_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        interests_entry.insert(0, ", ".join(personality["interests"]))
        
        # Descripción larga
        long_desc_frame = ttk.LabelFrame(main_frame, text="Descripción Larga")
        long_desc_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        long_desc_text = scrolledtext.ScrolledText(long_desc_frame)
        long_desc_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        long_desc_text.insert("1.0", personality["long_description"])
        
        # Botones
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        def save_personality():
            try:
                # Recopilar datos
                new_personality = {
                    "name": name_entry.get().strip(),
                    "description": desc_entry.get().strip(),
                    "traits": [t.strip() for t in traits_entry.get().split(",") if t.strip()],
                    "speaking_style": style_entry.get().strip(),
                    "interests": [i.strip() for i in interests_entry.get().split(",") if i.strip()],
                    "long_description": long_desc_text.get("1.0", tk.END).strip()
                }
                
                # Validar datos mínimos
                if not new_personality["name"]:
                    raise ValueError("El nombre no puede estar vacío")
                    
                if not new_personality["traits"]:
                    raise ValueError("Debe incluir al menos un rasgo")
                    
                # Actualizar personalidad
                result = self.chatbot.update_personality(new_personality)
                messagebox.showinfo("Personalidad Actualizada", result)
                edit_window.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al actualizar la personalidad: {str(e)}")
        
        ttk.Button(button_frame, text="Guardar", command=save_personality).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancelar", command=edit_window.destroy).pack(side=tk.RIGHT, padx=5)

    def view_personality(self):
        """Muestra la personalidad actual"""
        if not self.is_chatbot_loaded:
            messagebox.showinfo("Chatbot no iniciado", 
                               "Por favor cargue un modelo primero.")
            return
            
        # Formatear personalidad para mostrar
        personality = self.chatbot.personality
        detail = f"""Nombre: {personality['name']}
Descripción: {personality['description']}
Rasgos: {', '.join(personality['traits'])}
Estilo de habla: {personality['speaking_style']}
Intereses: {', '.join(personality['interests'])}

Descripción larga:
{personality['long_description']}
"""
        
        # Mostrar ventana con detalles
        messagebox.showinfo("Personalidad Actual", detail)

    def view_vocabulary_stats(self):
        """Muestra estadísticas del vocabulario aprendido"""
        if not self.is_chatbot_loaded:
            messagebox.showinfo("Chatbot no iniciado", 
                               "Por favor cargue un modelo primero.")
            return
            
        stats = self.chatbot.get_vocabulary_stats()
        detail = f"""Palabras aprendidas: {stats['total_words']}
Frases aprendidas: {stats['total_phrases']}

Palabras recientes:
{', '.join(stats['recent_words'])}
"""
        
        messagebox.showinfo("Estadísticas de Vocabulario", detail)

    def search_word(self):
        """Busca una palabra en el diccionario"""
        if not self.is_chatbot_loaded:
            messagebox.showinfo("Chatbot no iniciado", 
                              "Por favor cargue un modelo primero.")
            return
            
        # Solicitar palabra
        word = simpledialog.askstring("Buscar Palabra", "Ingrese la palabra a buscar:")
        
        if not word:
            return
            
        # Buscar en el diccionario
        dictionary = self.chatbot.learning_system.dictionary
        
        if word in dictionary:
            definitions = dictionary[word]["definitions"]
            detail = f"Definiciones para '{word}':\n\n"
            
            for i, definition in enumerate(definitions, 1):
                detail += f"{i}. {definition}\n"
                
            messagebox.showinfo(f"Definición de '{word}'", detail)
        else:
            # Preguntar si desea buscar la palabra
            if messagebox.askyesno("Palabra no encontrada", 
                                  f"La palabra '{word}' no está en el diccionario. ¿Desea buscarla ahora?"):
                # Buscar palabra
                self.status_var.set(f"Buscando definición para '{word}'...")
                
                definitions = self.chatbot.learning_system.search_word_definition(word)
                
                if definitions:
                    # Agregar al diccionario
                    self.chatbot.learning_system.dictionary[word] = {
                        "definitions": definitions,
                        "learned_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    self.chatbot.learning_system.save_dictionary()
                    
                    # Mostrar definiciones
                    detail = f"Definiciones encontradas para '{word}':\n\n"
                    
                    for i, definition in enumerate(definitions, 1):
                        detail += f"{i}. {definition}\n"
                    
                    messagebox.showinfo(f"Definición de '{word}'", detail)
                else:
                    messagebox.showinfo("Sin resultados", f"No se encontraron definiciones para '{word}'.")
                
                self.status_var.set("Listo")

print("Python Path:", sys.path)
print("Working Directory:", os.getcwd())