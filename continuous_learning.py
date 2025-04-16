import json
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime

class ContinuousLearningSystem:
    def __init__(self, data_dir="data", config_dir="config"):
        self.data_dir = data_dir
        self.config_dir = config_dir
        self.dictionary = {}
        self.phrases = []
        
        # Crear directorios si no existen
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        
        self.load_data()

    def load_data(self):
        # Cargar diccionario
        dict_path = os.path.join(self.data_dir, "dictionary.json")
        if os.path.exists(dict_path):
            with open(dict_path, 'r', encoding='utf-8') as f:
                self.dictionary = json.load(f)

    def save_dictionary(self):
        dict_path = os.path.join(self.data_dir, "dictionary.json")
        with open(dict_path, 'w', encoding='utf-8') as f:
            json.dump(self.dictionary, f, ensure_ascii=False, indent=2)

    def search_web(self, query, num_results=3):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(
                f"https://duckduckgo.com/html/?q={query}",
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            for result in soup.find_all('div', class_='result', limit=num_results):
                title = result.find('h2').get_text(strip=True) if result.find('h2') else "No title"
                snippet = result.find('a', class_='result__snippet')
                snippet_text = snippet.get_text(strip=True) if snippet else "No summary"
                link = result.find('a', href=True)['href'] if result.find('a', href=True) else "#"
                
                results.append({
                    'title': title,
                    'summary': snippet_text,
                    'url': link
                })
            
            return results
            
        except Exception as e:
            print(f"Error in web search: {e}")
            return []

    def save_interaction(self, user_input, response, interaction_type="chat"):
        """
        Guarda la interacción (mensaje de usuario y respuesta) en un archivo JSON dentro del directorio
        'conversation_history' en self.data_dir.
        """
        # Directorio para guardar el historial de conversación:
        history_folder = os.path.join(self.data_dir, "conversation_history")
        os.makedirs(history_folder, exist_ok=True)

        # Archivo donde se almacenarán las interacciones
        history_file = os.path.join(history_folder, "conversation_history.json")
        data = {
            "timestamp": datetime.now().isoformat(),
            "type": interaction_type,
            "input": user_input,
            "output": response
        }

        # Leer interacciones previas o iniciar con lista vacía
        if os.path.exists(history_file):
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []
        else:
            history = []

        # Aquí está el error - usar 'data' en lugar de 'interaction'
        history.append(data)  # Cambiado de 'interaction' a 'data'

        # Guardar el historial actualizado
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)