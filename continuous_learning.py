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
                title = result.find('h2').get_text(strip=True) if result.find('h2') else "Sin título"
                snippet = result.find('a', class_='result__snippet')
                snippet_text = snippet.get_text(strip=True) if snippet else "Sin resumen"
                link = result.find('a', href=True)['href'] if result.find('a', href=True) else "#"
                
                results.append({
                    'title': title,
                    'summary': snippet_text,
                    'url': link
                })
            
            return results
            
        except Exception as e:
            print(f"Error en búsqueda web: {e}")
            return []