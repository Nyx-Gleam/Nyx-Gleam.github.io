# AI ChatBot

An offline AI-powered chatbot designed for local use with full customization of its personality. Built using Flask for the backend and a lightweight web frontend, this project offers a modular and extensible architecture for experimenting with local LLMs.

---

## 🚀 Features

- 🧠 **Local AI Chatbot** using a pre-trained model such as LLaMA or GPT-2 (`.gguf` format)
- 🌐 **Web Frontend** for interacting with the chatbot
- 🔧 **Customizable Personality** via configuration files
- 📜 **Conversation Logging** for continuous learning and fine-tuning
- 📁 **Modular Structure** for easy maintenance and scalability

---

## 📁 Project Structure

```
AI-ChatBot/
├── backend/               # Flask API and core AI logic
│   ├── app.py             # Main backend server
│   ├── chatbot_core.py    # Handles local model logic
│   └── requirements.txt   # Python dependencies
│
├── frontend/              # Web-based user interface
│   ├── index.html         # Main UI
│   ├── app.js             # Chat logic
│   ├── style.css          # UI styling
│   └── main.py            # Optional Python frontend logic
│
├── config/                # Custom chatbot behavior
│   ├── personality.json   # Personality config
│   └── settings.json      # Global settings
│
├── data/                  # Dictionary and phrases
│   ├── dictionary.json
│   ├── phrases.json
│   └── conversation_history/
│
├── logs/                  # Log files
│   └── chatbot.logs       # Stores previous chats for learning
│
├── models/                # LLM model file (must be added manually)
│   └── mistral-7b-instruct-v0.1.Q4_K_M.gguf
│
├── src/                   # Additional scripts and utilities
│   ├── gui.py
│   └── web_search.py
│
├── continuous_learning.py # Experimental learning logic
├── test_imports.py        # Import check script
├── tree.py                # Prints project structure
├── estructura.txt         # Textual project overview
└── README.md              # You are here!
```

---

## 🔧 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Nyx-Gleam/AI-ChatBot.git
cd AI-ChatBot
```

### 2. Add a Model

Download a compatible `.gguf` model (such as LLaMA, GPT-2, or Mistral) and place it in the `models/` directory.  
Example:

```bash
models/
└── mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

> ⚠️ Models are **not included** in this repository due to file size.

### 3. Create Required Folders and Files

Make sure the following exist:

```bash
logs/
└── chatbot.logs
```

This file will store previous chat conversations for training and learning purposes.

### 4. Backend Setup

```bash
cd backend
pip install -r requirements.txt
python app.py
```

> The server will run on `http://localhost:5000`

### 5. Frontend

Open `frontend/index.html` directly in a browser or host it using a local web server.

---

## 📦 Pre-packaged Release

If you download the chatbot from the **[Releases](https://github.com/Nyx-Gleam/AI-ChatBot/releases)** section, it will come with a **default lightweight model** already placed in the `models/` directory.  
This model is optimized for **low resource usage**, making it ideal for testing, development, or running on less powerful machines.

> 🧠 For better performance or more advanced capabilities, you can manually replace it with a larger model (e.g., LLaMA, GPT-2, Mistral) of your choice.

---

## 📌 API Endpoints

### `POST /api/chat`

**Request:**

```json
{
  "message": "Hello, who are you?"
}
```

**Response:**

```json
{
  "response": "I'm your personal assistant."
}
```

### `GET /health`

Used for health checks. Returns:

```json
{
  "status": "ok"
}
```

---

## 🧬 Personality Customization

Edit `config/personality.json` to modify how the chatbot behaves. You can change tone, vocabulary, humor, empathy, and more.

---

## 📄 License

[MIT License](LICENSE) — Free to use, modify, and distribute.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, open an issue first to discuss your ideas.

---

## 🌟 Author

Made with 💻 by [Nyx-Gleam](https://github.com/Nyx-Gleam)
