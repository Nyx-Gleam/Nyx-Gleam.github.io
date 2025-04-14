# Model Management for AI ChatBot

This document explains how to obtain and manage the language models used by the AI ChatBot. The project is designed to load models automatically from Hugging Face if they are not present locally, but you also have the option to run it completely offline.

---

## 1. Automatic Model Loading

When you run the application, the code in `chatbot_core.py` checks for the specified model by its identifier. By default, the model is loaded via the Hugging Face Model Hub using the repo ID (e.g., `"NyxGleam/mistral-7b-instruct-v0.1.Q4_K_M"`).

### How It Works:
- The method `_load_model()` in `chatbot_core.py` calls:
  
  ```python
  self.llm = AutoModelForCausalLM.from_pretrained(
      "NyxGleam/mistral-7b-instruct-v0.1.Q4_K_M",
      model_type=model_type,
      context_length=2048,
      gpu_layers=0,
      token=os.getenv("HF_TOKEN")
  )
  ```
  
- If the model is not available locally, or if you're running the application in the cloud (e.g., Render), it automatically downloads the model from Hugging Face.

- For public repositories, no authentication is required. If your model is private or gated, be sure to set an environment variable `HF_TOKEN` with your Hugging Face access token.

---

## 2. Manual Model Download (Offline Mode)

If you prefer to run the chatbot offline (without an Internet connection), follow these steps:

### Step-by-Step Instructions:
1. **Visit the Model Repository:**
   - Go to [https://huggingface.co/NyxGleam/mistral-7b-instruct-v0.1.Q4_K_M](https://huggingface.co/NyxGleam/mistral-7b-instruct-v0.1.Q4_K_M) to access the model repository.

2. **Download the Model:**
   - Use the "Download" button on the model page to download the entire model file (in `.gguf` format).
   - Note that the file size is large (around 4 GB), so the download may take some time.

3. **Place the Model in the Correct Folder:**
   - Once downloaded, place the model file in the `models/` folder of your project.
   - Ensure that the model filename matches exactly what the code expects. For example:
     
     ```
     models/mistral-7b-instruct-v0.1.Q4_K_M.gguf
     ```

     or

     ```
     models/Llama-3.1-8B-UltraLong-4M-Instruct.gguf
     ```

4. **Run the Application Offline:**
   - With the model file present in the `models/` folder, when you run the application, it will load the model locally without attempting to download it from Hugging Face.
   
   - Execute your backend normally:
     
     ```bash
     python backend/app.py
     ```

---

## 3. Release Packages and Offline Usage

For distribution, you can create a GitHub Release that includes:
- The project code (without the 4 GB model file, to keep the repository lightweight).
- A packaged version (ZIP) that includes the model file in the `models/` folder.
- Alternatively, you can attach the model file as an asset in the release, and include instructions in this document on how to place it in the `models/` folder.

---

## 4. Environment Variables and Configuration

To ensure the correct automatic download from Hugging Face:
- Set the environment variable `HF_TOKEN` if required (for private or gated repositories). In your deployment service (Render, for example), add the variable in your settings:
  
  ```
  HF_TOKEN=<your_huggingface_access_token>
  ```
  
- The code handles the token by reading it via `os.getenv("HF_TOKEN")`.

---

## Conclusion

This setup allows you to choose between:
- **Online Mode:** The model is automatically downloaded from Hugging Face if needed.
- **Offline Mode:** Download the model manually and place it in the `models/` folder, so the application loads the model locally without requiring an internet connection.

Make sure to follow these instructions to optimize the performance of your AI ChatBot and ensure compatibility across different deployment environments.