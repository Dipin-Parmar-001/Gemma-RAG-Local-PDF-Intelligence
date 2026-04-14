# 📄 RAG Document Q&A with Gemma 4

A powerful Retrieval-Augmented Generation (RAG) application that allows you to chat with your PDF documents locally. This project uses **Streamlit** for the UI, **ChromaDB** for vector storage, and **Gemma 4** via Ollama for intelligent local processing.

## 🚀 Features
* **Local Processing:** Keep your data private by running the LLM locally.
* **PDF Support:** Upload any PDF and get instant answers.
* **MMR Retrieval:** Uses Maximum Marginal Relevance for diverse and accurate context fetching.
* **Persistent Storage:** Saves your document embeddings in a local `vector_db`.

## 🛠️ Prerequisites
Before running the app, ensure you have the following installed:
1. [Python 3.9+](https://www.python.org/)
2. [Ollama](https://ollama.com/)

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Pull the Gemma 4 Model:**
   Open your terminal and run:
   ```bash
   ollama pull gemma4:31b-cloud
   ```

## 🔑 Environment Setup
Create a `.env` file in the root directory and add your API keys:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
OLLAMA_API_KEY=your_ollama_api_key_here
```
*Note: The Mistral key is required for generating document embeddings.*

## 🏃 How to Run
Start the Streamlit application:
```bash
streamlit run app.py
```

## 📂 Project Structure
* `app.py`: The main Streamlit application code.
* `vector_db/`: Directory where processed document chunks are stored.
* `requirements.txt`: List of required Python packages.
* `.env`: (Hidden) Stores your private API keys.

---
