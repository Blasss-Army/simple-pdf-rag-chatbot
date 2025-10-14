---
title: Simple PDF RAG Chatbot
emoji: ⚡
colorFrom: red
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Privacy-first RAG chatbot to query your PDFs with citations.
sdk_version: 5.49.1
---
# 📘 Simple PDF RAG Chatbot (Gradio + Gemini + Qdrant)

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot designed to answer questions based on your own **PDF documents**.  
It uses **Google Gemini** for language generation, **text-embedding-004** for semantic embeddings, and a **local Qdrant vector database** for efficient similarity search — all wrapped in a clean **Gradio** web interface.

> 🚀 Built as a portfolio-ready project to demonstrate skills in LLM orchestration, retrieval pipelines, embeddings, and end-to-end GenAI application development.

---

## ✨ Features
- 📄 **Ask questions about your PDFs** and get **accurate, cited answers** with page references.  
- 🧠 **Retrieval-Augmented Generation pipeline** built with LangChain, Gemini, and Qdrant.  
- 🗃️ **Local vector storage** (Qdrant Path Mode) — no external database or cloud required.  
- 🔁 **Dynamic retriever settings** adjustable directly from the UI.  
- 🧬 **Deterministic chunk IDs** ensure reproducible citations and safe re-indexing.  
- 💬 **Conversational memory** remembers previous context across turns.  
- 🌐 **Cross-platform compatible** and ready for deployment or containerization.
- 📄**Drop PDFs in a folder** → ask questions and get **cited answers** with page numbers.

---

## 🧱 Tech Stack

- **LLM:** [Google Gemini](https://deepmind.google/technologies/gemini/) (`gemini-2.0-flash`)
- **Embeddings:** Google `text-embedding-004`
- **Vector Database:** [Qdrant](https://qdrant.tech/) (local path mode)
- **Frameworks:** [LangChain](https://www.langchain.com/) + [Gradio](https://www.gradio.app/)
- **Language:** Python 3.10+
- **Environment:** `.env` for API keys (`GOOGLE_API_KEY`)

---

## 🏗️ System Architecture

```text
User (Gradio UI)
   └── Chat()  ──(ConversationalRetrievalChain)──► LLM (Gemini)
         │
         ├─► Retriever (Qdrant + Embeddings)
         │      ├─ Load PDFs → Split into chunks
         │      ├─ Embed with Google text-embedding-004
         │      └─ Store vectors in local Qdrant (path = ./index)
         │
         └─► Memory (ConversationBufferMemory)
```

---

## 📂 Project Structure

```
P1-Simple PDF RAG Chatbot/
├─ app_core/
│  ├─ __init__.py
│  ├─ prompt.py           # Prompt templates for the LLM
│  └─ llm_call.py         # Main Chat class (chain, memory, retriever orchestration)
├─ create_retriever/
│  ├─ __init__.py
│  ├─ conf.py             # RetrieverConfig dataclass
│  └─ retriever.py        # PDF loading, chunking, embedding, Qdrant setup
├─ ui/
│  ├─ __init__.py
│  ├─ gradio_app.py       # Gradio UI logic
│  └─ style.py            # Reusable HTML UI components
├─ data/                  # Place your PDFs here
├─ index/                 # Local Qdrant vector store
├─ app.py                 # runner for Hugging Face
├─ main.py                # runner for Git Hub
├─ .env                   # API keys (GOOGLE_API_KEY)
└─ README.md
```

---

## ⚙️ Configuration

The behavior of the retriever is controlled via `create_retriever/conf.py`:

```python
@dataclass
class RetrieverConfig:
    data_path: Path = PROJECT_ROOT / "data"
    index_path: Path = PROJECT_ROOT / "index"
    collection_name: str = "my_collection"
    embed_model: str = "models/text-embedding-004"
    distance: Distance = Distance.COSINE
    prefer_grpc: bool = True
    reset_collection: bool = False
    vectore_store_search_type: str = "mmr"
    vectore_store_k: int = 5
    vectore_store_fetch_k: int = 30
    vectore_store_lambda_mult: float = 0.5
```

💡 **Tip:** Set `reset_collection=True` the first time or whenever you want to rebuild the index from scratch.

---

## 🚀 Getting Started

### 1️⃣ Install Requirements

```bash
git clone <your-repo-url>
cd P1-Simple-PDF-RAG-Chatbot
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -U pip
pip install langchain langchain-community langchain-google-genai qdrant-client gradio python-dotenv
```

---

### 2️⃣ Set Environment Variables

Create a `.env` file in the project root:

```bash
GOOGLE_API_KEY=your_api_key_here
```

---

### 3️⃣ Add PDFs

Place your documents inside the `data/` folder:

```
data/
├─ my_file.pdf
├─ research_paper.pdf
└─ report.pdf
```

---

### 4️⃣ Run the App

```bash
python app.py
```

Once started, open your browser at:  
👉 http://127.0.0.1:7860

---

## 🧠 How It Works

1. PDFs are loaded and split into overlapping text chunks (`RecursiveCharacterTextSplitter`).
2. Each chunk is embedded using `text-embedding-004`.
3. Vectors are stored in Qdrant and retrieved with **MMR search**.
4. Retrieved context + user query are passed to **Gemini**, which generates a grounded, cited answer.
5. Responses include references (file + page) and use conversational memory for context-aware conversations.

---

## 🛠️ Key Improvements in This Version

- ✅ **Chain bug fix:** `_make_chain()` now returns a valid `ConversationalRetrievalChain`.  
- ✅ **Retriever refactor:** duplicate insertions removed and retrieval logic improved.  
- ✅ **Dynamic updates:** retriever settings can be modified from the UI without restarting the app.  
- ✅ **Cross-platform:** source file paths now handled with `os.path.basename()`.
- ✅ **English documentation:** all code comments and docstrings are in English for readability.  
- ✅ **Cleaner UI:** all interface labels standardized and improved for demo purposes.

---

## 🧪 Recommended Tests

- Upload multiple PDFs and query across them.  
- Adjust retrieval settings (e.g., `k`, `lambda_mult`) to see real-time impact.  
- Clear memory and test context retention across turns.  
- Verify citation stability across repeated runs (deterministic IDs).  

---

## 📈 Future Enhancements

- 🔍 Hybrid lexical-vector retrieval  
- 📊 Cross-encoder reranking  
- 🌐 Multilingual question answering  
- 📑 Clickable citations with direct PDF previews  
- ⚡ FastAPI backend and REST API endpoints  
- 🐳 Docker deployment with one-command startup

---

## 📜 License

This project is released under the **MIT License**.  
See [LICENSE](./LICENSE) for details.

---

## 🙌 Acknowledgements

- [LangChain](https://python.langchain.com/)  
- [Qdrant](https://qdrant.tech/)  
- [Gradio](https://www.gradio.app/)  
- [Google Gemini](https://deepmind.google/technologies/gemini/)

---

### ⭐ Quick Demo

```bash
# Clone & run in minutes
git clone <your-repo-url>
cd P1-Simple-PDF-RAG-Chatbot
pip install -r requirements.txt
echo "GOOGLE_API_KEY=..." > .env
mkdir data && cp your.pdf data/
python app.py
```
