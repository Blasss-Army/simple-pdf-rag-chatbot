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
- 📄 **Drop PDFs in a folder** → ask questions and get **cited answers** with page numbers.
- 🔀 **Optional reranking stage** that rescrores and reorders retrieved chunks to improve answer quality (keeps the most relevant `k` from a larger candidate pool).

---

## 🧱 Tech Stack

- **LLM:** [Google Gemini](https://deepmind.google/technologies/gemini/) (`gemini-2.0-flash`)
- **Embeddings:** Google `text-embedding-004`
- **Vector Database:** [Qdrant](https://qdrant.tech/) (local path mode)
- **Frameworks:** [LangChain](https://www.langchain.com/) + [Gradio](https://www.gradio.app/)
- **Language:** Python 3.10+
- **Environment:** `.env` for API keys (`GOOGLE_API_KEY`)
- **(New) Reranker:** Optional reranking layer that operates on top of the vector search results (model/strategy is configurable in code).

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
         ├─► (Optional) Reranker
         │      └─ Rescore top-N candidates from vector search; keep top-k
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
│  └─ llm_call.py         # Main Chat class (chain, memory, retriever + (optional) reranking)
├─ create_retriever/
│  ├─ __init__.py
│  ├─ conf.py             # RetrieverConfig dataclass
│  └─ retriever.py        # PDF loading, chunking, embedding, Qdrant setup (+ rerank orchestration)
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
    # Rerank is optional and configured in code/UI. See notes below.
```

### 🔀 Reranking (optional)

- **What it does:** After the initial vector search returns a candidate pool of size **`vectore_store_fetch_k`**, the reranker **rescoring** step reorders candidates by query–passage relevance and keeps the most relevant **`vectore_store_k`** chunks that are passed to the LLM.
- **Why it helps:** Improves factual grounding and reduces noisy contexts, especially when using larger `fetch_k`.
- **How to enable:** Toggle it in your code or UI (depending on your implementation). The reranker runs **on top of** your current `vectore_store_search_type` (e.g., `"mmr"` or `"similarity"`), so you can keep MMR for diversity and still benefit from relevance reranking.
- **Recommended values:** Start with `vectore_store_k = 5` and `vectore_store_fetch_k = 30`. If you need more precision, increase `fetch_k` (e.g., 50–100) and keep `k` modest (5–8).
- **Performance tip:** If you combine MMR + Rerank, a common heuristic is to MMR-select a pool and then rerank that pool. You can also use a split like `ceil(k/2)` from MMR + `ceil(k/2)` from pure similarity before the final rerank to balance diversity and precision.

> Implementation details (model, thresholds, toggles) are intentionally left in code so you can swap different rerankers (e.g., cross-encoders, LLM scoring).

---

## 🚀 Getting Started

### 1️⃣ Install Requirements

```bash
git clone <your-repo-url>
cd P1-Simple-PDF-RAG-Chatbot
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scriptsctivate

pip install -U pip
pip install langchain langchain-community langchain-google-genai qdrant-client gradio python-dotenv
# If your reranker needs extras (e.g., sentence-transformers), install them here.
```

---

### 2️⃣ Set Environment Variables

Create a `.env` file in the project root:

```bash
GOOGLE_API_KEY=your_api_key_here
```

Add any reranker-specific environment variables here if your implementation uses them.

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
3. **Vector search** in Qdrant retrieves a **candidate pool** of size `vectore_store_fetch_k` using your chosen search strategy (e.g., MMR or similarity).
4. **(Optional) Reranking** **rescoring** step orders those candidates by fine-grained relevance and keeps the top `vectore_store_k` for the LLM.
5. Retrieved context + user query are passed to **Gemini**, which generates a grounded, cited answer.
6. Responses include references (file + page) and use conversational memory for context-aware conversations.

---

## 🛠️ Key Improvements in This Version

- ✅ **Reranking stage added:** improves precision by rescoring retrieved chunks before passing them to the LLM.  
- ✅ **Chain bug fix:** `_make_chain()` now returns a valid `ConversationalRetrievalChain`.  
- ✅ **Retriever refactor:** duplicate insertions removed and retrieval logic improved.  
- ✅ **Dynamic updates:** retriever settings can be modified from the UI without restarting the app.  
- ✅ **Cross-platform:** source file paths now handled with `os.path.basename()`.
- ✅ **English documentation:** all code comments and docstrings are in English for readability.  
- ✅ **Cleaner UI:** all interface labels standardized and improved for demo purposes.

---

## 🧪 Recommended Tests

- Upload multiple PDFs and query across them.  
- Adjust retrieval settings (e.g., `k`, `fetch_k`, `lambda_mult`) to see real-time impact.  
- Toggle reranking ON/OFF and measure answer quality and latency.  
- Clear memory and test context retention across turns.  
- Verify citation stability across repeated runs (deterministic IDs).  

---

## 📈 Future Enhancements

- 🔍 Hybrid lexical–vector retrieval  
- 📊 Learned/LLM-assisted reranking strategies with calibration
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
