---
title: Simple PDF RAG Chatbot
emoji: ⚡
colorFrom: red
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: "Privacy-first RAG chatbot to query your PDFs with citations."
---

# 📘 Simple PDF RAG Chatbot (Gradio + Qdrant + Gemini)

A privacy-friendly **RAG** (Retrieval-Augmented Generation) chatbot for **question answering over your own PDFs**.  
It uses **Google Gemini** (via `langchain_google_genai`) for generation, **text-embedding-004** for embeddings, and a **local Qdrant** vector store for similarity search. The UI is built with **Gradio**.

> Perfect for a portfolio: clean architecture, local vector DB, deterministic chunk IDs, multilingual ready, and easy to extend with reranking/hybrid search.

---

## ✨ Features

- **Drop PDFs in a folder** → ask questions and get **cited answers** with page numbers.
- **Local vector store (Qdrant Path mode)**: no external DB needed.
- **Deterministic chunk IDs**: idempotent ingestion, reproducible citations, selective reindexing.
- **Conversational memory** using `ConversationBufferMemory`.
- **Gradio UI**: simple chat, copy-button on answers, source panel (file + page).
- **Configurable** chunking and retrieval via a single dataclass.

---

## 🏗️ Architecture

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

- **Generation**: `ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)`
- **Embeddings**: `GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")`
- **Vector DB**: `qdrant-client` (path mode), cosine distance
- **Retriever**: LangChain `QdrantVectorStore.as_retriever(search_type="mmr")`

---

## 📂 Project Structure

```
P1-Simple PDF RAG Chatbot/
├─ app_core/
│  ├─ __init__.py
│  ├─ prompt.py                   # qa_prompt, document_prompt (templates)
│  └─ llm_call.py                 # Chat class: memory + chain
├─ create_retriever/
│  ├─ __init__.py
│  ├─ conf.py                     # RetrieverConfig (paths, models, k, etc.)
│  └─ retriever.py                # Loader, splitter, Qdrant build, deterministic IDs
├─ ui/
│  ├─ __init__.py
│  └─ gradio_app.py               # Gradio Blocks UI
├─ data/                          # Put your PDFs here
├─ index/                         # Local Qdrant storage (auto-created)
├─ .env                           # GOOGLE_API_KEY=...
└─ README.md
```

---

## 🧰 Requirements

- **Python** 3.10+ (recommended)
- A **Google API key** with access to Gemini (Generative AI Studio)

### Install

```bash
# 1) Create and activate a virtualenv (Windows)
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install -U pip
pip install langchain langchain-community langchain-google-genai qdrant-client gradio python-dotenv
```

> If you use additional splitters/loaders/rerankers, include them here.

---

## 🔐 Environment Variables

Create a `.env` file at project root:

```bash
GOOGLE_API_KEY=your_api_key_here
```

> The code uses `dotenv.load_dotenv()` so the key is picked up automatically.

---

## ⚙️ Configuration (RetrieverConfig)

`create_retriever/conf.py`:

```python
@dataclass
class RetrieverConfig:
    collection_name: str = "my_collection"
    index_path: Path = PROJECT_ROOT / "index"           # local Qdrant storage
    embed_model: str = "models/text-embedding-004"
    distance: Distance = Distance.COSINE
    prefer_grpc: bool = True
    recreate: bool = False
    default_k: int = 4
    temperature: float = 0.2
    data_path: Path = PROJECT_ROOT / "data"             # PDFs live here
    reset_collection: bool = False                      # reindex from scratch if True
```

> Set `reset_collection=True` the first time or when you want to **rebuild** the index.  
> For idempotent ingestion across runs, the project generates **deterministic chunk IDs**.

---

## 🧠 Deterministic Chunk IDs (Why they matter)

This project builds stable IDs from chunk metadata and hashes them with **UUIDv5**:

- **Idempotent ingestion**: re-running over the same PDFs won’t create duplicates.
- **Selective updates/deletes**: re-embed or remove chunks per document/page.
- **Reproducibility**: citations and logs point to an exact chunk across runs.
- **Better UX**: bind feedback and analytics to a stable chunk identity.

Snippet (`create_retriever/retriever.py`):

```python
def _make_ids(self, chunks):
    ids = []
    for i, d in enumerate(chunks):
        src   = d.metadata.get("source", "doc")
        page  = d.metadata.get("page", -1)
        start = d.metadata.get("start_index", i)
        key = f"{src}|p{page}|s{start}"
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, key)))
    return ids
```

---

## ▶️ How to Run

1. Put your **PDFs inside `data/`**.
2. (Optional) In `conf.py`, set `reset_collection=True` for a fresh index.
3. Launch the Gradio app:

```bash
python main.py          # Windows
# or
python main.py          # macOS/Linux
```

Open the URL printed in the console (usually `http://127.0.0.1:7860`).

---

## 🗨️ Using the App

- Type your question in **Input**.
- Click **Enviar/Send**.
- The **Answer** box shows the LLM response.
- The **Sources** panel lists files and **pages** that supported the answer.

### Clear memory (optional UI button)
If you expose a “Clear” button, wire it to:

```python
def clear_ui(chat_state):
    if chat_state is not None:
        chat_state.clear_memory()
    return "", "", None, None  # clears input, output, sources, and state
```

Inside `Chat` (in `llm_call.py`):

```python
class Chat:
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer",
        )
        # ... build retriever and chain with memory=self.memory

    def clear_memory(self):
        self.memory.clear()
        return "All memory stored has been deleted"
```

---

## 🔎 How Retrieval Works (default)

1. Load PDFs from `data/` with `DirectoryLoader` + `PyPDFLoader`.
2. Split into chunks (`RecursiveCharacterTextSplitter`, default `chunk_size=1000`, `chunk_overlap=200`).
3. Embed with Google `text-embedding-004`.
4. Store in **Qdrant** (path mode) with **cosine** similarity.
5. Use an **MMR retriever** (`k=5`, `fetch_k=30`, `lambda_mult=0.5` by default).
6. Feed retrieved chunks + user question into **Gemini** to generate an answer.

---

## 🚧 Troubleshooting

- **`AttributeError: 'Chat' object has no attribute 'memory'`**  
  Ensure memory is stored as `self.memory` in `Chat.__init__` and passed to the chain (`memory=self.memory`).

- **No vectors indexed on first run**  
  If `reset_collection=False` and the collection is empty, make sure you still call `add_documents(...)`.  
  A safe pattern is to index when `reset_collection` **or** `points_count == 0`.

- **Windows path separators in sources**  
  Normalize source paths when displaying: `src = src.replace("/", "\\").split("\\")[-1]`.

- **GOOGLE_API_KEY not found**  
  Check `.env` location and that `dotenv.load_dotenv()` runs before initializing clients.

---

## 🧪 Testing Ideas (quick wins)

- **Chunking & metadata**: correct page and offsets present after split.
- **Deterministic IDs**: same PDFs → same IDs across runs.
- **Indexing logic**: indexes when empty or on `reset_collection=True`.
- **Retrieval quality**: top-K contains known relevant pages for test queries.
- **Memory**: messages accumulate and clear correctly.

---

## 🗺️ Suggested Improvements (Roadmap)

- **Hybrid search**: BM25 + vectors with reciprocal rank fusion.
- **Reranking**: cross-encoder (e.g., bge-reranker) or LLM scoring for top-K.
- **Multilingual** (DE/FR/IT/EN): auto-detect question language and respond accordingly.
- **Clickable citations**: open the exact PDF page/region; add thumbnails in UI.
- **File upload in UI**: ingest on the fly with progress bars.
- **Prompt guardrails**: answer strictly from context unless user opts out.
- **Evaluation**: integrate **RAGAS** and report Faithfulness/Context Precision.
- **API**: wrap core in **FastAPI** (`/chat`, `/ingest`, `/health`, `/feedback`).
- **Docker/Compose**: one-command bring-up with Qdrant + app.
- **Telemetry**: latency/tokens per turn, feedback storage.

---

## 📄 License

This project is licensed under the MIT License.  
See the [LICENSE](./LICENSE) file for details

---

## 🙏 Acknowledgments

- [LangChain](https://python.langchain.com/)
- [Qdrant](https://qdrant.tech/)
- [Gradio](https://www.gradio.app/)
- Google **Gemini** & **text-embedding-004**

---

### Quick Start TL;DR

```bash
git clone <your-repo>
cd P1-Simple\ PDF\ RAG\ Chatbot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip && pip install -r requirements.txt  # or the packages listed above
echo "GOOGLE_API_KEY=..." > .env
mkdir -p data && cp your.pdf data/
python ui/gradio_app.py
```