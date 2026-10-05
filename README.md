# ⚖️ LexAI — AI-Powered Legal Document Analyzer


> An AI-powered legal document analyzer built with **Streamlit**, **Groq API** (Llama 3.3 70B), and **LangChain RAG** — with a full CI/CD pipeline and Streamlit Cloud deployment.

---

## ✨ Features

| Tab | Feature | Description |
|-----|---------|-------------|
| 📤 | **Upload & Analyze** | Auto-generate document summaries, extract parties, dates, obligations |
| 🚩 | **Red Flag Detector** | AI scans for risky clauses with severity scoring (0–100 risk score) |
| ⚖️ | **Compare Documents** | Upload 2 versions → AI highlights additions, removals, and changes |
| 🌍 | **Jurisdiction Check** | Flag clauses illegal/unenforceable in 6+ countries and regions |
| 💬 | **Chat with Document** | RAG-powered Q&A with source citations from your document |

---

## 🛠️ Tech Stack

```
Frontend:    Streamlit
LLM:         Groq API (llama-3.3-70b-versatile)
RAG:         LangChain + FAISS
Embeddings:  HuggingFace (all-MiniLM-L6-v2) — FREE, runs locally
PDF:         PyMuPDF
CI/CD:       GitHub Actions
Deploy:      Streamlit Cloud (free)
```

---

## 🚀 Quick Start (Local)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/lexai.git
cd lexai
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Groq API key
# GROQ_API_KEY=gsk_your_key_here
```

> 🔑 Get your **free** Groq API key at [console.groq.com](https://console.groq.com) — no credit card needed!

### 5. Run the app
```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=utils --cov=components --cov-report=term-missing

# Run linter
ruff check .

# Check formatting
black --check .
```

---

## 🌐 Deploy to Streamlit Cloud

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: LexAI"
git remote add origin https://github.com/yourusername/lexai.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repository and set:
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
5. Click **"Deploy!"**

### Step 3: Add API Key Secret
In Streamlit Cloud dashboard → **Settings** → **Secrets**, add:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```

### Step 4: Set up GitHub Secrets (for CI/CD)
In GitHub → **Settings** → **Secrets and variables** → **Actions**, add:
```
GROQ_API_KEY = gsk_your_key_here
```

---

## 🔄 CI/CD Pipeline

```
Every PR/Push to main:
  → GitHub Actions CI runs
      → Ruff linter check
      → Black formatter check
      → pytest test suite
  → If all pass → Streamlit Cloud auto-deploys ✅
```

The pipeline is defined in:
- [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — Lint & Test
- [`.github/workflows/cd.yml`](.github/workflows/cd.yml) — Deploy gate

---

## 📁 Project Structure

```
lexai/
├── streamlit_app.py           # 🚀 Main entry point
├── components/
│   ├── tab_analyze.py         # 📤 Upload & Analyze
│   ├── tab_risk.py            # 🚩 Red Flag Detector
│   ├── tab_compare.py         # ⚖️ Compare Documents
│   ├── tab_jurisdiction.py    # 🌍 Jurisdiction Check
│   └── tab_chat.py            # 💬 Chat Interface
├── utils/
│   ├── pdf_processor.py       # PDF loading & chunking
│   ├── groq_client.py         # Groq API wrapper
│   ├── embeddings.py          # FAISS vector store
│   └── rag_chain.py           # LangChain RAG pipeline
├── tests/
│   ├── test_pdf_processor.py  # PDF utility tests
│   └── test_utils.py          # Groq/RAG/embedding tests
├── .github/workflows/
│   ├── ci.yml                 # CI pipeline
│   └── cd.yml                 # CD pipeline
├── .streamlit/config.toml     # Streamlit theme
├── pyproject.toml             # Ruff + Black + pytest config
├── requirements.txt
└── .env.example
```

---

## ⚠️ Disclaimer

LexAI is for **informational and educational purposes only**. It does not constitute legal advice. Always consult a qualified attorney for legal matters.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
