# 🚀 SupportNet-X: Multi-Domain Support Triage Agent

SupportNet-X is an intelligent support triage agent designed for the **HackerRank Orchestrate Hackathon**. It provides an end-to-end pipeline for classifying, grounding, and responding to support tickets across three major ecosystems: **HackerRank**, **Claude**, and **Visa**.

---

## 🛠️ Approach Overview

SupportNet-X is built on a **modular RAG (Retrieval-Augmented Generation)** architecture designed for safety and domain-specific accuracy.

### 1. Hybrid Classification & Domain Inference
- **Keyword Heuristics**: Uses a tiered keyword matching system to categorize tickets into product areas (e.g., `account/billing`, `assessments`, `fraud/security`).
- **Domain Routing**: Automatically infers the target company (HackerRank, Claude, or Visa) even when missing from the metadata, ensuring the retriever only searches the relevant knowledge base.

### 2. Grounded Retrieval Pipeline
- **Vector Search**: Uses `all-MiniLM-L6-v2` embeddings and `FAISS` for semantic similarity search.
- **Contextual Chunking**: Support documents are processed with metadata-aware chunking to maintain source context.
- **RRF (Reciprocal Rank Fusion)**: Combines vector search with lexical matching for robust retrieval.

### 3. Safety & Grounding Guardrails
- **Risk Detection**: High-risk topics like Identity Theft or Refund requests are automatically escalated to ensure human oversight.
- **Post-Generation Validation**: Every AI-generated response is validated against the retrieved chunks. If the AI fails to cite its sources (`[chunk_id]`), the system triggers a **Smart Fallback** that provides a safe, grounded guidance message.
- **hallucination Prevention**: The system strictly enforces that no information outside the provided support corpus is used.

---

## 🚀 Setup Instructions

### 1. Prerequisites
- Python 3.10+
- A Hugging Face API Token (or OpenAI/Anthropic API Key)

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/Monika-Bhardwaj/SupportNet-X
cd SupportNet-X

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the project root with your credentials:
```text
HUGGINGFACE_API_KEY=your_token_here
LLM_PROVIDER=huggingface
LLM_MODEL=HuggingFaceH4/zephyr-7b-beta
```

### 4. Data Preparation
Place the support documentation (markdown/text files) in the `data/` directory:
- `data/hackerrank/`
- `data/claude/`
- `data/visa/`

---

## 🏃 Running the Agent

### Full Pipeline Execution
The easiest way to run the entire project (indexing, triage, and evaluation) is using the orchestration script:
```powershell
python scripts/run_all.py
```

### Generating Submission Files
To generate the final files for the HackerRank platform:
```powershell
python scripts/make_submission_bundle.py
```
This will create a `supportnetx_submission.zip` containing:
1. `code.zip` (Source code)
2. `output.csv` (Agent predictions)
3. `log.txt` (Full chat/trace log)

---

## 📊 Technical Stack
- **Engine**: Python, Pydantic
- **Vector DB**: FAISS
- **Embeddings**: Sentence-Transformers
- **LLM**: Hugging Face Inference API (Mistral/Zephyr)
- **Reporting**: Rich (Console UI)
