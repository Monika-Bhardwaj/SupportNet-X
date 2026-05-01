# 🚀 SupportNet-X: Advanced Support Triage Agent

SupportNet-X is a high-performance, multi-domain support triage agent built for the **HackerRank Orchestrate Hackathon**. It handles support tickets across the **HackerRank**, **Claude**, and **Visa** ecosystems with a focus on safety, grounding, and automated routing.

## ✨ Core Differentiators
- **Multi-Domain Intelligence**: Automatically routes tickets to the correct support corpus using domain inference and keyword heuristics.
- **Grounded RAG Pipeline**: Uses `sentence-transformers` and `FAISS` to retrieve the most relevant support documentation and generates responses with **inline citations** (e.g., `[chunk_id]`).
- **Safety First Architecture**: 
  - **Risk Classification**: Detects PII and high-risk topics (fraud, billing disputes) to trigger automated escalation.
  - **Grounding Guardrails**: Validates that every AI response is strictly grounded in retrieved facts.
  - **Fallback Safety**: Implements a "Smart Fallback" that cites documentation even when LLM APIs are unreachable.
- **Hybrid Performance**: Combines LLM reasoning with fast, deterministic keyword matching for classification.

## 🛠️ Project Structure
- `code/`: Core agent logic (`main.py` and `supportnetx/` package).
- `data/`: Support corpus organized by company.
- `scripts/`: Utility scripts for evaluation and bundling.
- `support_tickets/`: Input and output CSV data.

## 🚀 Final Submission Instructions
1. **Prepare Data**: Ensure the support corpus is in `data/`.
2. **Environment**: Ensure `.env` contains your `HUGGINGFACE_API_KEY` (or OpenAI/Anthropic).
3. **Run Pipeline**:
   ```powershell
   python scripts/run_all.py
   ```
4. **Generate Bundle**:
   ```powershell
   python scripts/make_submission_bundle.py
   ```
5. **Upload**: Upload `code.zip`, `output.csv`, and `log.txt` from the generated bundle to the HackerRank platform.
