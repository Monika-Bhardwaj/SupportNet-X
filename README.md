# 🚀 SupportNet-X: Multi-Domain Support Triage Agent

SupportNet-X is an intelligent, safety-first support triage agent. It provides an end-to-end pipeline for classifying, grounding, and responding to support tickets across multiple distinct corporate ecosystems (e.g., HackerRank, Claude, Visa). 

---

## 🛑 Problem Statement

Customer support queues are frequently bottlenecked by high volumes of repetitive queries. While automation can alleviate this, traditional chatbots suffer from severe limitations:
1. **Hallucination Risk:** Providing incorrect refund policies or API limits creates massive legal and financial liabilities.
2. **Context Bleed:** When supporting multiple products or domains, bots often cross-contaminate advice (e.g., giving HackerRank advice to a Visa customer).
3. **Missed Escalations:** High-risk issues (fraud, identity theft, security breaches) are often mishandled by AI when human intervention is critically required.

---

## 🛠️ Approach & Detailed Solution

SupportNet-X is built on a **modular RAG (Retrieval-Augmented Generation)** architecture designed specifically for safety and domain-specific accuracy. Rather than relying on a single LLM call to solve everything, the system uses a multi-layered pipeline:

1. **Risk Triage (Circuit Breaker):** Immediately scans raw tickets for sensitive topics (fraud, PII) using regex and keyword heuristics. High-risk tickets are escalated instantly.
2. **Hybrid Classification & Domain Inference:** Classifies the ticket into a product area and infers the target company even if the user forgets to mention it. 
3. **Query Rewriting:** Uses a deterministic fallback or an LLM to clean up the ticket into a search-optimized query.
4. **Grounded Retrieval Pipeline:** Uses `FAISS` and `all-MiniLM-L6-v2` for semantic search, combined with lexical signals via **Reciprocal Rank Fusion (RRF)**. Searches are strictly partitioned by the inferred domain.
5. **Post-Generation Validation:** Enforces that the LLM explicitly cites its sources (`[chunk_id]`). If it fails, the system triggers a **Smart Fallback** instead of hallucinating.

---

## ✨ Core Functionalities

- **Automated Ticket Resolution:** Automatically drafts responses for safe, standard "how-to" and billing queries.
- **Dynamic Routing:** Categorizes tickets into product areas (e.g., `account/billing`, `assessments`, `fraud/security`).
- **Confidence-Based Escalation:** Escalates tickets if the vector search fails to find highly relevant documentation (below a strict threshold).
- **Graceful Degradation:** The pipeline never breaks; if the LLM API times out, it falls back to deterministic rules and safe guidance templates.

---

## 🌟 Core Differentiator Features

1. **Implicit Domain Inference:** The ability to dynamically partition the vector search space based on inferred company context, preventing cross-contamination.
2. **The Dual-Guardrail System:** Combining deterministic upfront risk heuristics with strict post-generation citation validation.
3. **RRF (Reciprocal Rank Fusion):** Merging dense semantic search with sparse lexical search to handle both conceptual queries and exact error codes (`ERR-1092`).

---

## 💻 Implementation & Code Snippets

### 1. Post-Generation Grounding Validation
The system validates that the AI explicitly used a retrieved chunk.
```python
def validate_grounding(payload: ResponsePayload, chunks: list[RetrievedChunk]) -> tuple[bool, str]:
    if not chunks:
        return False, "No retrieved support context was available."
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    cited = set(re.findall(r"\[([^\]]+)\]", payload.response))
    valid_cited = cited.intersection(chunk_ids)
    if not valid_cited:
        return False, "Response missing valid chunk citations."

    return True, "Grounding validation passed."
```

### 2. Upfront Risk Detection
Before the LLM even sees the text, it is scanned for risk.
```python
def evaluate_text_risk(self, text: str) -> RiskResult:
    lowered = text.lower()
    for keyword in RISK_KEYWORDS: # "fraud", "hacked", "security breach"
        if keyword in lowered:
            return RiskResult(escalated=True, reason=f"Escalated: sensitive topic detected.")
    if CARD_RE.search(text):
        return RiskResult(escalated=True, reason="Escalated: possible payment card number detected.")
    return RiskResult(escalated=False, reason="Safe")
```

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
Create a `.env` file in the project root:
```text
HUGGINGFACE_API_KEY=your_token_here
LLM_PROVIDER=huggingface
LLM_MODEL=HuggingFaceH4/zephyr-7b-beta
```

### 4. Running the Agent
Run the entire orchestration pipeline:
```powershell
python scripts/run_all.py
```

---

## 📊 Technical Stack
- **Engine**: Python, Pydantic
- **Vector DB**: FAISS
- **Embeddings**: Sentence-Transformers
- **LLM**: Hugging Face Inference API / Anthropic / OpenAI
- **Reporting**: Rich (Console UI)
