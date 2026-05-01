# SupportNet-X

SupportNet-X is an intelligent multi-domain support triage assistant that processes ticket CSV files and produces safe, grounded responses with escalation when needed.

## Features

- Modular pipeline: `ingestion -> escalation -> classifier -> retriever -> responder`.
- Hybrid retrieval with semantic search (`FAISS`) + lexical retrieval (`BM25`) merged by Reciprocal Rank Fusion.
- **Smarter Domain Inference**: Automatically infers company context (HackerRank, Claude, Visa) from ticket text if the `company` field is missing.
- **Enhanced Classification**: Robust request type detection and expanded product areas (including Account Access and Permissions) aligned with hackathon requirements.
- **Risk-Aware Escalation**: Confidence-based and sensitive-keyword escalation logic to handle PII, fraud, and out-of-scope queries safely.
- Hybrid retrieval with company-aware bias.
- Query rewriting pass for noisy ticket text.
- Pydantic output validation + retry loop for LLM responses.
- Per-response source citations (doc + chunk id).
- Post-generation grounding validator to prevent unsupported claims.
- Async ticket processing and Rich CLI summary.

... (rest of setup/run sections) ...

## Submission

To prepare your files for the HackerRank platform, run:

```bash
python scripts/make_submission_bundle.py
```

This will create a `supportnetx_submission.zip` containing the exact 3 files you need to upload separately:
1. `code.zip`: Your source code (excluding data and artifacts).
2. `output.csv`: Your agent's predictions.
3. `log.txt`: The full chat transcript/log.

## Project Structure

- `code/main.py`: thin CLI entrypoint.
- `code/supportnetx/config.py`: runtime config and environment settings.
- `code/supportnetx/models.py`: ticket/result/chunk schemas.
- `code/supportnetx/logging_utils.py`: AGENTS-style prompt/response log writer.
- `code/supportnetx/ingestion.py`: data loading, chunking, embedding, index/cache.
- `code/supportnetx/classifier.py`: request type/product area classification.
- `code/supportnetx/escalation.py`: risk/PII and confidence escalation logic.
- `code/supportnetx/retriever.py`: hybrid retrieval with company routing.
- `code/supportnetx/responder.py`: grounded answer + justification generation.
- `code/supportnetx/pipeline.py`: orchestration for a single ticket.

## Setup

1. Create environment and install dependencies:
   - `python -m venv .venv`
   - Windows: `.venv\\Scripts\\activate`
   - `pip install -r requirements.txt`
2. Add API keys in `.env` (optional but recommended):
   - `OPENAI_API_KEY=...`
   - `ANTHROPIC_API_KEY=...`
3. Place corpora under `data/` and input tickets under `support_tickets/support_tickets.csv`.

## Run

From `code/`:

```bash
python main.py
```

Generates `support_tickets/output.csv` with exact columns:

`status, product_area, response, justification, request_type`

## Evaluation Harness

Run rubric-aligned checks after generating `output.csv`:

```bash
python scripts/evaluate.py --input-csv support_tickets/support_tickets.csv --output-csv support_tickets/output.csv
```

Generate a judge-friendly quality report:

```bash
python scripts/demo_report.py --current support_tickets/output.csv
```

Compare against a baseline run:

```bash
python scripts/demo_report.py --current support_tickets/output.csv --baseline support_tickets/output_baseline.csv
```

Run everything in one command (triage + evaluation + demo report):

```bash
python scripts/run_all.py
```

Use a baseline in the one-command flow:

```bash
python scripts/run_all.py --baseline support_tickets/output_baseline.csv
```

Create a final submission zip with output, logs, and report:

```bash
python scripts/make_submission_bundle.py
```

Create bundle with baseline comparison:

```bash
python scripts/make_submission_bundle.py --baseline support_tickets/output_baseline.csv
```

Create bundle and open the output location automatically:

```bash
python scripts/make_submission_bundle.py --open
```

## Determinism and Reproducibility

- Seed is configurable via `SUPPORTNETX_SEED` (default `42`).
- Cached embeddings/index metadata are stored under `data/index_cache/`.
- `STRICT_COMPANY_ROUTING=true` enforces hard corpus isolation by company.
- `ENABLE_QUERY_REWRITE=true` controls the rewrite pre-pass.
- Dependency versions are pinned in `requirements.txt` for reproducible installs.
- All prompt/response events are appended to:
  - `%USERPROFILE%\\hackerrank_orchestrate\\log.txt` on Windows
  - `$HOME/hackerrank_orchestrate/log.txt` on macOS/Linux
