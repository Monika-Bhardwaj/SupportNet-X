from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from supportnetx.classifier import Classifier
from supportnetx.config import load_config, seed_everything
from supportnetx.escalation import EscalationLogic
from supportnetx.ingestion import build_or_load_index
from supportnetx.logging_utils import PromptLogger
from supportnetx.models import ProcessMeta, Ticket
from supportnetx.pipeline import Pipeline
from supportnetx.query_rewriter import QueryRewriter
from supportnetx.responder import Responder
from supportnetx.retriever import Retriever

REQUIRED_COLUMNS = ["issue", "subject", "company"]
OUTPUT_COLUMNS = ["status", "product_area", "response", "justification", "request_type"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SupportNet-X triage CLI")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    return parser.parse_args()


def _validate_input_columns(frame: pd.DataFrame) -> dict[str, str]:
    columns_map = {col.lower(): col for col in frame.columns}
    missing = [col for col in REQUIRED_COLUMNS if col not in columns_map]
    if missing:
        raise ValueError(f"Input CSV missing required columns (case-insensitive): {missing}")
    return {col: columns_map[col] for col in REQUIRED_COLUMNS}


async def _process_all_tickets(pipeline: Pipeline, tickets: list[Ticket], concurrency: int):
    semaphore = asyncio.Semaphore(concurrency)

    async def _worker(ticket: Ticket):
        async with semaphore:
            return await asyncio.to_thread(pipeline.process_ticket, ticket)

    return await asyncio.gather(*[_worker(t) for t in tickets])


def _render_summary(console: Console, frame: pd.DataFrame, tickets_df: pd.DataFrame, metas: list[ProcessMeta]) -> None:
    # Ensure tickets_df has lowercase company for grouping
    tickets_copy = tickets_df.copy()
    company_col = next((c for c in tickets_df.columns if c.lower() == "company"), "company")
    tickets_copy["company"] = tickets_copy[company_col].fillna("None")
    
    total = len(frame)
    escalated = int((frame["status"] == "escalated").sum())
    replied = total - escalated
    avg_conf = (sum(meta.confidence for meta in metas) / max(len(metas), 1)) if metas else 0.0
    low_conf = sum(1 for meta in metas if meta.confidence < 0.35)

    table = Table(title="SupportNet-X Run Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total tickets", str(total))
    table.add_row("Replied", str(replied))
    table.add_row("Escalated", str(escalated))
    table.add_row("Escalation rate", f"{(escalated / max(total, 1)) * 100:.1f}%")
    table.add_row("Average confidence", f"{avg_conf:.3f}")
    table.add_row("Low-confidence tickets (<0.35)", str(low_conf))
    console.print(table)

    company_table = Table(title="Per-company status breakdown")
    company_table.add_column("Company", style="magenta")
    company_table.add_column("Replied", style="green")
    company_table.add_column("Escalated", style="yellow")
    
    combined = pd.DataFrame({"company": tickets_copy["company"], "status": frame["status"]})
    grouped = combined.groupby(["company", "status"]).size().unstack(fill_value=0)
    for company, row in grouped.iterrows():
        company_table.add_row(
            str(company),
            str(int(row.get("replied", 0))),
            str(int(row.get("escalated", 0))),
        )
    console.print(company_table)

    risk_table = Table(title="Escalation categories")
    risk_table.add_column("Risk category", style="red")
    risk_table.add_column("Count", style="yellow")
    risk_counts: dict[str, int] = {}
    for meta in metas:
        risk_counts[meta.risk_category] = risk_counts.get(meta.risk_category, 0) + 1
    for category, count in sorted(risk_counts.items(), key=lambda item: item[1], reverse=True):
        risk_table.add_row(category, str(count))
    console.print(risk_table)


async def async_main() -> None:
    args = _parse_args()
    project_root = Path(args.project_root)
    config = load_config(project_root)
    seed_everything(config.seed)

    console = Console()
    logger = PromptLogger(config.log_file)
    index = build_or_load_index(config.data_dir, config.cache_dir, config.embedding_model)

    classifier = Classifier()
    escalation = EscalationLogic(min_confidence=config.min_confidence)
    rewriter = QueryRewriter(
        provider=config.llm_provider,
        model=config.query_rewrite_model,
        logger=logger,
        enabled=config.enable_query_rewrite,
    )
    retriever = Retriever(
        index=index,
        top_k=config.top_k,
        rrf_k=config.rrf_k,
        rewriter=rewriter,
        strict_company_routing=config.strict_company_routing,
    )
    responder = Responder(provider=config.llm_provider, model=config.llm_model, logger=logger)
    pipeline = Pipeline(
        escalation=escalation,
        classifier=classifier,
        retriever=retriever,
        responder=responder,
    )

    if not config.tickets_csv.exists():
        raise FileNotFoundError(f"Ticket CSV not found: {config.tickets_csv}")

    frame = pd.read_csv(config.tickets_csv)
    col_map = _validate_input_columns(frame)
    
    # Map input columns to lowercase Ticket fields
    tickets = []
    for _, row in frame.iterrows():
        tickets.append(Ticket(
            issue=str(row[col_map["issue"]]),
            subject=str(row[col_map["subject"]]),
            company=str(row[col_map["company"]])
        ))
    processed = await _process_all_tickets(pipeline, tickets, config.max_concurrency)
    outputs = [item[0] for item in processed]
    metas = [item[1] for item in processed]

    out_df = pd.DataFrame([output.model_dump() for output in outputs])
    out_df = out_df[OUTPUT_COLUMNS]
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(config.output_csv, index=False)

    _render_summary(console, out_df, frame, metas)
    console.print(f"\nOutput written to: [bold]{config.output_csv}[/bold]")


if __name__ == "__main__":
    asyncio.run(async_main())
