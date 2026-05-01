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
from supportnetx.models import Ticket
from supportnetx.pipeline import Pipeline
from supportnetx.responder import Responder
from supportnetx.retriever import Retriever

REQUIRED_COLUMNS = ["issue", "subject", "company"]
OUTPUT_COLUMNS = ["status", "product_area", "response", "justification", "request_type"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SupportNet-X triage CLI")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    return parser.parse_args()


def _validate_input_columns(frame: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")


async def _process_all_tickets(pipeline: Pipeline, tickets: list[Ticket], concurrency: int):
    semaphore = asyncio.Semaphore(concurrency)

    async def _worker(ticket: Ticket):
        async with semaphore:
            return await asyncio.to_thread(pipeline.process_ticket, ticket)

    return await asyncio.gather(*[_worker(t) for t in tickets])


def _render_summary(console: Console, frame: pd.DataFrame, tickets_df: pd.DataFrame) -> None:
    total = len(frame)
    escalated = int((frame["status"] == "escalated").sum())
    replied = total - escalated

    table = Table(title="SupportNet-X Run Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total tickets", str(total))
    table.add_row("Replied", str(replied))
    table.add_row("Escalated", str(escalated))
    table.add_row("Escalation rate", f"{(escalated / max(total, 1)) * 100:.1f}%")
    console.print(table)

    company_table = Table(title="Per-company status breakdown")
    company_table.add_column("Company", style="magenta")
    company_table.add_column("Replied", style="green")
    company_table.add_column("Escalated", style="yellow")
    combined = tickets_df[["company"]].copy()
    combined["status"] = frame["status"]
    grouped = combined.groupby(["company", "status"]).size().unstack(fill_value=0)
    for company, row in grouped.iterrows():
        company_table.add_row(
            str(company),
            str(int(row.get("replied", 0))),
            str(int(row.get("escalated", 0))),
        )
    console.print(company_table)


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
    retriever = Retriever(index=index, top_k=config.top_k, rrf_k=config.rrf_k)
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
    _validate_input_columns(frame)
    tickets = [Ticket(**row) for row in frame[REQUIRED_COLUMNS].to_dict(orient="records")]
    outputs = await _process_all_tickets(pipeline, tickets, config.max_concurrency)

    out_df = pd.DataFrame([output.model_dump() for output in outputs])
    out_df = out_df[OUTPUT_COLUMNS]
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(config.output_csv, index=False)

    _render_summary(console, out_df, frame)
    console.print(f"\nOutput written to: [bold]{config.output_csv}[/bold]")


if __name__ == "__main__":
    asyncio.run(async_main())
