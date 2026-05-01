from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

ALLOWED_STATUS = {"replied", "escalated"}
ALLOWED_REQUEST_TYPE = {"product_issue", "feature_request", "bug", "invalid"}
REQUIRED_COLUMNS = ["status", "product_area", "response", "justification", "request_type"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SupportNet-X output quality checks.")
    parser.add_argument("--input-csv", required=True, help="Path to support_tickets input CSV.")
    parser.add_argument("--output-csv", required=True, help="Path to generated output CSV.")
    return parser.parse_args()


def schema_checks(output_df: pd.DataFrame) -> dict[str, int]:
    checks: dict[str, int] = {}
    checks["columns_valid"] = int(output_df.columns.tolist() == REQUIRED_COLUMNS)
    checks["status_invalid_rows"] = int((~output_df["status"].isin(ALLOWED_STATUS)).sum())
    checks["request_type_invalid_rows"] = int((~output_df["request_type"].isin(ALLOWED_REQUEST_TYPE)).sum())
    checks["empty_response_rows"] = int(output_df["response"].fillna("").str.strip().eq("").sum())
    checks["empty_justification_rows"] = int(output_df["justification"].fillna("").str.strip().eq("").sum())
    return checks


def faithfulness_checks(output_df: pd.DataFrame) -> dict[str, int]:
    citation_re = re.compile(r"\[[^\]]+\]")
    checks: dict[str, int] = {}
    replied = output_df[output_df["status"] == "replied"].copy()
    if replied.empty:
        checks["replied_rows"] = 0
        checks["replied_without_citation"] = 0
        checks["replied_without_grounding_signal"] = 0
        return checks
    checks["replied_rows"] = int(len(replied))
    has_citation = replied["response"].fillna("").str.contains(citation_re)
    has_grounding_signal = replied["justification"].fillna("").str.contains("retrieval_confidence=", regex=False)
    checks["replied_without_citation"] = int((~has_citation).sum())
    checks["replied_without_grounding_signal"] = int((~has_grounding_signal).sum())
    return checks


def safety_checks(output_df: pd.DataFrame) -> dict[str, int]:
    pii_re = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
    checks: dict[str, int] = {}
    checks["possible_card_leaks_in_response"] = int(output_df["response"].fillna("").str.contains(pii_re).sum())
    checks["escalated_rows"] = int((output_df["status"] == "escalated").sum())
    checks["escalations_missing_reason"] = int(
        (
            (output_df["status"] == "escalated")
            & output_df["justification"].fillna("").str.strip().eq("")
        ).sum()
    )
    return checks


def print_table(console: Console, title: str, checks: dict[str, int]) -> None:
    table = Table(title=title)
    table.add_column("Check", style="cyan")
    table.add_column("Value", style="green")
    for key, value in checks.items():
        table.add_row(key, str(value))
    console.print(table)


def main() -> None:
    args = parse_args()
    console = Console()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not output_path.exists():
        raise FileNotFoundError(f"Output CSV not found: {output_path}")

    input_df = pd.read_csv(input_path)
    output_df = pd.read_csv(output_path)

    if len(input_df) != len(output_df):
        console.print(f"[red]Row mismatch: input={len(input_df)} output={len(output_df)}[/red]")

    missing = [col for col in REQUIRED_COLUMNS if col not in output_df.columns]
    if missing:
        console.print(f"[red]Output missing required columns: {missing}[/red]")
        return

    schema = schema_checks(output_df)
    faithfulness = faithfulness_checks(output_df)
    safety = safety_checks(output_df)

    print_table(console, "Schema checks", schema)
    print_table(console, "Faithfulness checks", faithfulness)
    print_table(console, "Safety checks", safety)

    score = 100
    score -= schema["status_invalid_rows"] * 5
    score -= schema["request_type_invalid_rows"] * 5
    score -= schema["empty_response_rows"] * 3
    score -= schema["empty_justification_rows"] * 3
    score -= faithfulness["replied_without_citation"] * 2
    score -= faithfulness["replied_without_grounding_signal"] * 2
    score -= safety["possible_card_leaks_in_response"] * 10
    score = max(score, 0)
    console.print(f"[bold]Rubric-aligned local quality score: {score}/100[/bold]")


if __name__ == "__main__":
    main()
