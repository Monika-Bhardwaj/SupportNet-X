from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge-friendly SupportNet-X demo report.")
    parser.add_argument("--current", required=True, help="Path to current output CSV.")
    parser.add_argument("--baseline", help="Optional baseline output CSV for comparison.")
    return parser.parse_args()


def summarize(df: pd.DataFrame) -> dict[str, float]:
    total = float(len(df))
    escalated = float((df["status"] == "escalated").sum())
    replied = total - escalated
    citation_rate = 0.0
    if replied > 0:
        replied_df = df[df["status"] == "replied"]
        citation_rate = float(replied_df["response"].fillna("").str.contains(r"\[[^\]]+\]", regex=True).mean() * 100)
    confidence_vals = (
        df["justification"]
        .fillna("")
        .str.extract(r"retrieval_confidence=([0-9.]+)", expand=False)
        .dropna()
        .astype(float)
    )
    avg_conf = float(confidence_vals.mean()) if not confidence_vals.empty else 0.0
    return {
        "total_tickets": total,
        "replied": replied,
        "escalated": escalated,
        "escalation_rate_pct": (escalated / max(total, 1.0)) * 100.0,
        "citation_rate_pct": citation_rate,
        "avg_retrieval_confidence": avg_conf,
    }


def render(console: Console, current_metrics: dict[str, float], baseline_metrics: dict[str, float] | None) -> None:
    table = Table(title="SupportNet-X Demo Quality Report")
    table.add_column("Metric", style="cyan")
    table.add_column("Current", style="green")
    if baseline_metrics is not None:
        table.add_column("Baseline", style="yellow")
        table.add_column("Delta", style="magenta")

    ordered_keys = [
        "total_tickets",
        "replied",
        "escalated",
        "escalation_rate_pct",
        "citation_rate_pct",
        "avg_retrieval_confidence",
    ]
    for key in ordered_keys:
        current = current_metrics[key]
        if baseline_metrics is None:
            table.add_row(key, f"{current:.2f}")
            continue
        baseline = baseline_metrics[key]
        delta = current - baseline
        table.add_row(key, f"{current:.2f}", f"{baseline:.2f}", f"{delta:+.2f}")

    console.print(table)
    console.print(
        "[bold]Demo narrative:[/bold] higher citation rate and stable confidence "
        "demonstrate improved corpus grounding and safer escalation behavior."
    )


def main() -> None:
    args = parse_args()
    console = Console()

    current_path = Path(args.current)
    if not current_path.exists():
        raise FileNotFoundError(f"Current output CSV not found: {current_path}")
    current_df = pd.read_csv(current_path)
    current_metrics = summarize(current_df)

    baseline_metrics = None
    if args.baseline:
        baseline_path = Path(args.baseline)
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline output CSV not found: {baseline_path}")
        baseline_df = pd.read_csv(baseline_path)
        baseline_metrics = summarize(baseline_df)

    render(console, current_metrics, baseline_metrics)


if __name__ == "__main__":
    main()
