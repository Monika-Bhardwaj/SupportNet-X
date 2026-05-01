from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SupportNet-X end-to-end: triage, evaluate, and demo report."
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root path.",
    )
    parser.add_argument(
        "--baseline",
        help="Optional baseline output CSV for comparison in demo report.",
    )
    parser.add_argument(
        "--skip-triage",
        action="store_true",
        help="Skip triage and use existing support_tickets/output.csv.",
    )
    return parser.parse_args()


def run_step(console: Console, label: str, command: list[str], cwd: Path) -> tuple[bool, str]:
    console.print(f"[cyan]Running:[/cyan] {label}")
    process = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
    if process.returncode == 0:
        console.print(f"[green]PASS[/green] {label}")
        return True, process.stdout.strip()
    console.print(f"[red]FAIL[/red] {label}")
    if process.stdout.strip():
        console.print(f"[yellow]stdout:[/yellow]\n{process.stdout.strip()}")
    if process.stderr.strip():
        console.print(f"[yellow]stderr:[/yellow]\n{process.stderr.strip()}")
    return False, process.stderr.strip() or process.stdout.strip()


def render_summary(console: Console, results: list[tuple[str, bool]]) -> None:
    table = Table(title="SupportNet-X Submission Checklist")
    table.add_column("Step", style="cyan")
    table.add_column("Status", style="green")
    for name, ok in results:
        table.add_row(name, "PASS" if ok else "FAIL")
    console.print(table)


def main() -> int:
    args = parse_args()
    console = Console()
    root = Path(args.project_root)

    code_dir = root / "code"
    input_csv = root / "support_tickets" / "support_tickets.csv"
    output_csv = root / "support_tickets" / "output.csv"

    results: list[tuple[str, bool]] = []
    all_ok = True

    if not args.skip_triage:
        ok, _ = run_step(console, "Run triage pipeline", [sys.executable, "main.py"], code_dir)
        results.append(("Run triage pipeline", ok))
        all_ok = all_ok and ok
    else:
        console.print("[yellow]Skipped triage pipeline.[/yellow]")

    ok, _ = run_step(
        console,
        "Run evaluation harness",
        [
            sys.executable,
            "scripts/evaluate.py",
            "--input-csv",
            str(input_csv),
            "--output-csv",
            str(output_csv),
        ],
        root,
    )
    results.append(("Run evaluation harness", ok))
    all_ok = all_ok and ok

    demo_cmd = [sys.executable, "scripts/demo_report.py", "--current", str(output_csv)]
    if args.baseline:
        demo_cmd.extend(["--baseline", args.baseline])
    ok, _ = run_step(console, "Generate demo report", demo_cmd, root)
    results.append(("Generate demo report", ok))
    all_ok = all_ok and ok

    render_summary(console, results)
    if all_ok:
        console.print("[bold green]Submission-ready checks passed.[/bold green]")
        return 0
    console.print("[bold red]One or more checks failed. Review logs above.[/bold red]")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
