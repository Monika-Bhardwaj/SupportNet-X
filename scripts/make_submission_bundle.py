from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a submission zip with output.csv, logs, and REPORT.md."
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root path.",
    )
    parser.add_argument(
        "--bundle-name",
        default="supportnetx_submission.zip",
        help="Name of generated bundle zip.",
    )
    parser.add_argument(
        "--baseline",
        help="Optional baseline output CSV for demo comparison in REPORT.md.",
    )
    parser.add_argument(
        "--skip-run-all",
        action="store_true",
        help="Skip executing scripts/run_all.py before bundling.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open generated bundle location after creation.",
    )
    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def open_bundle_location(bundle_path: Path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(bundle_path.parent))  # type: ignore[attr-defined]
            return
        if sys.platform == "darwin":
            subprocess.run(["open", str(bundle_path.parent)], check=False)
            return
        subprocess.run(["xdg-open", str(bundle_path.parent)], check=False)
    except Exception:
        # Non-fatal convenience step.
        pass


def build_report(project_root: Path, baseline: str | None, run_all_output: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# SupportNet-X Submission Report",
        "",
        f"- Generated at: {now}",
        f"- Project root: `{project_root}`",
        f"- Baseline provided: `{'yes' if baseline else 'no'}`",
        "",
        "## Included Artifacts",
        "",
        "- `support_tickets/output.csv`",
        "- `logs/log.txt` (if present)",
        "- `run_all_output.txt`",
        "- `REPORT.md`",
        "",
        "## Run Summary",
        "",
        "```text",
        run_all_output.strip() or "run_all output unavailable",
        "```",
        "",
        "## Submission Checklist",
        "",
        "- [x] Output CSV generated",
        "- [x] Evaluation and demo scripts executed",
        "- [x] Logs attached when available",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_csv = project_root / "support_tickets" / "output.csv"
    log_file = Path.home() / "hackerrank_orchestrate" / "log.txt"
    bundle_path = project_root / args.bundle_name

    run_all_output = ""
    if not args.skip_run_all:
        cmd = [sys.executable, "scripts/run_all.py"]
        if args.baseline:
            cmd.extend(["--baseline", args.baseline])
        rc, out, err = run_command(cmd, project_root)
        run_all_output = (out or "") + ("\n" + err if err else "")
        if rc != 0:
            print("run_all.py failed; aborting bundle creation.")
            print(run_all_output.strip())
            return rc

    if not output_csv.exists():
        print(f"Missing output CSV: {output_csv}")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "submission_bundle"
        staging.mkdir(parents=True, exist_ok=True)

        (staging / "support_tickets").mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_csv, staging / "support_tickets" / "output.csv")

        (staging / "logs").mkdir(parents=True, exist_ok=True)
        if log_file.exists():
            shutil.copy2(log_file, staging / "logs" / "log.txt")

        run_all_path = staging / "run_all_output.txt"
        run_all_path.write_text(run_all_output.strip() or "run_all step was skipped.", encoding="utf-8")

        report = build_report(project_root, args.baseline, run_all_output)
        (staging / "REPORT.md").write_text(report, encoding="utf-8")

        if bundle_path.exists():
            bundle_path.unlink()
        with ZipFile(bundle_path, "w", compression=ZIP_DEFLATED) as zf:
            for file_path in staging.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(staging))

    print(f"Created submission bundle: {bundle_path}")
    if args.open:
        open_bundle_location(bundle_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
