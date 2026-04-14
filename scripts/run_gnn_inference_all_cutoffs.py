#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a per-cutoff GNN inference command over all frozen cutoffs."
    )
    p.add_argument(
        "--cutoffs-csv",
        default="data/usecase_cyberspace/gnn_llm_comparison/outputs/cutoff_months.csv",
        help="CSV with cutoff_date column.",
    )
    p.add_argument(
        "--cmd-template",
        required=True,
        help=(
            "Command template executed per cutoff. "
            "Supports placeholders: {cutoff_date}, {cutoff_month}, {idx}."
        ),
    )
    p.add_argument(
        "--workdir",
        default=".",
        help="Working directory used to execute commands.",
    )
    p.add_argument(
        "--log-dir",
        default="data/usecase_cyberspace/gnn_llm_comparison/gnn_cutoff_batch_logs",
        help="Directory for per-cutoff logs and summary.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    p.add_argument("--stop-on-error", action="store_true", help="Stop at first failure.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    cutoffs_path = (root / args.cutoffs_csv).resolve()
    workdir = (root / args.workdir).resolve()
    log_dir = (root / args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cutoffs_path)
    if "cutoff_date" not in df.columns:
        raise ValueError(f"{cutoffs_path} must contain column cutoff_date")

    rows = []
    for idx, cutoff_date in enumerate(df["cutoff_date"].astype(str).tolist(), start=1):
        cutoff_month = cutoff_date[:7]
        cmd = args.cmd_template.format(cutoff_date=cutoff_date, cutoff_month=cutoff_month, idx=idx)
        print(f"[{idx:02d}] {cmd}")
        rec = {
            "idx": idx,
            "cutoff_date": cutoff_date,
            "cutoff_month": cutoff_month,
            "command": cmd,
            "status": "DRY_RUN" if args.dry_run else "PENDING",
            "returncode": None,
            "duration_sec": None,
            "log_file": str(log_dir / f"{idx:02d}_{cutoff_month}.log"),
        }

        if args.dry_run:
            rows.append(rec)
            continue

        t0 = time.time()
        cp = subprocess.run(
            cmd,
            shell=True,
            cwd=str(workdir),
            capture_output=True,
            text=True,
        )
        dt = time.time() - t0
        rec["duration_sec"] = round(dt, 3)
        rec["returncode"] = int(cp.returncode)
        rec["status"] = "OK" if cp.returncode == 0 else "ERROR"

        log_path = Path(rec["log_file"])
        log_path.write_text(
            f"$ {cmd}\n\n[stdout]\n{cp.stdout}\n\n[stderr]\n{cp.stderr}\n",
            encoding="utf-8",
        )
        rows.append(rec)

        if cp.returncode != 0 and args.stop_on_error:
            break

    out_json = log_dir / "batch_summary.json"
    out_csv = log_dir / "batch_summary.csv"
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=True), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[done] summary: {out_json}")
    print(f"[done] summary csv: {out_csv}")


if __name__ == "__main__":
    main()


