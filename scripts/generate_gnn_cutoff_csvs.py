#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run per-cutoff GNN inference and collect ranking CSVs into model folders."
    )
    p.add_argument(
        "--cutoffs-csv",
        default="data/usecase_cyberspace/gnn_llm_comparison/outputs/cutoff_months.csv",
    )
    p.add_argument(
        "--cmd-template",
        required=True,
        help="Command run per cutoff. Supports {cutoff_date}, {cutoff_month}, {idx}.",
    )
    p.add_argument("--workdir", default=".")
    p.add_argument(
        "--capture-root",
        default="data/usecase_cyberspace",
        help="Root folder to search produced ranking CSVs after each run.",
    )
    p.add_argument(
        "--output-root",
        default="data/usecase_cyberspace/gnn_llm_comparison/gnn_cutoff_csvs",
        help="Output root with one folder per model.",
    )
    p.add_argument(
        "--models",
        default="GraphModel,NoGraphModel",
        help="Comma-separated model labels to collect.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    return p.parse_args()


def _pick_latest(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    cutoffs = pd.read_csv((root / args.cutoffs_csv).resolve())["cutoff_date"].astype(str).tolist()
    workdir = (root / args.workdir).resolve()
    capture_root = (root / args.capture_root).resolve()
    output_root = (root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    summary = []
    for idx, cutoff_date in enumerate(cutoffs, start=1):
        cutoff_month = cutoff_date[:7]
        cmd = args.cmd_template.format(cutoff_date=cutoff_date, cutoff_month=cutoff_month, idx=idx)
        print(f"[{idx:02d}] cutoff={cutoff_date}")
        print(f"      cmd={cmd}")

        rec = {
            "idx": idx,
            "cutoff_date": cutoff_date,
            "cutoff_month": cutoff_month,
            "command": cmd,
            "status": "DRY_RUN" if args.dry_run else "PENDING",
            "returncode": None,
            "duration_sec": None,
            "copied": {},
        }

        if not args.dry_run:
            t0 = time.time()
            cp = subprocess.run(cmd, shell=True, cwd=str(workdir), capture_output=True, text=True)
            rec["duration_sec"] = round(time.time() - t0, 3)
            rec["returncode"] = int(cp.returncode)
            rec["status"] = "OK" if cp.returncode == 0 else "ERROR"
            run_log = output_root / f"run_{idx:02d}_{cutoff_month}.log"
            run_log.write_text(
                f"$ {cmd}\n\n[stdout]\n{cp.stdout}\n\n[stderr]\n{cp.stderr}\n",
                encoding="utf-8",
            )
            if cp.returncode != 0:
                summary.append(rec)
                if args.stop_on_error:
                    break
                continue

        # Collect produced CSVs for requested models.
        for model in models:
            pattern = f"**/ranking_{model}_RECENT_{cutoff_month}.csv"
            candidates = list(capture_root.glob(pattern))
            pick = _pick_latest(candidates)
            if pick is None:
                rec["copied"][model] = ""
                continue
            dst_dir = output_root / model
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"predictions_{cutoff_month}.csv"
            if not args.dry_run:
                shutil.copy2(pick, dst)
            rec["copied"][model] = str(dst)

        summary.append(rec)

    (output_root / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    pd.DataFrame(summary).to_csv(output_root / "generation_summary.csv", index=False)
    print(f"[done] output_root={output_root}")


if __name__ == "__main__":
    main()


