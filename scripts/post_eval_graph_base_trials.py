from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


GRAPH_RE = re.compile(r"Graph Model Test MAE:\s*([0-9.]+),\s*RMSE:\s*([0-9.]+)")


def _as_bool_str(v: str) -> str:
    s = str(v).strip().lower()
    return "true" if s in {"1", "true", "yes"} else "false"


def _run_trial(
    repo_root: Path,
    trial_row: Dict[str, str],
    run_id_prefix: str,
    top_keywords: Optional[int],
) -> tuple[float, float, str]:
    trial = int(trial_row["trial"])
    run_id = f"{run_id_prefix}{trial:02d}"
    run_dir = repo_root / "data" / "usecase_cyberspace" / "05_train_gnn" / "outputs" / "runs" / run_id
    log_path = run_dir / "cell_outputs.txt"

    if log_path.exists():
        txt = log_path.read_text(encoding="utf-8", errors="ignore")
        m = GRAPH_RE.search(txt)
        if m:
            return float(m.group(1)), float(m.group(2)), run_id

    overrides = [
        f"params.graph_train.run_id={run_id}",
        "params.graph_train.sweep.enabled=false",
        "params.graph_train.diagnostics.enabled=false",
        "params.graph_train.plot.show_figs=false",
        "params.graph_train.plot.topk_recent=false",
        "params.graph_train.plot.topk_wordcloud=0",
        "params.graph_train.plot.topk_heatmap=0",
        "params.graph_train.cfg_defaults.FORECAST=24",
        f"params.graph_train.cfg_defaults.TEMP_WINDOW={int(float(trial_row['TEMP_WINDOW']))}",
        f"params.graph_train.cfg_defaults.HIDDEN_CHANNELS={int(float(trial_row['HIDDEN_CHANNELS']))}",
        f"params.graph_train.cfg_defaults.NUM_HEADS={int(float(trial_row['NUM_HEADS']))}",
        f"params.graph_train.cfg_defaults.DROPOUT={float(trial_row['DROPOUT'])}",
        f"params.graph_train.cfg_defaults.LR={float(trial_row['LR'])}",
        f"params.graph_train.cfg_defaults.WD={float(trial_row['WD'])}",
        "params.graph_train.cfg_defaults.NODE_ONLY=false",
        "params.graph_train.cfg_defaults.GRAPH_MIX=false",
        "params.graph_train.cfg_defaults.GRAPH_MULTI=false",
        "params.graph_train.cfg_defaults.GRAPH_MIX_LAMBDA=0.0",
        f"params.graph_train.cfg_defaults.EDGE_NORM={trial_row['EDGE_NORM']}",
        f"params.graph_train.cfg_defaults.EDGE_SELF_LOOPS={_as_bool_str(trial_row['EDGE_SELF_LOOPS'])}",
        f"params.graph_train.cfg_defaults.GRAPH_GATE={_as_bool_str(trial_row['GRAPH_GATE'])}",
    ]
    if top_keywords is None:
        overrides.append("params.graph_train.preprocess.top_keywords=null")
    else:
        overrides.append(f"params.graph_train.preprocess.top_keywords={top_keywords}")

    args = [
        sys.executable,
        "-m",
        "src.pipeline_runner",
        "--usecase",
        "usecase_cyberspace",
        "--from-step",
        "05_train_gnn",
        "--up-to",
        "05_train_gnn",
        "--force",
        "--override",
    ]
    args.extend(overrides)
    print(f"[post-eval] running trial={trial} run_id={run_id}")
    subprocess.run(args, cwd=repo_root, check=True)

    if not log_path.exists():
        raise RuntimeError(f"Missing log file after run: {log_path}")
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    m = GRAPH_RE.search(txt)
    if not m:
        raise RuntimeError(f"Could not parse graph test metrics in {log_path}")
    return float(m.group(1)), float(m.group(2)), run_id


def _to_float_or_none(v: str) -> Optional[float]:
    s = (v or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _parse_top_keywords(v: str) -> Optional[int]:
    s = (v or "").strip().lower()
    if not s or s in {"null", "none", "all", "-1"}:
        return None
    return int(float(s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/usecase_cyberspace/05_train_gnn/outputs/runs/exp_graphbase_optuna_top500_20260304/sweep_results_graph_base.csv"),
    )
    ap.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/usecase_cyberspace/05_train_gnn/outputs/runs/exp_graphbase_optuna_top500_20260304/post_eval_graph_base_trials.csv"),
    )
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument(
        "--top-keywords",
        type=str,
        default="500",
        help="Integer cap, or one of: null, none, all, -1 for no cap.",
    )
    ap.add_argument("--run-id-prefix", type=str, default="exp_posteval_optuna_trial")
    ap.add_argument(
        "--trials",
        type=str,
        default="",
        help="Optional comma-separated trial ids to run (e.g. 0,1,2). Default: all complete trials.",
    )
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    top_keywords = _parse_top_keywords(args.top_keywords)
    rows: List[Dict[str, str]] = []
    with args.input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    requested: Optional[set[int]] = None
    if args.trials.strip():
        requested = {int(x.strip()) for x in args.trials.split(",") if x.strip()}

    out_rows: List[Dict[str, str]] = []
    for row in rows:
        trial = int(float(row["trial"]))
        val_mean = _to_float_or_none(row.get("val_mean", ""))
        status = "complete" if val_mean is not None else "pruned"
        rec: Dict[str, str] = {
            "trial": str(trial),
            "status": status,
            "objective": row.get("objective", ""),
            "val_mean": row.get("val_mean", ""),
            "HIDDEN_CHANNELS": row.get("HIDDEN_CHANNELS", ""),
            "NUM_HEADS": row.get("NUM_HEADS", ""),
            "TEMP_WINDOW": row.get("TEMP_WINDOW", ""),
            "FORECAST": row.get("FORECAST", ""),
            "DROPOUT": row.get("DROPOUT", ""),
            "LR": row.get("LR", ""),
            "WD": row.get("WD", ""),
            "EDGE_NORM": row.get("EDGE_NORM", ""),
            "EDGE_SELF_LOOPS": row.get("EDGE_SELF_LOOPS", ""),
            "GRAPH_GATE": row.get("GRAPH_GATE", ""),
            "test_mae": "",
            "test_rmse": "",
            "run_id": "",
        }
        if status == "complete" and (requested is None or trial in requested):
            mae, rmse, run_id = _run_trial(repo_root, row, args.run_id_prefix, top_keywords)
            rec["test_mae"] = f"{mae:.6f}"
            rec["test_rmse"] = f"{rmse:.6f}"
            rec["run_id"] = run_id
        out_rows.append(rec)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial",
        "status",
        "objective",
        "val_mean",
        "test_mae",
        "test_rmse",
        "HIDDEN_CHANNELS",
        "NUM_HEADS",
        "TEMP_WINDOW",
        "FORECAST",
        "DROPOUT",
        "LR",
        "WD",
        "EDGE_NORM",
        "EDGE_SELF_LOOPS",
        "GRAPH_GATE",
        "run_id",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"[post-eval] wrote: {args.output_csv}")


if __name__ == "__main__":
    main()
