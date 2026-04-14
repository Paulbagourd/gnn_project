#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


DEFAULT_CUTOFFS_CSV = "data/usecase_cyberspace/gnn_llm_comparison/outputs/cutoff_months.csv"
DEFAULT_GRAPH_DATA_DIR = "data/usecase_cyberspace/04_build_graph/outputs"


def _parse_int_list(raw: str) -> list[int]:
    vals: list[int] = []
    for part in str(raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        vals.append(int(token))
    if not vals:
        raise ValueError("No forecast horizon provided.")
    return vals


def _parse_str_list(raw: str) -> list[str]:
    vals: list[str] = []
    for part in str(raw or "").split(","):
        token = part.strip()
        if token:
            vals.append(token)
    if not vals:
        raise ValueError("No value provided.")
    return vals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train reusable GNN checkpoints per horizon, then run inference for all frozen cutoffs "
            "without retraining."
        )
    )
    p.add_argument("--phase", choices=["train", "infer"], required=True)
    p.add_argument("--usecase", default="usecase_cyberspace")
    p.add_argument("--forecasts", default="12,24,36,48", help="Comma-separated horizons in months.")
    p.add_argument(
        "--target-modes",
        default="residual",
        help="Comma-separated TARGET_MODE values (e.g. drift_residual,smooth_relative).",
    )
    p.add_argument("--cutoffs-csv", default=DEFAULT_CUTOFFS_CSV)
    p.add_argument(
        "--checkpoint-root",
        default="data/usecase_cyberspace/gnn_llm_comparison/gnn_checkpoints",
        help="Root directory; checkpoints stored under fXX/.",
    )
    p.add_argument(
        "--output-root",
        default="data/usecase_cyberspace/gnn_llm_comparison/gnn_cutoff_csvs_from_ckpt",
        help="Used in infer phase to collect predictions by model/horizon/cutoff.",
    )
    p.add_argument(
        "--graph-data-dir",
        default=DEFAULT_GRAPH_DATA_DIR,
        help="Directory containing precomputed graph tensors (1_raw_data,2_active_data,3_corrected_data).",
    )
    p.add_argument("--models", default="GraphModel,NoGraphModel")
    p.add_argument(
        "--include-graph-mix",
        action="store_true",
        help="Run an additional graph_mix profile (separate checkpoint + inference outputs).",
    )
    p.add_argument(
        "--graph-mix-lambda",
        type=float,
        default=0.10,
        help="GRAPH_MIX_LAMBDA value used when --include-graph-mix is enabled.",
    )
    p.add_argument(
        "--train-start-ym",
        default="2005-01",
        help="Training time filter start month (YYYY-MM). Empty to disable.",
    )
    p.add_argument(
        "--train-end-ym",
        default="",
        help="Training time filter end month (YYYY-MM). Empty keeps latest month.",
    )
    p.add_argument(
        "--top-keywords",
        default=529,
        type=int,
        help="Cap active nodes to top-k keywords when reusing graph tensors. Use <=0 to disable cap.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    return p.parse_args()


def _run_cmd(cmd: list[str], cwd: Path, dry_run: bool) -> tuple[int, float, str, str]:
    if dry_run:
        print("$ " + " ".join(cmd))
        return 0, 0.0, "", ""
    t0 = time.time()
    cp = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return cp.returncode, time.time() - t0, cp.stdout, cp.stderr


def _build_common_cmd(usecase: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.pipeline_runner",
        "--usecase",
        usecase,
        "--from-step",
        "06_predict",
        "--up-to",
        "06_predict",
        "--force",
        "--override",
    ]


def _load_cutoffs(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "cutoff_date" not in df.columns:
        raise ValueError(f"{path} must contain cutoff_date column")
    return df["cutoff_date"].astype(str).tolist()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    forecasts = _parse_int_list(args.forecasts)
    target_modes = _parse_str_list(args.target_modes)
    cutoffs_path = (root / args.cutoffs_csv).resolve()
    checkpoint_root = (root / args.checkpoint_root).resolve()
    output_root = (root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    profiles = ["base"]
    if args.include_graph_mix:
        profiles.append("graph_mix")

    summary_rows: list[dict[str, object]] = []

    if args.phase == "train":
        for f in forecasts:
            for target_mode in target_modes:
                for profile in profiles:
                    ckpt_dir = checkpoint_root / f"f{f}" / target_mode / profile
                    profile_overrides = [f"params.predict.cfg_overrides.TARGET_MODE={target_mode}"]
                    if profile == "graph_mix":
                        profile_overrides.extend(
                            [
                                "params.predict.cfg_overrides.GRAPH_MIX=true",
                                "params.predict.cfg_overrides.GRAPH_GATE=false",
                                f"params.predict.cfg_overrides.GRAPH_MIX_LAMBDA={args.graph_mix_lambda}",
                            ]
                        )
                    overrides = [
                        "params.predict.reuse_graph_outputs=false",
                        "params.predict.reuse_existing_graph=true",
                        f"params.predict.graph_data_dir={args.graph_data_dir}",
                        f"params.predict.train_time_filter.start_year_month={args.train_start_ym}" if args.train_start_ym else "",
                        f"params.predict.train_time_filter.end_year_month={args.train_end_ym}" if args.train_end_ym else "",
                        f"params.predict.preprocess.top_keywords={args.top_keywords}" if args.top_keywords and args.top_keywords > 0 else "",
                        f"params.predict.cfg_overrides.FORECAST={f}",
                        "params.predict.cfg_overrides.LOAD_CHECKPOINT=false",
                        "params.predict.cfg_overrides.SAVE_CHECKPOINT=true",
                        f"params.predict.cfg_overrides.CHECKPOINT_DIR={ckpt_dir}",
                        "params.predict.cfg_overrides.TOPK_RECENT=false",
                        "params.predict.cfg_overrides.TOPK_RECENT_INFERENCE=false",
                        "params.predict.cfg_overrides.SHOW_FIGS=false",
                    ] + profile_overrides
                    overrides = [x for x in overrides if x]
                    cmd = _build_common_cmd(args.usecase) + overrides
                    print(f"[train {target_mode} {profile} f{f}]")
                    rc, dt, out, err = _run_cmd(cmd, root, args.dry_run)
                    log_file = output_root / f"train_{target_mode}_{profile}_f{f}.log"
                    if not args.dry_run:
                        log_file.write_text(f"$ {' '.join(cmd)}\n\n[stdout]\n{out}\n\n[stderr]\n{err}\n", encoding="utf-8")
                    rec = {
                        "phase": "train",
                        "target_mode": target_mode,
                        "profile": profile,
                        "forecast": f,
                        "cutoff_date": "",
                        "status": "DRY_RUN" if args.dry_run else ("OK" if rc == 0 else "ERROR"),
                        "returncode": rc,
                        "duration_sec": round(dt, 3),
                        "checkpoint_dir": str(ckpt_dir),
                        "copied": "",
                        "log_file": str(log_file),
                    }
                    summary_rows.append(rec)
                    if rc != 0 and args.stop_on_error and not args.dry_run:
                        break

    if args.phase == "infer":
        cutoffs = _load_cutoffs(cutoffs_path)
        plot_dir = (root / f"data/{args.usecase}/06_predict/outputs/plots").resolve()
        for f in forecasts:
            for target_mode in target_modes:
                for profile in profiles:
                    ckpt_dir = checkpoint_root / f"f{f}" / target_mode / profile
                    profile_overrides = [f"params.predict.cfg_overrides.TARGET_MODE={target_mode}"]
                    profile_models = list(models)
                    if profile == "graph_mix":
                        profile_overrides.extend(
                            [
                                "params.predict.cfg_overrides.GRAPH_MIX=true",
                                "params.predict.cfg_overrides.GRAPH_GATE=false",
                                f"params.predict.cfg_overrides.GRAPH_MIX_LAMBDA={args.graph_mix_lambda}",
                            ]
                        )
                        # ranking file remains ranking_GraphModel_... ; we store it under GraphMixModel folder.
                        profile_models = ["GraphMixModel"]
                    for idx, cutoff_date in enumerate(cutoffs, start=1):
                        cutoff_month = cutoff_date[:7]
                        overrides = [
                            "params.predict.reuse_graph_outputs=false",
                            "params.predict.reuse_existing_graph=true",
                            f"params.predict.graph_data_dir={args.graph_data_dir}",
                            f"params.predict.train_time_filter.start_year_month={args.train_start_ym}" if args.train_start_ym else "",
                            f"params.predict.train_time_filter.end_year_month={args.train_end_ym}" if args.train_end_ym else "",
                            f"params.predict.preprocess.top_keywords={args.top_keywords}" if args.top_keywords and args.top_keywords > 0 else "",
                            f"params.predict.cfg_overrides.FORECAST={f}",
                            "params.predict.cfg_overrides.LOAD_CHECKPOINT=true",
                            "params.predict.cfg_overrides.SAVE_CHECKPOINT=false",
                            f"params.predict.cfg_overrides.CHECKPOINT_DIR={ckpt_dir}",
                            f"params.predict.cfg_overrides.RECENT_CUTOFF_DATE={cutoff_date}",
                            "params.predict.cfg_overrides.TOPK_RECENT=true",
                            "params.predict.cfg_overrides.TOPK_RECENT_INFERENCE=false",
                            "params.predict.cfg_overrides.SHOW_FIGS=false",
                        ] + profile_overrides
                        overrides = [x for x in overrides if x]
                        cmd = _build_common_cmd(args.usecase) + overrides
                        print(f"[infer {target_mode} {profile} f{f} #{idx:02d}] cutoff={cutoff_date}")
                        rc, dt, out, err = _run_cmd(cmd, root, args.dry_run)
                        log_file = output_root / f"infer_{target_mode}_{profile}_f{f}_{cutoff_month}.log"
                        if not args.dry_run:
                            log_file.write_text(f"$ {' '.join(cmd)}\n\n[stdout]\n{out}\n\n[stderr]\n{err}\n", encoding="utf-8")

                        copied: dict[str, str] = {}
                        if rc == 0 and not args.dry_run:
                            for model in profile_models:
                                src_model = model
                                if model == "GraphMixModel":
                                    src_model = "GraphModel"
                                src = plot_dir / f"ranking_{src_model}_RECENT_{cutoff_month}.csv"
                                if src.exists():
                                    dst_dir = output_root / f"f{f}" / target_mode / model
                                    dst_dir.mkdir(parents=True, exist_ok=True)
                                    dst = dst_dir / f"predictions_{cutoff_month}.csv"
                                    shutil.copy2(src, dst)
                                    copied[model] = str(dst)
                                else:
                                    copied[model] = ""

                        rec = {
                            "phase": "infer",
                            "target_mode": target_mode,
                            "profile": profile,
                            "forecast": f,
                            "cutoff_date": cutoff_date,
                            "status": "DRY_RUN" if args.dry_run else ("OK" if rc == 0 else "ERROR"),
                            "returncode": rc,
                            "duration_sec": round(dt, 3),
                            "checkpoint_dir": str(ckpt_dir),
                            "copied": copied,
                            "log_file": str(log_file),
                        }
                        summary_rows.append(rec)

                        if rc != 0 and args.stop_on_error and not args.dry_run:
                            break

    summary_json = output_root / f"checkpoint_{args.phase}_summary.json"
    summary_csv = output_root / f"checkpoint_{args.phase}_summary.csv"
    summary_json.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=True), encoding="utf-8")

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phase",
                "target_mode",
                "profile",
                "forecast",
                "cutoff_date",
                "status",
                "returncode",
                "duration_sec",
                "checkpoint_dir",
                "copied",
                "log_file",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"[done] summary json: {summary_json}")
    print(f"[done] summary csv: {summary_csv}")


if __name__ == "__main__":
    main()

