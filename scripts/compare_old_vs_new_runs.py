#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


METRIC_PATTERNS = {
    "graph": re.compile(r"Graph Model Test MAE:\s*([0-9eE+\-.]+),\s*RMSE:\s*([0-9eE+\-.]+)"),
    "nograph": re.compile(r"No-Graph Model Test MAE:\s*([0-9eE+\-.]+),\s*RMSE:\s*([0-9eE+\-.]+)"),
    "persist": re.compile(r"Persistence baseline\s+MAE:\s*([0-9eE+\-.]+),\s*RMSE:\s*([0-9eE+\-.]+)"),
    "drift": re.compile(r"Drift baseline\s+MAE:\s*([0-9eE+\-.]+),\s*RMSE:\s*([0-9eE+\-.]+)"),
}


@dataclass
class Metrics:
    mae: float
    rmse: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare old GNN runs with new checkpoint workflow runs. "
            "Outputs metric deltas and optional prediction coherence stats."
        )
    )
    p.add_argument("--forecasts", default="12,24,36,48", help="Comma-separated horizons.")
    p.add_argument(
        "--old-runs-root",
        default="data/usecase_cyberspace/05_train_gnn/outputs/runs",
        help="Root folder containing old training run directories.",
    )
    p.add_argument(
        "--new-root",
        default="data/usecase_cyberspace/gnn_llm_comparison/gnn_cutoff_csvs_from_ckpt",
        help="Root folder created by run_gnn_checkpoint_workflow.py.",
    )
    p.add_argument(
        "--old-run-map-json",
        default="",
        help="Optional JSON mapping forecast->absolute/relative old run path, e.g. {'12':'...','24':'...'}.",
    )
    p.add_argument(
        "--prefer-token",
        default="",
        help="Optional token to prioritize when auto-picking old runs (e.g., drift_residual).",
    )
    p.add_argument(
        "--cutoff-month",
        default="2020-04",
        help="Cutoff month used for optional prediction coherence checks (YYYY-MM).",
    )
    p.add_argument(
        "--models",
        default="GraphModel,NoGraphModel",
        help="Comma-separated model names for prediction coherence checks.",
    )
    p.add_argument(
        "--output-dir",
        default="data/usecase_cyberspace/gnn_llm_comparison/gnn_cutoff_csvs_from_ckpt/comparison_old_vs_new",
        help="Output folder for comparison artifacts.",
    )
    return p.parse_args()


def _parse_forecasts(raw: str) -> list[int]:
    out: list[int] = []
    for part in str(raw).split(","):
        token = part.strip()
        if token:
            out.append(int(token))
    if not out:
        raise ValueError("No forecasts provided.")
    return out


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_metrics(text: str) -> dict[str, Metrics]:
    out: dict[str, Metrics] = {}
    for key, pattern in METRIC_PATTERNS.items():
        m = pattern.search(text)
        if not m:
            continue
        out[key] = Metrics(mae=float(m.group(1)), rmse=float(m.group(2)))
    return out


def _choose_old_run(old_root: Path, forecast: int, prefer_token: str) -> Path | None:
    cands = [p for p in old_root.iterdir() if p.is_dir() and f"_f{forecast}_" in p.name]
    if not cands:
        return None
    if prefer_token:
        token = prefer_token.lower()
        preferred = [p for p in cands if token in p.name.lower()]
        if preferred:
            cands = preferred
    return max(cands, key=lambda p: p.stat().st_mtime)


def _load_old_run_map(path: str) -> dict[str, str]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in dict(payload).items()}


def _to_float(val: Any) -> float | None:
    try:
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def _prediction_coherence(old_csv: Path, new_csv: Path) -> dict[str, Any]:
    old_df = pd.read_csv(old_csv)
    new_df = pd.read_csv(new_csv)
    if "name" not in old_df.columns or "prediction" not in old_df.columns:
        return {"ok": False, "reason": f"old file missing required columns: {old_csv}"}
    if "name" not in new_df.columns or "prediction" not in new_df.columns:
        return {"ok": False, "reason": f"new file missing required columns: {new_csv}"}

    merged = old_df[["name", "prediction"]].rename(columns={"prediction": "pred_old"}).merge(
        new_df[["name", "prediction"]].rename(columns={"prediction": "pred_new"}), on="name", how="inner"
    )
    if merged.empty:
        return {"ok": False, "reason": "no overlapping keywords"}

    diff = merged["pred_new"] - merged["pred_old"]
    rmse = float((diff.pow(2).mean()) ** 0.5)
    mae = float(diff.abs().mean())
    pearson = _to_float(merged["pred_old"].corr(merged["pred_new"], method="pearson"))
    spearman = _to_float(merged["pred_old"].corr(merged["pred_new"], method="spearman"))
    return {
        "ok": True,
        "n_overlap": int(len(merged)),
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson,
        "spearman": spearman,
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    forecasts = _parse_forecasts(args.forecasts)
    old_root = (root / args.old_runs_root).resolve()
    new_root = (root / args.new_root).resolve()
    out_dir = (root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    old_map = _load_old_run_map(args.old_run_map_json)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    metric_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []

    for f in forecasts:
        old_run = Path(old_map.get(str(f), "")).resolve() if str(f) in old_map else _choose_old_run(old_root, f, args.prefer_token)
        new_log = new_root / f"train_f{f}.log"

        if old_run is None:
            metric_rows.append(
                {
                    "forecast": f,
                    "status": "MISSING_OLD_RUN",
                    "old_run": "",
                    "new_log": str(new_log),
                }
            )
            continue

        old_log = old_run / "cell_outputs.txt"
        if not old_log.exists() or not new_log.exists():
            metric_rows.append(
                {
                    "forecast": f,
                    "status": "MISSING_LOG",
                    "old_run": str(old_run),
                    "old_log_exists": old_log.exists(),
                    "new_log_exists": new_log.exists(),
                }
            )
            continue

        old_metrics = _extract_metrics(_read_text(old_log))
        new_metrics = _extract_metrics(_read_text(new_log))

        for key in ["graph", "nograph", "persist", "drift"]:
            om = old_metrics.get(key)
            nm = new_metrics.get(key)
            row = {
                "forecast": f,
                "metric_block": key,
                "old_run": str(old_run),
                "old_mae": om.mae if om else None,
                "old_rmse": om.rmse if om else None,
                "new_mae": nm.mae if nm else None,
                "new_rmse": nm.rmse if nm else None,
                "delta_mae_new_minus_old": (nm.mae - om.mae) if (om and nm) else None,
                "delta_rmse_new_minus_old": (nm.rmse - om.rmse) if (om and nm) else None,
                "status": "OK" if (om and nm) else "PARTIAL",
            }
            metric_rows.append(row)

        # Optional prediction coherence at cutoff month
        for model in models:
            new_pred = new_root / f"f{f}" / model / f"predictions_{args.cutoff_month}.csv"
            old_pred_cut = old_run / "plots" / f"ranking_{model}_RECENT_{args.cutoff_month}.csv"
            old_pred_recent = old_run / "plots" / f"ranking_{model}_RECENT.csv"

            source = ""
            old_pred = None
            if old_pred_cut.exists():
                old_pred = old_pred_cut
                source = "cutoff_specific"
            elif old_pred_recent.exists():
                old_pred = old_pred_recent
                source = "recent_fallback"

            rec = {
                "forecast": f,
                "model": model,
                "cutoff_month": args.cutoff_month,
                "old_run": str(old_run),
                "old_pred": str(old_pred) if old_pred else "",
                "new_pred": str(new_pred),
                "old_source": source,
                "status": "MISSING_FILES",
                "n_overlap": None,
                "mae": None,
                "rmse": None,
                "pearson": None,
                "spearman": None,
                "note": "",
            }
            if old_pred and new_pred.exists():
                coh = _prediction_coherence(old_pred, new_pred)
                if coh.get("ok"):
                    rec.update(
                        {
                            "status": "OK",
                            "n_overlap": coh.get("n_overlap"),
                            "mae": coh.get("mae"),
                            "rmse": coh.get("rmse"),
                            "pearson": coh.get("pearson"),
                            "spearman": coh.get("spearman"),
                        }
                    )
                else:
                    rec.update({"status": "ERROR", "note": str(coh.get("reason", ""))})
            pred_rows.append(rec)

    metrics_df = pd.DataFrame(metric_rows)
    preds_df = pd.DataFrame(pred_rows)

    metrics_csv = out_dir / "metrics_old_vs_new.csv"
    preds_csv = out_dir / "prediction_coherence_old_vs_new.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    preds_df.to_csv(preds_csv, index=False)

    summary = {
        "forecasts": forecasts,
        "old_runs_root": str(old_root),
        "new_root": str(new_root),
        "cutoff_month": args.cutoff_month,
        "metrics_rows": int(len(metrics_df)),
        "prediction_rows": int(len(preds_df)),
        "metrics_csv": str(metrics_csv),
        "prediction_csv": str(preds_csv),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"[ok] metrics csv: {metrics_csv}")
    print(f"[ok] prediction coherence csv: {preds_csv}")
    print(f"[ok] summary json: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

