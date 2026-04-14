#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HORIZONS = [12, 24, 36, 48]

# -----------------------------------------------------------------------------
# User defaults (edit here to avoid retyping long command lines)
# -----------------------------------------------------------------------------
USER_DEFAULTS: dict[str, Any] = {
    "output_run_dir": "data/usecase_cyberspace/gnn_llm_comparison/converted_gnn_runs/run_demo_convert",  # Where JSON outputs are written
    "model_id": "gnn_graphmodel_recent_demo",  # Model identifier saved in predictions JSON
    "run_id": "gnn_demo_001",  # Run identifier saved in predictions JSON
    "input_csv": "data/usecase_cyberspace/05_train_gnn/outputs/runs/tmp_sti_2027_raw_driftresidual_20260313/plots/ranking_GraphModel_RECENT.csv",  # Single CSV source (use this OR input_dir)
    "input_dir": "",  # Directory source for multi-file conversion (leave empty when using input_csv)
    "glob": "*.csv",  # File pattern when input_dir is used
    "cutoff_date": "all",  # Cutoff date tied to input_csv (YYYY-MM-DD) or "all" for all available cutoffs
    "keyword_col": "name",  # Column containing keyword names in CSV
    "mode": "single_score",  # Conversion mode: auto | full | single_score
    "score_col": "prediction",  # Score column used in single_score mode
    "score_semantic": "abs",  # Meaning of score_col: abs | rel | both
    "inputs_by_cutoff_dir": "data/usecase_cyberspace/gnn_llm_comparison/outputs/inputs_by_cutoff",  # Needed to compute E(t) for abs<->rel conversion
    "epsilon": 1e-8,  # Stability constant used in relative conversions
    "feature_weights": "0.3831,0.5189,0.0980",  # Weights to compute E(t) from feature history
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert GNN CSV predictions to predictions_YYYY-MM.json format."
    )
    p.add_argument("--output-run-dir", default=USER_DEFAULTS["output_run_dir"], help="Output run directory root.")
    p.add_argument("--model-id", default=USER_DEFAULTS["model_id"], help="model_id value in output JSON.")
    p.add_argument("--run-id", default=USER_DEFAULTS["run_id"], help="run_id value in output JSON.")

    p.add_argument("--input-csv", default=USER_DEFAULTS["input_csv"], help="Single CSV file for one cutoff.")
    p.add_argument("--input-dir", default=USER_DEFAULTS["input_dir"], help="Directory with multiple CSVs.")
    p.add_argument("--glob", default=USER_DEFAULTS["glob"], help="Glob pattern when using --input-dir.")
    p.add_argument("--cutoff-date", default=USER_DEFAULTS["cutoff_date"], help="Cutoff date for --input-csv (YYYY-MM-DD).")

    p.add_argument("--keyword-col", default=USER_DEFAULTS["keyword_col"], help="Keyword column name (auto by default).")

    p.add_argument(
        "--mode",
        choices=["auto", "full", "single_score"],
        default=USER_DEFAULTS["mode"],
        help=(
            "full: CSV already has abs_12..abs_48 and rel_12..rel_48. "
            "single_score: CSV has one score column. "
            "auto: detect."
        ),
    )
    p.add_argument(
        "--score-col",
        default=USER_DEFAULTS["score_col"],
        help="Score column for single_score mode (e.g., prediction/emergence_score).",
    )
    p.add_argument(
        "--score-semantic",
        choices=["abs", "rel", "both"],
        default=USER_DEFAULTS["score_semantic"],
        help=(
            "How to interpret single score: abs => abs_h=score; rel inferred via E(t). "
            "rel => rel_h=score; abs inferred via E(t). both => both=score."
        ),
    )

    p.add_argument(
        "--inputs-by-cutoff-dir",
        default=USER_DEFAULTS["inputs_by_cutoff_dir"],
        help="Directory containing llm_input_cutoff_YYYY-MM.json (used to infer E(t)).",
    )
    p.add_argument("--epsilon", type=float, default=float(USER_DEFAULTS["epsilon"]))
    p.add_argument(
        "--feature-weights",
        default=USER_DEFAULTS["feature_weights"],
        help="Comma-separated feature weights to compute E(t); default uniform.",
    )
    return p.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _pick_keyword_col(df: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"keyword column '{requested}' not found in {list(df.columns)}")
        return requested
    for c in ["keyword", "name", "Keyword", "node", "term"]:
        if c in df.columns:
            return c
    raise ValueError("Could not detect keyword column. Use --keyword-col.")


def _infer_cutoff_from_name(path: Path) -> str:
    m = re.search(r"(20\d{2})[-_](\d{2})", path.stem)
    if not m:
        raise ValueError(f"Could not infer cutoff YYYY-MM from filename: {path.name}")
    return f"{m.group(1)}-{m.group(2)}-01"


def _normalize_cutoff_date(raw: str) -> str:
    s = str(raw).strip()
    if not s:
        raise ValueError("Empty cutoff value.")
    m = re.match(r"^(20\d{2})-(\d{2})(?:-(\d{2}))?$", s)
    if not m:
        raise ValueError(f"Invalid cutoff format: {raw!r}. Expected YYYY-MM or YYYY-MM-DD.")
    yyyy = m.group(1)
    mm = m.group(2)
    return f"{yyyy}-{mm}-01"


def _collect_available_cutoffs(inputs_dir: Path) -> list[str]:
    out = []
    for p in sorted(inputs_dir.glob("llm_input_cutoff_20??-??.json")):
        m = re.search(r"(20\d{2}-\d{2})", p.stem)
        if m:
            out.append(f"{m.group(1)}-01")
    return out


def _split_csv_by_cutoff(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    cutoff_col = None
    for c in ["cutoff_date", "cutoff_month", "cutoff", "date_cutoff"]:
        if c in df.columns:
            cutoff_col = c
            break
    if cutoff_col is None:
        raise ValueError(
            "cutoff_date='all' with --input-csv requires a cutoff column "
            "(one of: cutoff_date, cutoff_month, cutoff, date_cutoff)."
        )

    pairs: list[tuple[str, pd.DataFrame]] = []
    for raw, g in df.groupby(cutoff_col):
        cd = _normalize_cutoff_date(str(raw))
        pairs.append((cd, g.copy()))
    pairs.sort(key=lambda x: x[0])
    return pairs


def _load_cutoff_inputs(inputs_dir: Path, cutoff_date: str) -> dict[str, Any]:
    month = cutoff_date[:7]
    p = inputs_dir / f"llm_input_cutoff_{month}.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing cutoff input file: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _parse_weights(raw: str, n_features: int) -> np.ndarray:
    if raw.strip():
        vals = np.asarray([float(x.strip()) for x in raw.split(",") if x.strip()], dtype=np.float64)
        if vals.size != n_features:
            raise ValueError(f"--feature-weights expects {n_features} values, got {vals.size}.")
        s = float(vals.sum())
        if abs(s) < 1e-12:
            raise ValueError("Feature weights sum to zero.")
        return vals / s
    return np.ones(n_features, dtype=np.float64) / float(n_features)


def _build_e_t_map(cutoff_payload: dict[str, Any], feature_weights_arg: str) -> dict[str, float]:
    feature_names = cutoff_payload.get("feature_names", [])
    n_features = len(feature_names)
    if n_features <= 0:
        raise ValueError("Invalid cutoff payload: missing feature_names.")
    w = _parse_weights(feature_weights_arg, n_features)

    out: dict[str, float] = {}
    for rec in cutoff_payload.get("keywords", []):
        kw = str(rec.get("keyword", "")).strip()
        vals = np.asarray(rec.get("history", {}).get("values", []), dtype=np.float64)
        if not kw or vals.ndim != 2 or vals.shape[0] == 0 or vals.shape[1] != n_features:
            continue
        e_t = float(np.dot(vals[-1], w))
        out[kw] = e_t
    return out


def _convert_full(df: pd.DataFrame, kw_col: str) -> list[dict[str, Any]]:
    required = [f"abs_{h}" for h in HORIZONS] + [f"rel_{h}" for h in HORIZONS]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Full mode selected but missing columns: {miss}")
    rows = []
    for _, r in df.iterrows():
        kw = str(r[kw_col]).strip()
        if not kw:
            continue
        row = {"keyword": kw}
        for h in HORIZONS:
            row[f"abs_{h}"] = float(r[f"abs_{h}"])
            row[f"rel_{h}"] = float(r[f"rel_{h}"])
        rows.append(row)
    return rows


def _convert_single_score(
    df: pd.DataFrame,
    kw_col: str,
    score_col: str,
    score_semantic: str,
    e_t_map: dict[str, float],
    epsilon: float,
) -> list[dict[str, Any]]:
    if score_col not in df.columns:
        raise ValueError(f"Score column '{score_col}' not found.")
    rows = []
    for _, r in df.iterrows():
        kw = str(r[kw_col]).strip()
        if not kw:
            continue
        score = float(r[score_col])
        e_t = float(e_t_map.get(kw, 0.0))
        row = {"keyword": kw}
        for h in HORIZONS:
            if score_semantic == "abs":
                abs_v = score
                rel_v = abs_v / (e_t + epsilon)
            elif score_semantic == "rel":
                rel_v = score
                abs_v = rel_v * (e_t + epsilon)
            else:  # both
                abs_v = score
                rel_v = score
            row[f"abs_{h}"] = float(abs_v)
            row[f"rel_{h}"] = float(rel_v)
        rows.append(row)
    return rows


def _detect_mode(df: pd.DataFrame, score_col: str) -> str:
    required = {f"abs_{h}" for h in HORIZONS} | {f"rel_{h}" for h in HORIZONS}
    if required.issubset(set(df.columns)):
        return "full"
    if score_col in df.columns:
        return "single_score"
    raise ValueError(
        "Could not auto-detect mode. Need full columns or score column. "
        f"Columns seen: {list(df.columns)}"
    )


def _write_output(
    output_run_dir: Path,
    run_id: str,
    model_id: str,
    cutoff_date: str,
    predictions: list[dict[str, Any]],
) -> Path:
    pred_dir = output_run_dir / "predictions_by_cutoff"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "run_id": run_id,
        "model_id": model_id,
        "cutoff_date": cutoff_date,
        "predictions": predictions,
    }
    out_path = pred_dir / f"predictions_{cutoff_date[:7]}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output_run_dir = (root / args.output_run_dir).resolve()
    inputs_dir = (root / args.inputs_by_cutoff_dir).resolve()

    cutoff_arg = str(args.cutoff_date).strip().lower()

    if args.input_dir and args.input_csv:
        raise ValueError("Use either --input-csv or --input-dir, not both.")

    if args.input_dir:
        in_dir = (root / args.input_dir).resolve()
        paths = sorted(in_dir.glob(args.glob))
        if not paths:
            raise FileNotFoundError(f"No files for pattern {args.glob} in {in_dir}")
        csv_paths = [(p.resolve(), "") for p in paths]
    elif args.input_csv:
        csv_path = Path(root / args.input_csv).resolve()
        if cutoff_arg == "all":
            # Special path: one CSV with explicit cutoff column, split internally.
            csv_paths = [(csv_path, "all")]
        else:
            cutoff_date = _normalize_cutoff_date(args.cutoff_date)
            csv_paths = [(csv_path, cutoff_date)]
    else:
        raise ValueError("No input source provided. Set USER_DEFAULTS or pass --input-csv/--input-dir.")

    out_files = []
    for csv_path, cutoff in csv_paths:
        if cutoff == "all":
            df_all = _read_csv(csv_path)
            per_cut = _split_csv_by_cutoff(df_all)
            avail = set(_collect_available_cutoffs(inputs_dir))
            if avail:
                per_cut = [(cd, d) for cd, d in per_cut if cd in avail]
            if not per_cut:
                raise ValueError(
                    "No cutoff rows matched available frozen setup cutoffs in inputs_by_cutoff_dir."
                )
            for cutoff_date, df in per_cut:
                kw_col = _pick_keyword_col(df, args.keyword_col)
                mode = args.mode if args.mode != "auto" else _detect_mode(df, args.score_col)
                if mode == "full":
                    preds = _convert_full(df, kw_col)
                else:
                    cutoff_payload = _load_cutoff_inputs(inputs_dir, cutoff_date)
                    e_t_map = _build_e_t_map(cutoff_payload, args.feature_weights)
                    preds = _convert_single_score(
                        df=df,
                        kw_col=kw_col,
                        score_col=args.score_col,
                        score_semantic=args.score_semantic,
                        e_t_map=e_t_map,
                        epsilon=float(args.epsilon),
                    )
                out_path = _write_output(
                    output_run_dir=output_run_dir,
                    run_id=args.run_id,
                    model_id=args.model_id,
                    cutoff_date=cutoff_date,
                    predictions=preds,
                )
                out_files.append(out_path)
                print(f"[ok] {csv_path.name} [{cutoff_date}] -> {out_path}")
        else:
            cutoff_date = cutoff if cutoff else _infer_cutoff_from_name(csv_path)
            df = _read_csv(csv_path)
            kw_col = _pick_keyword_col(df, args.keyword_col)
            mode = args.mode if args.mode != "auto" else _detect_mode(df, args.score_col)

            if mode == "full":
                preds = _convert_full(df, kw_col)
            else:
                cutoff_payload = _load_cutoff_inputs(inputs_dir, cutoff_date)
                e_t_map = _build_e_t_map(cutoff_payload, args.feature_weights)
                preds = _convert_single_score(
                    df=df,
                    kw_col=kw_col,
                    score_col=args.score_col,
                    score_semantic=args.score_semantic,
                    e_t_map=e_t_map,
                    epsilon=float(args.epsilon),
                )

            out_path = _write_output(
                output_run_dir=output_run_dir,
                run_id=args.run_id,
                model_id=args.model_id,
                cutoff_date=cutoff_date,
                predictions=preds,
            )
            out_files.append(out_path)
            print(f"[ok] {csv_path.name} -> {out_path}")

    meta = {
        "run_id": args.run_id,
        "model_id": args.model_id,
        "mode": args.mode,
        "score_col": args.score_col,
        "score_semantic": args.score_semantic,
        "epsilon": float(args.epsilon),
        "n_files": len(out_files),
        "output_run_dir": str(output_run_dir),
    }
    output_run_dir.mkdir(parents=True, exist_ok=True)
    (output_run_dir / "conversion_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(f"[done] output_run_dir={output_run_dir}")


if __name__ == "__main__":
    main()

