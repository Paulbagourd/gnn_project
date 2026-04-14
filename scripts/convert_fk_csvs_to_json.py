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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert all fk CSV predictions (f12/f24/f36/f48 trees) to evaluator-ready JSON, "
            "preserving folder structure under *_json folders."
        )
    )
    p.add_argument(
        "--input-root",
        default="data/usecase_cyberspace/gnn_llm_comparison/gnn_cutoff_csvs_from_ckpt",
        help="Root containing f12/f24/f36/f48 prediction CSV folders.",
    )
    p.add_argument(
        "--output-root",
        default="data/usecase_cyberspace/gnn_llm_comparison/gnn_cutoff_csvs_from_ckpt_json",
        help="Output root for mirrored JSON tree.",
    )
    p.add_argument(
        "--inputs-by-cutoff-dir",
        default="data/usecase_cyberspace/gnn_llm_comparison/handoff_andrea/handoff_andrea_temporal_only/data/inputs_by_cutoff",
        help="Directory containing llm_input_cutoff_YYYY-MM.json (for E(t) conversion).",
    )
    p.add_argument(
        "--csv-glob",
        default="predictions_20??-??.csv",
        help="Pattern for prediction CSV files inside model folders.",
    )
    p.add_argument(
        "--mode",
        choices=["auto", "full", "single_score"],
        default="single_score",
        help="Conversion mode for CSV schema.",
    )
    p.add_argument("--keyword-col", default="auto")
    p.add_argument("--score-col", default="prediction")
    p.add_argument("--score-semantic", choices=["abs", "rel", "both"], default="abs")
    p.add_argument("--epsilon", type=float, default=1e-8)
    p.add_argument(
        "--feature-weights",
        default="0.3831,0.5189,0.0980",
        help="Comma-separated weights to compute E(t) from llm_input histories.",
    )
    p.add_argument(
        "--run-id-prefix",
        default="gnn_fk",
        help="Prefix used in JSON run_id.",
    )
    p.add_argument(
        "--forecast-dir-regex",
        default=r"^f\d+$",
        help="Regex to recognize forecast folders to convert.",
    )
    p.add_argument(
        "--horizon-aware",
        action="store_true",
        help=(
            "If set, only the horizon matching the source forecast folder (f12/f24/...) "
            "is filled with a predicted value; other horizons are written as null."
        ),
    )
    return p.parse_args()


def _pick_keyword_col(df: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"keyword column '{requested}' not found in {list(df.columns)}")
        return requested
    for c in ["keyword", "name", "Keyword", "node", "term"]:
        if c in df.columns:
            return c
    raise ValueError("Could not detect keyword column. Use --keyword-col.")


def _normalize_cutoff_date(raw: str) -> str:
    s = str(raw).strip()
    m = re.match(r"^(20\d{2})-(\d{2})(?:-(\d{2}))?$", s)
    if not m:
        raise ValueError(f"Invalid cutoff format: {raw!r}. Expected YYYY-MM or YYYY-MM-DD.")
    return f"{m.group(1)}-{m.group(2)}-01"


def _cutoff_from_filename(path: Path) -> str:
    m = re.search(r"(20\d{2})-(\d{2})", path.stem)
    if not m:
        raise ValueError(f"Could not parse cutoff from filename: {path.name}")
    return f"{m.group(1)}-{m.group(2)}-01"


def _parse_weights(raw: str, n_features: int) -> np.ndarray:
    vals = np.asarray([float(x.strip()) for x in raw.split(",") if x.strip()], dtype=np.float64)
    if vals.size != n_features:
        raise ValueError(f"--feature-weights expects {n_features} values, got {vals.size}.")
    s = float(vals.sum())
    if abs(s) < 1e-12:
        raise ValueError("Feature weights sum to zero.")
    return vals / s


def _load_cutoff_inputs(inputs_dir: Path, cutoff_date: str) -> dict[str, Any]:
    month = cutoff_date[:7]
    p = inputs_dir / f"llm_input_cutoff_{month}.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing cutoff input file: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


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
        out[kw] = float(np.dot(vals[-1], w))
    return out


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


def _convert_full(df: pd.DataFrame, kw_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        kw = str(r[kw_col]).strip()
        if not kw:
            continue
        rec = {"keyword": kw}
        for h in HORIZONS:
            rec[f"abs_{h}"] = float(r[f"abs_{h}"])
            rec[f"rel_{h}"] = float(r[f"rel_{h}"])
        rows.append(rec)
    return rows


def _convert_single_score(
    df: pd.DataFrame,
    kw_col: str,
    score_col: str,
    score_semantic: str,
    e_t_map: dict[str, float],
    epsilon: float,
    active_horizon: int | None = None,
) -> list[dict[str, Any]]:
    if score_col not in df.columns:
        raise ValueError(f"Score column '{score_col}' not found.")
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        kw = str(r[kw_col]).strip()
        if not kw:
            continue
        score = float(r[score_col])
        e_t = float(e_t_map.get(kw, 0.0))
        rec = {"keyword": kw}
        for h in HORIZONS:
            if active_horizon is not None and h != int(active_horizon):
                rec[f"abs_{h}"] = None
                rec[f"rel_{h}"] = None
                continue
            if score_semantic == "abs":
                abs_v = score
                rel_v = abs_v / (e_t + epsilon)
            elif score_semantic == "rel":
                rel_v = score
                abs_v = rel_v * (e_t + epsilon)
            else:
                abs_v = score
                rel_v = score
            rec[f"abs_{h}"] = float(abs_v)
            rec[f"rel_{h}"] = float(rel_v)
        rows.append(rec)
    return rows


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    input_root = (root / args.input_root).resolve()
    output_root = (root / args.output_root).resolve()
    inputs_dir = (root / args.inputs_by_cutoff_dir).resolve()

    if not input_root.exists():
        raise FileNotFoundError(input_root)
    if not inputs_dir.exists():
        raise FileNotFoundError(inputs_dir)

    f_re = re.compile(args.forecast_dir_regex)
    converted = 0
    skipped = 0
    errors: list[str] = []

    forecast_dirs = [p for p in sorted(input_root.iterdir()) if p.is_dir() and f_re.match(p.name)]
    if not forecast_dirs:
        raise ValueError(f"No forecast directories matching '{args.forecast_dir_regex}' found in {input_root}")

    for fdir in forecast_dirs:
        out_fdir = output_root / f"{fdir.name}_json"
        try:
            forecast_h = int(fdir.name[1:])
        except Exception:  # noqa: BLE001
            forecast_h = None
        csv_files = sorted(fdir.rglob(args.csv_glob))
        if not csv_files:
            continue
        for csv_path in csv_files:
            rel = csv_path.relative_to(fdir)
            model_parts = rel.parts[:-1]
            model_id = "__".join(model_parts) if model_parts else "model"
            cutoff_date = _normalize_cutoff_date(_cutoff_from_filename(csv_path))
            run_id = f"{args.run_id_prefix}_{fdir.name}_{model_id}_{cutoff_date[:7]}"

            try:
                df = pd.read_csv(csv_path)
                kw_col = _pick_keyword_col(df, args.keyword_col)
                mode = args.mode if args.mode != "auto" else _detect_mode(df, args.score_col)
                if mode == "full":
                    preds = _convert_full(df, kw_col)
                else:
                    cutoff_payload = _load_cutoff_inputs(inputs_dir, cutoff_date)
                    e_t_map = _build_e_t_map(cutoff_payload, args.feature_weights)
                    active_h = forecast_h if args.horizon_aware else None
                    preds = _convert_single_score(
                        df=df,
                        kw_col=kw_col,
                        score_col=args.score_col,
                        score_semantic=args.score_semantic,
                        e_t_map=e_t_map,
                        epsilon=float(args.epsilon),
                        active_horizon=active_h,
                    )

                out_dir = out_fdir.joinpath(*model_parts, "predictions_by_cutoff")
                out_dir.mkdir(parents=True, exist_ok=True)
                out_json = out_dir / f"predictions_{cutoff_date[:7]}.json"
                payload = {
                    "run_id": run_id,
                    "model_id": model_id,
                    "cutoff_date": cutoff_date,
                    "predictions": preds,
                }
                out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
                converted += 1
                print(f"[ok] {csv_path.relative_to(input_root)} -> {out_json.relative_to(output_root)}")
            except Exception as exc:
                skipped += 1
                msg = f"[skip] {csv_path}: {exc}"
                errors.append(msg)
                print(msg)

    meta = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "inputs_by_cutoff_dir": str(inputs_dir),
        "converted_files": converted,
        "skipped_files": skipped,
        "mode": args.mode,
        "score_col": args.score_col,
        "score_semantic": args.score_semantic,
        "epsilon": float(args.epsilon),
        "horizon_aware": bool(args.horizon_aware),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "conversion_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    if errors:
        (output_root / "conversion_errors.log").write_text("\n".join(errors), encoding="utf-8")
    print(
        f"[done] converted={converted} skipped={skipped} "
        f"meta={output_root / 'conversion_meta.json'}"
    )


if __name__ == "__main__":
    main()

