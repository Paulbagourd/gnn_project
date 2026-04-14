#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HORIZONS = [12, 24, 36, 48]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Gather forecast-specific CSV predictions (f12/f24/f36/f48) into consolidated "
            "JSON runs by model, with genuine horizon values in abs_k/rel_k."
        )
    )
    p.add_argument(
        "--input-root",
        default="data/usecase_cyberspace/gnn_llm_comparison/gnn_cutoff_csvs_from_ckpt",
        help="Root containing f12/f24/f36/f48 CSV predictions.",
    )
    p.add_argument(
        "--output-root",
        default="data/usecase_cyberspace/gnn_llm_comparison/gnn_json_gathered_model",
        help="Output root for consolidated JSON runs by model.",
    )
    p.add_argument(
        "--inputs-by-cutoff-dir",
        default="data/usecase_cyberspace/gnn_llm_comparison/handoff_andrea/handoff_andrea_temporal_only/data/inputs_by_cutoff",
        help="Directory containing llm_input_cutoff_YYYY-MM.json (for E(t) conversion).",
    )
    p.add_argument("--score-col", default="prediction")
    p.add_argument("--keyword-col", default="name")
    p.add_argument("--epsilon", type=float, default=1e-8)
    p.add_argument(
        "--feature-weights",
        default="0.3831,0.5189,0.0980",
        help="Comma-separated feature weights to compute E(t).",
    )
    p.add_argument("--run-id-prefix", default="gnn_gathered")
    return p.parse_args()


def _parse_weights(raw: str, n_features: int) -> np.ndarray:
    vals = np.asarray([float(x.strip()) for x in raw.split(",") if x.strip()], dtype=np.float64)
    if vals.size != n_features:
        raise ValueError(f"--feature-weights expects {n_features} values, got {vals.size}.")
    s = float(vals.sum())
    if abs(s) < 1e-12:
        raise ValueError("Feature weights sum to zero.")
    return vals / s


def _load_cutoff_inputs(inputs_dir: Path, cutoff_month: str) -> dict[str, Any]:
    p = inputs_dir / f"llm_input_cutoff_{cutoff_month}.json"
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


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    input_root = (root / args.input_root).resolve()
    output_root = (root / args.output_root).resolve()
    inputs_dir = (root / args.inputs_by_cutoff_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    fdirs = sorted([p for p in input_root.iterdir() if p.is_dir() and re.match(r"^f\d+$", p.name)])
    if not fdirs:
        raise ValueError(f"No forecast directories found in {input_root}")

    # grouped[(model_path, cutoff_month, keyword)][h] = {"abs": v, "rel": v}
    grouped: dict[tuple[str, str, str], dict[int, dict[str, float]]] = defaultdict(dict)
    model_cutoffs: dict[str, set[str]] = defaultdict(set)

    for fdir in fdirs:
        h = int(fdir.name[1:])
        if h not in HORIZONS:
            continue
        csv_files = sorted(fdir.rglob("predictions_20??-??.csv"))
        for csv_path in csv_files:
            rel = csv_path.relative_to(fdir)
            if len(rel.parts) < 2:
                continue
            model_parts = rel.parts[:-1]  # keep nested path (target_mode/model)
            model_path = "/".join(model_parts)
            m = re.search(r"(20\d{2}-\d{2})", csv_path.stem)
            if not m:
                continue
            cutoff_month = m.group(1)
            cutoff_payload = _load_cutoff_inputs(inputs_dir, cutoff_month)
            e_t_map = _build_e_t_map(cutoff_payload, args.feature_weights)

            df = pd.read_csv(csv_path)
            if args.keyword_col not in df.columns or args.score_col not in df.columns:
                continue

            for _, row in df.iterrows():
                kw = str(row[args.keyword_col]).strip()
                if not kw:
                    continue
                abs_v = float(row[args.score_col])
                e_t = float(e_t_map.get(kw, 0.0))
                rel_v = float(abs_v / (e_t + float(args.epsilon)))
                grouped[(model_path, cutoff_month, kw)][h] = {"abs": abs_v, "rel": rel_v}
                model_cutoffs[model_path].add(cutoff_month)

    written = 0
    for model_path, cutoff_set in sorted(model_cutoffs.items()):
        for cutoff_month in sorted(cutoff_set):
            kw_records = {}
            for (m_path, c_month, kw), h_map in grouped.items():
                if m_path != model_path or c_month != cutoff_month:
                    continue
                rec = {"keyword": kw}
                for h in HORIZONS:
                    hv = h_map.get(h)
                    rec[f"abs_{h}"] = None if hv is None else float(hv["abs"])
                    rec[f"rel_{h}"] = None if hv is None else float(hv["rel"])
                kw_records[kw] = rec

            predictions = sorted(kw_records.values(), key=lambda x: x["keyword"])
            out_payload = {
                "run_id": f"{args.run_id_prefix}_{model_path.replace('/', '__')}_{cutoff_month}",
                "model_id": model_path.replace("/", "__"),
                "cutoff_date": f"{cutoff_month}-01",
                "predictions": predictions,
            }
            out_dir = output_root.joinpath(*model_path.split("/"), "predictions_by_cutoff")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"predictions_{cutoff_month}.json"
            out_file.write_text(json.dumps(out_payload, indent=2, ensure_ascii=True), encoding="utf-8")
            written += 1

    meta = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "inputs_by_cutoff_dir": str(inputs_dir),
        "written_files": written,
        "models": sorted(model_cutoffs.keys()),
        "horizons_expected": HORIZONS,
        "epsilon": float(args.epsilon),
        "feature_weights": args.feature_weights,
    }
    (output_root / "gather_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[done] written_files={written} output_root={output_root}")


if __name__ == "__main__":
    main()


