#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class FrozenSetup:
    usecase: str
    split_source: str
    test_start: str
    test_end: str
    data_start: str
    data_end: str
    common_eval_start: str
    common_eval_end: str
    horizons_months: list[int]
    epsilon: float
    n_cutoffs: int
    n_keywords: int
    feature_names: list[str]
    feature_tensor_path: str
    timestamps_path: str
    keywords_path: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze split+horizons and export LLM dry-run inputs."
    )
    parser.add_argument("--usecase", default="usecase_cyberspace")
    parser.add_argument("--config", default="config/usecases/usecase_cyberspace.yaml")
    parser.add_argument("--horizons", default="12,24,36,48")
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument(
        "--output-dir",
        default="data/usecase_cyberspace/gnn_llm_comparison/outputs",
    )
    parser.add_argument(
        "--max-keywords",
        type=int,
        default=0,
        help="0 means all keywords; otherwise keep first N.",
    )
    return parser.parse_args()


def _month_floor(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts).to_period("M").to_timestamp(how="start")


def _load_usecase_cfg(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML root in {path}")
    return payload


def _read_graph_paths(project_root: Path, usecase: str) -> tuple[Path, Path, Path, Path]:
    base = project_root / "data" / usecase / "04_build_graph" / "outputs" / "1_raw_data"
    feat_path = base / "stacked_features.npy"
    ts_path = base / "feature_timestamps.npy"
    kw_path = base / "keywords_final.txt"
    used_cfg = project_root / "data" / usecase / "04_build_graph" / "outputs" / "used_config.json"
    return feat_path, ts_path, kw_path, used_cfg


def _load_feature_names(used_cfg_path: Path) -> list[str]:
    obj = json.loads(used_cfg_path.read_text(encoding="utf-8"))
    graph_params = obj.get("graph_params", {})
    preview = graph_params.get("preview", {})
    names = preview.get("feature_names")
    if isinstance(names, list) and names:
        return [str(x) for x in names]
    return ["feature_0", "feature_1", "feature_2"]


def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    if end < start:
        return []
    return list(pd.date_range(start=start, end=end, freq="MS"))


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = (project_root / args.config).resolve()
    out_root = (project_root / args.output_dir).resolve()
    out_inputs = out_root / "inputs_by_cutoff"
    out_inputs.mkdir(parents=True, exist_ok=True)

    cfg = _load_usecase_cfg(cfg_path)
    params = cfg.get("params", {})
    graph_train = params.get("graph_train", {})
    cfg_defaults = graph_train.get("cfg_defaults", {})
    split_dates = cfg_defaults.get("SPLIT_DATES", {})
    if not split_dates or not split_dates.get("enabled", False):
        raise ValueError("SPLIT_DATES.enabled must be true in usecase config for this dry-run.")

    test_start = _month_floor(str(split_dates["test_start"]))
    test_end = _month_floor(str(split_dates["test_end"]))

    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    if not horizons:
        raise ValueError("No horizons provided.")
    if any(h <= 0 for h in horizons):
        raise ValueError("Horizons must be positive months.")
    max_h = max(horizons)

    feat_path, ts_path, kw_path, used_cfg_path = _read_graph_paths(project_root, args.usecase)
    features = np.load(feat_path)
    timestamps = np.load(ts_path, allow_pickle=True)
    keywords = [ln.strip() for ln in kw_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    feature_names = _load_feature_names(used_cfg_path)

    if features.ndim != 3:
        raise ValueError(f"Expected 3D feature tensor, got shape={features.shape}")
    t_dim, n_dim, f_dim = features.shape
    if len(timestamps) != t_dim:
        raise ValueError("Mismatch between timestamps length and feature tensor time axis.")
    if len(keywords) != n_dim:
        raise ValueError("Mismatch between keywords length and feature tensor node axis.")
    if len(feature_names) != f_dim:
        feature_names = [f"feature_{i}" for i in range(f_dim)]

    ts_months = [_month_floor(str(x)) for x in timestamps.tolist()]
    data_start = ts_months[0]
    data_end = ts_months[-1]

    common_eval_start = max(test_start, data_start)
    common_eval_end = min(test_end, data_end - pd.DateOffset(months=max_h))
    cutoffs = _month_range(common_eval_start, common_eval_end)
    if not cutoffs:
        raise ValueError(
            f"No valid cutoff month. Check split dates and horizons. "
            f"test=[{test_start.date()}..{test_end.date()}], data_end={data_end.date()}, max_h={max_h}"
        )

    if args.max_keywords > 0:
        keep = min(args.max_keywords, len(keywords))
        keywords = keywords[:keep]
        features = features[:, :keep, :]

    ts_index = {m.strftime("%Y-%m"): i for i, m in enumerate(ts_months)}
    months_full = [m.strftime("%Y-%m") for m in ts_months]

    for cutoff in cutoffs:
        cutoff_key = cutoff.strftime("%Y-%m")
        cutoff_idx = ts_index[cutoff_key]
        months_slice = months_full[: cutoff_idx + 1]
        tensor_slice = features[: cutoff_idx + 1, :, :]

        records = []
        for j, kw in enumerate(keywords):
            vals = np.asarray(tensor_slice[:, j, :], dtype=np.float64)
            records.append(
                {
                    "keyword": kw,
                    "history": {
                        "months": months_slice,
                        "values": vals.round(8).tolist(),
                    },
                }
            )

        payload = {
            "cutoff_date": f"{cutoff_key}-01",
            "epsilon": args.epsilon,
            "horizons_months": horizons,
            "feature_names": feature_names,
            "keywords": records,
        }
        out_file = out_inputs / f"llm_input_cutoff_{cutoff_key}.json"
        out_file.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

    setup = FrozenSetup(
        usecase=args.usecase,
        split_source=str(cfg_path.relative_to(project_root)),
        test_start=test_start.strftime("%Y-%m-%d"),
        test_end=test_end.strftime("%Y-%m-%d"),
        data_start=data_start.strftime("%Y-%m-%d"),
        data_end=data_end.strftime("%Y-%m-%d"),
        common_eval_start=common_eval_start.strftime("%Y-%m-%d"),
        common_eval_end=common_eval_end.strftime("%Y-%m-%d"),
        horizons_months=horizons,
        epsilon=float(args.epsilon),
        n_cutoffs=len(cutoffs),
        n_keywords=len(keywords),
        feature_names=feature_names,
        feature_tensor_path=str(feat_path.relative_to(project_root)),
        timestamps_path=str(ts_path.relative_to(project_root)),
        keywords_path=str(kw_path.relative_to(project_root)),
    )
    (out_root / "frozen_setup.json").write_text(
        json.dumps(setup.__dict__, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    with (out_root / "cutoff_months.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cutoff_month", "cutoff_date", "n_history_months", "n_keywords"])
        for c in cutoffs:
            key = c.strftime("%Y-%m")
            writer.writerow([key, f"{key}-01", ts_index[key] + 1, len(keywords)])

    print(f"[ok] frozen setup written to: {out_root / 'frozen_setup.json'}")
    print(f"[ok] cutoff list written to: {out_root / 'cutoff_months.csv'}")
    print(f"[ok] input files directory: {out_inputs}")
    print(f"[info] n_cutoffs={len(cutoffs)} n_keywords={len(keywords)} horizons={horizons}")


if __name__ == "__main__":
    main()

