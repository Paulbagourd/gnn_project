#!/usr/bin/env python
"""Export a full graph JSON (all active nodes + top edges) for Cytoscape viewer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from matplotlib import colormaps, colors as mcolors

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def _export_full_graph_json(
    df: pd.DataFrame,
    fname: str,
    freeze_idx: int,
    freeze_actual: pd.Timestamp,
    column_info: list[dict[str, Any]],
    mats: np.ndarray,
    ts: pd.DatetimeIndex,
    save_dir: Path | None,
    tag: str,
    bubble_window: int,
    bubble_trend_mode: str,
    topk_k: int,
    bubble_diff_tol: float,
) -> Path:
    if save_dir is None:
        raise ValueError("save_dir must be provided")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    column_labels = [info["label"] for info in column_info]
    active_indices = [info.get("active_idx", idx) for idx, info in enumerate(column_info)]
    df = df.loc[:, column_labels]

    freeze_idx = int(freeze_idx)
    if freeze_idx < 0 or freeze_idx >= len(df):
        raise ValueError("freeze index outside DataFrame range")

    freeze_actual = pd.to_datetime(freeze_actual)
    node_recent_start = max(0, freeze_idx - bubble_window + 1) if bubble_window > 0 else 0
    node_past_start = max(0, node_recent_start - bubble_window) if bubble_window > 0 else 0
    node_past_end = node_recent_start - 1

    node_metrics: dict[str, dict[str, float]] = {}
    node_max_slope = 0.0
    node_max_pct = 0.0
    for label in column_labels:
        values = df[label].to_numpy(dtype=float)
        recent_vals = values[node_recent_start:freeze_idx + 1]
        recent_vals = recent_vals[np.isfinite(recent_vals)]
        recent_mean = float(np.nanmean(recent_vals)) if recent_vals.size else 0.0
        past_mean = None
        if bubble_window > 0 and node_past_end >= node_past_start:
            past_vals = values[node_past_start:node_past_end + 1]
            past_vals = past_vals[np.isfinite(past_vals)]
            if past_vals.size:
                past_mean = float(np.nanmean(past_vals))
        slope = 0.0
        if recent_vals.size >= 2:
            x_idx = np.arange(recent_vals.size, dtype=float)
            try:
                slope = float(np.polyfit(x_idx, recent_vals, 1)[0])
            except Exception:
                slope = 0.0
        pct_delta = 0.0
        if past_mean is not None and past_mean > 0:
            pct_delta = (recent_mean - past_mean) / past_mean
        node_max_slope = max(node_max_slope, abs(slope))
        node_max_pct = max(node_max_pct, abs(pct_delta))
        node_metrics[label] = {
            "recent_mean": recent_mean,
            "past_mean": past_mean,
            "slope": slope,
            "pct": pct_delta,
            "diff": recent_mean - (past_mean if past_mean is not None else 0.0),
        }

    mats = np.asarray(mats, dtype=float)
    edges = []
    T = mats.shape[2]
    slice_start = node_recent_start
    for i, label_i in enumerate(column_labels):
        idx_i = active_indices[i]
        for j in range(i + 1, len(column_labels)):
            idx_j = active_indices[j]
            recent_vals = mats[idx_i, idx_j, slice_start:freeze_idx + 1]
            recent_vals = np.clip(recent_vals[np.isfinite(recent_vals)], a_min=0.0, a_max=None)
            recent_mean = float(np.nanmean(recent_vals)) if recent_vals.size else 0.0
            if recent_mean <= 0.0:
                continue
            past_mean = None
            if bubble_window > 0 and slice_start > 0:
                past_end = slice_start - 1
                past_start = max(0, past_end - bubble_window + 1)
                past_vals = mats[idx_i, idx_j, past_start:past_end + 1]
                past_vals = np.clip(past_vals[np.isfinite(past_vals)], a_min=0.0, a_max=None)
                if past_vals.size:
                    past_mean = float(np.nanmean(past_vals))
            slope = 0.0
            if recent_vals.size >= 2:
                x_idx = np.arange(recent_vals.size, dtype=float)
                try:
                    slope = float(np.polyfit(x_idx, recent_vals, 1)[0])
                except Exception:
                    slope = 0.0
            pct_delta = 0.0
            if past_mean is not None and past_mean > 0:
                pct_delta = (recent_mean - past_mean) / past_mean
            edges.append((label_i, column_labels[j], recent_mean, slope, pct_delta, past_mean))

    edges.sort(key=lambda x: x[2], reverse=True)
    max_edges = max(200, topk_k * 20)
    if len(edges) > max_edges:
        edges = edges[:max_edges]

    G = nx.Graph()
    max_node_value = max((metrics["recent_mean"] for metrics in node_metrics.values()), default=1.0)
    for label, metrics in node_metrics.items():
        size = metrics["recent_mean"]
        G.add_node(label, score=df.iloc[freeze_idx][label], size=size, value=size / max_node_value if max_node_value else 0.0)

    cmap_slope = colormaps.get_cmap("coolwarm")
    max_edge_slope = max((abs(e[3]) for e in edges), default=0.0)
    max_edge_pct = max((abs(e[4]) for e in edges), default=0.0)
    for u, v, weight, slope, pct, past_mean in edges:
        delta = 0.0
        color = "#bfbfbf"
        if bubble_trend_mode == "slope":
            delta = slope
            if max_edge_slope > 0:
                norm = np.clip(delta / max_edge_slope, -1.0, 1.0)
                color = mcolors.to_hex(cmap_slope((norm + 1.0) / 2.0))
        elif bubble_trend_mode == "percentage":
            delta = pct
            if abs(delta) <= bubble_diff_tol:
                color = "#bfbfbf"
            elif max_edge_pct > 0:
                norm = np.clip(delta / max_edge_pct, -1.0, 1.0)
                color = mcolors.to_hex(cmap_slope((norm + 1.0) / 2.0))
        else:
            if past_mean is not None:
                delta = weight - past_mean
                if delta > bubble_diff_tol:
                    color = "#b40426"
                elif delta < -bubble_diff_tol:
                    color = "#3b4cc0"
                else:
                    color = "#bfbfbf"
        G.add_edge(u, v, weight=weight, delta=delta, color=color)

    try:
        communities = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
    except Exception:
        communities = [set(G.nodes())]
    community_map = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            community_map[node] = idx
    palette = colormaps.get_cmap("Set3")
    node_entries = []
    node_max_slope = max((abs(m["slope"]) for m in node_metrics.values()), default=0.0)
    for label in column_labels:
        metrics = node_metrics[label]
        community = community_map.get(label, 0)
        community_color = mcolors.to_hex(palette((community % 12) / 12))
        growth_color = "#bfbfbf"
        if bubble_trend_mode == "slope":
            delta = metrics["slope"]
            if node_max_slope > 0:
                norm = np.clip(delta / node_max_slope, -1.0, 1.0)
                growth_color = mcolors.to_hex(cmap_slope((norm + 1.0) / 2.0))
        elif bubble_trend_mode == "percentage":
            delta = metrics["pct"]
            if node_max_pct > 0:
                norm = np.clip(delta / node_max_pct, -1.0, 1.0)
                growth_color = mcolors.to_hex(cmap_slope((norm + 1.0) / 2.0))
        else:
            delta = metrics["diff"]
            if delta > bubble_diff_tol:
                growth_color = "#b40426"
            elif delta < -bubble_diff_tol:
                growth_color = "#3b4cc0"
        node_entries.append({
            "id": label,
            "score": float(df.iloc[freeze_idx][label]),
            "size": metrics["recent_mean"],
            "community": community,
            "community_color": community_color,
            "growth_color": growth_color,
            "recent_mean": metrics["recent_mean"],
            "past_mean": metrics["past_mean"],
            "slope": metrics["slope"],
            "pct": metrics["pct"],
            "diff": metrics["diff"],
            "is_top": False,
        })

    edge_entries = [
        {"source": u, "target": v, "weight": float(data["weight"]), "delta": float(data["delta"]),
         "trend": float(data["delta"]), "color": data["color"]}
        for u, v, data in G.edges(data=True)
    ]

    metadata = {
        "tag": tag,
        "feature": fname,
        "freeze_date": freeze_actual.strftime("%Y-%m-%d"),
        "bubble_window": bubble_window,
        "trend_mode": bubble_trend_mode,
        "topk_k": topk_k,
        "bubble_diff_tol": bubble_diff_tol,
        "total_nodes": len(node_entries),
        "total_edges": len(edge_entries),
    }

    safe_name = fname.replace(" ", "_")
    out_path = save_dir / f"{tag}__graph_full_{safe_name}_{freeze_actual.strftime('%Y-%m')}.json"
    out_path.write_text(json.dumps({
        "metadata": metadata,
        "nodes": node_entries,
        "edges": edge_entries,
    }, indent=2))
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usecase", default="usecase_cyberspace", nargs="?",
                        help="Usecase identifier under data/<usecase>/04_graph/outputs")
    parser.add_argument("--freeze", default=None,
                        help="Freeze date (YYYY-MM) to override metadata date")
    parser.add_argument("--source-json", dest="source_json", default=None,
                        help="Existing bubble JSON (default: first *graph_fw_aggregate*.json)")
    return parser.parse_args()


def load_metadata(bubble_dir: Path, source_json: Path | None) -> tuple[dict, Path]:
    if source_json is None:
        candidates = sorted(bubble_dir.glob("*__graph_full_fw_aggregate_*.json"))
        if not candidates:
            raise FileNotFoundError(f"No legacy graph JSON found under {bubble_dir}")
        source_json = candidates[0]
    if not source_json.exists():
        raise FileNotFoundError(source_json)
    data = json.loads(source_json.read_text())
    metadata = data.get("metadata", {})
    return metadata, source_json


def derive_metadata_from_config(usecase_cfg: dict[str, Any], outputs_dir: Path, freeze: str | None) -> dict[str, Any]:
    if not freeze:
        raise RuntimeError("Freeze date must be provided when no legacy metadata exists (use --freeze YYYY-MM).")
    params = usecase_cfg.get("params", {})
    predict_cfg = params.get("predict", {})
    defaults = predict_cfg.get("cfg_defaults", {})
    emergence = str(defaults.get("EMERGENCE_MODE", "raw"))
    epsilon = str(defaults.get("EPSILON", "1e-8"))
    forecast = defaults.get("FORECAST", 8)
    window = defaults.get("TEMP_WINDOW", 12)
    tag = f"EM_{emergence}_Eps{epsilon}_F{forecast}_W{window}"
    topk_cfg = defaults
    bubble_window = int(topk_cfg.get("TOPK_BUBBLE_WINDOW_MONTHS", 24) or 24)
    bubble_trend = str(topk_cfg.get("TOPK_BUBBLE_TREND_MODE", "diff") or "diff").lower()
    bubble_topk = int(topk_cfg.get("TOPK_BUBBLE_K", 20) or 20)
    bubble_diff_tol = float(topk_cfg.get("TOPK_BUBBLE_DIFF_TOL", 0.0) or 0.0)
    metadata = {
        "tag": tag,
        "feature": "fw_aggregate",
        "freeze_date": f"{freeze}-01" if len(freeze) == 7 else freeze,
        "bubble_window": bubble_window,
        "trend_mode": bubble_trend,
        "topk_k": bubble_topk,
        "bubble_diff_tol": bubble_diff_tol,
    }
    return metadata


def main() -> None:
    args = parse_args()
    usecase_dir = Path("data") / args.usecase
    outputs_dir = usecase_dir / "04_graph" / "outputs"
    bubble_dir = outputs_dir / "plots" / "inputs_seen" / "bubble"
    bubble_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path("config") / "usecases" / f"{args.usecase}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Usecase config not found at {config_path}")
    usecase_cfg = yaml.safe_load(config_path.read_text())

    try:
        metadata, source_json = load_metadata(bubble_dir, Path(args.source_json) if args.source_json else None)
    except FileNotFoundError:
        metadata = derive_metadata_from_config(usecase_cfg, outputs_dir, args.freeze)
        source_json = None
    tag = metadata.get("tag")
    if not tag:
        raise RuntimeError(f"Metadata in {source_json} is missing 'tag'")

    freeze_date = args.freeze or metadata.get("freeze_date")
    if not freeze_date:
        raise RuntimeError("Freeze date not provided")
    freeze_ts = pd.to_datetime(freeze_date).to_period("M").to_timestamp(how="start")

    bubble_window = int(metadata.get("bubble_window", 24))
    bubble_trend_mode = str(metadata.get("trend_mode", "diff")).lower()
    bubble_topk = int(metadata.get("topk_k", 20))
    bubble_diff_tol = float(metadata.get("bubble_diff_tol", 0.0) or 0.0)

    used_cfg = json.loads((outputs_dir / "used_config.json").read_text())
    graph_params = used_cfg.get("graph_params", {})
    preview_cfg = graph_params.get("preview", {})
    fw = np.asarray(preview_cfg.get("feature_weights", [0.0, 0.5, 0.5]), dtype=float)

    feats_path = outputs_dir / "3_corrected_data" / "stacked_features_active_corrected.npy"
    mats_path = outputs_dir / "3_corrected_data" / "stacked_matrices_corrected.npy"
    ts_path = outputs_dir / "1_raw_data" / "feature_timestamps.npy"
    names_path = outputs_dir / "2_active_data" / "keywords_active.txt"

    Xt = np.load(feats_path)
    mats = np.load(mats_path)
    timestamps = np.load(ts_path)
    names = [line.strip() for line in names_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if Xt.shape[1] != len(names):
        raise RuntimeError(f"Name count ({len(names)}) != feature count ({Xt.shape[1]})")

    try:
        ts = pd.PeriodIndex(timestamps.astype(str), freq="M").to_timestamp(how="start")
    except Exception:
        ts = pd.to_datetime(timestamps)
    T = min(len(ts), Xt.shape[0], mats.shape[2])
    ts = ts[:T]
    Xt = Xt[:T]
    mats = mats[:, :, :T]

    agg_vals = np.tensordot(Xt, fw, axes=(2, 0))
    df_fw = pd.DataFrame(agg_vals, index=ts, columns=names)

    try:
        freeze_idx = df_fw.index.get_indexer([freeze_ts], method="nearest")[0]
    except Exception:
        raise RuntimeError(f"Freeze date {freeze_ts.date()} not in index range")
    freeze_actual = df_fw.index[freeze_idx]

    column_info = [
        {"label": nm, "name": nm, "active_idx": i, "orig_idx": i}
        for i, nm in enumerate(names)
    ]

    _export_full_graph_json(
        df=df_fw,
        fname="fw_aggregate",
        freeze_idx=freeze_idx,
        freeze_actual=freeze_actual,
        column_info=column_info,
        mats=mats,
        ts=ts,
        save_dir=bubble_dir,
        tag=tag,
        bubble_window=bubble_window,
        bubble_trend_mode=bubble_trend_mode,
        topk_k=bubble_topk,
        bubble_diff_tol=bubble_diff_tol,
    )


if __name__ == "__main__":
    main()
