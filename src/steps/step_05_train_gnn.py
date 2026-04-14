from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List

from . import step_04_05_graph as graph

STEP_NAME = "train_gnn"
STEP_CODE_VERSION = "2"
inputs_from_prev = True


def external_inputs(cfg: Dict[str, Any], usecase_dir: Path, prev_dir: Path | None) -> List[Path]:
    return graph.external_inputs(cfg, usecase_dir, prev_dir)


def relevant_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    params = cfg.get("params", {}) if isinstance(cfg, dict) else {}
    graph_train = params.get("graph_train") if isinstance(params, dict) else None
    graph_params = graph_train if isinstance(graph_train, dict) else params.get("graph", {})
    params = dict(graph_params) if isinstance(graph_params, dict) else {}
    params["mode"] = "train_only"
    return params


def run(cfg: Dict[str, Any], usecase_dir: Path, prev_dir: Path, step_dir: Path) -> Dict[str, Any]:
    cfg_copy = copy.deepcopy(cfg)
    params = cfg_copy.setdefault("params", {})
    graph_train = params.get("graph_train") if isinstance(params, dict) else None
    if isinstance(graph_train, dict):
        graph_params = copy.deepcopy(graph_train)
        params["graph"] = graph_params
    else:
        graph_params = params.setdefault("graph", {})
    graph_params["skip_training"] = False
    # Training should reuse tensors produced by 04_build_graph.
    graph_params.setdefault("reuse_existing_graph", True)
    # Write each training execution into outputs/runs/<run_id>.
    graph_params.setdefault("run_id_enabled", True)
    graph_params.setdefault("run_id", "auto")
    # Disable input plots/topk/bubble during training-only runs.
    preview_cfg = graph_params.setdefault("preview", {})
    preview_cfg["plot_preview"] = False
    plot_cfg = graph_params.setdefault("plot", {})
    plot_cfg["topk_heatmap"] = 0
    cfg_overrides = graph_params.setdefault("cfg_overrides", {})
    cfg_overrides.setdefault("TOPK_HEATMAP_K", 0)
    cfg_overrides.setdefault("TOPK_BUBBLE_K", 0)
    cfg_overrides.setdefault("PLOT_LINE_BEFORE_HEATMAPS", False)
    return graph.run(cfg_copy, usecase_dir, prev_dir, step_dir)
