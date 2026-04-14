from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List

from . import step_04_05_graph as graph

STEP_NAME = "build_graph"
STEP_CODE_VERSION = "2"
inputs_from_prev = True


def external_inputs(cfg: Dict[str, Any], usecase_dir: Path, prev_dir: Path | None) -> List[Path]:
    return graph.external_inputs(cfg, usecase_dir, prev_dir)


def relevant_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    params = cfg.get("params", {}) if isinstance(cfg, dict) else {}
    graph_build = params.get("graph_build") if isinstance(params, dict) else None
    graph_params = graph_build if isinstance(graph_build, dict) else params.get("graph", {})
    params = dict(graph_params) if isinstance(graph_params, dict) else {}
    params["mode"] = "build_only"
    return params


def run(cfg: Dict[str, Any], usecase_dir: Path, prev_dir: Path, step_dir: Path) -> Dict[str, Any]:
    cfg_copy = copy.deepcopy(cfg)
    params = cfg_copy.setdefault("params", {})
    graph_build = params.get("graph_build") if isinstance(params, dict) else None
    if isinstance(graph_build, dict):
        graph_params = copy.deepcopy(graph_build)
        params["graph"] = graph_params
    else:
        graph_params = params.setdefault("graph", {})
    # Apply fast_diagnostic.preprocess overrides during build (if enabled).
    diag_cfg = graph_params.get("diagnostics")
    if isinstance(diag_cfg, dict):
        fast_diag = diag_cfg.get("fast_diagnostic")
        if isinstance(fast_diag, dict) and fast_diag.get("enabled"):
            fast_pre = fast_diag.get("preprocess")
            if isinstance(fast_pre, dict) and fast_pre:
                pre_cfg = graph_params.get("preprocess", {})
                pre_cfg = dict(pre_cfg) if isinstance(pre_cfg, dict) else {}
                pre_cfg.update(fast_pre)
                graph_params["preprocess"] = pre_cfg
    graph_params["skip_training"] = True
    # Build step must always (re)build tensors.
    graph_params["reuse_existing_graph"] = False

    # Disable sweep/diagnostics for build-only.
    diag = graph_params.setdefault("diagnostics", {})
    diag["enabled"] = False
    sweep = graph_params.setdefault("sweep", {})
    sweep["enabled"] = False

    return graph.run(cfg_copy, usecase_dir, prev_dir, step_dir)
