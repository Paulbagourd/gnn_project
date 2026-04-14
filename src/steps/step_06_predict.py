from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .step_04_05_graph import _TeeStream, run_graph_pipeline

STEP_NAME = "predict"
STEP_CODE_VERSION = "5"
inputs_from_prev = True


def _as_dict(obj: Any) -> Dict[str, Any]:
    return obj if isinstance(obj, dict) else {}


def _resolve_prepared_dir(prev_dir: Path) -> Path:
    outputs_root = prev_dir / "outputs"
    latest_manifest = outputs_root / "latest_run.json"
    if latest_manifest.exists():
        try:
            payload = json.loads(latest_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        run_dir_raw = payload.get("run_dir")
        if isinstance(run_dir_raw, str) and run_dir_raw.strip():
            candidate = Path(run_dir_raw)
            if not candidate.is_absolute():
                candidate = (outputs_root / candidate).resolve()
            if candidate.exists():
                return candidate
        run_id = payload.get("run_id")
        if isinstance(run_id, str) and run_id.strip():
            candidate = outputs_root / "runs" / run_id.strip()
            if candidate.exists():
                return candidate
    runs_dir = outputs_root / "runs"
    if runs_dir.exists():
        run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
        if run_dirs:
            return max(run_dirs, key=lambda p: p.stat().st_mtime)
    return outputs_root


def external_inputs(cfg: Dict[str, Any], usecase_dir: Path, prev_dir: Path | None) -> List[Path]:
    assert prev_dir is not None, "step_06_predict requires step_05_train_gnn outputs"
    inputs: List[Path] = [prev_dir / "outputs"]
    # Also depend on upstream raw artefacts to ensure signature invalidates correctly.
    inputs.append(usecase_dir / "02_keywords" / "outputs")
    inputs.append(usecase_dir / "01_extract" / "outputs")
    return inputs


def relevant_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    params = _as_dict(cfg.get("params"))
    predict_params = _as_dict(params.get("predict"))
    graph_params = _as_dict(params.get("graph"))
    # Merge shallowly so signature reflects both sections
    combined = dict(graph_params)
    combined.update({f"predict.{k}": v for k, v in predict_params.items()})
    return combined


def _resolve_keyword_path(
    usecase_dir: Path,
    name: str | None,
    summary: Dict[str, Any],
) -> Path:
    if isinstance(summary.get("keywords_csv"), str):
        p = Path(summary["keywords_csv"])
        if p.exists():
            return p
    fallback = name or "cleaned_keywords_to_build_graphs.csv"
    refined_dir = usecase_dir / "03_refined" / "outputs"
    primary_dir = usecase_dir / "02_keywords" / "outputs"
    for base in (refined_dir, primary_dir):
        candidate = (base / fallback).resolve()
        if candidate.exists():
            return candidate
        refined_variant = candidate.with_name(f"{candidate.stem}_refined{candidate.suffix}")
        if refined_variant.exists():
            return refined_variant
    return (primary_dir / fallback).resolve()


def _resolve_papers_path(
    usecase_dir: Path,
    name: str | None,
    summary: Dict[str, Any],
) -> Path:
    if isinstance(summary.get("papers_csv"), str):
        p = Path(summary["papers_csv"])
        if p.exists():
            return p
    refined_dir = usecase_dir / "03_refined" / "outputs"
    refined_name = "refined_papers.csv"
    refined_path = (refined_dir / refined_name).resolve()
    if refined_path.exists():
        return refined_path
    fallback = name or "papers.csv"
    return (usecase_dir / "01_extract" / "outputs" / fallback).resolve()


def run(cfg: Dict[str, Any], usecase_dir: Path, prev_dir: Path, step_dir: Path) -> Dict[str, Any]:
    out_dir = step_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    prepared_root = prev_dir / "outputs"
    if not prepared_root.exists():
        raise FileNotFoundError(f"Expected prepared tensors under {prepared_root}")
    prepared_dir = _resolve_prepared_dir(prev_dir)
    if not prepared_dir.exists():
        raise FileNotFoundError(f"Resolved training outputs directory does not exist: {prepared_dir}")
    if prepared_dir != prepared_root:
        print(f"[predict] Using latest training run folder: {prepared_dir}")

    params = _as_dict(cfg.get("params"))
    graph_params = _as_dict(params.get("graph"))
    predict_params = _as_dict(params.get("predict"))

    preprocess_cfg = _as_dict(graph_params.get("preprocess"))
    predict_preprocess_cfg = _as_dict(predict_params.get("preprocess"))
    if predict_preprocess_cfg:
        merged_preprocess = dict(preprocess_cfg)
        merged_preprocess.update(predict_preprocess_cfg)
        preprocess_cfg = merged_preprocess
    tail_correction_cfg = _as_dict(graph_params.get("tail_correction"))
    preview_cfg = _as_dict(graph_params.get("preview"))

    plot_cfg = _as_dict(predict_params.get("plot"))
    if not plot_cfg:
        plot_cfg = _as_dict(graph_params.get("plot"))

    cfg_defaults = predict_params.get("cfg_defaults")
    if not isinstance(cfg_defaults, dict):
        cfg_defaults = graph_params.get("cfg_defaults")

    cfg_overrides = predict_params.get("cfg_overrides", {})
    if not isinstance(cfg_overrides, dict) or not cfg_overrides:
        cfg_overrides = graph_params.get("cfg_overrides", {})

    additional_keywords = predict_params.get("additional_keywords")
    remove_keywords = predict_params.get("remove_keywords")
    drop_keywords_cfg = predict_params.get("drop_keywords")
    if additional_keywords is None:
        additional_keywords = graph_params.get("additional_keywords")
    if remove_keywords is None:
        remove_keywords = graph_params.get("remove_keywords")
    if drop_keywords_cfg is None:
        drop_keywords_cfg = graph_params.get("drop_keywords", "drop_fully_inactive")
    if drop_keywords_cfg is None:
        drop_keywords_cfg = "drop_fully_inactive"

    keywords_name = predict_params.get("keywords_csv") or graph_params.get("keywords_csv") or "cleaned_keywords_to_build_graphs.csv"
    papers_name = predict_params.get("papers_csv") or graph_params.get("papers_csv") or "papers.csv"
    topk_predict = int(predict_params.get("topk_predict", 20))

    summary_path = prepared_dir / "used_config.json"
    summary_data: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary_data = {}

    keywords_path = _resolve_keyword_path(usecase_dir, keywords_name, summary_data)
    papers_path = _resolve_papers_path(usecase_dir, papers_name, summary_data)

    if not keywords_path.exists():
        raise FileNotFoundError(f"Missing keywords file for training: {keywords_path}")
    if not papers_path.exists():
        raise FileNotFoundError(f"Missing papers file for training: {papers_path}")

    default_overrides = {"SHOW_FIGS": False}
    merged_overrides = dict(default_overrides)
    if isinstance(cfg_overrides, dict):
        merged_overrides.update(cfg_overrides)

    plot_dir_name = "plots"
    if isinstance(cfg_defaults, dict) and isinstance(cfg_defaults.get("PLOT_DIR"), str):
        plot_dir_name = cfg_defaults["PLOT_DIR"]
    if isinstance(merged_overrides, dict) and isinstance(merged_overrides.get("PLOT_DIR"), str):
        plot_dir_name = merged_overrides["PLOT_DIR"]

    reuse_graph_outputs = bool(predict_params.get("reuse_graph_outputs"))
    train_time_filter = _as_dict(predict_params.get("train_time_filter"))
    if not train_time_filter:
        train_time_filter = _as_dict(graph_params.get("train_time_filter"))
    reuse_existing_graph = bool(predict_params.get("reuse_existing_graph", False))
    graph_data_dir_raw = predict_params.get("graph_data_dir")
    graph_data_dir = prepared_dir
    if reuse_existing_graph:
        project_root = usecase_dir.parents[1]
        if isinstance(graph_data_dir_raw, str) and graph_data_dir_raw.strip():
            candidate = Path(graph_data_dir_raw.strip()).expanduser()
            if not candidate.is_absolute():
                candidate = (project_root / candidate).resolve()
            graph_data_dir = candidate
        else:
            graph_data_dir = (usecase_dir / "04_build_graph" / "outputs").resolve()
        feats_ok = (graph_data_dir / "3_corrected_data" / "stacked_features_active_corrected.npy").exists()
        mats_ok = (graph_data_dir / "3_corrected_data" / "stacked_matrices_corrected.npy").exists()
        if not (feats_ok and mats_ok):
            raise FileNotFoundError(
                "reuse_existing_graph=true but corrected tensors are missing under "
                f"{graph_data_dir / '3_corrected_data'}"
            )
        print(f"[predict] Reusing corrected graph tensors from: {graph_data_dir}")

    log_path = out_dir / ("reuse.log" if reuse_graph_outputs else "cell_outputs.txt")
    plot_source_dir = out_dir

    if reuse_graph_outputs:
        plot_source_dir = prepared_dir
        message = (
            "[predict] reuse_graph_outputs=true -> skipping pipeline run and "
            "reusing artifacts from step_05_train_gnn."
        )
        print(message)
        log_path.write_text(message + "\n", encoding="utf-8")
    else:
        ctx = {
            "path_keyword_counts": keywords_path,
            "path_papers": papers_path,
            "base_dir": out_dir,
            "data_base_dir": graph_data_dir,
            "cfg_overrides": merged_overrides,
            "cfg_defaults": cfg_defaults,
            "preprocess_cfg": preprocess_cfg,
            "tail_correction_cfg": tail_correction_cfg,
            "preview_cfg": preview_cfg,
            "plot_cfg": plot_cfg,
            "additional_keywords": additional_keywords,
            "remove_keywords": remove_keywords,
            "drop_keywords": drop_keywords_cfg,
            "reuse_existing_graph": reuse_existing_graph,
            "train_time_filter": train_time_filter,
        }

        orig_stdout = sys.stdout
        orig_stderr = sys.stderr

        with log_path.open("w", encoding="utf-8") as log_file:
            tee_stdout = _TeeStream(orig_stdout, log_file)
            tee_stderr = _TeeStream(orig_stderr, log_file)
            with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
                run_graph_pipeline(ctx)

    topk_predictions_path: Path | None = None
    plot_dir = plot_source_dir / plot_dir_name
    if plot_dir.exists():
        ranking_files = sorted(plot_dir.glob("ranking_*_RECENT.csv"))
        ranking_file = next((p for p in ranking_files if "GraphModel" in p.stem), ranking_files[0]) if ranking_files else None
        if ranking_file is not None:
            try:
                df_rank = pd.read_csv(ranking_file)
                if not df_rank.empty:
                    topk_df = df_rank.head(max(0, topk_predict)).copy()
                    if "prediction" in topk_df.columns:
                        topk_df.rename(columns={"prediction": "emergence_score"}, inplace=True)
                    topk_predictions_path = out_dir / f"topk_predictions_graphmodel_top{topk_predict}.csv"
                    topk_df.to_csv(topk_predictions_path, index=False)
                    print(f"[predict] Saved top-{topk_predict} predictions to {topk_predictions_path}")
                else:
                    print(f"[predict] Ranking file {ranking_file} is empty; skipping top-k export.")
            except Exception as exc:  # pragma: no cover - best-effort
                print(f"[predict] Warning: unable to build top-k predictions from {ranking_file}: {exc}")
        else:
            print(f"[predict] No ranking CSV found in {plot_dir}; skipping top-k export.")
    else:
        print(f"[predict] Plot directory {plot_dir} not found; skipping top-k export.")

    summary = {
        "keywords_csv": str(keywords_path),
        "papers_csv": str(papers_path),
        "log_path": str(log_path),
        "base_dir": str(out_dir),
        "data_base_dir": str(graph_data_dir),
        "reuse_existing_graph": reuse_existing_graph,
        "train_time_filter": train_time_filter,
    }
    (out_dir / "used_config.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    return {
        "state": "DONE",
        "outputs": str(out_dir),
        "log": str(log_path),
        "topk_predictions": str(topk_predictions_path) if topk_predictions_path else None,
    }
