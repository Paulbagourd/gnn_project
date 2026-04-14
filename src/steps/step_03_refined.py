from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any, Dict, List

import ahocorasick
import numpy as np
import pandas as pd

from .step_02_core import (
    DEFAULT_STOP_WORDS,
    make_extractor,
    extract_keywords_batch,
    build_keyword_tables,
    normalize_text,
)
from .step_02_keywords import (
    _normalize_keyword_series,
    _sync_refined_keyword_list,
    relevant_params as primary_keyword_params,
)

STEP_NAME = "refined"
STEP_CODE_VERSION = "2"
inputs_from_prev = True

ROOT = Path(__file__).resolve().parents[2]

KEYWORD_FILES_TO_COPY = [
    "cleaned_keywords_to_build_graphs.csv",
    "cleaned_keyword_list.csv",
    "cleaned_keyword_list_refined.csv",
    "keyword_presence_matches.csv",
    "keyword_presence_summary.csv",
    "keywords_from_abstract.csv",
    "keywords.parquet",
    "keywords_summary.csv",
]


def _as_dict(obj: Any) -> Dict[str, Any]:
    return obj if isinstance(obj, dict) else {}


def _coerce_list(raw, default=None) -> list[str]:
    if raw is None:
        return list(default) if default is not None else []
    if isinstance(raw, (list, tuple, set)):
        return [str(x) for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return [raw] if raw.strip() else (list(default) if default is not None else [])
    return list(default) if default is not None else []


def _coerce_stop_words(raw, fallback):
    if raw is None:
        return list(fallback)
    if raw is True:
        return list(DEFAULT_STOP_WORDS)
    if isinstance(raw, str):
        token = raw.strip()
        if not token:
            return list(fallback)
        if token.lower() == "english":
            try:
                from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            except Exception:
                return list(DEFAULT_STOP_WORDS)
            else:
                return sorted(set(ENGLISH_STOP_WORDS))
        return [token]
    if isinstance(raw, (list, tuple, set)):
        combined: set[str] = set()
        for item in raw:
            if not isinstance(item, str):
                continue
            token = item.strip()
            if not token:
                continue
            if token.lower() == "english":
                try:
                    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
                except Exception:
                    combined.update(DEFAULT_STOP_WORDS)
                else:
                    combined.update(ENGLISH_STOP_WORDS)
            else:
                combined.add(token)
        return sorted(combined) if combined else list(fallback)
    return list(fallback)


def _load_custom_keywords(prev_outputs: Path, custom_cfg: Any) -> tuple[list[str] | None, Path | None]:
    """
    When custom refinement is enabled, load keywords from a text file under the
    step_02 outputs directory. Supports either a boolean or a dict config with
    keys {enabled, keywords_file}.
    """
    if not custom_cfg:
        return None, None
    enabled = bool(custom_cfg) if not isinstance(custom_cfg, dict) else bool(custom_cfg.get("enabled", True))
    if not enabled:
        return None, None
    filename = "custom_keywords.txt"
    if isinstance(custom_cfg, dict):
        filename = custom_cfg.get("keywords_file") or filename
    path = Path(filename)
    if not path.is_absolute():
        path = prev_outputs / path
    if not path.exists():
        raise FileNotFoundError(f"Custom refinement enabled but missing keyword file: {path}")
    keywords = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if token:
                keywords.append(token)
    if not keywords:
        raise ValueError(f"Custom refinement keyword file is empty: {path}")
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for kw in keywords:
        if kw not in seen:
            deduped.append(kw)
            seen.add(kw)
    return deduped, path


def external_inputs(cfg: Dict[str, Any], usecase_dir: Path, prev_dir: Path | None) -> List[Path]:
    assert prev_dir is not None, "step_03_refined requires step_02 outputs"
    inputs: List[Path] = [prev_dir / "outputs"]
    inputs.append(usecase_dir / "01_extract" / "outputs")
    return inputs


def relevant_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    params = _as_dict(cfg.get("params"))
    graph_params = _as_dict(params.get("graph"))
    return {
        "refined_collection": _as_dict(params.get("refined_collection")),
        "refined_keywords": _as_dict(params.get("refined_keywords")),
        "graph_keywords_csv": graph_params.get("keywords_csv"),
    }


def _on_rm_error(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass
    func(path)


def _copy_keyword_artifacts(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for file in src.iterdir():
        if file.name in KEYWORD_FILES_TO_COPY or file.name.startswith("cleaned_keywords_to_build_graphs"):
            shutil.copy2(file, dst / file.name)


def _resolve_topics_path(raw: str | None, extract_dir: Path) -> Path | None:
    if not raw:
        return None
    # Prefer the use-case specific refined topics produced by step_01.
    fallback = (extract_dir / raw).resolve()
    if fallback.exists():
        return fallback
    candidate = Path(raw).expanduser()
    if candidate.exists():
        return candidate
    return None


def _compute_monthly_reweight_factors(monthly: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series | None:
    if not isinstance(cfg, dict) or not cfg.get("enabled"):
        return None
    series = monthly.set_index("publication_date")["papers"].sort_index()
    if series.empty:
        return None
    method = str(cfg.get("method", "ema") or "ema").lower()
    span = max(1, int(cfg.get("smooth_span_months", 12) or 12))
    if method == "rolling":
        smooth = series.rolling(window=span, min_periods=1).mean()
    else:
        smooth = series.ewm(span=span, adjust=False).mean()
    base = series.clip(lower=1.0)
    weights = smooth / base
    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    min_factor = float(cfg.get("min_factor", 0.2) or 0.2)
    max_factor = float(cfg.get("max_factor", 5.0) or 5.0)
    weights = weights.clip(lower=min_factor, upper=max_factor)
    return weights.reindex(series.index, fill_value=1.0)


def _plot_monthly_refined_counts(csv_path: Path, out_dir: Path, reweight_cfg: Dict[str, Any] | None) -> dict[str, str] | None:
    if not csv_path.exists():
        return None
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[03_refined] Skipping monthly plot (matplotlib unavailable): {exc}")
        return None
    df = pd.read_csv(csv_path, usecols=["publication_date"], low_memory=False)
    if df.empty or "publication_date" not in df.columns:
        return None
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    monthly = (
        df.dropna(subset=["publication_date"])
        .groupby(pd.Grouper(key="publication_date", freq="MS"))
        .size()
        .rename("papers")
        .reset_index()
    )
    if monthly.empty:
        return None
    monthly["papers_ma6"] = monthly["papers"].rolling(6, min_periods=1).mean()
    if reweight_cfg:
        weights = _compute_monthly_reweight_factors(monthly, reweight_cfg)
    else:
        weights = None
    if weights is not None:
        aligned = weights.reindex(monthly["publication_date"]).fillna(1.0).values
        monthly["papers_corrected"] = monthly["papers"] * aligned
    else:
        monthly["papers_corrected"] = monthly["papers"].ewm(span=12, adjust=False).mean()
    pdf_path = out_dir / "refined_papers_monthly.pdf"
    fig = plt.figure(figsize=(9, 4))
    plt.plot(monthly["publication_date"], monthly["papers"], label="Refined papers / month")
    plt.plot(monthly["publication_date"], monthly["papers_ma6"], label="6-month MA")
    if weights is not None:
        plt.plot(
            monthly["publication_date"],
            monthly["papers_corrected"],
            linestyle="--",
            color="red",
            label="Volume-corrected (reweight)",
        )
    else:
        plt.plot(
            monthly["publication_date"],
            monthly["papers_corrected"],
            linestyle="--",
            color="red",
            label="Virtual correction (EMA)",
        )
    plt.title("Refined papers per month")
    plt.xlabel("Month")
    plt.ylabel("# papers")
    plt.legend()
    plt.tight_layout()
    fig.savefig(pdf_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    csv_out = out_dir / "refined_papers_monthly.csv"
    monthly.to_csv(csv_out, index=False)
    return {"pdf": str(pdf_path), "csv": str(csv_out)}


def _load_scan_topic_ids(
    collection_cfg: Dict[str, Any], params: Dict[str, Any], extract_dir: Path
) -> tuple[set[str], list[str]]:
    topics_csv = collection_cfg.get("topics_csv") or params.get("refined_topics_csv")
    path = _resolve_topics_path(topics_csv, extract_dir)
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"refined topics CSV not found (looked for '{topics_csv}'); "
            "set params.refined_collection.topics_csv explicitly."
        )
    df_topics = pd.read_csv(path)
    column = collection_cfg.get("scan_column", "scan")
    values = collection_cfg.get("accepted_values", ["x"]) or ["x"]
    norm_values = {str(v).strip().lower() for v in values if str(v).strip()}
    if not norm_values:
        norm_values = {"x"}
    if column not in df_topics.columns:
        raise KeyError(f"Column '{column}' not found in {path}")
    mask = df_topics[column].astype(str).str.strip().str.lower().isin(norm_values)
    id_column = "id" if "id" in df_topics.columns else "Tid"
    if id_column not in df_topics.columns:
        raise KeyError(f"Neither 'id' nor 'Tid' columns were found in {path}")
    scan_ids = set(df_topics.loc[mask, id_column].dropna().astype(str).str.strip())
    label_column = "display_name" if "display_name" in df_topics.columns else id_column
    scan_labels = (
        df_topics.loc[mask, label_column]
        .fillna(df_topics.loc[mask, id_column])
        .astype(str)
        .str.strip()
        .tolist()
    )
    return scan_ids, scan_labels


def _resolve_keywords_csv(prev_outputs: Path, base_name: str | None) -> Path:
    name = base_name or "cleaned_keywords_to_build_graphs.csv"
    path = prev_outputs / name
    if not path.stem.endswith("_refined"):
        candidate = path.with_name(f"{path.stem}_refined{path.suffix}")
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise FileNotFoundError(f"Missing keyword counts file: {path}")
    return path


def _build_automaton(keywords: List[str]) -> ahocorasick.Automaton:
    auto = ahocorasick.Automaton()
    for kw in keywords:
        norm = normalize_text(kw)
        if norm:
            auto.add_word(norm, norm)
    if len(auto) == 0:
        raise RuntimeError("No keywords available to build the refined collection.")
    auto.make_automaton()
    return auto


def _match_count(text: Any, automaton: ahocorasick.Automaton, min_hits: int) -> int:
    if not isinstance(text, str):
        return 0
    hits = 0
    for _end, _value in automaton.iter(text):
        hits += 1
        if hits >= min_hits:
            break
    return hits


def _merge_keyword_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    base_params = primary_keyword_params(cfg)
    merged = dict(base_params)
    overrides = _as_dict(cfg.get("params", {}).get("refined_keywords"))
    for key, value in overrides.items():
        if value is None:
            continue
        if key in {"remove_keywords", "remove_patterns"}:
            merged[key] = _coerce_list(value, default=merged.get(key, []))
        elif key in {"additional_keywords", "forced_keywords"}:
            merged[key] = _coerce_list(value, default=[])
        elif key == "keyword_stop_words":
            merged[key] = _coerce_stop_words(value, merged.get(key, DEFAULT_STOP_WORDS))
        else:
            merged[key] = value
    return merged


def _artifact_summary(out_dir: Path) -> Dict[str, str]:
    return {
        "cleaned_counts_csv": str(out_dir / "cleaned_keywords_to_build_graphs.csv"),
        "cleaned_list_csv": str(out_dir / "cleaned_keyword_list.csv"),
        "presence_summary_csv": str(out_dir / "keyword_presence_summary.csv"),
        "keywords_parquet": str(out_dir / "keywords.parquet"),
        "raw_keywords_csv": str(out_dir / "keywords_from_abstract.csv"),
    }


def run(cfg: Dict[str, Any], usecase_dir: Path, prev_dir: Path, step_dir: Path) -> Dict[str, Any]:
    prev_outputs = prev_dir / "outputs"
    extract_dir = usecase_dir / "01_extract" / "outputs"

    params = _as_dict(cfg.get("params"))
    collection_cfg = _as_dict(params.get("refined_collection"))
    graph_cfg = _as_dict(params.get("graph"))
    keywords_csv_name = graph_cfg.get("keywords_csv", "cleaned_keywords_to_build_graphs.csv")
    keyword_params = _merge_keyword_params(cfg)
    colab_mode = bool(keyword_params.get("colab", False))

    out_dir = step_dir / "outputs"
    done_marker = out_dir / "done.marker"
    if colab_mode and done_marker.exists():
        return {"state": "DONE", "outputs": str(out_dir), "artifacts": _artifact_summary(out_dir)}

    if out_dir.exists():
        shutil.rmtree(out_dir, onerror=_on_rm_error)
    out_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = step_dir / "remote"
    remote_dir.mkdir(parents=True, exist_ok=True)
    done_marker.unlink(missing_ok=True)

    used_config = {
        "refined_collection": collection_cfg,
        "refined_keywords": keyword_params,
        "graph_keywords_csv": keywords_csv_name,
    }
    (out_dir / "used_config.json").write_text(json.dumps(used_config, indent=2, ensure_ascii=False), encoding="utf-8")

    enabled = bool(collection_cfg.get("enabled"))
    if not enabled:
        _copy_keyword_artifacts(prev_outputs, out_dir)
        info = {"enabled": False, "reason": "refined_collection.disabled"}
        (out_dir / "collection_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        return {"state": "DONE", "outputs": str(out_dir), "artifacts": _artifact_summary(out_dir)}

    custom_keywords, custom_path = _load_custom_keywords(prev_outputs, collection_cfg.get("custom_refinement"))
    if custom_keywords is not None:
        keyword_list = custom_keywords
        keyword_source = str(custom_path)
    else:
        keywords_path = _resolve_keywords_csv(prev_outputs, keywords_csv_name)
        keywords_df = pd.read_csv(keywords_path)
        if "Keyword" not in keywords_df.columns:
            raise KeyError(f"'Keyword' column missing from {keywords_path}")
        keyword_list = keywords_df["Keyword"].dropna().astype(str).str.strip().tolist()
        keyword_source = str(keywords_path)
    automaton = _build_automaton(keyword_list)

    papers_csv = collection_cfg.get("papers_csv") or graph_cfg.get("papers_csv") or "papers.csv"
    papers_path = extract_dir / papers_csv if not Path(papers_csv).is_absolute() else Path(papers_csv)
    if not papers_path.exists():
        raise FileNotFoundError(f"Missing papers file for refined collection: {papers_path}")
    df_papers = pd.read_csv(papers_path, engine="c", low_memory=False)
    if "primary_topic.id" not in df_papers.columns:
        raise KeyError("Column 'primary_topic.id' not found in papers.csv")
    text_column = collection_cfg.get("text_column", "Abstract")
    if text_column not in df_papers.columns:
        raise KeyError(f"Column '{text_column}' not found in papers.csv")

    if bool(collection_cfg.get("scan_all_topics", False)):
        scan_ids = set(df_papers["primary_topic.id"].dropna().astype(str))
        scan_labels = ["ALL TOPICS"]
    else:
        scan_ids, scan_labels = _load_scan_topic_ids(collection_cfg, params, extract_dir)
        if not scan_ids:
            raise RuntimeError(
                "No topics flagged for refined collection. "
                "Mark topics with scan='x' in refined_topics or enable scan_all_topics."
            )
    print("[03_refined] Topics selected for refined scan:")
    for name in scan_labels:
        print(f"  - {name}")

    df_scan = df_papers[df_papers["primary_topic.id"].astype(str).isin(scan_ids)].copy()
    if df_scan.empty:
        raise RuntimeError("No papers belong to topics flagged for refined scanning. Nothing to refine.")

    use_norm = bool(collection_cfg.get("use_normalized_text", True))
    apply_norm = bool(collection_cfg.get("apply_norm_if_needed", True))
    text_series_name = text_column
    if use_norm and "Abstract_norm" in df_scan.columns:
        text_series_name = "Abstract_norm"
    elif use_norm and apply_norm:
        text_series_name = "__refined_norm"
        df_scan[text_series_name] = df_scan[text_column].map(normalize_text)
    else:
        text_series_name = text_column
    df_scan[text_series_name] = df_scan[text_series_name].fillna("").astype(str)

    min_hits = max(1, int(collection_cfg.get("min_keyword_hits", 1)))
    df_scan["_kw_hits"] = df_scan[text_series_name].apply(lambda txt: _match_count(txt, automaton, min_hits))
    df_refined = df_scan[df_scan["_kw_hits"] >= min_hits].copy()
    df_refined.drop(columns=["_kw_hits"], inplace=True)
    if "__refined_norm" in df_refined.columns and text_series_name != "__refined_norm":
        df_refined.drop(columns=["__refined_norm"], inplace=True)

    if df_refined.empty:
        raise RuntimeError("Refined collection produced zero papers. Relax the filter or disable the feature.")

    print(f"[03_refined] Total scan candidates: {len(df_scan)}")
    print(f"[03_refined] Refined papers retained: {len(df_refined)} (min_keyword_hits={min_hits})")

    output_name = collection_cfg.get("output_papers_csv") or "refined_papers.csv"
    refined_papers_path = out_dir / output_name
    df_refined.to_csv(refined_papers_path, index=False)
    monthly_artifacts = _plot_monthly_refined_counts(
        refined_papers_path, out_dir, _as_dict(graph_cfg.get("paper_volume_reweight"))
    )
    if monthly_artifacts:
        print(f"[03_refined] Monthly refined papers plot -> {monthly_artifacts['pdf']}")
    refined_abstracts_path = out_dir / "refined_abstracts.parquet"
    abstracts_df = pd.DataFrame(
        {
            "paper_id": df_refined.get("id", df_refined.index),
            "abstract": df_refined[text_column].fillna("").astype(str),
        }
    )
    abstracts_df.to_parquet(refined_abstracts_path, index=False)

    info = {
        "enabled": True,
        "total_papers": int(len(df_papers)),
        "scan_candidates": int(len(df_scan)),
        "refined_papers": int(len(df_refined)),
        "min_keyword_hits": int(min_hits),
        "keyword_source": keyword_source,
        "papers_csv": str(refined_papers_path),
        "abstracts_parquet": str(refined_abstracts_path),
    }
    if monthly_artifacts:
        info["monthly_plot_pdf"] = monthly_artifacts.get("pdf")
        info["monthly_counts_csv"] = monthly_artifacts.get("csv")
    (out_dir / "collection_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    if colab_mode:
        request = {
            "usecase": usecase_dir.name,
            "project_root": str(ROOT),
            "refined_papers_csv": str(refined_papers_path.resolve()),
            "outputs_dir": str(out_dir.resolve()),
            "text_column": text_column,
            "params": keyword_params,
            "expected_outputs": [
                "keywords_from_abstract.csv",
                "cleaned_keywords_to_build_graphs.csv",
                "cleaned_keyword_list.csv",
                "cleaned_keyword_list_refined.csv",
                "keyword_presence_matches.csv",
                "keyword_presence_summary.csv",
                "keywords.parquet",
                "keywords_summary.csv",
                "done.marker",
            ],
            "instructions": (
                "Run src/steps/step_03_refined_colab.ipynb in Colab, pointing it at this request. "
                "After it finishes, copy the generated keyword files back into the outputs "
                "directory and create an empty done.marker file."
            ),
        }
        request_path = remote_dir / "request.json"
        request_path.write_text(json.dumps(request, indent=2, ensure_ascii=False), encoding="utf-8")
        message = (
            "[03_refined] Colab mode enabled. Execute the generated request via the "
            "step_03_refined_colab.ipynb notebook, then place the resulting files and "
            "done.marker into the outputs directory before rerunning."
        )
        return {
            "state": "WAITING",
            "message": message,
            "wait_for": str(done_marker),
            "request_path": str(request_path),
            "outputs": str(out_dir),
        }

    progress_enabled = bool(keyword_params.get("show_progress", True))
    raw_kw_csv = out_dir / "keywords_from_abstract.csv"
    sample_df = df_refined.reset_index(drop=True)
    kw_text_column = keyword_params.get("abstracts_text_col", text_column)
    if kw_text_column not in sample_df.columns:
        kw_text_column = text_column
    sample_limit = keyword_params.get("sample_papers")
    if sample_limit and int(sample_limit) > 0 and len(sample_df) > int(sample_limit):
        sample_df = sample_df.sample(n=int(sample_limit), random_state=42).reset_index(drop=True)

    if keyword_params.get("compute", True):
        kind, extractor = make_extractor(keyword_params)
        extraction = extract_keywords_batch(
            sample_df[kw_text_column].fillna("").astype(str).tolist(),
            keyword_params,
            extractor=extractor,
            kind=kind,
            progress_desc="[03_refined] extracting keywords",
            progress_enabled=progress_enabled,
            threads=int(keyword_params.get("threads", 1)),
            return_per_doc=False,
        )
        flat_keywords = extraction.flat_keywords
        pd.Series(flat_keywords, name="raw").to_csv(raw_kw_csv, index=False)
    else:
        if not raw_kw_csv.exists():
            raise FileNotFoundError(f"{raw_kw_csv} not found and compute=false for refined keywords")
        raw_df = pd.read_csv(raw_kw_csv)
        if "raw" in raw_df.columns:
            flat_keywords = raw_df["raw"].astype(str).tolist()
        else:
            flat_keywords = raw_df.iloc[:, 0].astype(str).tolist()

    tables = build_keyword_tables(flat_keywords, keyword_params)

    df_keyword_list = pd.DataFrame({"Keyword": tables.cleaned_list})
    out_list = out_dir / "cleaned_keyword_list.csv"
    df_keyword_list.to_csv(out_list, index=False)

    refined_list_path = out_dir / "cleaned_keyword_list_refined.csv"
    refined_df = _sync_refined_keyword_list(df_keyword_list, refined_list_path)
    keep_mask = refined_df["removal"].astype(str).str.strip().eq("")
    keep_norms = set(_normalize_keyword_series(refined_df.loc[keep_mask, "Keyword"]))
    if not keep_norms:
        keep_norms = set(_normalize_keyword_series(df_keyword_list["Keyword"]))

    keyword_norms = _normalize_keyword_series(tables.df_keys["Keyword"])
    mask_keep = keyword_norms.isin(keep_norms)
    df_keys_filtered = tables.df_keys.loc[mask_keep, ["Keyword", "Keyword_norm", "Count"]]

    out_clean_counts = out_dir / "cleaned_keywords_to_build_graphs.csv"
    df_keys_filtered.to_csv(out_clean_counts, index=False)

    matches_path = out_dir / "keyword_presence_matches.csv"
    if not tables.df_hits.empty:
        tables.df_hits.to_csv(matches_path, index=False)
    else:
        matches_path.unlink(missing_ok=True)

    tables.presence_summary.to_csv(out_dir / "keyword_presence_summary.csv", index=False)
    df_keys_filtered.to_parquet(out_dir / "keywords.parquet", index=False)

    pd.DataFrame(
        {
            "n_papers": [int(len(sample_df))],
            "n_keywords_raw": [int(len(flat_keywords))],
            "n_keywords_unique": [int(df_keys_filtered.shape[0])],
            "topn": [int(keyword_params.get("keyword_topn", 0))],
            "source_file": [str(refined_papers_path)],
        }
    ).to_csv(out_dir / "keywords_summary.csv", index=False)

    (out_dir / "done.marker").write_text("ok", encoding="utf-8")

    return {
        "state": "DONE",
        "outputs": str(out_dir),
        "artifacts": _artifact_summary(out_dir),
    }
