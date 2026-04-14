# src/steps/step_02_keywords.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
from ast import literal_eval
import json
import traceback

import pandas as pd

from .step_02_core import (
    DEFAULT_MODEL,
    DEFAULT_TOP_N,
    DEFAULT_RANGE,
    DEFAULT_NR_CANDIDATES,
    DEFAULT_DIVERSITY,
    DEFAULT_STOP_WORDS,
    DEFAULT_STRICT_LITERAL,
    DEFAULT_JACCARD_THRESHOLD,
    DEFAULT_REMOVE_KEYWORDS,
    load_abstracts,
    ensure_cols,
    make_extractor,
    extract_keywords_batch,
    build_keyword_tables,
)

STEP_NAME = "keywords"
STEP_CODE_VERSION = "12"     # bump when logic/outputs change
inputs_from_prev = True


# ------------------------- config plumbing -------------------------

def external_inputs(cfg: Dict[str, Any], usecase_dir: Path, prev_dir: Path | None) -> List[Path]:
    assert prev_dir is not None, "step_02 requires step_01 outputs"
    return [prev_dir / "outputs"]


def relevant_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    p = (cfg or {}).get("params", {})
    if "ARE_THERE_KEYWORDS" not in p or not p["ARE_THERE_KEYWORDS"]:
        raise ValueError("params.ARE_THERE_KEYWORDS must be set in the usecase config.")

    def _coerce_list(raw, default):
        if raw is None:
            return list(default)
        if isinstance(raw, (list, tuple, set)):
            return [str(x) for x in raw if str(x).strip()]
        if isinstance(raw, str):
            return [raw] if raw.strip() else list(default)
        return list(default)

    graph_cfg = p.get("graph", {}) if isinstance(p, dict) else {}
    add_source = p.get("additional_keywords") if isinstance(p, dict) else None
    if not add_source and isinstance(graph_cfg, dict):
        add_source = graph_cfg.get("additional_keywords")
    forced_source = p.get("forced_keywords") if isinstance(p, dict) else None
    if forced_source is None and isinstance(graph_cfg, dict):
        forced_source = graph_cfg.get("forced_keywords")

    def _coerce_stop_words(raw):
        if raw is None or raw is True:
            return DEFAULT_STOP_WORDS
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
                        combined.add("english")
                    else:
                        combined.update(ENGLISH_STOP_WORDS)
                else:
                    combined.add(token)
            return sorted(combined) if combined else DEFAULT_STOP_WORDS
        return raw

    return {
        "ARE_THERE_KEYWORDS": p["ARE_THERE_KEYWORDS"],
        "show_progress": bool(p.get("show_progress", True)),
        "remove_keywords": _coerce_list(p.get("remove_keywords"), DEFAULT_REMOVE_KEYWORDS),
        "remove_patterns": _coerce_list(p.get("remove_patterns"), []),
        "additional_keywords": _coerce_list(add_source, []) + _coerce_list(forced_source, []),
        "keyword_topn": int(p.get("keyword_topn", DEFAULT_TOP_N)),
        "sample_papers": int(p.get("sample_papers", 10000)),
        "compute": bool(p.get("compute", True)),
        "threads": int(p.get("threads", 2)),
        "abstracts_candidates": p.get("abstracts_candidates", ["abstracts.parquet", "abstracts.jsonl", "abstracts.csv"]),
        "abstracts_text_col": p.get("abstracts_text_col", "Abstract"),
        "paper_id_candidates": p.get("paper_id_candidates", ["paper_id", "id", "openalex_id", "paperId", "uid"]),
        "device": p.get("device", "auto"),
        "keybert_model": p.get("keybert_model", DEFAULT_MODEL),
        "keybert_model_path": p.get("keybert_model_path", None),
        "keyword_ngram": list(p.get("keyword_ngram", list(DEFAULT_RANGE))),
        "keyword_nr_candidates": int(p.get("keyword_nr_candidates", DEFAULT_NR_CANDIDATES)),
        "keyword_diversity": float(p.get("keyword_diversity", DEFAULT_DIVERSITY)),
        "keyword_stop_words": _coerce_stop_words(p.get("keyword_stop_words", DEFAULT_STOP_WORDS)),
        "keyword_strict_literal": bool(p.get("keyword_strict_literal", DEFAULT_STRICT_LITERAL)),
        "keyword_jaccard_threshold": float(p.get("keyword_jaccard_threshold", DEFAULT_JACCARD_THRESHOLD)),
        "colab": bool(p.get("colab", False)),
    }


def _normalize_keyword_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def _sync_refined_keyword_list(raw_df: pd.DataFrame, refined_path: Path) -> pd.DataFrame:
    raw_df = raw_df.copy()
    raw_df["_norm"] = _normalize_keyword_series(raw_df["Keyword"])
    if refined_path.exists():
        refined_df = pd.read_csv(refined_path)
        if "removal" not in refined_df.columns:
            refined_df["removal"] = ""
        refined_df["_norm"] = _normalize_keyword_series(refined_df["Keyword"])
        merged = raw_df.merge(refined_df[["_norm", "removal"]], on="_norm", how="left")
        merged["removal"] = merged["removal"].fillna("")
    else:
        merged = raw_df.assign(removal="")
    refined_out = pd.DataFrame({
        "Keyword": raw_df["Keyword"],
        "removal": merged["removal"]
    })
    refined_out.to_csv(refined_path, index=False)
    return refined_out


# ------------------------- main run -------------------------

def run(cfg: Dict[str, Any], usecase_dir: Path, prev_dir: Path, step_dir: Path):
    params = dict(relevant_params(cfg))
    progress_enabled = bool(params.get("show_progress", True))
    colab_mode = params.pop("colab", False)

    out_dir = step_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = step_dir / "remote"
    remote_dir.mkdir(parents=True, exist_ok=True)

    params_to_store = dict(params)
    (out_dir / "used_config.json").write_text(json.dumps(params_to_store, indent=2), encoding="utf-8")

    artifacts = {
        "cleaned_counts_csv": str(out_dir / "cleaned_keywords_to_build_graphs.csv"),
        "cleaned_list_csv": str(out_dir / "cleaned_keyword_list.csv"),
        "presence_summary_csv": str(out_dir / "keyword_presence_summary.csv"),
        "keywords_parquet": str(out_dir / "keywords.parquet"),
        "raw_keywords_csv": str(out_dir / "keywords_from_abstract.csv"),
        "keywords_summary_csv": str(out_dir / "keywords_summary.csv"),
    }
    done_marker = out_dir / "done.marker"
    if colab_mode and done_marker.exists():
        return {
            "state": "DONE",
            "outputs": str(out_dir),
            "artifacts": artifacts,
        }

    if colab_mode:
        request = {
            "usecase": usecase_dir.name,
            "inputs_dir": str((prev_dir / "outputs").resolve()),
            "outputs_dir": str(out_dir.resolve()),
            "params": params_to_store,
            "expected_outputs": [
                "keywords_from_abstract.csv",
                "cleaned_keywords_to_build_graphs.csv",
                "cleaned_keyword_list.csv",
                "keyword_presence_matches.csv",
                "keyword_presence_summary.csv",
                "keywords.parquet",
                "keywords_summary.csv",
                "done.marker",
            ],
            "instructions": (
                "Run the step_02 keyword extraction notebook in Google Colab using these inputs. "
                "Copy the resulting files back to the outputs folder and create done.marker when finished."
            ),
        }
        request_path = remote_dir / "request.json"
        request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
        message = (
            "[02_keywords] Colab mode enabled. Open the generated request.json in Colab, "
            "run the keyword extraction notebook, and drop the outputs plus done.marker into "
            f"{out_dir} before rerunning the pipeline."
        )
        return {
            "state": "WAITING",
            "message": message,
            "wait_for": str(done_marker),
            "request_path": str(request_path),
            "outputs": str(out_dir),
        }

    done_marker.unlink(missing_ok=True)

    print("[02_keywords] step 2 being processedÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦", flush=True)
    (out_dir / "STATUS.RUNNING").write_text("running", encoding="utf-8")

    try:
        in_dir = prev_dir / "outputs"
        df_raw, src_path = load_abstracts(in_dir, params["abstracts_candidates"])
        df_raw = ensure_cols(df_raw, params["abstracts_text_col"], params["paper_id_candidates"])

        approved_col = "topic_is_approved"
        if params["sample_papers"] and params["sample_papers"] > 0 and len(df_raw) > params["sample_papers"]:
            df = df_raw.sample(n=params["sample_papers"], random_state=42).reset_index(drop=True)
        else:
            df = df_raw.reset_index(drop=True)

        raw_kw_csv = out_dir / "keywords_from_abstract.csv"
        if params["compute"]:
            kind, extractor = make_extractor(params)
            extraction = extract_keywords_batch(
                df["abstract"].tolist(),
                params,
                extractor=extractor,
                kind=kind,
                progress_desc="[02_keywords] extracting",
                progress_enabled=progress_enabled,
                threads=int(params["threads"]),
                return_per_doc=False,
            )
            flat_keywords = extraction.flat_keywords
            pd.Series(flat_keywords, name="raw").to_csv(raw_kw_csv, index=False)
        else:
            if not raw_kw_csv.exists():
                raise FileNotFoundError(f"{raw_kw_csv} not found and compute=False")
            raw_df = pd.read_csv(raw_kw_csv)
            if "raw" in raw_df.columns:
                flat_keywords = raw_df["raw"].astype(str).tolist()
            elif "keywords" in raw_df.columns:
                # Older schema stored per-document keyword lists
                def _coerce_list(val):
                    if isinstance(val, str):
                        try:
                            parsed = literal_eval(val)
                        except (ValueError, SyntaxError):
                            return [val]
                        val = parsed
                    if isinstance(val, (list, tuple, set)):
                        return list(val)
                    return [str(val)]

                lists = raw_df["keywords"].apply(_coerce_list)
                flat_keywords = [kw for sub in lists for kw in sub if str(kw).strip()]
            else:
                # Fallback: assume single-column CSV without header
                flat_keywords = raw_df.iloc[:, 0].astype(str).tolist()

        tables = build_keyword_tables(flat_keywords, params)

        df_keyword_list = pd.DataFrame({"Keyword": tables.cleaned_list})
        out_list = out_dir / "cleaned_keyword_list.csv"
        df_keyword_list.to_csv(out_list, index=False)

        refined_list_path = out_dir / "cleaned_keyword_list_refined.csv"
        refined_df = _sync_refined_keyword_list(df_keyword_list, refined_list_path)
        keep_mask = refined_df["removal"].astype(str).str.strip().eq("")
        keep_norms = set(_normalize_keyword_series(refined_df.loc[keep_mask, "Keyword"]))
        if not keep_norms:
            print("[keywords] Warning: refined list removed every keyword; keeping full set.")
            keep_norms = set(_normalize_keyword_series(df_keyword_list["Keyword"]))

        keyword_norms = _normalize_keyword_series(tables.df_keys["Keyword"])
        mask_keep = keyword_norms.isin(keep_norms)
        df_keys_filtered = tables.df_keys.loc[mask_keep, ["Keyword", "Keyword_norm", "Count"]]

        out_clean_counts = out_dir / "cleaned_keywords_to_build_graphs.csv"
        df_keys_filtered.to_csv(out_clean_counts, index=False)

        matches_path = out_dir / "keyword_presence_matches.csv"
        if not tables.df_hits.empty:
            tables.df_hits.to_csv(matches_path, index=False)
        elif matches_path.exists():
            matches_path.unlink()

        tables.presence_summary.to_csv(out_dir / "keyword_presence_summary.csv", index=False)

        df_keys_filtered.to_parquet(out_dir / "keywords.parquet", index=False)
        pd.DataFrame({
            "n_papers": [int(len(df))],
            "n_keywords_raw": [int(len(flat_keywords))],
            "n_keywords_unique": [int(df_keys_filtered.shape[0])],
            "topn": [int(params["keyword_topn"])],
            "source_file": [str(src_path)]
        }).to_csv(out_dir / "keywords_summary.csv", index=False)

        (out_dir / "done.marker").write_text("ok")
        (out_dir / "STATUS.DONE").write_text("done", encoding="utf-8")
        if (out_dir / "STATUS.RUNNING").exists():
            (out_dir / "STATUS.RUNNING").unlink(missing_ok=True)

        return {
            "state": "DONE",
            "outputs": str(out_dir),
            "artifacts": {
                "cleaned_counts_csv": str(out_clean_counts),
                "cleaned_list_csv": str(out_list),
                "presence_summary_csv": str(out_dir / "keyword_presence_summary.csv"),
                "keywords_parquet": str(out_dir / "keywords.parquet"),
                "raw_keywords_csv": str(out_dir / "keywords_from_abstract.csv"),
            },
        }

    except Exception as e:
        (out_dir / "STATUS.FAILED").write_text(
            f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
            encoding="utf-8"
        )
        raise
