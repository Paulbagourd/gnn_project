# Cell 1
from __future__ import annotations

import contextlib
from datetime import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

STEP_NAME = "graph"
STEP_CODE_VERSION = "10"
inputs_from_prev = True

class _TeeStream:
    """Multiplex writes to several file-like streams (used for live logging)."""

    def __init__(self, *streams):
        self._streams = [s for s in streams if s is not None]

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


class _FilteredStream:
    """Forward only lines passing `predicate` to the wrapped stream."""

    def __init__(self, stream, predicate):
        self._stream = stream
        self._predicate = predicate
        self._buffer = ""

    def write(self, data: str) -> int:
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if self._predicate(line):
                self._stream.write(line + "\n")
        return len(data)

    def flush(self) -> None:
        if self._buffer:
            if self._predicate(self._buffer):
                self._stream.write(self._buffer)
        self._buffer = ""
        self._stream.flush()


def _sanitize_run_id(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    chars: list[str] = []
    for ch in raw:
        if ch.isalnum() or ch in {"-", "_"}:
            chars.append(ch)
        else:
            chars.append("_")
    cleaned = "".join(chars).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned[:80]


def _build_auto_run_id(graph_params: Dict[str, Any]) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_defaults = graph_params.get("cfg_defaults") if isinstance(graph_params.get("cfg_defaults"), dict) else {}
    target_mode = _sanitize_run_id(cfg_defaults.get("TARGET_MODE") or "train")
    forecast = cfg_defaults.get("FORECAST")
    temp_window = cfg_defaults.get("TEMP_WINDOW")
    parts = [stamp, target_mode]
    if forecast is not None:
        parts.append(f"f{forecast}")
    if temp_window is not None:
        parts.append(f"w{temp_window}")
    return "_".join(str(p) for p in parts if str(p))


def _resolve_training_output_dir(
    step_dir: Path,
    graph_params: Dict[str, Any],
    *,
    skip_training: bool,
) -> tuple[Path, Path, str | None]:
    outputs_root = step_dir / "outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)

    if skip_training:
        return outputs_root, outputs_root, None

    run_id_enabled = bool(graph_params.get("run_id_enabled", True))
    if not run_id_enabled:
        return outputs_root, outputs_root, None

    run_id = _sanitize_run_id(graph_params.get("run_id", "auto"))
    if not run_id or run_id.lower() == "auto":
        run_id = _build_auto_run_id(graph_params)

    runs_dir = outputs_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / run_id
    if run_dir.exists():
        suffix = 1
        while True:
            candidate = runs_dir / f"{run_id}_{suffix:02d}"
            if not candidate.exists():
                run_id = candidate.name
                run_dir = candidate
                break
            suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, outputs_root, run_id


def _normalize_keyword_series_simple(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def _apply_refined_removals_to_counts(keywords_path: Path, outputs_dir: Path) -> None:
    refined_path = outputs_dir / "cleaned_keyword_list_refined.csv"
    if not refined_path.exists():
        return
    try:
        refined_df = pd.read_csv(refined_path)
    except Exception:
        return
    if "Keyword" not in refined_df.columns or "removal" not in refined_df.columns:
        return
    keep_mask = refined_df["removal"].fillna("").astype(str).str.strip().eq("")
    keep_norms = set(_normalize_keyword_series_simple(refined_df.loc[keep_mask, "Keyword"]))
    if not keep_norms:
        base_path = outputs_dir / "cleaned_keyword_list.csv"
        if base_path.exists():
            try:
                base_df = pd.read_csv(base_path)
                keep_norms = set(_normalize_keyword_series_simple(base_df.get("Keyword", pd.Series(dtype=str))))
            except Exception:
                keep_norms = set()
    if not keep_norms:
        return
    try:
        counts_df = pd.read_csv(keywords_path)
    except Exception:
        return
    if "Keyword" not in counts_df.columns:
        return
    counts_df["_norm"] = _normalize_keyword_series_simple(counts_df["Keyword"])
    mask = counts_df["_norm"].isin(keep_norms)
    removed = int((~mask).sum())
    if removed <= 0:
        return
    filtered = counts_df.loc[mask].drop(columns="_norm")
    tmp_path = keywords_path.with_suffix(".tmp")
    filtered.to_csv(tmp_path, index=False)
    tmp_path.replace(keywords_path)
    print(f"[graph] Applied refined keyword removals (removed {removed})")


def _compute_volume_reweight_factors(ts: pd.DatetimeIndex, ctx: Dict[str, Any]) -> np.ndarray | None:
    cfg_rw = ctx.get("paper_volume_reweight") or {}
    if not isinstance(cfg_rw, dict) or not cfg_rw.get("enabled"):
        return None
    counts_path_raw = cfg_rw.get("counts_csv")
    if counts_path_raw:
        counts_path = Path(counts_path_raw)
        if not counts_path.is_absolute():
            base_dir = ctx.get("counts_base_dir")
            if base_dir:
                base_dir = Path(base_dir)
            else:
                base_dir = Path(ctx["path_papers"]).parent
            counts_path = (base_dir / counts_path).resolve()
    else:
        counts_path = Path(ctx["path_papers"])
    if not counts_path.exists():
        print(f"[volume_reweight] counts file not found: {counts_path}")
        return None
    date_col = cfg_rw.get("date_column", "publication_date")
    try:
        df_counts = pd.read_csv(counts_path, usecols=[date_col])
    except Exception as exc:
        print(f"[volume_reweight] unable to read counts file ({exc})")
        return None
    if date_col not in df_counts.columns:
        return None
    dates = pd.to_datetime(df_counts[date_col], errors="coerce")
    if dates.notna().sum() == 0:
        return None
    counts_series = dates.dt.to_period("M").value_counts().sort_index()
    ts_period = pd.PeriodIndex(pd.to_datetime(ts), freq="M")
    counts_aligned = counts_series.reindex(ts_period, fill_value=0).astype(float)
    if counts_aligned.empty:
        return None
    series = counts_aligned.to_timestamp(how="start").sort_index().reindex(ts, fill_value=0.0)
    if series.empty:
        return None
    method = str(cfg_rw.get("method", "ema") or "ema").lower()
    span = max(1, int(cfg_rw.get("smooth_span_months", 12) or 12))
    if method == "rolling":
        smooth = series.rolling(window=span, min_periods=1).mean()
    else:
        smooth = series.ewm(span=span, adjust=False).mean()
    base = np.maximum(series.values, 1.0)
    weights = smooth.values / base
    weights = np.where(np.isfinite(weights), weights, 1.0)
    max_factor = float(cfg_rw.get("max_factor", 5.0) or 5.0)
    min_factor = float(cfg_rw.get("min_factor", 0.2) or 0.2)
    weights = np.clip(weights, min_factor, max_factor)
    print(f"[volume_reweight] applied monthly factors (median={np.median(weights):.2f}, "
          f"max={weights.max():.2f}) from {counts_path.name}")
    return weights


def _normalize_fw_weights(
    fw: np.ndarray,
    *,
    enabled: bool = True,
    where: str = "fw",
) -> np.ndarray:
    arr = np.asarray(fw, dtype=float).reshape(-1)
    if not enabled:
        return arr
    if not np.all(np.isfinite(arr)):
        print(f"[fw] {where}: non-finite weights detected; replacing with 0 before normalization.")
        arr = np.where(np.isfinite(arr), arr, 0.0)
    s = float(arr.sum())
    if abs(s) < 1e-12:
        print(f"[fw] {where}: sum(FW) is ~0; skipping normalization.")
        return arr
    out = arr / s
    if not np.allclose(arr, out):
        print(f"[fw] {where}: normalized FW to sum=1.")
    return out


def _coerce_list_any(raw, default=None) -> list[str]:
    values = raw if raw is not None else default
    if values is None:
        return []
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    coerced: list[str] = []
    for item in values:
        if isinstance(item, str):
            token = item.strip()
            if token:
                coerced.append(token)
        else:
            coerced.append(str(item).strip())
    return [token for token in coerced if token]


# Cell 2
def _run_graph_notebook(ctx):
    emit = ctx.get("emit")
    skip_training = bool(ctx.get("skip_training", False))
    reuse_graph = bool(ctx.get("reuse_existing_graph", False))

    def _announce(label: str):
        message = f"[graph] {label}"
        if emit:
            emit(message)
        print(message)

    preprocess_cfg = ctx.get("preprocess_cfg") or {}
    tail_cfg = ctx.get("tail_correction_cfg") or {}
    preview_cfg = ctx.get("preview_cfg") or {}
    plot_cfg = ctx.get("plot_cfg") or {}
    cfg_defaults_ctx = ctx.get("cfg_defaults") or {}
    sweep_cfg = ctx.get("sweep_cfg") or {}
    sweep_enabled = bool(sweep_cfg.get("enabled"))
    diag_cfg = ctx.get("diag_cfg") or {}
    diag_enabled = bool(diag_cfg.get("enabled")) if isinstance(diag_cfg, dict) else False
    # Apply the fast_diagnostic profile only when diagnostics mode is explicitly enabled.
    # This prevents accidental top-k/preprocess overrides during sweep-only runs.
    if diag_enabled and isinstance(diag_cfg, dict):
        diag_fast = diag_cfg.get("fast_diagnostic")
        if isinstance(diag_fast, dict) and diag_fast.get("enabled"):
            diag_cfg = {**diag_cfg, **diag_fast}
            diag_cfg["enabled"] = True
        diag_enabled = bool(diag_cfg.get("enabled"))

    if sweep_enabled:
        # Allow sweep to override top_keywords before tensor generation.
        if "top_keywords" in sweep_cfg and sweep_cfg["top_keywords"] is not None:
            preprocess_cfg = dict(preprocess_cfg)
            preprocess_cfg["top_keywords"] = int(sweep_cfg["top_keywords"])
    if diag_enabled:
        # Allow diagnostics to override preprocess settings before tensor generation.
        if isinstance(diag_cfg, dict):
            preprocess_override = diag_cfg.get("preprocess")
            if isinstance(preprocess_override, dict):
                preprocess_cfg = {**preprocess_cfg, **preprocess_override}
            if "top_keywords" in diag_cfg and diag_cfg["top_keywords"] is not None:
                preprocess_cfg = dict(preprocess_cfg)
                preprocess_cfg["top_keywords"] = int(diag_cfg["top_keywords"])

    def _get(cfg, key, default=None):
        return cfg.get(key, default) if isinstance(cfg, dict) else default

    def _get_bool(cfg, key, default):
        if not isinstance(cfg, dict) or key not in cfg or cfg[key] is None:
            return default
        return bool(cfg[key])

    def _get_int(cfg, key, default=None):
        if not isinstance(cfg, dict) or key not in cfg or cfg[key] is None:
            return default
        try:
            return int(cfg[key])
        except (TypeError, ValueError):
            return default

    def _get_float(cfg, key, default=None):
        if not isinstance(cfg, dict) or key not in cfg or cfg[key] is None:
            return default
        try:
            return float(cfg[key])
        except (TypeError, ValueError):
            return default

    _announce("Cell 1: imports and helpers")
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx
    try:
        from wordcloud import WordCloud
    except ImportError:  # pragma: no cover - optional dependency
        WordCloud = None

    from tqdm import tqdm
    from ast import literal_eval
    from itertools import permutations
    from collections import Counter, defaultdict

    from difflib import SequenceMatcher
    from flashtext import KeywordProcessor
    import ahocorasick

    import os
    import random
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from ast import literal_eval
    from itertools import permutations
    from collections import Counter, defaultdict
    import matplotlib.pyplot as plt

    from difflib import SequenceMatcher
    from flashtext import KeywordProcessor
    import ahocorasick


# Cell 3 (markdown)
#     # ## scanning the abstracts

# Cell 4
    # -*- coding: utf-8 -*-
    """
    Build monthly co-occurrence and per-keyword features from papers.csv using a fixed keyword list.
    Adds causal k-month trailing averages for features (optional for co-occurrence).
    """

    import os, re, unicodedata
    import numpy as np
    import pandas as pd
    import ahocorasick
    from collections import defaultdict, Counter
    from itertools import combinations
    from tqdm import tqdm

    _announce("Cell 2: configuration and keyword setup")
    # =========================
    # Config
    # =========================
    PATH_KEYWORD_COUNTS = os.fspath(ctx['path_keyword_counts'])
    PATH_PAPERS         = os.fspath(ctx['path_papers'])

    TOP_KEYWORDS        = _get_int(preprocess_cfg, "top_keywords", 1000) or 1000
    NUMBER_MONTHS       = _get_int(preprocess_cfg, "number_months", None)
    START_YM            = _get(preprocess_cfg, "start_year_month", None)
    END_YM              = _get(preprocess_cfg, "end_year_month", None)
    USE_NORMALIZED_TEXT = _get_bool(preprocess_cfg, "use_normalized_text", True)
    APPLY_NORM_IF_NEEDED= _get_bool(preprocess_cfg, "apply_norm_if_needed", True)
    NORMALIZE_KEYWORDS  = _get_bool(preprocess_cfg, "normalize_keywords", True)

    # occurrence mode
    #   "doc"   -> 1 per abstract if keyword appears at least once (original behavior)
    #   "token" -> count ALL matches (total occurrences)
    OCCURRENCE_MODE     = str(_get(preprocess_cfg, "occurrence_mode", "token") or "token").lower()
    # Optional: how to sum citations when OCCURRENCE_MODE="token"
    #   "per_doc" (default) -> add the doc's cited_by_count once per keyword present
    #   "per_token"         -> add cited_by_count * token_count
    CITATION_WEIGHTING  = str(_get(preprocess_cfg, "citation_weighting", "per_doc") or "per_doc").lower()
    # Citation signal source:
    #   "snapshot"          -> legacy behavior using paper-level cited_by_count at extraction time
    #   "causal_flow"       -> monthly incoming citation events from in-corpus references
    #   "causal_cumulative" -> cumulative incoming citations up to month t (as-of signal)
    CITATION_SOURCE = str(_get(preprocess_cfg, "citation_source", "snapshot") or "snapshot").lower()
    # Citation feature variant when using causal citation source:
    #   "base"       -> standard in-corpus citation flow
    #   "frac"       -> flow weighted by 1/|R(p)|
    #   "frac_split" -> flow weighted by 1/|R(p)| * 1/|K(r)|
    CITATION_FEATURE_VARIANT = str(
        _get(preprocess_cfg, "citation_feature_variant", "base") or "base"
    ).strip().lower()

    # NEW: rolling-average knobs
    K_ROLL        = _get_int(preprocess_cfg, "k_roll", 6)
    MIN_PERIODS   = _get_int(preprocess_cfg, "min_periods", 1)
    SMOOTH_FEATURES = _get_bool(preprocess_cfg, "smooth_features", True)
    SMOOTH_COOCC    = _get_bool(preprocess_cfg, "smooth_coocc", False)
    WRITE_BOTH      = _get_bool(preprocess_cfg, "write_raw_tensors", True)

    def _coerce_list(raw, default=None):
        values = raw if raw is not None else default
        if values is None:
            return []
        if not isinstance(values, (list, tuple, set)):
            values = [values]
        return [str(item) for item in values if str(item).strip()]

    def _normalize_corr_modes(raw) -> list[str]:
        if isinstance(raw, (list, tuple, set)):
            items = list(raw)
        elif raw is None:
            items = []
        else:
            items = [raw]
        modes: list[str] = []
        for val in items:
            mode = "none"
            if isinstance(val, bool):
                mode = "log" if val else "none"
            elif isinstance(val, (int, float)):
                mode = "log" if val else "none"
            elif isinstance(val, str):
                norm_val = val.strip().lower()
                if norm_val in {"", "none", "level", "false", "0", "off", "null", "na"}:
                    mode = "none"
                elif norm_val in {"log", "logdiff", "log_diff", "dlog", "ln", "delta_log"}:
                    mode = "log"
                elif norm_val in {"raw", "diff", "delta", "value", "linear"}:
                    mode = "raw"
                else:
                    print(f"[correlation] unrecognised TOPK_CORR_USE_DIFF entry '{val}', defaulting to log.")
                    mode = "log"
            elif val:
                mode = "log"
            if mode not in modes:
                modes.append(mode)
        if not modes:
            modes.append("none")
        return modes

    Additional_keywords = _coerce_list(ctx.get("additional_keywords"), default=[])
    drop_keywords_cfg = ctx.get("drop_keywords", "drop_fully_inactive")
    forced_normals: set[str] = set()
    forced_indices: list[int] = []
    Remove_keywords = _coerce_list(ctx.get("remove_keywords"), default=[])
    Keyword_aliases_cfg = ctx.get("keyword_aliases", {})
    progress_enabled = bool(ctx.get("show_progress", False))

    BASE_DIR = ctx['base_dir']
    OUT_DIR  = os.path.join(BASE_DIR, "1_raw_data")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Save targets (under 1_raw_data/)
    OUT_TOP_KEYWORDS   = os.path.join(OUT_DIR, "top_keywords.csv")
    OUT_KEYWORDS_FIXED = os.path.join(OUT_DIR, "keywords_final.txt")
    OUT_MATS           = os.path.join(OUT_DIR, "stacked_matrices.npy")
    OUT_FEATS          = os.path.join(OUT_DIR, "stacked_features.npy")
    OUT_TIMES          = os.path.join(OUT_DIR, "feature_timestamps.npy")

    # Optional raw (pre-smoothing) dumps
    RAW_MATS           = os.path.join(OUT_DIR, "stacked_matrices_raw.npy")
    RAW_FEATS          = os.path.join(OUT_DIR, "stacked_features_raw.npy")
    OUT_CIT_FLOW_RAW   = os.path.join(OUT_DIR, "citation_flow_in_corpus.npy")
    OUT_CIT_FLOW_FRAC  = os.path.join(OUT_DIR, "citation_flow_fractional.npy")
    OUT_CIT_FLOW_SPLIT = os.path.join(OUT_DIR, "citation_flow_fractional_split.npy")
    OUT_CIT_FLOW_CSV   = os.path.join(OUT_DIR, "citation_flow_overall.csv")

    # =========================
    # Normalization
    # =========================
    def norm(s: str) -> str:
        """Lower, NFKC, unify hyphens/spaces, keep [a-z0-9 ] only, collapse spaces."""
        if not isinstance(s, str): 
            return ""
        s = unicodedata.normalize("NFKC", s).lower()
        s = re.sub(r"[\u2010-\u2015\u2212\-_]+", " ", s)
        s = re.sub(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000]", " ", s)
        s = re.sub(r"[^a-z0-9 ]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def normalize_kw(value) -> str:
        """Apply the configured keyword normalisation pipeline (or basic strip)."""
        if value is None:
            return ""
        return norm(value) if NORMALIZE_KEYWORDS else str(value).strip()

    alias_variants_map: dict[str, list[str]] = {}
    variant_to_canonical: dict[str, str] = {}
    if Keyword_aliases_cfg and not isinstance(Keyword_aliases_cfg, dict):
        print("Warning: keyword_aliases should be a mapping of canonical keys to variant lists. Ignoring misconfigured value.")
    if isinstance(Keyword_aliases_cfg, dict):
        for canonical_raw, variants_raw in Keyword_aliases_cfg.items():
            canonical_norm = normalize_kw(canonical_raw)
            if not canonical_norm:
                continue
            variants_iter = variants_raw if isinstance(variants_raw, (list, tuple, set)) else [variants_raw]
            collected: list[str] = []
            for variant in variants_iter:
                variant_norm = normalize_kw(variant)
                if variant_norm:
                    collected.append(variant_norm)
            collected.append(canonical_norm)

            unique_variants: list[str] = []
            seen_variants: set[str] = set()
            for alias in collected:
                if alias in seen_variants:
                    continue
                seen_variants.add(alias)
                unique_variants.append(alias)

            if not unique_variants:
                continue

            alias_variants_map[canonical_norm] = unique_variants
            for alias in unique_variants:
                existing = variant_to_canonical.get(alias)
                if existing and existing != canonical_norm:
                    print(
                        f"Warning: alias '{alias}' mapped to both '{existing}' and '{canonical_norm}'. "
                        f"Keeping '{existing}' and ignoring the conflicting entry."
                    )
                    continue
                variant_to_canonical[alias] = canonical_norm

    ban: set[str] = set()
    if Remove_keywords:
        for item in Remove_keywords:
            normalized = normalize_kw(item)
            if normalized:
                ban.add(normalized)

    # =========================
    # Rolling helper (causal, trailing)
    # =========================
    def kmonth_roll(arr: np.ndarray, k: int, min_periods: int = 1, axis: int = 0) -> np.ndarray:
        """
        Trailing window mean of size k along `axis`, using shorter window at start if needed.
        Example: with min_periods=1, t=2 uses average over indices [0..2].
        """
        if k is None or k <= 1:
            return arr.astype(np.float32, copy=False)
        x = np.asarray(arr, dtype=np.float64)
        x = np.moveaxis(x, axis, 0)           # put time first -> (T, ...)
        T = x.shape[0]
        cs = np.cumsum(x, axis=0)
        out = np.empty_like(x, dtype=np.float64)
        for t in range(T):
            s = max(0, t - k + 1)
            total = cs[t] - (cs[s-1] if s > 0 else 0.0)
            count = t - s + 1
            out[t] = np.nan if count < min_periods else total / count
        return np.moveaxis(out, 0, axis).astype(np.float32)

    # =========================
    # 1) Load keyword counts → fixed keyword list
    # =========================
    dfk = pd.read_csv(PATH_KEYWORD_COUNTS, low_memory=False)
    if "Keyword" not in dfk.columns:
        dfk = pd.read_csv(PATH_KEYWORD_COUNTS, header=None, names=["Keyword"])
    if "Count" not in dfk.columns:
        dfk["Count"] = 1
    dfk["Keyword"] = dfk["Keyword"].map(normalize_kw)
    dfk["Count"] = pd.to_numeric(dfk["Count"], errors="coerce")
    dfk = dfk[(dfk["Keyword"] != "") & dfk["Count"].notna()].copy()
    dfk["Count"] = dfk["Count"].astype(int)

    if variant_to_canonical:
        dfk["Keyword"] = dfk["Keyword"].map(lambda k: variant_to_canonical.get(k, k))

    dfk = dfk.groupby("Keyword", as_index=False)["Count"].sum()

    keyword_debug = False

    dist = Counter(dfk["Count"])
    if keyword_debug:
        for c, n in sorted(dist.items()):
            print(f"Count {c}: {n} keywords")

    top = dfk.sort_values("Count", ascending=False).head(TOP_KEYWORDS)["Keyword"].tolist()

    seen, keywords = set(), []
    for kw in top:
        k = normalize_kw(kw)
        if k and k not in seen:
            seen.add(k); keywords.append(k)

    if alias_variants_map:
        missing_canonicals = [canon for canon in alias_variants_map if canon not in seen and canon not in ban]
        if missing_canonicals:
            keywords.extend(missing_canonicals)
            seen.update(missing_canonicals)
            msg = f"Force-included {len(missing_canonicals)} canonical keyword(s) from alias map"
            if keyword_debug:
                msg += f": {missing_canonicals}"
            print(msg)

    if variant_to_canonical:
        filtered_keywords: list[str] = []
        collapsed_aliases: list[str] = []
        for kw in keywords:
            canonical = variant_to_canonical.get(kw)
            if canonical and canonical != kw:
                collapsed_aliases.append(kw)
                continue
            filtered_keywords.append(kw)
        if collapsed_aliases:
            collapsed_unique = sorted(set(collapsed_aliases))
            msg = f"Collapsed {len(collapsed_unique)} alias keyword(s) into canonical forms"
            if keyword_debug:
                msg += f": {collapsed_unique}"
            print(msg)
        keywords = filtered_keywords
        seen = set(keywords)

    if Additional_keywords:
        added = []
        for kw in Additional_keywords:
            k = normalize_kw(kw)
            if not k:
                continue
            canonical_k = variant_to_canonical.get(k, k)
            forced_normals.add(canonical_k)
            if canonical_k not in seen:
                seen.add(canonical_k)
                keywords.append(canonical_k)
                added.append(canonical_k)
        if added:
            msg = f"Force-included {len(added)} additional keyword(s)"
            if keyword_debug:
                msg += f": {added}"
            print(msg)

    if ban:
        before = len(keywords)
        keywords = [k for k in keywords if k not in ban]
        if len(keywords) != before:
            removed = before - len(keywords)
            msg = f"Removed {removed} keyword(s)"
            if keyword_debug:
                msg += f": {sorted(ban)}"
            print(msg)
        seen = set(keywords)

    if forced_normals:
        forced_normals = {variant_to_canonical.get(name, normalize_kw(name)) for name in forced_normals}
        forced_indices = [idx for idx, name in enumerate(keywords) if name in forced_normals]

    pd.Series(keywords).to_csv(OUT_TOP_KEYWORDS, index=False, header=False)
    N = len(keywords)
    print(f"\nUsing {N} unique keywords (saved -> {OUT_TOP_KEYWORDS})")

    if reuse_graph:
        _announce("Skipping graph build; reusing existing tensors.")
    else:
        _announce("Cell 3: loading papers and selecting months")
        # =========================
        # 2) Load papers and select months
        # =========================
        df_papers = pd.read_csv(PATH_PAPERS, engine="c", low_memory=False)
        print("number of papers to scan:", len(df_papers))

        scan_filter_cfg = ctx.get("scan_filter_cfg") or {}
        scan_enabled = bool(scan_filter_cfg.get("enabled"))
        if scan_enabled:
            import pathlib

            scan_column = str(scan_filter_cfg.get("column") or "").strip()
            accepted_values_raw = scan_filter_cfg.get("accepted_values") or []
            accepted_values = {
            str(val).strip().lower() for val in accepted_values_raw if str(val).strip()
            }
            fallback_all = bool(scan_filter_cfg.get("fallback_to_all", True))
            raw_topics_path = scan_filter_cfg.get("topics_path")
            topics_path = pathlib.Path(raw_topics_path) if raw_topics_path else None
            flagged_topic_ids: set[str] = set()
            if topics_path and topics_path.exists():
                try:
                    df_topic_meta = pd.read_csv(topics_path)
                except Exception as exc:
                    print(f"[graph] Warning: unable to read scan topics metadata ({topics_path}): {exc}")
                else:
                    missing_cols = [col for col in [scan_column, "id"] if col not in df_topic_meta.columns]
                    if missing_cols:
                        print(f"[graph] Warning: scan metadata missing required columns: {missing_cols}")
                    else:
                        mask = df_topic_meta[scan_column].astype(str).str.strip().str.lower().isin(accepted_values)
                        flagged_topic_ids = set(
                            df_topic_meta.loc[mask, "id"].dropna().astype(str).str.strip()
                        )
            else:
                if raw_topics_path:
                    print(f"[graph] Warning: scan topics file not found at {raw_topics_path}")

            if flagged_topic_ids and "primary_topic.id" in df_papers.columns:
                before = len(df_papers)
                df_papers = df_papers[df_papers["primary_topic.id"].isin(flagged_topic_ids)].copy()
                after = len(df_papers)
                print(
                    f"[graph] Scan filter retained {after} / {before} papers "
                    f"across {len(flagged_topic_ids)} flagged topics."
                )
                if after == 0:
                    raise RuntimeError("Scan filter removed all papers. Check the scan column values.")
            elif not flagged_topic_ids:
                message = "[graph] Scan filter enabled but no topics were flagged; "
                if fallback_all:
                    message += "keeping all papers."
                    print(message)
                else:
                    message += "aborting per configuration."
                    raise RuntimeError(message)
            else:
                print("[graph] primary_topic.id is missing from papers; skipping scan filter.")
        months_all = sorted(df_papers["year_month"].dropna().astype(str).unique())
        if START_YM or END_YM:
            months = [
                m
                for m in months_all
                if (START_YM is None or m >= START_YM) and (END_YM is None or m <= END_YM)
            ]
        else:
            months = months_all if NUMBER_MONTHS is None else months_all[-NUMBER_MONTHS:]
        T = len(months)
        print(f"months to scan: {T} (from {months[0] if T else 'N/A'} to {months[-1] if T else 'N/A'})")

        TEXT_COL = "Abstract_norm" if (USE_NORMALIZED_TEXT and "Abstract_norm" in df_papers.columns) else "Abstract"
        if TEXT_COL != "Abstract_norm" and APPLY_NORM_IF_NEEDED:
            df_papers["Abstract_norm"] = df_papers["Abstract"].map(norm)
            TEXT_COL = "Abstract_norm"

        # ensure cited_by_count exists & numeric
        if "cited_by_count" not in df_papers.columns:
            df_papers["cited_by_count"] = 0
        df_papers["cited_by_count"] = (
            pd.to_numeric(df_papers["cited_by_count"], errors="coerce").fillna(0).astype(np.int64)
        )

        if OCCURRENCE_MODE not in {"doc", "token"}:
            raise ValueError("OCCURRENCE_MODE must be 'doc' or 'token'")
        if CITATION_WEIGHTING not in {"per_doc", "per_token"}:
            raise ValueError("CITATION_WEIGHTING must be 'per_doc' or 'per_token'")
        if CITATION_SOURCE not in {"snapshot", "causal_flow", "causal_cumulative"}:
            print(f"Warning: unknown citation_source='{CITATION_SOURCE}', falling back to 'snapshot'.")
            CITATION_SOURCE = "snapshot"
        if CITATION_FEATURE_VARIANT not in {"base", "frac", "frac_split"}:
            print(
                f"Warning: unknown citation_feature_variant='{CITATION_FEATURE_VARIANT}', "
                "falling back to 'base'."
            )
            CITATION_FEATURE_VARIANT = "base"
        if CITATION_SOURCE == "snapshot" and CITATION_FEATURE_VARIANT != "base":
            print("[citation] snapshot source ignores citation_feature_variant; using 'base'.")
            CITATION_FEATURE_VARIANT = "base"

        need_cols = ["year_month", TEXT_COL, "cited_by_count", "id"]
        if "referenced_works" in df_papers.columns:
            need_cols.append("referenced_works")
        df_papers = df_papers.loc[:, [c for c in need_cols if c in df_papers.columns]].copy()

        sub = df_papers[df_papers["year_month"].isin(months)].copy()
        has_referenced_works = "referenced_works" in sub.columns
        if CITATION_SOURCE != "snapshot" and not has_referenced_works:
            print(
                "[citation] citation_source requires 'referenced_works' in papers.csv, "
                "falling back to snapshot citations."
            )
            CITATION_SOURCE = "snapshot"

        group_cols = [TEXT_COL, "cited_by_count"]
        if CITATION_SOURCE != "snapshot":
            group_cols = ["id", TEXT_COL, "cited_by_count", "referenced_works"]

        groups = {
            ym: g.reindex(columns=group_cols).fillna({"cited_by_count": 0, "referenced_works": "[]"})
            for ym, g in sub.groupby("year_month", sort=False)
        }

        _announce("Cell 4: building Aho-Corasick automaton and scanning abstracts")
        # =========================
        # 3) Aho–Corasick automaton
        # =========================
        keyword_variants_for_automaton: list[list[str]] = []
        for kw in keywords:
            variants = alias_variants_map.get(kw)
            if not variants:
                variants = [kw]
            keyword_variants_for_automaton.append(variants)

        A = ahocorasick.Automaton()
        for i, variants in enumerate(keyword_variants_for_automaton):
            for variant in variants:
                A.add_word(variant, (i, len(variant)))
        A.make_automaton()

        def find_kw_ids(text: str) -> list[int]:
            """Unique keyword IDs present in text (document-level presence)."""
            if not isinstance(text, str):
                return []
            ids = set()
            for end, payload in A.iter(text):
                kid, L = payload
                start = end - L + 1
                before_ok = (start == 0) or (text[start - 1] == " ")
                after_ok = (end + 1 == len(text)) or (text[end + 1] == " ")
                if before_ok and after_ok:
                    ids.add(kid)
            return list(ids)

        def find_kw_counts(text: str) -> dict[int, int]:
            """Token-level counts per keyword ID."""
            if not isinstance(text, str):
                return {}
            counts = defaultdict(int)
            for end, payload in A.iter(text):
                kid, L = payload
                start = end - L + 1
                before_ok = (start == 0) or (text[start - 1] == " ")
                after_ok = (end + 1 == len(text)) or (text[end + 1] == " ")
                if before_ok and after_ok:
                    counts[kid] += 1
            return counts

        def _norm_work_id(v: Any) -> str:
            if v is None:
                return ""
            s = str(v).strip().strip('"').strip("'")
            if not s:
                return ""
            if s.startswith("http://openalex.org/"):
                s = "https://openalex.org/" + s.split("http://openalex.org/", 1)[1]
            if s.startswith("https://openalex.org/W"):
                return s
            if re.fullmatch(r"W\d+", s):
                return f"https://openalex.org/{s}"
            return ""

        def _parse_referenced_works(raw: Any) -> list[str]:
            if raw is None:
                return []
            if isinstance(raw, float) and np.isnan(raw):
                return []
            vals: list[Any]
            if isinstance(raw, (list, tuple, set)):
                vals = list(raw)
            else:
                txt = str(raw).strip()
                if txt in {"", "[]", "None", "none", "nan", "NaN", "null"}:
                    return []
                parsed = None
                try:
                    parsed = literal_eval(txt)
                except Exception:
                    try:
                        parsed = json.loads(txt)
                    except Exception:
                        parsed = [tok.strip() for tok in re.split(r"[;,|]", txt) if tok.strip()]
                vals = list(parsed) if isinstance(parsed, (list, tuple, set)) else [parsed]
            out = []
            for item in vals:
                norm_id = _norm_work_id(item)
                if norm_id:
                    out.append(norm_id)
            return out

        _announce("Cell 5: computing monthly tensors")
        # =========================
        # 4) Compute monthly tensors
        # =========================
        cooc = np.zeros((N, N, T), dtype=np.int32)  # co-occurrence per month (doc-level)
        feat = np.zeros((T, N, 3), dtype=np.float64)  # [citation_signal, occurrences, cooc_sum]
        times = list(months)

        citation_flow = None
        citation_flow_frac = None
        citation_flow_frac_split = None
        doc_kw_ids: dict[str, list[int]] = {}
        doc_kw_counts: dict[str, dict[int, int]] = {}
        doc_month_idx: dict[str, int] = {}
        if CITATION_SOURCE != "snapshot":
            citation_flow = np.zeros((T, N), dtype=np.int64)
            citation_flow_frac = np.zeros((T, N), dtype=np.float64)
            citation_flow_frac_split = np.zeros((T, N), dtype=np.float64)

        for t, ym in enumerate(tqdm(months, desc="months", disable=not progress_enabled)):
            g = groups.get(ym)
            if g is None or g.empty:
                continue

            # kid -> [occurrences, cited_sum]
            feat_tmp = defaultdict(lambda: [0, 0])

            for row in g.itertuples(index=False, name=None):
                if CITATION_SOURCE == "snapshot":
                    abs_text, cited = row
                    paper_id = None
                    refs_raw = None
                else:
                    paper_id, abs_text, cited, refs_raw = row
                cited_i = int(cited) if pd.notna(cited) else 0

                if OCCURRENCE_MODE == "doc":
                    ids = find_kw_ids(abs_text)
                    counts = None
                    if not ids:
                        continue

                    # occurrences: +1 per abstract
                    for i in ids:
                        feat_tmp[i][0] += 1

                    # snapshot citations: add once per keyword present
                    if CITATION_SOURCE == "snapshot":
                        for i in ids:
                            feat_tmp[i][1] += cited_i

                    # co-occurrence: document-level (count each pair once)
                    for i, j in combinations(ids, 2):
                        cooc[i, j, t] += 1
                        cooc[j, i, t] += 1

                else:  # OCCURRENCE_MODE == "token"
                    counts = find_kw_counts(abs_text)  # {kid: token_count}
                    if not counts:
                        continue

                    # occurrences: +token_count
                    for i, tok in counts.items():
                        feat_tmp[i][0] += tok

                    # snapshot citations: per-doc or per-token
                    if CITATION_SOURCE == "snapshot":
                        if CITATION_WEIGHTING == "per_doc":
                            for i in counts:
                                feat_tmp[i][1] += cited_i
                        else:  # per_token
                            for i, tok in counts.items():
                                feat_tmp[i][1] += cited_i * tok

                    # co-occurrence: still document-level (pairs counted once)
                    ids = list(counts.keys())
                    for i, j in combinations(ids, 2):
                        cooc[i, j, t] += 1
                        cooc[j, i, t] += 1

                if CITATION_SOURCE != "snapshot":
                    pid = _norm_work_id(paper_id)
                    if pid:
                        doc_month_idx[pid] = t
                        doc_kw_ids[pid] = list(ids)
                        if counts:
                            doc_kw_counts[pid] = dict(counts)

                        refs = _parse_referenced_works(refs_raw)
                        inv_refs = (1.0 / float(len(refs))) if refs else 0.0
                        for ref in refs:
                            ref_month = doc_month_idx.get(ref)
                            if ref_month is None or ref_month > t:
                                continue
                            ref_ids = doc_kw_ids.get(ref)
                            if not ref_ids:
                                continue
                            if CITATION_WEIGHTING == "per_token":
                                ref_counts = doc_kw_counts.get(ref)
                                if ref_counts:
                                    for rid, tok in ref_counts.items():
                                        citation_flow[t, rid] += int(tok)
                                else:
                                    for rid in ref_ids:
                                        citation_flow[t, rid] += 1
                            else:
                                for rid in ref_ids:
                                    citation_flow[t, rid] += 1

                            # Fractional variants (independent from per_doc/per_token):
                            # 1) 1/|R(p)| * 1[c in K(r)]
                            if citation_flow_frac is not None and inv_refs > 0.0:
                                for rid in ref_ids:
                                    citation_flow_frac[t, rid] += inv_refs

                            # 2) 1/|R(p)| * 1/|K(r)| * 1[c in K(r)]
                            if citation_flow_frac_split is not None and inv_refs > 0.0:
                                k_ref = len(ref_ids)
                                if k_ref > 0:
                                    split_w = inv_refs / float(k_ref)
                                    for rid in ref_ids:
                                        citation_flow_frac_split[t, rid] += split_w

            # copy to tensor
            for i, (occ, cited_sum) in feat_tmp.items():
                feat[t, i, 1] = occ
                if CITATION_SOURCE == "snapshot":
                    feat[t, i, 0] = cited_sum

            # cooccurrence_sum per node (sum over partners j)
            feat[t, :, 2] = cooc[:, :, t].sum(axis=1)

        if CITATION_SOURCE != "snapshot" and citation_flow is not None:
            flow_selected = citation_flow
            if CITATION_FEATURE_VARIANT == "frac":
                if citation_flow_frac is not None:
                    flow_selected = citation_flow_frac
            elif CITATION_FEATURE_VARIANT == "frac_split":
                if citation_flow_frac_split is not None:
                    flow_selected = citation_flow_frac_split

            if CITATION_SOURCE == "causal_cumulative":
                feat[:, :, 0] = np.cumsum(flow_selected, axis=0)
            else:  # causal_flow
                feat[:, :, 0] = flow_selected

        # =========================
        # 4.5) Apply k-month trailing averages
        # =========================
        feat_raw = feat.copy()
        cooc_raw = cooc.copy()

        if SMOOTH_FEATURES and K_ROLL and K_ROLL > 1:
            feat = kmonth_roll(feat, k=K_ROLL, min_periods=MIN_PERIODS, axis=0)

        if SMOOTH_COOCC and K_ROLL and K_ROLL > 1:
            cooc = kmonth_roll(cooc, k=K_ROLL, min_periods=MIN_PERIODS, axis=-1)

        # =========================
        # 5) Save outputs
        # =========================
        if WRITE_BOTH:
            np.save(RAW_FEATS, feat_raw)  # int/float, pre-roll
            np.save(RAW_MATS, cooc_raw)  # int32

        if CITATION_SOURCE != "snapshot" and citation_flow is not None:
            np.save(OUT_CIT_FLOW_RAW, citation_flow.astype(np.float32))
            if citation_flow_frac is not None:
                np.save(OUT_CIT_FLOW_FRAC, citation_flow_frac.astype(np.float32))
            if citation_flow_frac_split is not None:
                np.save(OUT_CIT_FLOW_SPLIT, citation_flow_frac_split.astype(np.float32))

            overall = pd.DataFrame({
                "year_month": [str(x) for x in times],
                "flow_in_corpus_mean": np.nanmean(citation_flow, axis=1),
                "flow_fractional_mean": (
                    np.nanmean(citation_flow_frac, axis=1)
                    if citation_flow_frac is not None else np.nan
                ),
                "flow_fractional_split_mean": (
                    np.nanmean(citation_flow_frac_split, axis=1)
                    if citation_flow_frac_split is not None else np.nan
                ),
            })
            overall.to_csv(OUT_CIT_FLOW_CSV, index=False, encoding="utf-8")
        else:
            for stale in [OUT_CIT_FLOW_RAW, OUT_CIT_FLOW_FRAC, OUT_CIT_FLOW_SPLIT, OUT_CIT_FLOW_CSV]:
                try:
                    if os.path.exists(stale):
                        os.remove(stale)
                except Exception:
                    pass

        np.save(OUT_MATS, cooc.astype(np.float32) if (SMOOTH_COOCC and K_ROLL and K_ROLL > 1) else cooc)
        np.save(OUT_FEATS, feat.astype(np.float32))
        np.save(OUT_TIMES, np.array(times))
        pd.Series(keywords).to_csv(OUT_KEYWORDS_FIXED, index=False, header=False)

        print("\nSaved to:", OUT_DIR)
        print(f" - {OUT_MATS}   shape={cooc.shape}   (N, N, T)   dtype={cooc.dtype}")
        print(f" - {OUT_FEATS}  shape={feat.shape}   (T, N, 3)   dtype={feat.dtype}   (K_ROLL={K_ROLL}, min={MIN_PERIODS})")
        print(f" - {OUT_TIMES}  len={len(times)}")
        print(f" - {OUT_KEYWORDS_FIXED}  (exact keyword list used)")
        if CITATION_SOURCE != "snapshot":
            print(f" - {OUT_CIT_FLOW_RAW}, {OUT_CIT_FLOW_FRAC}, {OUT_CIT_FLOW_SPLIT}")
            print(f" - {OUT_CIT_FLOW_CSV}")
        if WRITE_BOTH:
            print(f" - {RAW_FEATS}, {RAW_MATS} (pre-roll)")
        print(
            f"Occurrence mode = {OCCURRENCE_MODE}, citation weighting = {CITATION_WEIGHTING}, "
            f"citation source = {CITATION_SOURCE}, citation feature variant = {CITATION_FEATURE_VARIANT}"
        )

        _announce("Cell 6: filtering inactive keywords and building active tensors")

        BASE_DIR_PATH = Path(BASE_DIR)
        RAW_DIR = BASE_DIR_PATH / "1_raw_data"
        ACTIVE_DIR = BASE_DIR_PATH / "2_active_data"
        ACTIVE_DIR.mkdir(parents=True, exist_ok=True)

        KW_PATH = RAW_DIR / "keywords_final.txt"
        FEATS_PATH = RAW_DIR / "stacked_features.npy"
        MATS_PATH = RAW_DIR / "stacked_matrices.npy"

        raw_node_features = np.load(FEATS_PATH)  # (T, N, F)
        all_matrices = np.load(MATS_PATH)  # (N, N, T)
        Tn, Nn, F = raw_node_features.shape
        assert all_matrices.shape == (Nn, Nn, Tn), f"{all_matrices.shape=} != {(Nn, Nn, Tn)}"

        keywords_loaded = pd.read_csv(KW_PATH, header=None, names=["kw"], dtype=str, engine="python")["kw"].astype(str).to_list()
        assert len(keywords_loaded) == Nn, f"len(keywords)={len(keywords_loaded)} != N={Nn}. Check {KW_PATH}."

        cited_by = raw_node_features[:, :, 0]  # (T, N)
        oc_freq = raw_node_features[:, :, 1]  # (T, N)

        never_cited_nodes = np.flatnonzero(np.all(cited_by == 0, axis=0))
        never_occur_nodes = np.flatnonzero(np.all(oc_freq == 0, axis=0))

        deg_total = all_matrices.sum(axis=1).sum(axis=-1)  # (N,)
        never_connected_nodes = np.flatnonzero(deg_total == 0)

        fully_inactive_nodes = np.intersect1d(
            never_cited_nodes, np.intersect1d(never_occur_nodes, never_connected_nodes)
        )

        drop_modes_cfg = drop_keywords_cfg
        if isinstance(drop_modes_cfg, str):
            drop_modes = [drop_modes_cfg] if drop_modes_cfg else []
        elif isinstance(drop_modes_cfg, (list, tuple, set)):
            drop_modes = [str(m) for m in drop_modes_cfg if str(m).strip()]
        else:
            drop_modes = []
        drop_modes = [m.lower().strip() for m in drop_modes]

        idx_all = np.arange(Nn)
        never_cited_bool = np.isin(idx_all, never_cited_nodes)
        never_occur_bool = np.isin(idx_all, never_occur_nodes)
        fully_inactive_bool = np.isin(idx_all, fully_inactive_nodes)

        drop_mask = np.zeros(Nn, dtype=bool)
        if not drop_modes:
            drop_mask |= fully_inactive_bool
        else:
            for mode in drop_modes:
                if mode in ("", "none", "keep"):
                    continue
                if mode == "drop_never_cited":
                    drop_mask |= never_cited_bool
                elif mode == "drop_never_occured":
                    drop_mask |= never_occur_bool
                elif mode == "drop_never_cited_or_occured":
                    drop_mask |= (never_cited_bool | never_occur_bool)
                elif mode == "drop_fully_inactive":
                    drop_mask |= fully_inactive_bool

        active_mask = ~drop_mask
        active_node_indices = np.flatnonzero(active_mask)

        raw_node_features_active = raw_node_features[:, active_mask, :]
        all_matrices_active = all_matrices[active_mask][:, active_mask, :]

        np.save(ACTIVE_DIR / "stacked_features_active.npy", raw_node_features_active)
        np.save(ACTIVE_DIR / "stacked_matrices_active.npy", all_matrices_active)
        pd.Series([keywords_loaded[i] for i in active_node_indices]).to_csv(
            ACTIVE_DIR / "keywords_active.txt", index=False, header=False
        )

        print("Filtered shapes (ACTIVE):")
        print("  stacked_features_active:", raw_node_features_active.shape)
        print("  stacked_matrices_active:", all_matrices_active.shape)

        status_cols = {
            "never_cited": [keywords_loaded[i] for i in never_cited_nodes],
            "never_occur": [keywords_loaded[i] for i in never_occur_nodes],
            "never_connected": [keywords_loaded[i] for i in never_connected_nodes],
            "fully_inactive": [keywords_loaded[i] for i in fully_inactive_nodes],
            "active": [keywords_loaded[i] for i in active_node_indices],
        }
        max_len = max(len(v) for v in status_cols.values()) if status_cols else 0
        for k, v in status_cols.items():
            if len(v) < max_len:
                status_cols[k] = v + [""] * (max_len - len(v))
        pd.DataFrame(status_cols).to_csv(ACTIVE_DIR / "node_status_columns.csv", index=False)
        print(f"=== Node status (columns) ===")
        print(f"  rows={max_len} cols={len(status_cols)} -> {ACTIVE_DIR / 'node_status_columns.csv'}")

        CORR_DIR = BASE_DIR_PATH / "3_corrected_data"
        PLOT_DIR = CORR_DIR / "plots"
        CORR_DIR.mkdir(parents=True, exist_ok=True)
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        # Inputs
        MATS_IN = ACTIVE_DIR / "stacked_matrices_active.npy"     # (N, N, T)
        FEATS_IN = ACTIVE_DIR / "stacked_features_active.npy"    # (T, N, F)
        TIME_PATH = RAW_DIR / "feature_timestamps.npy"           # (T,)

        # Optional feature-name files
        FEAT_JSON = RAW_DIR / "feature_names.json"
        FEAT_NPY = RAW_DIR / "feature_names.npy"
        FEAT_TXT = RAW_DIR / "feature_names.txt"

        # --------------------------
        # Preview + correction config
        # --------------------------
        emergence_mode_cfg = _get(preview_cfg, "emergence_mode", _get(cfg_defaults_ctx, "EMERGENCE_MODE", "raw"))
        EMERGENCE_MODE = str(emergence_mode_cfg or "raw").lower()
        epsilon_cfg = _get_float(preview_cfg, "epsilon", None)
        if epsilon_cfg is None:
            epsilon_cfg = _get_float(cfg_defaults_ctx, "EPSILON", 1e-8)
        EPSILON = epsilon_cfg
        fw_cfg = _get(preview_cfg, "feature_weights", None)
        if fw_cfg is None:
            fw_cfg = _get(cfg_defaults_ctx, "FW", [0.0, 0.5, 0.5])
        normalize_fw_cfg = _get_bool(preview_cfg, "normalize_feature_weights", None)
        if normalize_fw_cfg is None:
            normalize_fw_cfg = _get_bool(cfg_defaults_ctx, "NORMALIZE_FW", True)
        FW = _normalize_fw_weights(np.array(fw_cfg, dtype=float), enabled=normalize_fw_cfg, where="preview")
        DPI = _get_int(preview_cfg, "dpi", _get_int(cfg_defaults_ctx, "DPI", 180))
        PLOT_PREVIEW = _get_bool(preview_cfg, "plot_preview", True)

        # --------------------------
        # Load
        # --------------------------
        A = np.load(MATS_IN)                                    # (N, N, T)
        X = np.load(FEATS_IN)                                   # (T, N, F)
        ts = np.load(TIME_PATH, allow_pickle=True).astype(str)  # (T,)
        dates = pd.to_datetime(ts, errors="coerce").to_period("M").to_timestamp(how="start")

        # Basic shape checks
        N, N2, Tm = A.shape
        Tf, Nf, F = X.shape
        assert N == N2, "Adjacency must be square per slice."
        assert Tm == Tf == len(dates), "Time mismatch between matrices, features, and timestamps."
        assert Nf > 0 and F > 0, "Feature tensor looks empty."

        # --------------------------
        # Feature names
        # --------------------------
        def _load_feature_names(F_count: int) -> list[str]:
            names = None
            cfg_names = _get(preview_cfg, "feature_names", None)
            if cfg_names is None:
                cfg_names = _get(cfg_defaults_ctx, "FEATURE_NAMES_DEFAULT", None)
            if cfg_names:
                names = [str(x) for x in cfg_names]
            elif FEAT_JSON.exists():
                with open(FEAT_JSON, "r", encoding="utf-8") as f:
                    names = list(json.load(f))
            elif FEAT_NPY.exists():
                names = list(np.load(FEAT_NPY, allow_pickle=True))
            elif FEAT_TXT.exists():
                names = [ln.strip() for ln in FEAT_TXT.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not names:
                names = ["citation", "cooc", "occ"] if F_count == 3 else [f"feat{i}" for i in range(F_count)]
            if len(names) != F_count:
                names = (names + [f"feat{i}" for i in range(len(names), F_count)])[:F_count]
            return names

        FEATURE_NAMES = _load_feature_names(F)

        # --------------------------
        # Tail-drop correction helpers
        # --------------------------
        ROLL_WIN_DETECT = _get_int(tail_cfg, "roll_win_detect", 6)
        FIT_MONTHS_MIN = _get_int(tail_cfg, "fit_months_min", 24)
        FIT_MONTHS_MAX = _get_int(tail_cfg, "fit_months_max", 84)
        DROP_TOL_PCT = _get_float(tail_cfg, "drop_tol_pct", 0.05)
        DECLINE_RUN = _get_int(tail_cfg, "decline_run", 3)
        SMOOTH_COEFF_W = _get_int(tail_cfg, "smooth_coeff_window", 3)
        CLIP_MIN = _get_float(tail_cfg, "clip_min", 0.5)
        clip_max_val = _get(tail_cfg, "clip_max", None)
        CLIP_MAX = None if clip_max_val in (None, "", "null") else _get_float(tail_cfg, "clip_max", None)
        EPS = _get_float(tail_cfg, "epsilon", 1e-8)
        FEATURE_COEFF_SOURCE = str(
            _get(tail_cfg, "feature_coeff_source", "active")
        ).strip().lower()
        CORRECT_CUMULATIVE_VIA_INCREMENTS = _get_bool(
            tail_cfg, "correct_cumulative_via_increments", True
        )
        cumulative_tokens_cfg = _get(
            tail_cfg, "cumulative_feature_tokens", ["xcum", "cumulative"]
        )
        if isinstance(cumulative_tokens_cfg, str):
            cumulative_tokens = [cumulative_tokens_cfg]
        elif isinstance(cumulative_tokens_cfg, (list, tuple, set)):
            cumulative_tokens = [str(x) for x in cumulative_tokens_cfg]
        else:
            cumulative_tokens = ["xcum", "cumulative"]
        cumulative_tokens = [
            tok.lower().strip() for tok in cumulative_tokens if str(tok).strip()
        ]

        def rolling_mean(x, win, minp=1):
            return pd.Series(x, index=dates).rolling(win, min_periods=minp).mean().values

        def _first_sustained_break(y, t_peak, drop_tol_pct=DROP_TOL_PCT, run=DECLINE_RUN):
            peak_val = y[t_peak]
            thr = (1.0 - drop_tol_pct) * peak_val
            for t in range(t_peak + 1, len(y) - run + 1):
                if np.all(y[t:t + run] <= thr):
                    return t
            return t_peak + 1

        def _fit_line_with_huber_weights(idx, y_log):
            b1, b0 = np.polyfit(idx, y_log, 1)
            resid = y_log - (b1 * idx + b0)
            mad = np.median(np.abs(resid)) + 1e-9
            w = np.clip(1.0 / np.maximum(1.0, np.abs(resid) / (1.345 * mad)), 0.2, 1.0)
            W = np.sqrt(w)
            b1, b0 = np.polyfit(idx * W, y_log * W, 1)
            return b1, b0

        def _coeff_series_from_avg(y_avg):
            Tn = y_avg.shape[0]
            if not np.isfinite(y_avg).any() or np.nanmax(y_avg) <= 0:
                return np.ones(Tn, dtype=float), 0, (0.0, 0.0)

            y_log = np.log1p(y_avg + EPS)
            y_s_log = rolling_mean(y_log, ROLL_WIN_DETECT, minp=1)
            y_s = np.expm1(y_s_log)

            t_peak = int(np.nanargmax(y_s_log))
            t_break = _first_sustained_break(y_avg, t_peak)

            if t_break >= Tn:
                c = np.ones(Tn, dtype=float)
                c[:t_break] = 1.0
                return c, t_break, (0.0, y_log[-1])

            m = min(max(FIT_MONTHS_MIN, t_break), FIT_MONTHS_MAX)
            t0 = max(0, (t_break - 1) - m + 1)
            if t_break - t0 < 2:
                t0 = max(0, t_break - 2)

            idx_fit = np.arange(t0, t_break, dtype=float)
            y_fit = y_log[t0:t_break]
            b1, b0 = _fit_line_with_huber_weights(idx_fit, y_fit)

            # anchor
            t_anchor = float(t_break - 1)
            y_anchor = y_log[t_break - 1]
            b0 = y_anchor - b1 * t_anchor

            idx_future_i = np.arange(t_break, Tn, dtype=int)
            idx_future_f = idx_future_i.astype(float)

            y_hat_log = np.empty(Tn)
            y_hat_log[:] = np.nan
            if idx_future_i.size > 0:
                y_hat_log[idx_future_i] = b1 * idx_future_f + b0

            c = np.ones(Tn, dtype=float)
            if idx_future_i.size > 0:
                target = np.expm1(y_hat_log[idx_future_i])
                obs = y_avg[idx_future_i]
                safeobs = np.where(obs <= 0, EPS, obs)
                c_tail = target / safeobs
                c_tail = np.maximum.accumulate(c_tail)
                c[idx_future_i] = c_tail

            c = pd.Series(c, index=dates).rolling(SMOOTH_COEFF_W, min_periods=1).mean().values
            c = np.maximum(c, CLIP_MIN)
            if CLIP_MAX is not None:
                c = np.minimum(c, CLIP_MAX)
            c[:t_break] = 1.0
            return c, t_break, (b1, b0)

        def _is_cumulative_feature(feature_name: str) -> bool:
            nm = str(feature_name or "").lower()
            return any(tok in nm for tok in cumulative_tokens)

        # Feature-coefficient source:
        # - "active": 2_active_data/stacked_features_active.npy (post-roll)
        # - "raw_pre_roll": 1_raw_data/stacked_features_raw.npy filtered by active_mask
        X_coeff_source = X
        if FEATURE_COEFF_SOURCE in {"raw_pre_roll", "raw", "original"}:
            feats_raw_pre = RAW_DIR / "stacked_features_raw.npy"
            if feats_raw_pre.exists():
                X_raw_pre = np.load(feats_raw_pre)
                if (
                    X_raw_pre.ndim == 3
                    and X_raw_pre.shape[0] == Tf
                    and X_raw_pre.shape[2] >= F
                    and X_raw_pre.shape[1] == active_mask.shape[0]
                ):
                    X_coeff_source = X_raw_pre[:, active_mask, :F]
                    print(
                        "[tail-correction] feature coefficient source: "
                        "1_raw_data/stacked_features_raw.npy (active-filtered)."
                    )
                else:
                    print(
                        "[tail-correction] WARNING: stacked_features_raw.npy shape mismatch; "
                        "falling back to active features."
                    )
            else:
                print(
                    "[tail-correction] WARNING: stacked_features_raw.npy missing; "
                    "falling back to active features."
                )
        else:
            print("[tail-correction] feature coefficient source: 2_active_data (post-roll).")

        # --------------------------
        # 1) Matrix multipliers from mean edge signal
        # --------------------------
        iu, ju = np.triu_indices(N, k=1)
        y_cooc_edges = A[iu, ju, :].mean(axis=0).astype(float)
        c_cooc, t_break_cooc, _ = _coeff_series_from_avg(y_cooc_edges)

        A_corr = A.astype(np.float32, copy=True)
        for t in range(A_corr.shape[2]):
            A_corr[:, :, t] *= float(c_cooc[t])

        # --------------------------
        # 2) Per-feature multipliers from node-avg of each feature
        # --------------------------
        avg_feat_over_nodes = np.nanmean(X_coeff_source, axis=1)  # (T, F)
        coeffs_feat = np.zeros((Tf, F), dtype=float)
        X_corr = X.astype(np.float32, copy=True)
        for fi in range(F):
            y_feat = avg_feat_over_nodes[:, fi].astype(float)
            feat_name = FEATURE_NAMES[fi] if fi < len(FEATURE_NAMES) else f"feat{fi}"

            if CORRECT_CUMULATIVE_VIA_INCREMENTS and _is_cumulative_feature(feat_name):
                x_src = X_coeff_source[:, :, fi].astype(float, copy=False)  # (T, N)
                dx_src = np.diff(x_src, axis=0, prepend=x_src[:1, :])
                dx_src = np.clip(dx_src, a_min=0.0, a_max=None)
                y_feat_increments = np.nanmean(dx_src, axis=1).astype(float)
                c_f, _, _ = _coeff_series_from_avg(y_feat_increments)

                # Apply the inferred monthly multipliers to the series effectively used by GTAN.
                x_apply = X[:, :, fi].astype(float, copy=False)
                dx_apply = np.diff(x_apply, axis=0, prepend=x_apply[:1, :])
                dx_apply = np.clip(dx_apply, a_min=0.0, a_max=None)
                dx_corr = dx_apply * c_f[:, None]
                x_corr_f = x_apply[:1, :] + np.cumsum(dx_corr, axis=0)
                X_corr[:, :, fi] = x_corr_f.astype(np.float32)

                y_corr = np.nanmean(x_corr_f, axis=1).astype(float)
                safe_y = np.where(y_feat <= EPS, np.nan, y_feat)
                with np.errstate(divide="ignore", invalid="ignore"):
                    c_eff = np.where(np.isfinite(safe_y), y_corr / safe_y, 1.0)
                c_eff = np.where(np.isfinite(c_eff), c_eff, 1.0)
                c_eff = np.maximum(c_eff, 1.0)
                coeffs_feat[:, fi] = c_eff
                print(
                    f"[tail-correction] feature '{feat_name}' corrected via monthly increments "
                    f"(effective max multiplier={float(np.nanmax(c_eff)):.4f})."
                )
            else:
                c_f, _, _ = _coeff_series_from_avg(y_feat)
                coeffs_feat[:, fi] = c_f
                X_corr[:, :, fi] *= c_f[:, None]

        # --------------------------
        # Save multipliers (NPY + CSV with dates & feature names)
        # --------------------------
        np.save(CORR_DIR / "cooc_month_coefficients.npy", c_cooc.astype(np.float32))
        np.save(CORR_DIR / "month_coefficients_per_feature.npy", coeffs_feat.astype(np.float32))
        (pd.DataFrame({"date": dates, "multiplier": c_cooc})
         .to_csv(CORR_DIR / "cooc_month_coefficients.csv", index=False))
        (pd.DataFrame(coeffs_feat, index=dates, columns=FEATURE_NAMES)
         .to_csv(CORR_DIR / "month_coefficients_per_feature.csv", index_label="date"))

        # --------------------------
        # Save corrected tensors (as before)
        # --------------------------
        out_mats = CORR_DIR / "stacked_matrices_corrected.npy"
        out_feats = CORR_DIR / "stacked_features_active_corrected.npy"
        np.save(out_mats, A_corr)
        np.save(out_feats, X_corr)

        print(f"[OK] Saved corrected matrices to: {out_mats}  shape={A_corr.shape}  dtype={A_corr.dtype}")
        print(f"[OK] Saved corrected features to: {out_feats}  shape={X_corr.shape}  dtype={X_corr.dtype}")

        # ==============================================================\n        # === EXACT inputs the GTAN will see (apply EMERGENCE_MODE) ===\n        # ==============================================================

        def ratio_curr_prev(X_all, eps):
            X_all = X_all.astype(float, copy=False)
            out = np.zeros_like(X_all)
            if X_all.shape[0] <= 1:
                return out
            prev, curr = X_all[:-1], X_all[1:]
            denom = np.maximum(prev, eps) if eps > 0.0 else prev
            mask = denom > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(mask, curr / denom, 0.0)
                out[1:] = r
            return out

        def pct_change_curr_prev(X_all, eps):
            X_all = X_all.astype(float, copy=False)
            out = np.zeros_like(X_all)
            if X_all.shape[0] <= 1:
                return out
            prev, curr = X_all[:-1], X_all[1:]
            denom = np.maximum(prev, eps) if eps > 0.0 else prev
            mask = denom > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(mask, curr / denom - 1.0, 0.0)
                out[1:] = r
            return out

        def transform_by_mode_3d(X_all, mode, eps):
            mode = (mode or "raw").lower()
            if mode == "raw":
                return X_all.astype(float, copy=False)
            Tn, Nn, Fn = X_all.shape
            Xt = np.zeros_like(X_all, dtype=float)
            for fi in range(Fn):
                Xf = X_all[:, :, fi]
                Xt[:, :, fi] = ratio_curr_prev(Xf, eps) if mode == "ratio" else pct_change_curr_prev(Xf, eps)
            return Xt

        # Raw vs corrected as SEEN BY GTAN
        Xt_raw = transform_by_mode_3d(X, EMERGENCE_MODE, EPSILON)  # (T, N, F)
        Xt_cor = transform_by_mode_3d(X_corr, EMERGENCE_MODE, EPSILON)

        # --------------------------
        # Plots of the actual inputs (averaged just for visualization)
        # --------------------------
        if PLOT_PREVIEW:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            plt.rcParams.update({"figure.dpi": DPI})

            def vis(x):
                return np.log1p(np.clip(x, a_min=0.0, a_max=None))

            avg_corr_full = np.nanmean(X_corr, axis=1)
            fig, ax = plt.subplots(figsize=(10, 4))
            for fi, nm in enumerate(FEATURE_NAMES):
                ax.plot(dates, vis(avg_corr_full[:, fi]), lw=1.6, label=f"{nm} (corrected, full range)")
            ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.legend(ncol=2, fontsize=8)
            plt.tight_layout()
            fig_path_full = PLOT_DIR / "inputs_corrected_per_feature_full.pdf"
            plt.savefig(fig_path_full, bbox_inches="tight")
            plt.close()
            print(f"[plot] {fig_path_full}")

            agg_corr_full = np.nanmean((X_corr @ FW), axis=1)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(dates, vis(agg_corr_full), lw=1.6, label="sum(FW*features) (corrected, full range)")
            ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.legend()
            plt.tight_layout()
            fig_path_full_agg = PLOT_DIR / "inputs_corrected_aggregate_full.pdf"
            plt.savefig(fig_path_full_agg, bbox_inches="tight")
            plt.close()
            print(f"[plot] {fig_path_full_agg}")

            # Per-feature averages over nodes, AFTER transform (what the model sees)
            avg_raw = np.nanmean(Xt_raw, axis=1)  # (T, F)
            avg_cor = np.nanmean(Xt_cor, axis=1)

            fig, ax = plt.subplots(figsize=(10, 4))
            for fi, nm in enumerate(FEATURE_NAMES):
                ax.plot(
                    dates,
                    vis(avg_raw[:, fi]),
                    ls="--",
                    lw=1.1,
                    alpha=0.85,
                    label=f"{nm} (former, pre-correction -> GTAN)",
                )
                ax.plot(dates, vis(avg_cor[:, fi]), lw=1.6, label=f"{nm} (corrected -> GTAN)")
            ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.legend(ncol=2, fontsize=8)
            plt.tight_layout()
            fig_path1 = PLOT_DIR / f"inputs_seen_per_feature__mode_{EMERGENCE_MODE}.pdf"
            plt.savefig(fig_path1, bbox_inches="tight")
            plt.close()
            print(f"[plot] {fig_path1}")

            # FW-weighted aggregate (common in your targets)
            assert FW.size == F, f"FW length {FW.size} must match feature dim {F}"
            agg_raw = np.nanmean(Xt_raw @ FW, axis=1)  # (T,)
            agg_cor = np.nanmean(Xt_cor @ FW, axis=1)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(
                dates,
                vis(agg_raw),
                ls="--",
                lw=1.2,
                alpha=0.9,
                label="sum(FW*features) (former, pre-correction -> GTAN)",
            )
            ax.plot(dates, vis(agg_cor), lw=1.8, label="sum(FW*features) (corrected -> GTAN)")
            ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.legend()
            plt.tight_layout()
            fig_path2 = PLOT_DIR / f"inputs_seen_aggregate__mode_{EMERGENCE_MODE}.pdf"
            plt.savefig(fig_path2, bbox_inches="tight")
            plt.close()
            print(f"[plot] {fig_path2}")
    _announce("Cell 8: configuring GTAN models and utilities")
    import os, math, random
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import TransformerConv
    from torch_geometric.data import Data
    from torch_geometric.utils import add_self_loops, remove_self_loops
    from matplotlib.colors import PowerNorm
    from matplotlib import colormaps

    # ================================================================
    # Config — one place to rule them all
    # ================================================================
    cfg_defaults_fallback = {
        "EMERGENCE_MODE": "raw",
        "EPSILON": 1e-8,
        "LAST_YEARS": 0,
        "FORECAST": 8,
        "TEMP_WINDOW": 12,
        "FEATURE_NAMES_DEFAULT": ["cited_by_count", "oc_freq", "edge_weight"],
        "FW": [0.0, 0.5, 0.5],
        "NORMALIZE_FW": True,
        "HIDDEN_CHANNELS": 64,
        "NUM_HEADS": 4,
        "DROPOUT": 0.25,
        "USE_POSENC": True,
        "USE_CAUSAL_MASK": True,
        "NODE_ONLY": False,
        "GRAPH_GATE": True,
        "GRAPH_MIX": False,
        "GRAPH_MIX_LAMBDA": 0.0,
        "GRAPH_MIX_EPS": 1e-6,
        "GRAPH_MULTI": False,
        "GRAPH_MULTI_LAGS": [0, 6, 12],
        "GRAPH_MULTI_AUX_FORECASTS": [6, 12],
        "GRAPH_MULTI_AUX_WEIGHT": 0.1,
        "COPY_HEAD": False,
        "EDGE_SELF_LOOPS": False,
        "EDGE_NORM": "none",
        "EDGE_ATTR_MODE": "weight",
        "EPOCHS": 9,
        "LR": 1e-3,
        "WD": 1e-5,
        "PATIENCE": 3,
        "MIN_DELTA": 1e-3,
        "TARGET_MODE": "residual",
        "TARGET_RELATIVE_SMOOTH_WINDOW": 12,
        "TARGET_RELATIVE_LAG": 12,
        "TARGET_RELATIVE_EPS": 1e-8,
        "TARGET_RELATIVE_TAU": 0.0,
        "TARGET_RELATIVE_TAU_QUANTILE": 0.25,
        "TARGET_RELATIVE_DENOM_MIN_ABS": 0.0,
        "TARGET_RELATIVE_DENOM_MIN_QUANTILE": 0.0,
        "LOSS_SPACE": "raw",
        "PER_NODE_STD": False,
        "SPLIT_FRACS": [0.60, 0.20, 0.20],
        "LOSS_FN": "rmse",
        "HUBER_DELTA": 1.0,
        "PLOT_DIR": "plots",
        "INPUTS_DIR": "plots/inputs_seen",
        "PLOT_METRIC": "rmse",
        "PLOT_INPUT_PREVIEWS": True,
        "HEATMAP": "viridis",
        "DPI": 180,
        "XTICK_STEP": 100,
        "CAP_PCT": 95.0,
        "TOPK_HEATMAP_K": 0,
        "TOPK_HEATMAP_DATE": None,
        "TOPK_HEATMAP_DATE_FILE": None,
        "TOPK_HEATMAP_FROM_DATE": None,
        "TOPK_HEATMAP_FEATURE": "oc_freq",
        "SHOW_FIGS": True,
        "LOG_TRAIN_RAW": False,
        "TOPK_RECENT": True,
        "TOPK_RECENT_INFERENCE": True,
        "SAVE_CHECKPOINT": True,
        "LOAD_CHECKPOINT": False,
        "CHECKPOINT_DIR": "checkpoints",
        "CHECKPOINT_STRICT": True,
        "TOPK_K": 20,
        "TOPK_WORDCLOUD": 20,
        "RELATIVE_EVOLUTION_ENABLED": False,
        "RELATIVE_EVOLUTION_LAG": 12,
        "RELATIVE_EVOLUTION_EPS": 1e-8,
        "RELATIVE_EVOLUTION_CAP_PCT": 99.0,
        "RELATIVE_EVOLUTION_SUBDIR": "relative_evolution",
        "RELATIVE_EVOLUTION_SMOOTH_ENABLED": False,
        "RELATIVE_EVOLUTION_SMOOTH_WINDOW": 12,
        "RELATIVE_EVOLUTION_SMOOTH_SUBDIR": "relative_evolution_smooth",
        "RELATIVE_EVOLUTION_CMAP": "coolwarm",
        "RELATIVE_EVOLUTION_TAU": 0.0,
        "RELATIVE_EVOLUTION_TAU_QUANTILE": 0.0,
        "RELATIVE_EVOLUTION_IGNORE_FIRST_ACTIVE_LAG": True,
        "RELATIVE_EVOLUTION_IGNORE_REACTIVATION_STARTS": True,
        "RELATIVE_EVOLUTION_ACTIVITY_THRESHOLD": 1e-8,
        "RELATIVE_EVOLUTION_MASK_ACTIVATION_MONTHS": 1,
        "RELATIVE_EVOLUTION_DENOM_MIN_ABS": 0.0,
        "RELATIVE_EVOLUTION_DENOM_MIN_QUANTILE": 0.0,
    }
    cfg_defaults = dict(cfg_defaults_fallback)
    if isinstance(cfg_defaults_ctx, dict):
        cfg_defaults.update(cfg_defaults_ctx)
    # Ensure FW follows preview.feature_weights when provided
    if isinstance(preview_cfg, dict) and preview_cfg.get("feature_weights") is not None:
        fw_preview = preview_cfg.get("feature_weights")
        if "FW" in cfg_defaults and list(cfg_defaults.get("FW", [])) != list(fw_preview):
            print("[config] WARNING: preview.feature_weights differs from cfg_defaults.FW; using preview.feature_weights.")
        cfg_defaults["FW"] = fw_preview
    # Ensure feature labels follow preview.feature_names when provided
    if isinstance(preview_cfg, dict) and preview_cfg.get("feature_names") is not None:
        fn_preview = preview_cfg.get("feature_names")
        try:
            fn_list = [str(x) for x in list(fn_preview)]
            if "FEATURE_NAMES_DEFAULT" in cfg_defaults and list(cfg_defaults.get("FEATURE_NAMES_DEFAULT", [])) != fn_list:
                print(
                    "[config] WARNING: preview.feature_names differs from cfg_defaults.FEATURE_NAMES_DEFAULT; "
                    "using preview.feature_names."
                )
            cfg_defaults["FEATURE_NAMES_DEFAULT"] = fn_list
        except Exception:
            pass
    normalize_fw_cfg_runtime = _get_bool(preview_cfg, "normalize_feature_weights", None)
    if normalize_fw_cfg_runtime is None:
        normalize_fw_cfg_runtime = _get_bool(cfg_defaults_ctx, "NORMALIZE_FW", True)
    cfg_defaults["NORMALIZE_FW"] = bool(normalize_fw_cfg_runtime)
    if isinstance(plot_cfg, dict):
        plot_overrides = {}
        if "plot_dir" in plot_cfg:
            plot_overrides["PLOT_DIR"] = plot_cfg["plot_dir"]
        if "inputs_dir" in plot_cfg:
            plot_overrides["INPUTS_DIR"] = plot_cfg["inputs_dir"]
        if "plot_metric" in plot_cfg:
            plot_overrides["PLOT_METRIC"] = plot_cfg["plot_metric"]
        if "plot_input_previews" in plot_cfg:
            plot_overrides["PLOT_INPUT_PREVIEWS"] = plot_cfg["plot_input_previews"]
        if "heatmap" in plot_cfg:
            plot_overrides["HEATMAP"] = plot_cfg["heatmap"]
        if "dpi" in plot_cfg:
            plot_overrides["DPI"] = plot_cfg["dpi"]
        if "xtick_step" in plot_cfg:
            plot_overrides["XTICK_STEP"] = plot_cfg["xtick_step"]
        if "cap_pct" in plot_cfg:
            plot_overrides["CAP_PCT"] = plot_cfg["cap_pct"]
        if "show_figs" in plot_cfg:
            plot_overrides["SHOW_FIGS"] = plot_cfg["show_figs"]
        if "log_train_raw" in plot_cfg:
            plot_overrides["LOG_TRAIN_RAW"] = plot_cfg["log_train_raw"]
        if "topk_recent" in plot_cfg:
            plot_overrides["TOPK_RECENT"] = plot_cfg["topk_recent"]
        if "topk_k" in plot_cfg:
            plot_overrides["TOPK_K"] = plot_cfg["topk_k"]
        if "topk_wordcloud" in plot_cfg:
            plot_overrides["TOPK_WORDCLOUD"] = plot_cfg["topk_wordcloud"]
        if "topk_corr_k" in plot_cfg:
            plot_overrides["TOPK_CORR_K"] = plot_cfg["topk_corr_k"]
        if "topk_corr_from_date" in plot_cfg:
            plot_overrides["TOPK_CORR_FROM_DATE"] = plot_cfg["topk_corr_from_date"]
        if "topk_corr_window_months" in plot_cfg:
            plot_overrides["TOPK_CORR_WINDOW_MONTHS"] = plot_cfg["topk_corr_window_months"]
        if "topk_corr_use_diff" in plot_cfg:
            plot_overrides["TOPK_CORR_USE_DIFF"] = plot_cfg["topk_corr_use_diff"]
        if "topk_corr_linfits" in plot_cfg:
            plot_overrides["TOPK_CORR_LINFITS"] = plot_cfg["topk_corr_linfits"]
        if "topk_bubble_k" in plot_cfg:
            plot_overrides["TOPK_BUBBLE_K"] = plot_cfg["topk_bubble_k"]
        if "topk_bubble_window_months" in plot_cfg:
            plot_overrides["TOPK_BUBBLE_WINDOW_MONTHS"] = plot_cfg["topk_bubble_window_months"]
        if "topk_bubble_diff_tol" in plot_cfg:
            plot_overrides["TOPK_BUBBLE_DIFF_TOL"] = plot_cfg["topk_bubble_diff_tol"]
        if "topk_bubble_max_edges" in plot_cfg:
            plot_overrides["TOPK_BUBBLE_MAX_EDGES"] = plot_cfg["topk_bubble_max_edges"]
        if "topk_bubble_trend_mode" in plot_cfg:
            plot_overrides["TOPK_BUBBLE_TREND_MODE"] = plot_cfg["topk_bubble_trend_mode"]
        if "topk_heatmap" in plot_cfg:
            plot_overrides["TOPK_HEATMAP_K"] = plot_cfg["topk_heatmap"]
        if "topk_heatmap_date" in plot_cfg:
            plot_overrides["TOPK_HEATMAP_DATE"] = plot_cfg["topk_heatmap_date"]
        if "topk_heatmap_date_file" in plot_cfg:
            plot_overrides["TOPK_HEATMAP_DATE_FILE"] = plot_cfg["topk_heatmap_date_file"]
        if "topk_heatmap_from_date" in plot_cfg:
            plot_overrides["TOPK_HEATMAP_FROM_DATE"] = plot_cfg["topk_heatmap_from_date"]
        if "topk_heatmap_feature" in plot_cfg:
            plot_overrides["TOPK_HEATMAP_FEATURE"] = plot_cfg["topk_heatmap_feature"]
        if "relative_evolution" in plot_cfg:
            plot_overrides["RELATIVE_EVOLUTION_ENABLED"] = plot_cfg["relative_evolution"]
        if "relative_evolution_lag" in plot_cfg:
            plot_overrides["RELATIVE_EVOLUTION_LAG"] = plot_cfg["relative_evolution_lag"]
        if "relative_evolution_eps" in plot_cfg:
            plot_overrides["RELATIVE_EVOLUTION_EPS"] = plot_cfg["relative_evolution_eps"]
        if "relative_evolution_cap_pct" in plot_cfg:
            plot_overrides["RELATIVE_EVOLUTION_CAP_PCT"] = plot_cfg["relative_evolution_cap_pct"]
        if "relative_evolution_subdir" in plot_cfg:
            plot_overrides["RELATIVE_EVOLUTION_SUBDIR"] = plot_cfg["relative_evolution_subdir"]
        if "relative_evolution_smooth" in plot_cfg:
            plot_overrides["RELATIVE_EVOLUTION_SMOOTH_ENABLED"] = plot_cfg["relative_evolution_smooth"]
        if "relative_evolution_smooth_window" in plot_cfg:
            plot_overrides["RELATIVE_EVOLUTION_SMOOTH_WINDOW"] = plot_cfg["relative_evolution_smooth_window"]
        if "relative_evolution_smooth_subdir" in plot_cfg:
            plot_overrides["RELATIVE_EVOLUTION_SMOOTH_SUBDIR"] = plot_cfg["relative_evolution_smooth_subdir"]
        if "relative_evolution_cmap" in plot_cfg:
            plot_overrides["RELATIVE_EVOLUTION_CMAP"] = plot_cfg["relative_evolution_cmap"]
        if "relative_evolution_tau" in plot_cfg:
            plot_overrides["RELATIVE_EVOLUTION_TAU"] = plot_cfg["relative_evolution_tau"]
        if "relative_evolution_tau_quantile" in plot_cfg:
            plot_overrides["RELATIVE_EVOLUTION_TAU_QUANTILE"] = plot_cfg["relative_evolution_tau_quantile"]
        if "relative_evolution_ignore_first_active_lag" in plot_cfg:
            plot_overrides["RELATIVE_EVOLUTION_IGNORE_FIRST_ACTIVE_LAG"] = plot_cfg["relative_evolution_ignore_first_active_lag"]
        if "relative_evolution_ignore_reactivation_starts" in plot_cfg:
            plot_overrides["RELATIVE_EVOLUTION_IGNORE_REACTIVATION_STARTS"] = plot_cfg["relative_evolution_ignore_reactivation_starts"]
        if "relative_evolution_activity_threshold" in plot_cfg:
            plot_overrides["RELATIVE_EVOLUTION_ACTIVITY_THRESHOLD"] = plot_cfg["relative_evolution_activity_threshold"]
        if "relative_evolution_mask_activation_months" in plot_cfg:
            plot_overrides["RELATIVE_EVOLUTION_MASK_ACTIVATION_MONTHS"] = plot_cfg["relative_evolution_mask_activation_months"]
        if "relative_evolution_denom_min_abs" in plot_cfg:
            plot_overrides["RELATIVE_EVOLUTION_DENOM_MIN_ABS"] = plot_cfg["relative_evolution_denom_min_abs"]
        if "relative_evolution_denom_min_quantile" in plot_cfg:
            plot_overrides["RELATIVE_EVOLUTION_DENOM_MIN_QUANTILE"] = plot_cfg["relative_evolution_denom_min_quantile"]
        if "topk_corr_use_diff" in plot_cfg:
            plot_overrides["TOPK_CORR_USE_DIFF"] = plot_cfg["topk_corr_use_diff"]
        if "topk_corr_linfits" in plot_cfg:
            plot_overrides["TOPK_CORR_LINFITS"] = plot_cfg["topk_corr_linfits"]
        cfg_defaults.update({k: v for k, v in plot_overrides.items() if v is not None})
    CFG = dict(cfg_defaults)
    overrides = ctx.get('cfg_overrides', {}) or {}
    if overrides:
        CFG.update(overrides)
    if "FW" in CFG:
        normalize_fw_runtime = bool(CFG.get("NORMALIZE_FW", True))
        CFG["FW"] = _normalize_fw_weights(
            np.asarray(CFG["FW"], dtype=float),
            enabled=normalize_fw_runtime,
            where="runtime",
        )
    if isinstance(CFG.get("SPLIT_FRACS"), list):
        CFG["SPLIT_FRACS"] = tuple(CFG["SPLIT_FRACS"])

    # ================================================================
    # Paths (with CORR_DIR fallbacks)
    # ================================================================
    BASE_DIR   = Path(ctx['base_dir'])
    DATA_DIR   = Path(ctx.get("data_base_dir") or BASE_DIR)
    RAW_DIR    = DATA_DIR / "1_raw_data"
    ACTIVE_DIR = DATA_DIR / "2_active_data"
    CORR_DIR   = DATA_DIR / "3_corrected_data"

    # Preferred corrected tensors
    MATS_PATH  = CORR_DIR / "stacked_matrices_corrected.npy"        # (N,N,T)
    FEATS_PATH = CORR_DIR / "stacked_features_active_corrected.npy" # (T,N,F)
    TIME_PATH  = RAW_DIR  / "feature_timestamps.npy"                 # (T,)

    # Name-resolution resources
    FEATS_ACT  = ACTIVE_DIR / "stacked_features_active.npy"
    IDX_ACT    = ACTIVE_DIR / "active_node_indices.npy"
    KW_ACT     = ACTIVE_DIR / "keywords_active.txt"
    KW_FULL    = RAW_DIR   / "keywords_final.txt"
    NAMES_OVERRIDE = None

    PLOT_DIR   = BASE_DIR / CFG["PLOT_DIR"]
    INPUTS_DIR = BASE_DIR / CFG["INPUTS_DIR"]
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Utilities shared by viz + training
    # ================================================================
    _ALIAS = {"blue":"Blues","red":"Reds","green":"Greens","purple":"Purples","orange":"Oranges",
              "gray":"Greys","grey":"Greys","ylgnbu":"YlGnBu","ylorbr":"YlOrBr"}

    def resolve_cmap(name: str, fallback: str) -> str:
        if not isinstance(name, str) or not name:
            return fallback
        base = name.lower(); rev = base.endswith("_r"); key = base[:-2] if rev else base
        cmap_name = _ALIAS.get(key, name)
        try:
            _ = colormaps[cmap_name]
            return cmap_name
        except Exception:
            return fallback

    # ---- transforms ----

    def ratio_curr_prev(X, eps):
        X = X.astype(float, copy=False)
        out = np.zeros_like(X)
        if X.shape[0] <= 1:
            return out
        prev, curr = X[:-1], X[1:]
        denom = np.maximum(prev, eps) if eps > 0.0 else prev
        mask  = denom > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(mask, curr / denom, 0.0)
        out[1:] = r
        return out

    def pct_change_curr_prev(X, eps):
        X = X.astype(float, copy=False)
        out = np.zeros_like(X)
        if X.shape[0] <= 1:
            return out
        prev, curr = X[:-1], X[1:]
        denom = np.maximum(prev, eps) if eps > 0.0 else prev
        mask  = denom > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(mask, curr / denom - 1.0, 0.0)
        out[1:] = r
        return out

    def transform_by_mode_3d(X_all, mode, eps):
        """Apply raw/ratio/pct to (T,N,F); returns float array of same shape."""
        mode = (mode or "raw").lower()
        if mode == "raw":
            return X_all.astype(float, copy=False)
        T, N, F = X_all.shape
        Xt = np.zeros_like(X_all, dtype=float)
        for fi in range(F):
            X = X_all[:, :, fi]
            Xt[:, :, fi] = ratio_curr_prev(X, eps) if mode == "ratio" else pct_change_curr_prev(X, eps)
        return Xt

    # ---- name resolution ----

    def resolve_active_names(N_active: int, feature_names_default):
        if NAMES_OVERRIDE is not None and len(NAMES_OVERRIDE) == N_active:
            return NAMES_OVERRIDE, None, "override"
        # 1) keywords_active.txt
        if KW_ACT.exists():
            try:
                names = pd.read_csv(KW_ACT, header=None, names=["kw"], dtype=str, engine="python")["kw"].tolist()
                if len(names) == N_active:
                    orig_idx = np.load(IDX_ACT) if IDX_ACT.exists() else None
                    return names, orig_idx, "keywords_active.txt"
            except Exception:
                pass
        # 2) indices + keywords_final.txt
        if IDX_ACT.exists() and KW_FULL.exists():
            try:
                idx  = np.load(IDX_ACT)
                full = pd.read_csv(KW_FULL, header=None, names=["kw"], dtype=str, engine="python")["kw"].tolist()
                names = [full[int(i)] for i in idx]
                if len(names) == N_active:
                    return names, idx, "indices->keywords_final.txt"
            except Exception:
                pass
        # 3) legacy
        legacy = ACTIVE_DIR / "node_names.npy"
        if legacy.exists():
            try:
                names = np.load(legacy, allow_pickle=True).tolist()
                if len(names) == N_active:
                    orig_idx = np.load(IDX_ACT) if IDX_ACT.exists() else None
                    return [str(x) for x in names], orig_idx, "node_names.npy"
            except Exception:
                pass
        # 4) fallback
        return [f"n{i}" for i in range(N_active)], (np.load(IDX_ACT) if IDX_ACT.exists() else None), "fallback"

    def _select_topk_indices_from_active(N_active: int, topk: int | None):
        if topk is None or topk <= 0 or topk >= N_active:
            return None, None
        names = None
        if KW_ACT.exists():
            try:
                names = pd.read_csv(KW_ACT, header=None, names=["kw"], dtype=str, engine="python")["kw"].tolist()
                if len(names) != N_active:
                    names = None
            except Exception:
                names = None
        if names is None:
            idx = np.arange(N_active, dtype=int)[:topk]
            return idx, None
        name_to_idx = {n: i for i, n in enumerate(names)}
        order = []
        topk_path = RAW_DIR / "top_keywords.csv"
        if topk_path.exists():
            try:
                with topk_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        kw = line.strip()
                        if not kw:
                            continue
                        if kw in name_to_idx and kw not in order:
                            order.append(kw)
                        if len(order) >= topk:
                            break
            except Exception:
                pass
        if len(order) < topk:
            for nm in names:
                if nm not in order:
                    order.append(nm)
                if len(order) >= topk:
                    break
        idx = np.array([name_to_idx[nm] for nm in order[:topk]], dtype=int)
        return idx, [names[i] for i in idx]

    def _select_topk_indices_from_fw_ranking(N_active: int, topk: int | None):
        if topk is None or topk <= 0 or topk >= N_active:
            return None, None
        names = None
        if KW_ACT.exists():
            try:
                names = pd.read_csv(KW_ACT, header=None, names=["kw"], dtype=str, engine="python")["kw"].tolist()
                if len(names) != N_active:
                    names = None
            except Exception:
                names = None
        if names is None:
            return None, None

        # Find FW aggregate ranking file (prefer latest by name)
        rank_path = None
        try:
            inputs_root = DATA_DIR / "plots" / "inputs_seen"
            candidate_dirs = [inputs_root / "ranking", inputs_root]
            candidates = []
            for candidate_dir in candidate_dirs:
                if candidate_dir.exists():
                    candidates.extend(sorted(candidate_dir.glob("*ranking_fw_aggregate*.csv")))
            if candidates:
                rank_path = candidates[-1]
        except Exception:
            rank_path = None
        if rank_path is None or not rank_path.exists():
            return None, None

        try:
            df = pd.read_csv(rank_path)
        except Exception:
            return None, None
        name_col = "keyword" if "keyword" in df.columns else ("name" if "name" in df.columns else df.columns[0])
        order = [str(x) for x in df[name_col].tolist() if isinstance(x, str) and x]

        name_to_idx = {n: i for i, n in enumerate(names)}
        filtered = []
        for nm in order:
            if nm in name_to_idx and nm not in filtered:
                filtered.append(nm)
            if len(filtered) >= topk:
                break
        if len(filtered) < topk:
            for nm in names:
                if nm not in filtered:
                    filtered.append(nm)
                if len(filtered) >= topk:
                    break
        idx = np.array([name_to_idx[nm] for nm in filtered[:topk]], dtype=int)
        return idx, [names[i] for i in idx]

    # ---- plotting helpers ----

    def _format_year_axis(ax, idx):
        import matplotlib.dates as mdates
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.YearLocator(base=1))
        ax.set_xlim(idx.min() - pd.DateOffset(months=3), idx.max() + pd.DateOffset(months=3))
        ax.margins(x=0.0)

    def year_ticks(ax, dates):
        jan = np.flatnonzero(dates.month == 1)
        ax.set_yticks(jan + 0.5)
        ax.set_yticklabels(dates[jan].strftime("%Y"), rotation=0)

    def node_ticks(ax, N, step, *, label_start: int = 1):
        xt = np.arange(0, N, step)
        if xt.size == 0 or xt[-1] != N - 1:
            xt = np.append(xt, N - 1)
        ax.set_xticks(xt + 0.5)
        ax.set_xticklabels([str(int(x) + int(label_start)) for x in xt], rotation=0)

    # ================================================================
    # Data loading + global transform (single source of truth)
    # ================================================================

    def load_and_transform(cfg):
        nonlocal NAMES_OVERRIDE
        all_matrices      = np.load(MATS_PATH)
        raw_node_features = np.load(FEATS_PATH)
        timestamps        = np.load(TIME_PATH, allow_pickle=True)

        # monthly index
        try:
            ts = pd.PeriodIndex(timestamps.astype(str), freq="M").to_timestamp(how="start")
        except Exception:
            ts = pd.to_datetime(timestamps, errors="coerce")

        # align
        T_aligned = min(len(ts), raw_node_features.shape[0], all_matrices.shape[2])
        ts                = ts[:T_aligned]
        raw_node_features = raw_node_features[:T_aligned]
        all_matrices      = all_matrices[:, :, :T_aligned]

        # time cutoff
        cutoff = ts.max() - pd.DateOffset(years=cfg['LAST_YEARS'])
        keep = (ts <= cutoff) if cfg['LAST_YEARS'] > 0 else (ts == ts)
        ts_c   = ts[keep]
        feats  = raw_node_features[keep]
        mats   = all_matrices[:, :, keep]

        # Optional: train-only time filter when reusing an existing graph
        train_filter = ctx.get("train_time_filter") if reuse_graph else None
        if isinstance(train_filter, dict):
            start_ym = train_filter.get("start_year_month")
            end_ym = train_filter.get("end_year_month")
            if start_ym or end_ym:
                start_ts = pd.to_datetime(str(start_ym), errors="coerce") if start_ym else None
                end_ts = pd.to_datetime(str(end_ym), errors="coerce") if end_ym else None
                mask = np.ones(len(ts_c), dtype=bool)
                if start_ts is not None and not pd.isna(start_ts):
                    mask &= ts_c >= start_ts
                if end_ts is not None and not pd.isna(end_ts):
                    mask &= ts_c <= end_ts
                if not mask.any():
                    raise RuntimeError(
                        f"train_time_filter removed all months (start={start_ym}, end={end_ym})."
                    )
                ts_c = ts_c[mask]
                feats = feats[mask]
                mats = mats[:, :, mask]
                print(
                    f"[graph] Train time filter applied: {start_ym or 'min'} to {end_ym or 'max'} "
                    f"-> {len(ts_c)} months"
                )

        weights = _compute_volume_reweight_factors(ts_c, ctx)
        if weights is not None:
            weights = weights.astype(float)
            feats = feats * weights[:, None, None]
            mats = mats * weights[None, None, :]

        # Optional: cap number of active nodes when reusing an existing graph
        try:
            cap_topk = int(preprocess_cfg.get("top_keywords")) if isinstance(preprocess_cfg, dict) and preprocess_cfg.get("top_keywords") is not None else None
        except Exception:
            cap_topk = None
        if reuse_graph and cap_topk is not None and cap_topk > 0 and feats.shape[1] > cap_topk:
            use_fw_rank = False
            if isinstance(preprocess_cfg, dict):
                use_fw_rank = bool(preprocess_cfg.get("cap_from_fw_aggregate", False))
            if use_fw_rank:
                idx, names = _select_topk_indices_from_fw_ranking(feats.shape[1], cap_topk)
                if idx is None:
                    print("[graph] FW-aggregate ranking not found; falling back to active order for capping")
                    idx, names = _select_topk_indices_from_active(feats.shape[1], cap_topk)
            else:
                idx, names = _select_topk_indices_from_active(feats.shape[1], cap_topk)
            if idx is not None:
                feats = feats[:, idx, :]
                mats = mats[idx][:, idx, :]
                NAMES_OVERRIDE = names if names is not None else NAMES_OVERRIDE
                print(f"[graph] Capped active nodes to top {len(idx)} (reuse_existing_graph)")

        # === GLOBAL TRANSFORM for emergence mode ===
        Xt = transform_by_mode_3d(feats, cfg['EMERGENCE_MODE'], cfg['EPSILON'])

        return ts_c, Xt, mats, feats  # return feats (pre-transform) only for optional reference

    # ================================================================
    # Visualize inputs seen (uses *transformed* features Xt) —
    # match your plot_features_cfg.py style (aggregate line + per-feature heatmaps)
    # ================================================================

    # --- feature selection helpers to mimic your original script ---
    def attach_names_for_inputs(N:int):
        if NAMES_OVERRIDE is not None and len(NAMES_OVERRIDE) == N:
            return NAMES_OVERRIDE
        if KW_ACT.exists():
            try:
                names = pd.read_csv(KW_ACT, header=None, names=["kw"], dtype=str, engine="python")["kw"].tolist()
                if len(names) == N: return names
            except Exception:
                pass
        if IDX_ACT.exists() and KW_FULL.exists():
            try:
                idx  = np.load(IDX_ACT)
                full = pd.read_csv(KW_FULL, header=None, names=["kw"], dtype=str, engine="python")["kw"].tolist()
                names = [full[int(i)] for i in idx]
                if len(names) == N: return names
            except Exception:
                pass
        legacy = ACTIVE_DIR / "node_names.npy"
        if legacy.exists():
            try:
                names = np.load(legacy, allow_pickle=True).tolist()
                if len(names) == N: return [str(x) for x in names]
            except Exception:
                pass
        return None


    def get_feature_indices_for_inputs(X_all: np.ndarray, feature, feature_names_default):
        T, N, F = X_all.shape
        if isinstance(feature, str) and feature.lower() != "all":
            if feature.isdigit():
                idx = int(feature); return [idx], [f"f{idx}"]
            if F == 3 and feature in feature_names_default:
                return [feature_names_default.index(feature)], [feature]
            raise ValueError(f"Unknown feature '{feature}'. Use one of {feature_names_default}, an int, or 'all'.")
        names = feature_names_default if F == 3 else [f"f{i}" for i in range(F)]
        return list(range(F)), names


    def visualize_inputs(ts, Xt, mats, cfg, feature_names=None, save_dir=None, show=False):
        # ts: DatetimeIndex, Xt: (T,N,F) already transformed by EMERGENCE_MODE
        T, N, F = Xt.shape
        feature_names_all = cfg['FEATURE_NAMES_DEFAULT'] if F == 3 else [f"f{i}" for i in range(F)]
        if feature_names is None:
            feature_names = feature_names_all

        FW = None
        try:
            FW = np.asarray(cfg.get('FW', []), dtype=float)
            if FW.size != F:
                print(f"[heatmap] skipping FW aggregate: FW length {FW.size} != feature dim {F}")
                FW = None
        except Exception as exc:
            print(f"[heatmap] failed to read FW: {exc}")
            FW = None

        save_dir = Path(save_dir or INPUTS_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)
        heatmap_dir = save_dir / "heatmaps"
        ranking_dir = save_dir / "ranking"
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        ranking_dir.mkdir(parents=True, exist_ok=True)
        tag = f"EM_{cfg['EMERGENCE_MODE']}_Eps{cfg['EPSILON']}_F{cfg['FORECAST']}_W{cfg['TEMP_WINDOW']}"
        rel_enabled = bool(cfg.get("RELATIVE_EVOLUTION_ENABLED", False))
        rel_lag = int(cfg.get("RELATIVE_EVOLUTION_LAG", 12) or 12)
        rel_eps = float(cfg.get("RELATIVE_EVOLUTION_EPS", cfg.get("EPSILON", 1e-8)) or 0.0)
        rel_cap_pct = float(cfg.get("RELATIVE_EVOLUTION_CAP_PCT", 99.0) or 99.0)
        rel_subdir = str(cfg.get("RELATIVE_EVOLUTION_SUBDIR", "relative_evolution") or "relative_evolution").strip()
        rel_subdir = rel_subdir if rel_subdir else "relative_evolution"
        rel_smooth_enabled = bool(cfg.get("RELATIVE_EVOLUTION_SMOOTH_ENABLED", False))
        rel_smooth_window = int(cfg.get("RELATIVE_EVOLUTION_SMOOTH_WINDOW", rel_lag) or rel_lag)
        rel_smooth_subdir = str(
            cfg.get("RELATIVE_EVOLUTION_SMOOTH_SUBDIR", f"{rel_subdir}_smooth") or f"{rel_subdir}_smooth"
        ).strip()
        rel_smooth_subdir = rel_smooth_subdir if rel_smooth_subdir else f"{rel_subdir}_smooth"
        rel_cmap = resolve_cmap(str(cfg.get("RELATIVE_EVOLUTION_CMAP", "coolwarm") or "coolwarm"), "coolwarm")
        rel_tau = float(cfg.get("RELATIVE_EVOLUTION_TAU", 0.0) or 0.0)
        rel_tau_quantile = float(cfg.get("RELATIVE_EVOLUTION_TAU_QUANTILE", 0.0) or 0.0)
        rel_ignore_first = bool(cfg.get("RELATIVE_EVOLUTION_IGNORE_FIRST_ACTIVE_LAG", True))
        rel_ignore_reactivation = bool(cfg.get("RELATIVE_EVOLUTION_IGNORE_REACTIVATION_STARTS", True))
        rel_activity_thr = float(cfg.get("RELATIVE_EVOLUTION_ACTIVITY_THRESHOLD", rel_eps) or 0.0)
        rel_mask_activation_months = int(cfg.get("RELATIVE_EVOLUTION_MASK_ACTIVATION_MONTHS", 1) or 0)
        rel_denom_min_abs = float(cfg.get("RELATIVE_EVOLUTION_DENOM_MIN_ABS", 0.0) or 0.0)
        rel_denom_min_quantile = float(cfg.get("RELATIVE_EVOLUTION_DENOM_MIN_QUANTILE", 0.0) or 0.0)
        if rel_lag <= 0:
            print(f"[relative_evolution] invalid lag={rel_lag}; using 12.")
            rel_lag = 12
        if rel_smooth_window <= 0:
            print(f"[relative_evolution] invalid smooth_window={rel_smooth_window}; disabling smooth heatmaps.")
            rel_smooth_enabled = False
            rel_smooth_window = rel_lag
        if rel_mask_activation_months < 0:
            print(f"[relative_evolution] invalid mask_activation_months={rel_mask_activation_months}; using 0.")
            rel_mask_activation_months = 0
        if rel_denom_min_quantile < 0.0 or rel_denom_min_quantile > 1.0:
            print(
                f"[relative_evolution] invalid denom_min_quantile={rel_denom_min_quantile}; "
                "clipping to [0, 1]."
            )
            rel_denom_min_quantile = float(np.clip(rel_denom_min_quantile, 0.0, 1.0))
        if rel_tau < 0.0:
            print(f"[relative_evolution] invalid tau={rel_tau}; using 0.")
            rel_tau = 0.0
        if rel_tau_quantile < 0.0 or rel_tau_quantile > 1.0:
            print(
                f"[relative_evolution] invalid tau_quantile={rel_tau_quantile}; "
                "clipping to [0, 1]."
            )
            rel_tau_quantile = float(np.clip(rel_tau_quantile, 0.0, 1.0))
        rel_save_dir = save_dir / rel_subdir
        if rel_enabled:
            rel_save_dir.mkdir(parents=True, exist_ok=True)
        rel_smooth_save_dir = save_dir / rel_smooth_subdir
        if rel_enabled and rel_smooth_enabled:
            rel_smooth_save_dir.mkdir(parents=True, exist_ok=True)

        # --- Top-K heatmap controls ---
        topk_k = int(cfg.get("TOPK_HEATMAP_K", 0) or 0)
        raw_topk_feature = cfg.get("TOPK_HEATMAP_FEATURE", "")
        topk_features: set[str] = set()
        if isinstance(raw_topk_feature, (list, tuple, set)):
            for item in raw_topk_feature:
                val = (str(item) if item is not None else "").strip().lower()
                if val:
                    topk_features.add(val)
        else:
            val = (str(raw_topk_feature) if raw_topk_feature is not None else "").strip().lower()
            if val:
                topk_features.add(val)

        topk_date_str = cfg.get("TOPK_HEATMAP_DATE")
        topk_date_file = cfg.get("TOPK_HEATMAP_DATE_FILE")
        topk_freeze_ts = None
        topk_from_ts = None
        topk_corr_k = int(cfg.get("TOPK_CORR_K", 0) or 0)
        topk_wordcloud_cfg = int(cfg.get("TOPK_WORDCLOUD", 0) or 0)
        topk_corr_window = int(cfg.get("TOPK_CORR_WINDOW_MONTHS", 0) or 0)
        corr_modes = _normalize_corr_modes(cfg.get("TOPK_CORR_USE_DIFF", []))
        topk_corr_use_logdiff = "log" in corr_modes
        topk_corr_use_rawdiff = "raw" in corr_modes
        topk_corr_use_diff = topk_corr_use_logdiff or topk_corr_use_rawdiff
        topk_corr_linfits = bool(cfg.get("TOPK_CORR_LINFITS", False))
        if topk_k <= 0 and topk_corr_k > 0:
            topk_k = topk_corr_k
        if topk_k > 0 or topk_wordcloud_cfg > 0:
            freeze_str = None
            if topk_date_file not in (None, "", "null"):
                freeze_path = Path(str(topk_date_file))
                if not freeze_path.is_absolute():
                    freeze_path = (BASE_DIR / freeze_path).resolve()
                if freeze_path.exists():
                    try:
                        freeze_str = freeze_path.read_text(encoding="utf-8").strip()
                    except Exception as exc:
                        print(f"[topk_heatmap] failed to read date file '{freeze_path}': {exc}")
                else:
                    print(f"[topk_heatmap] date file not found: {freeze_path}")
            if not freeze_str and topk_date_str not in (None, "", "null"):
                freeze_str = str(topk_date_str)
            if freeze_str:
                freeze_candidate = pd.to_datetime(freeze_str, errors="coerce")
                if pd.isna(freeze_candidate):
                    print(f"[topk_heatmap] invalid date '{freeze_str}'")
                else:
                    try:
                        topk_freeze_ts = pd.Period(freeze_candidate, freq="M").to_timestamp(how="start")
                    except Exception:
                        topk_freeze_ts = freeze_candidate.normalize()
            from_str = cfg.get("TOPK_HEATMAP_FROM_DATE")
            if from_str not in (None, "", "null"):
                from_candidate = pd.to_datetime(str(from_str), errors="coerce")
                if pd.isna(from_candidate):
                    print(f"[topk_heatmap] invalid from_date '{from_str}'")
                else:
                    try:
                        topk_from_ts = pd.Period(from_candidate, freq="M").to_timestamp(how="start")
                    except Exception:
                        topk_from_ts = from_candidate.normalize()
        topk_corr_from_ts = None
        corr_from_str = cfg.get("TOPK_CORR_FROM_DATE")
        if corr_from_str not in (None, "", "null"):
            corr_candidate = pd.to_datetime(str(corr_from_str), errors="coerce")
            if pd.isna(corr_candidate):
                print(f"[correlation] invalid from_date '{corr_from_str}'")
            else:
                try:
                    topk_corr_from_ts = pd.Period(corr_candidate, freq="M").to_timestamp(how="start")
                except Exception:
                    topk_corr_from_ts = corr_candidate.normalize()
        if topk_corr_from_ts is None:
            topk_corr_from_ts = topk_from_ts

        resolved_names, resolved_orig_idx, _ = resolve_active_names(N, feature_names_all)
        if resolved_names is not None and len(resolved_names) == N:
            node_names = resolved_names
            orig_idx_arr = resolved_orig_idx if resolved_orig_idx is not None else None
        else:
            node_names = attach_names_for_inputs(N)
            orig_idx_arr = None

        column_info = []
        column_lookup = defaultdict(list)
        if node_names is not None and len(node_names) == N:
            for i, nm in enumerate(node_names):
                orig_idx_val = int(orig_idx_arr[i]) if (orig_idx_arr is not None and i < len(orig_idx_arr)) else i
                info = {"label": nm, "name": nm, "active_idx": i, "orig_idx": orig_idx_val}
                column_info.append(info)
                column_lookup[nm].append(info)
        else:
            for i in range(N):
                nm = f"node {i}"
                info = {"label": nm, "name": nm, "active_idx": i, "orig_idx": i}
                column_info.append(info)
                column_lookup[nm].append(info)
        column_labels = [info["label"] for info in column_info]

        # --- Aggregate line BEFORE heatmaps (exact semantics: weight per-feature then avg over nodes) ---
        if FW is not None and cfg.get("PLOT_LINE_BEFORE_HEATMAPS", True):
            Xt_w = Xt * FW[None, None, :]
            avg_per_feat_w = np.nanmean(Xt_w, axis=1)            # (T,F) avg over nodes AFTER FW
            weighted_sum   = np.nanmean(Xt_w.sum(axis=-1), axis=1) # (T,) mean over nodes of sum

            # Clip negatives then log1p to match your original style
            Y  = np.log1p(np.clip(avg_per_feat_w, a_min=0.0, a_max=None))
            Yw = np.log1p(np.clip(weighted_sum,   a_min=0.0, a_max=None))

            fig, ax = plt.subplots(figsize=(10,4))
            show_names = feature_names_all
            for i, nm in enumerate(show_names):
                ax.plot(ts, Y[:, i], label=f"{nm} × {FW[i]:.2f} (avg over nodes, log1p)")
            ax.plot(ts, Yw, ls="--", label="Σ (weighted features)")
            _format_year_axis(ax, ts)
            ax.set_xlabel("Year"); ax.set_ylabel("log1p(value)")
            ax.legend(ncol=2, fontsize=9)
            plt.tight_layout()
            out = save_dir / f"{tag}__aggregate_{cfg['EMERGENCE_MODE']}.pdf"
            plt.savefig(out, dpi=cfg['DPI'], bbox_inches="tight");
            if show: plt.show()
            else: plt.close()

        # --- Overall citation-flow variants (scale comparison) ---
        if cfg.get("PLOT_CITATION_FLOW_VARIANTS", True):
            try:
                flow_base_path = RAW_DIR / "citation_flow_in_corpus.npy"
                flow_frac_path = RAW_DIR / "citation_flow_fractional.npy"
                flow_split_path = RAW_DIR / "citation_flow_fractional_split.npy"

                if flow_base_path.exists() and flow_frac_path.exists() and flow_split_path.exists():
                    flow_base = np.load(flow_base_path)
                    flow_frac = np.load(flow_frac_path)
                    flow_split = np.load(flow_split_path)
                    if flow_base.ndim == flow_frac.ndim == flow_split.ndim == 2:
                        timestamps_raw = np.load(TIME_PATH, allow_pickle=True)
                        try:
                            ts_raw = pd.PeriodIndex(timestamps_raw.astype(str), freq="M").to_timestamp(how="start")
                        except Exception:
                            ts_raw = pd.to_datetime(timestamps_raw, errors="coerce")

                        Tm = min(len(ts_raw), flow_base.shape[0], flow_frac.shape[0], flow_split.shape[0])
                        ts_raw = pd.DatetimeIndex(ts_raw[:Tm])
                        s_base = pd.Series(np.nanmean(flow_base[:Tm], axis=1), index=ts_raw)
                        s_frac = pd.Series(np.nanmean(flow_frac[:Tm], axis=1), index=ts_raw)
                        s_split = pd.Series(np.nanmean(flow_split[:Tm], axis=1), index=ts_raw)

                        raw_df_flow = pd.DataFrame(
                            {
                                "x_flow": s_base,
                                "x_flow_frac": s_frac,
                                "x_flow_frac_split": s_split,
                            }
                        ).sort_index()
                        raw_df_cum = raw_df_flow.cumsum()

                        target_idx = pd.DatetimeIndex(ts)
                        df_flow = raw_df_flow.reindex(target_idx)
                        df_cum = raw_df_cum.reindex(target_idx)

                        if not (df_flow.dropna(how="all").empty and df_cum.dropna(how="all").empty):
                            Yf = np.log1p(np.clip(df_flow.to_numpy(), a_min=0.0, a_max=None))
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(df_flow.index, Yf[:, 0], label="x_flow")
                            ax.plot(df_flow.index, Yf[:, 1], label="x_flow_frac")
                            ax.plot(df_flow.index, Yf[:, 2], label="x_flow_frac_split")
                            _format_year_axis(ax, df_flow.index)
                            ax.set_xlabel("Year")
                            ax.set_ylabel("log1p(value)")
                            ax.legend(ncol=2, fontsize=9)
                            plt.tight_layout()
                            out_flow = heatmap_dir / f"{tag}__citation_flow_variants_overall.pdf"
                            plt.savefig(out_flow, dpi=cfg["DPI"], bbox_inches="tight")
                            if show:
                                plt.show()
                            else:
                                plt.close()

                            df_out = df_flow.copy()
                            df_out.insert(0, "date", df_out.index.strftime("%Y-%m-%d"))
                            df_out.to_csv(
                                ranking_dir / f"{tag}__citation_flow_variants_overall.csv",
                                index=False,
                                encoding="utf-8",
                            )

                            # 6-signal plot: three flows + three cumulative variants
                            df_six = pd.DataFrame(
                                {
                                    "x_flow": df_flow["x_flow"],
                                    "x_flow_frac": df_flow["x_flow_frac"],
                                    "x_flow_frac_split": df_flow["x_flow_frac_split"],
                                    "x_cum": df_cum["x_flow"],
                                    "x_cum_frac": df_cum["x_flow_frac"],
                                    "x_cum_frac_split": df_cum["x_flow_frac_split"],
                                },
                                index=target_idx,
                            ).dropna(how="all")

                            if not df_six.empty:
                                Y6 = np.log1p(np.clip(df_six.to_numpy(), a_min=0.0, a_max=None))
                                fig, ax = plt.subplots(figsize=(11, 4.5))
                                # Use the same color for each flow/cumulative pair.
                                pair_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
                                ax.plot(df_six.index, Y6[:, 0], color=pair_colors[0], label="x_flow")
                                ax.plot(df_six.index, Y6[:, 1], color=pair_colors[1], label="x_flow_frac")
                                ax.plot(df_six.index, Y6[:, 2], color=pair_colors[2], label="x_flow_frac_split")
                                ax.plot(df_six.index, Y6[:, 3], ls="--", color=pair_colors[0], label="x_cum")
                                ax.plot(df_six.index, Y6[:, 4], ls="--", color=pair_colors[1], label="x_cum_frac")
                                ax.plot(df_six.index, Y6[:, 5], ls="--", color=pair_colors[2], label="x_cum_frac_split")
                                _format_year_axis(ax, df_six.index)
                                ax.set_xlabel("Year")
                                ax.set_ylabel("log1p(value)")
                                ax.legend(ncol=3, fontsize=8)
                                plt.tight_layout()
                                out_six = heatmap_dir / f"{tag}__citation_flow_and_cumulative_overall.pdf"
                                plt.savefig(out_six, dpi=cfg["DPI"], bbox_inches="tight")
                                if show:
                                    plt.show()
                                else:
                                    plt.close()

                                df_six_out = df_six.copy()
                                df_six_out.insert(0, "date", df_six_out.index.strftime("%Y-%m-%d"))
                                df_six_out.to_csv(
                                    ranking_dir / f"{tag}__citation_flow_and_cumulative_overall.csv",
                                    index=False,
                                    encoding="utf-8",
                                )
                    else:
                        print("[citation] flow variant arrays must be 2D; skipping overall comparison plot.")
            except Exception as exc:
                print(f"[citation] failed to plot flow variants: {exc}")

        # --- Per-feature heatmaps (match labels/scales from your script) ---
        cmap_name = resolve_cmap(cfg['HEATMAP'], 'viridis')

        # FEATURE filter support like your original
        feat_idx, feat_names = get_feature_indices_for_inputs(Xt, cfg.get('FEATURE', 'all'), feature_names_all)
        topk_wordcloud = topk_wordcloud_cfg
        if topk_wordcloud > 0 and WordCloud is None:
            print("[wordcloud] wordcloud package not installed; skipping word clouds.")
            topk_wordcloud = 0
        wordcloud_dir = None
        if topk_wordcloud > 0:
            wordcloud_dir = save_dir / "wordcloud"
            wordcloud_dir.mkdir(parents=True, exist_ok=True)
        corr_dir = None
        if int(cfg.get("TOPK_CORR_K", 0) or 0) > 0 or topk_corr_k > 0:
            corr_dir = save_dir / "correlation"
            corr_dir.mkdir(parents=True, exist_ok=True)
        bubble_dir = None
        bubble_k = int(cfg.get("TOPK_BUBBLE_K", 0) or 0)
        if bubble_k <= 0:
            bubble_k = topk_corr_k if topk_corr_k > 0 else topk_k
        bubble_window = int(cfg.get("TOPK_BUBBLE_WINDOW_MONTHS", 0) or 0)
        bubble_diff_tol = float(cfg.get("TOPK_BUBBLE_DIFF_TOL", 0.0) or 0.0)
        bubble_max_edges = int(cfg.get("TOPK_BUBBLE_MAX_EDGES", 200) or 200)
        bubble_trend_mode = str(cfg.get("TOPK_BUBBLE_TREND_MODE", "diff") or "diff").strip().lower()
        if bubble_trend_mode not in {"diff", "slope", "percentage"}:
            print(f"[bubble] unknown TREND_MODE='{bubble_trend_mode}', defaulting to 'diff'.")
            bubble_trend_mode = "diff"
        if bubble_k > 0:
            bubble_dir = save_dir / "bubble"
            bubble_dir.mkdir(parents=True, exist_ok=True)

        def _generate_topk_outputs(df: pd.DataFrame, fname: str, allow_bubble: bool = True, force: bool = False) -> None:
            process_topk = (
                topk_k > 0 and topk_freeze_ts is not None and
                (force or not topk_features or fname.lower() in topk_features)
            )
            if not process_topk:
                return
            if df.index.size == 0:
                print("[topk_heatmap] empty dataframe, skipping.")
                return
            try:
                loc = df.index.get_indexer([topk_freeze_ts], method="nearest")[0]
            except Exception:
                loc = -1
            if loc == -1 or loc >= df.shape[0]:
                print(f"[topk_heatmap] date {topk_freeze_ts.date()} outside range, skipping.")
                return
            freeze_idx = loc
            freeze_actual = df.index[freeze_idx]
            row_vals = df.iloc[freeze_idx].dropna()
            if row_vals.empty:
                print(f"[topk_heatmap] no data for {freeze_actual.date()}, skipping.")
                return
            sorted_series = row_vals.sort_values(ascending=False)
            top_series = sorted_series.head(topk_k)
            ranking_rows: list[dict[str, Any]] = []
            for rank, (label, value) in enumerate(sorted_series.items(), 1):
                pos = df.columns.get_loc(label)
                if isinstance(pos, slice):
                    candidates = list(range(pos.start, pos.stop))
                elif isinstance(pos, np.ndarray):
                    candidates = pos.tolist()
                else:
                    candidates = [pos]
                info = column_info[candidates[0]] if candidates else {"name": label, "active_idx": None, "orig_idx": None}
                ranking_rows.append({
                    "rank": rank,
                    "keyword": info["name"],
                    "active_index": info["active_idx"],
                    "orig_index": info["orig_idx"],
                    "value": float(value),
                })
            ranking_df = pd.DataFrame(ranking_rows)
            ranking_path = ranking_dir / f"{tag}__ranking_{fname}_{freeze_actual.strftime('%Y-%m-%d')}.csv"
            ranking_df.to_csv(ranking_path, index=False, encoding="utf-8")
            print(f"[topk_heatmap] saved ranking CSV -> {ranking_path}")

            subset = df[top_series.index]
            if topk_from_ts is not None:
                subset = subset[subset.index >= topk_from_ts]
                if subset.empty:
                    print(f"[topk_heatmap] from_date {topk_from_ts.date()} removed all rows, skipping.")
                    return
            subset_display = subset.copy()
            try:
                subset_display.index = pd.to_datetime(subset_display.index).strftime("%Y-%m")
            except Exception:
                subset_display.index = subset_display.index.astype(str)
            heat_df = subset_display.transpose()

            txt_path = save_dir / f"{tag}__topk_{fname}_{freeze_actual.strftime('%Y-%m-%d')}.txt"
            with txt_path.open("w", encoding="utf-8") as fh:
                fh.write(f"Top {len(top_series)} '{fname}' keywords at {freeze_actual.strftime('%Y-%m-%d')}\n")
                fh.write("rank\tkeyword\tactive_index\torig_index\tvalue\n")
                for rank, (label, value) in enumerate(top_series.items(), 1):
                    pos = df.columns.get_loc(label)
                    if isinstance(pos, slice):
                        candidates = list(range(pos.start, pos.stop))
                    elif isinstance(pos, np.ndarray):
                        candidates = pos.tolist()
                    else:
                        candidates = [pos]
                    info = column_info[candidates[0]] if candidates else {"name": label, "active_idx": None, "orig_idx": None}
                    fh.write(f"{rank}\t{info['name']}\t{info['active_idx']}\t{info['orig_idx']}\t{float(value):.6g}\n")
            print(f"[topk_heatmap] saved list -> {txt_path}")

            plt.figure(figsize=(max(10, len(heat_df.columns) * 0.16), max(7.5, len(heat_df.index) * 0.55)))
            if cfg['EMERGENCE_MODE'] == 'raw':
                vals = np.log1p(np.clip(heat_df.values, a_min=0.0, a_max=None))
                vmin_h = 0.0
                finite_vals = vals[np.isfinite(vals)]
                vmax_h = np.nanpercentile(finite_vals, 99.5) if finite_vals.size else 1.0
                norm_h = PowerNorm(gamma=1.8, vmin=vmin_h, vmax=vmax_h)
                ax_top = sns.heatmap(
                    vals,
                    cmap=cmap_name,
                    norm=norm_h,
                    yticklabels=heat_df.index,
                    cbar_kws={"label": "log(1 + emergence)"}
                )
            elif cfg['EMERGENCE_MODE'] == 'ratio':
                vals = heat_df.values[np.isfinite(heat_df.values)]
                vmax_h = np.nanpercentile(vals, cfg['CAP_PCT']) if vals.size else 1.0
                vmax_h = max(vmax_h, 1.0)
                vmin_h = np.nanmin(vals) if vals.size else 0.0
                ax_top = sns.heatmap(
                    np.clip(heat_df.values, a_min=vmin_h, a_max=vmax_h),
                    cmap=cmap_name,
                    vmin=vmin_h,
                    vmax=vmax_h,
                    yticklabels=heat_df.index,
                    cbar_kws={"label": f"ratio = curr/(prev{' + ε' if cfg['EPSILON']>0 else ''})"}
                )
            else:
                vals = heat_df.values[np.isfinite(heat_df.values)]
                vmax_h = np.nanpercentile(vals, cfg['CAP_PCT']) if vals.size else 1.0
                vmax_h = max(vmax_h, 1.0)
                vmin_h = np.nanmin(vals) if vals.size else 0.0
                ax_top = sns.heatmap(
                    np.clip(heat_df.values, a_min=vmin_h, a_max=vmax_h),
                    cmap=cmap_name,
                    vmin=vmin_h,
                    vmax=vmax_h,
                    yticklabels=heat_df.index,
                    cbar_kws={"label": f"pct = curr/(prev{' + ε' if cfg['EPSILON']>0 else ''}) - 1"}
                )
            label_fontsize = 52
            tick_fontsize = 34
            cbar_label_fontsize = 42
            cbar_tick_fontsize = 32
            tick_step_months = 6
            all_positions = np.arange(len(heat_df.columns))
            tick_positions = all_positions[::tick_step_months] if tick_step_months > 0 else all_positions
            if tick_positions.size == 0:
                tick_positions = all_positions
            elif tick_positions[-1] != all_positions[-1]:
                tick_positions = np.append(tick_positions, all_positions[-1])
            tick_positions = np.unique(tick_positions)
            tick_labels = [heat_df.columns[i] for i in tick_positions]
            ax_top.set_xlabel("Date", fontsize=label_fontsize)
            ax_top.set_ylabel("Keyword", fontsize=label_fontsize)
            ax_top.set_xticks(tick_positions + 0.5)
            ax_top.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=tick_fontsize)
            ax_top.tick_params(axis="y", labelsize=tick_fontsize)
            ax_top.tick_params(axis="x", labelsize=tick_fontsize, pad=16)
            if ax_top.collections:
                cbar = ax_top.collections[0].colorbar
                if cbar is not None:
                    cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
                    cbar.ax.set_ylabel(cbar.ax.get_ylabel(), fontsize=cbar_label_fontsize)
            plt.tight_layout()
            out_top = heatmap_dir / f"{tag}__topk_heatmap_{fname}_{freeze_actual.strftime('%Y-%m-%d')}.pdf"
            plt.savefig(out_top, dpi=cfg['DPI'], bbox_inches="tight")
            if show: plt.show()
            else: plt.close()

            if not (allow_bubble and bubble_dir is not None):
                return

            bubble_series = sorted_series.head(bubble_k)
            if bubble_series.empty:
                print("[bubble] no keywords available for bubble graph, skipping.")
                return
            bubble_infos = []
            for nm in bubble_series.index:
                infos = column_lookup.get(nm)
                if infos:
                    bubble_infos.append(infos[0])
            if not bubble_infos:
                print("[bubble] no metadata for bubble keywords, skipping.")
                return
            bubble_active_idx = [info["active_idx"] for info in bubble_infos]
            bubble_labels = [info["label"] for info in bubble_infos]
            if bubble_window > 0:
                bubble_recent_start = max(0, freeze_idx - bubble_window + 1)
            else:
                bubble_recent_start = freeze_idx
            node_slice = df.iloc[bubble_recent_start:freeze_idx + 1]
            bubble_sizes = [
                float(node_slice[info["label"]].mean())
                if not node_slice.empty else float(bubble_series[info["label"]])
                for info in bubble_infos
            ]
            edges = []
            num_nodes = len(bubble_labels)
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    recent_start = bubble_recent_start
                    recent_vals = mats[bubble_active_idx[i], bubble_active_idx[j],
                                       recent_start:freeze_idx + 1].astype(float)
                    recent_vals = np.clip(recent_vals[np.isfinite(recent_vals)], a_min=0.0, a_max=None)
                    recent_mean = float(np.nanmean(recent_vals)) if recent_vals.size else 0.0
                    weight = recent_mean
                    if weight <= 0.0:
                        continue
                    past_mean = None
                    if bubble_window > 0 and recent_start > 0:
                        past_end = recent_start - 1
                        past_start = max(0, past_end - bubble_window + 1)
                        if past_end >= past_start:
                            past_vals = mats[bubble_active_idx[i], bubble_active_idx[j],
                                             past_start:past_end + 1].astype(float)
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
                    edges.append((i, j, weight, recent_mean, past_mean, slope, pct_delta))
            if not edges:
                print("[bubble] no edges above threshold, skipping graph.")
                return
            edges.sort(key=lambda x: x[2], reverse=True)
            if bubble_max_edges > 0 and len(edges) > bubble_max_edges:
                edges = edges[:bubble_max_edges]
            node_metrics: dict[str, dict[str, float]] = {}
            node_recent_start = max(0, freeze_idx - bubble_window + 1) if bubble_window > 0 else 0
            node_past_start = max(0, node_recent_start - bubble_window) if bubble_window > 0 else 0
            node_past_end = node_recent_start - 1
            node_max_slope = 0.0
            node_max_pct = 0.0
            for label in bubble_labels:
                series_vals = df[label].to_numpy(dtype=float)
                recent_vals = series_vals[node_recent_start:freeze_idx + 1]
                recent_vals = recent_vals[np.isfinite(recent_vals)]
                recent_mean = float(np.nanmean(recent_vals)) if recent_vals.size else 0.0
                past_mean = None
                if bubble_window > 0 and node_past_end >= node_past_start:
                    past_vals = series_vals[node_past_start:node_past_end + 1]
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

            G = nx.Graph()
            max_node_value = max(bubble_sizes) if bubble_sizes else 1.0
            for label, size in zip(bubble_labels, bubble_sizes):
                G.add_node(label, value=size, size=size / max_node_value if max_node_value else 0.0)
            max_weight = max(e[2] for e in edges)
            max_slope = max((abs(e[5]) for e in edges), default=0.0)
            max_pct = max((abs(e[6]) for e in edges), default=0.0)
            cmap_slope = plt.cm.get_cmap("coolwarm")
            for i, j, weight, recent_mean, past_mean, slope, pct_delta in edges:
                u = bubble_labels[i]
                v = bubble_labels[j]
                delta = 0.0
                color = "grey"
                if bubble_trend_mode == "slope":
                    delta = slope
                    if max_slope > 0:
                        norm = np.clip(delta / max_slope, -1.0, 1.0)
                        color = cmap_slope((norm + 1.0) / 2.0)
                    else:
                        color = "grey"
                elif past_mean is not None:
                    if bubble_trend_mode == "percentage":
                        delta = pct_delta
                        if abs(delta) <= bubble_diff_tol:
                            color = "grey"
                        elif max_pct > 0:
                            norm = np.clip(delta / max_pct, -1.0, 1.0)
                            color = cmap_slope((norm + 1.0) / 2.0)
                        else:
                            color = "grey"
                    else:
                        delta = recent_mean - past_mean
                        if delta > bubble_diff_tol:
                            color = "red"
                        elif delta < -bubble_diff_tol:
                            color = "blue"
                G.add_edge(u, v, weight=weight, delta=delta, color=color)
            try:
                communities = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
            except Exception:
                communities = [set(G.nodes())]
            community_map = {}
            for idx_comm, comm in enumerate(communities):
                for node in comm:
                    community_map[node] = idx_comm
            pos = nx.spring_layout(G, weight="weight", k=0.4, seed=cfg.get("RANDOM_STATE", 42))
            fig_bub, ax_bub = plt.subplots(figsize=(10, 8))
            node_colors = [plt.cm.Set2(community_map.get(node, 0) % 8) for node in G.nodes()]
            node_sizes = [3000 * G.nodes[node]["size"] if max_node_value else 300 for node in G.nodes()]
            edge_colors = [data["color"] for _, _, data in G.edges(data=True)]
            edge_widths = [2 + 2 * (data["weight"] / max_weight if max_weight else 0.0)
                           for _, _, data in G.edges(data=True)]
            nx.draw_networkx_edges(G, pos, ax=ax_bub, edge_color=edge_colors, width=edge_widths, alpha=0.7)
            nx.draw_networkx_nodes(G, pos, ax=ax_bub, node_color=node_colors, node_size=node_sizes, alpha=0.9)
            nx.draw_networkx_labels(G, pos, ax=ax_bub, font_size=10)
            ax_bub.axis("off")
            ax_bub.margins(0.15)
            safe_name = fname.replace(" ", "_")
            out_bubble = bubble_dir / f"{tag}__bubble_{safe_name}_{freeze_actual.strftime('%Y-%m')}.pdf"
            fig_bub.savefig(out_bubble, dpi=cfg['DPI'], bbox_inches="tight", pad_inches=0.5)
            plt.close(fig_bub)
            print(f"[bubble] saved network plot: {out_bubble}")

            cmap_growth = plt.cm.get_cmap("coolwarm")
            node_growth_colors = []
            for node in G.nodes():
                metrics = node_metrics.get(node)
                color = "grey"
                if metrics:
                    if bubble_trend_mode == "slope":
                        delta = metrics["slope"]
                        norm = (delta / node_max_slope) if node_max_slope > 0 else 0.0
                        color = cmap_growth((np.clip(norm, -1.0, 1.0) + 1.0) / 2.0)
                    elif bubble_trend_mode == "percentage":
                        delta = metrics["pct"]
                        norm = (delta / node_max_pct) if node_max_pct > 0 else 0.0
                        color = cmap_growth((np.clip(norm, -1.0, 1.0) + 1.0) / 2.0)
                    else:
                        delta = metrics["diff"]
                        if delta > bubble_diff_tol:
                            color = "red"
                        elif delta < -bubble_diff_tol:
                            color = "blue"
                        else:
                            color = "grey"
                node_growth_colors.append(color)
            fig_ng, ax_ng = plt.subplots(figsize=(10, 8))
            nx.draw_networkx_edges(G, pos, ax=ax_ng, edge_color=edge_colors, width=edge_widths, alpha=0.7)
            nx.draw_networkx_nodes(G, pos, ax=ax_ng, node_color=node_growth_colors,
                                   node_size=node_sizes, alpha=0.9)
            nx.draw_networkx_labels(G, pos, ax=ax_ng, font_size=10)
            ax_ng.axis("off")
            ax_ng.margins(0.15)
            out_growth = bubble_dir / f"{tag}__bubble_growth_{safe_name}_{freeze_actual.strftime('%Y-%m')}.pdf"
            fig_ng.savefig(out_growth, dpi=cfg['DPI'], bbox_inches="tight", pad_inches=0.5)
            plt.close(fig_ng)
            print(f"[bubble] saved growth network plot: {out_growth}")

        def _sort_columns_by_freeze(df_sort: pd.DataFrame) -> pd.DataFrame:
            if df_sort.empty:
                return df_sort
            loc = len(df_sort) - 1
            if topk_freeze_ts is not None:
                try:
                    loc = df_sort.index.get_indexer([topk_freeze_ts], method="nearest")[0]
                except Exception:
                    loc = len(df_sort) - 1
            loc = max(0, min(loc, df_sort.shape[0] - 1))
            try:
                order = (
                    df_sort.iloc[loc]
                    .fillna(-np.inf)
                    .sort_values(ascending=False)
                    .index
                )
                return df_sort.loc[:, order]
            except Exception:
                return df_sort

        def _resolve_freeze_idx(df_ref: pd.DataFrame) -> tuple[int, pd.Timestamp | None]:
            if df_ref.empty:
                return -1, None
            loc = len(df_ref) - 1
            if topk_freeze_ts is not None:
                try:
                    loc = df_ref.index.get_indexer([topk_freeze_ts], method="nearest")[0]
                except Exception:
                    loc = len(df_ref) - 1
            loc = max(0, min(loc, df_ref.shape[0] - 1))
            return loc, df_ref.index[loc]

        def _ordered_columns_from_reference(df_ref: pd.DataFrame) -> tuple[list[str], pd.Timestamp | None]:
            loc, freeze_actual = _resolve_freeze_idx(df_ref)
            if loc < 0:
                return [], None
            try:
                order = (
                    df_ref.iloc[loc]
                    .fillna(-np.inf)
                    .sort_values(ascending=False)
                    .index
                    .tolist()
                )
            except Exception:
                order = list(df_ref.columns)
            return order, freeze_actual

        def _plot_matrix_heatmap(df_plot: pd.DataFrame, *, yticklabels=False):
            mode = cfg['EMERGENCE_MODE']
            eps = float(cfg['EPSILON'])
            if mode == 'raw':
                values = np.log1p(np.clip(df_plot.values, a_min=0.0, a_max=None))
                finite_vals = values[np.isfinite(values)]
                vmax = np.nanpercentile(finite_vals, 99.5) if finite_vals.size else 1.0
                norm = PowerNorm(gamma=1.8, vmin=0.0, vmax=max(float(vmax), 1e-6))
                return sns.heatmap(
                    values,
                    cmap=cmap_name,
                    norm=norm,
                    yticklabels=yticklabels,
                    cbar_kws={"label": "log(1 + emergence)"},
                )
            values = df_plot.values[np.isfinite(df_plot.values)]
            vmax = np.nanpercentile(values, cfg['CAP_PCT']) if values.size else 1.0
            vmax = max(float(vmax), 1.0)
            vmin = np.nanmin(values) if values.size else 0.0
            clip_vals = np.clip(df_plot.values, a_min=vmin, a_max=vmax)
            if mode == 'ratio':
                return sns.heatmap(
                    clip_vals,
                    cmap=cmap_name,
                    vmin=vmin,
                    vmax=vmax,
                    yticklabels=yticklabels,
                    cbar_kws={"label": f"ratio = curr/(prev{' + ε' if eps > 0 else ''}), ε={eps:g}"},
                )
            if mode == 'pct':
                return sns.heatmap(
                    clip_vals,
                    cmap=cmap_name,
                    vmin=vmin,
                    vmax=vmax,
                    yticklabels=yticklabels,
                    cbar_kws={"label": f"pct = curr/(prev{' + ε' if eps > 0 else ''}) - 1, ε={eps:g}"},
                )
            raise ValueError("EMERGENCE_MODE must be 'raw', 'ratio', or 'pct'.")

        def _plot_relative_matrix_heatmap(
            df_plot: pd.DataFrame,
            *,
            yticklabels=False,
            label_prefix: str,
            tau_used: float,
        ):
            rel_array = np.asarray(df_plot.values, dtype=float)
            finite = rel_array[np.isfinite(rel_array)]
            if finite.size == 0:
                raise ValueError("No finite values available for relative heatmap.")
            vmax = np.nanpercentile(np.abs(finite), rel_cap_pct)
            if not np.isfinite(vmax) or vmax <= 0.0:
                vmax = np.nanmax(np.abs(finite))
            if not np.isfinite(vmax) or vmax <= 0.0:
                vmax = 1.0
            vmin = -vmax
            return sns.heatmap(
                np.clip(rel_array, a_min=vmin, a_max=vmax),
                cmap=rel_cmap,
                vmin=vmin,
                vmax=vmax,
                center=0.0,
                yticklabels=yticklabels,
                cbar_kws={
                    "label": f"{label_prefix}"
                },
            )

        def _save_global_heatmap(df_plot: pd.DataFrame, out_path: Path) -> None:
            plt.figure(figsize=(14, 6))
            ax = _plot_matrix_heatmap(df_plot, yticklabels=False)
            ax.set_xlabel("Ranked concept id")
            ax.set_ylabel("Date")
            year_ticks(ax, pd.DatetimeIndex(df_plot.index))
            node_ticks(ax, df_plot.shape[1], cfg['XTICK_STEP'])
            plt.tight_layout()
            plt.savefig(out_path, dpi=cfg['DPI'], bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close()

        def _style_topk_heatmap_axes(ax_top, heat_df: pd.DataFrame) -> None:
            label_fontsize = 52
            tick_fontsize = 34
            cbar_label_fontsize = 42
            cbar_tick_fontsize = 32
            tick_step_months = 6
            all_positions = np.arange(len(heat_df.columns))
            tick_positions = all_positions[::tick_step_months] if tick_step_months > 0 else all_positions
            if tick_positions.size == 0:
                tick_positions = all_positions
            elif tick_positions[-1] != all_positions[-1]:
                tick_positions = np.append(tick_positions, all_positions[-1])
            tick_positions = np.unique(tick_positions)
            tick_labels = [heat_df.columns[i] for i in tick_positions]
            ax_top.set_xlabel("Date", fontsize=label_fontsize)
            ax_top.set_ylabel("Keyword", fontsize=label_fontsize)
            ax_top.set_xticks(tick_positions + 0.5)
            ax_top.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=tick_fontsize)
            ax_top.tick_params(axis="y", labelsize=tick_fontsize)
            ax_top.tick_params(axis="x", labelsize=tick_fontsize, pad=16)
            if ax_top.collections:
                cbar = ax_top.collections[0].colorbar
                if cbar is not None:
                    cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
                    cbar.ax.set_ylabel(cbar.ax.get_ylabel(), fontsize=cbar_label_fontsize)

        def _save_selected_topk_heatmap(
            df_source: pd.DataFrame,
            selected_labels: list[str],
            out_path: Path,
            plotter: Callable[[pd.DataFrame], Any],
            *,
            context_label: str,
        ) -> None:
            if not selected_labels:
                print(f"[topk_heatmap] no labels available for {context_label}, skipping.")
                return
            subset = df_source.loc[:, selected_labels]
            if topk_from_ts is not None:
                subset = subset[subset.index >= topk_from_ts]
                if subset.empty:
                    print(
                        f"[topk_heatmap] from_date {topk_from_ts.date()} removed all rows for "
                        f"{context_label}, skipping."
                    )
                    return
            subset_display = subset.copy()
            try:
                subset_display.index = pd.to_datetime(subset_display.index).strftime("%Y-%m")
            except Exception:
                subset_display.index = subset_display.index.astype(str)
            heat_df = subset_display.transpose()
            plt.figure(figsize=(max(10, len(heat_df.columns) * 0.16), max(7.5, len(heat_df.index) * 0.55)))
            try:
                ax_top = plotter(heat_df)
            except Exception as exc:
                plt.close()
                print(f"[topk_heatmap] failed to plot {context_label}: {exc}")
                return
            _style_topk_heatmap_axes(ax_top, heat_df)
            plt.tight_layout()
            plt.savefig(out_path, dpi=cfg['DPI'], bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close()
            print(f"[topk_heatmap] saved heatmap -> {out_path}")

        def _save_reference_topk_heatmap(
            df_source: pd.DataFrame,
            fname: str,
            ref_order_labels: list[str],
            ref_name: str,
            ref_freeze_actual: pd.Timestamp | None,
            force: bool = False,
        ) -> None:
            process_topk = (
                topk_k > 0 and ref_freeze_actual is not None and
                (force or not topk_features or fname.lower() in topk_features)
            )
            if not process_topk:
                return
            selected_labels = [label for label in ref_order_labels if label in df_source.columns][:topk_k]
            out_top = heatmap_dir / (
                f"{tag}__topk_heatmap_{fname}_ordered_by_{ref_name}_{ref_freeze_actual.strftime('%Y-%m-%d')}.pdf"
            )
            _save_selected_topk_heatmap(
                df_source,
                selected_labels,
                out_top,
                lambda heat_df: _plot_matrix_heatmap(heat_df, yticklabels=heat_df.index),
                context_label=f"'{fname}' ordered by {ref_name}",
            )

        df_fw_reference = None
        fw_reference_order: list[str] = []
        fw_reference_freeze_actual = None
        rel_smooth_fw_reference_order: list[str] = []
        rel_smooth_fw_reference_freeze_actual = None
        if FW is not None:
            try:
                agg_vals_ref = np.tensordot(Xt, FW, axes=(2, 0))
                df_fw_reference = pd.DataFrame(agg_vals_ref, index=ts, columns=column_labels)
                fw_reference_order, fw_reference_freeze_actual = _ordered_columns_from_reference(df_fw_reference)
            except Exception as exc:
                print(f"[heatmap] failed to prepare FW aggregate ordering reference: {exc}")
                df_fw_reference = None
                fw_reference_order = []
                fw_reference_freeze_actual = None
                rel_smooth_fw_reference_order = []
                rel_smooth_fw_reference_freeze_actual = None

        def _resolve_relative_tau(prev_vals: np.ndarray) -> float:
            tau_used = rel_tau
            if rel_tau_quantile > 0.0:
                finite_prev = prev_vals[np.isfinite(prev_vals)]
                pos_prev = finite_prev[finite_prev > 0.0]
                if pos_prev.size:
                    q_tau = float(np.nanquantile(pos_prev, rel_tau_quantile))
                    if np.isfinite(q_tau):
                        tau_used = max(tau_used, q_tau)
            return float(max(tau_used, 0.0))

        def _compute_relative_lag_df(df_in: pd.DataFrame) -> tuple[pd.DataFrame, float]:
            vals = np.asarray(df_in.values, dtype=float)
            rel_vals = np.full_like(vals, np.nan, dtype=float)
            tau_used = 0.0
            if vals.shape[0] > rel_lag:
                prev = vals[:-rel_lag]
                curr = vals[rel_lag:]
                tau_used = _resolve_relative_tau(prev)
                denom = np.asarray(prev, dtype=float).copy() + tau_used
                denom_floor = max(rel_eps, rel_denom_min_abs)
                if rel_denom_min_quantile > 0.0:
                    finite_prev = prev[np.isfinite(prev)]
                    pos_prev = np.abs(finite_prev[np.abs(finite_prev) > 0.0])
                    if pos_prev.size:
                        q_floor = float(np.nanquantile(pos_prev, rel_denom_min_quantile))
                        if np.isfinite(q_floor):
                            denom_floor = max(denom_floor, q_floor)
                if denom_floor > 0.0:
                    denom = np.where(np.isfinite(denom), np.maximum(denom, denom_floor), np.nan)
                mask = np.isfinite(curr) & np.isfinite(denom) & (np.abs(denom) > 0.0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel_vals[rel_lag:] = np.where(mask, (curr - prev) / denom, np.nan)
            if rel_ignore_reactivation and rel_mask_activation_months > 0:
                active = np.isfinite(vals) & (vals > rel_activity_thr)
                if active.size:
                    starts = np.zeros_like(active, dtype=bool)
                    starts[0, :] = active[0, :]
                    starts[1:, :] = active[1:, :] & (~active[:-1, :])
                    start_rows, start_cols = np.where(starts)
                    if start_rows.size:
                        t_max = vals.shape[0]
                        for r, c in zip(start_rows.tolist(), start_cols.tolist()):
                            # Mask onset points (activity just started).
                            end_onset = min(t_max, r + rel_mask_activation_months)
                            rel_vals[r:end_onset, c] = np.nan
                            # Also mask lag-shifted points whose denominator comes
                            # from this fragile onset window.
                            start_denom = r + rel_lag
                            if start_denom < t_max:
                                end_denom = min(t_max, start_denom + rel_mask_activation_months)
                                rel_vals[start_denom:end_denom, c] = np.nan
            if rel_ignore_first:
                finite_mask = np.isfinite(rel_vals)
                has_finite = finite_mask.any(axis=0)
                if np.any(has_finite):
                    first_finite = np.argmax(finite_mask, axis=0)
                    cols = np.where(has_finite)[0]
                    rel_vals[first_finite[cols], cols] = np.nan
            rel_vals[~np.isfinite(rel_vals)] = np.nan
            return pd.DataFrame(rel_vals, index=df_in.index, columns=df_in.columns), tau_used

        def _compute_trailing_mean_df(df_in: pd.DataFrame, window: int) -> pd.DataFrame:
            if window <= 1:
                return df_in.copy()
            return df_in.rolling(window=window, min_periods=window).mean()

        if df_fw_reference is not None and rel_enabled and rel_smooth_enabled:
            try:
                df_fw_reference_smooth = _compute_trailing_mean_df(df_fw_reference, rel_smooth_window)
                rel_smooth_fw_reference, _rel_smooth_fw_tau = _compute_relative_lag_df(df_fw_reference_smooth)
                rel_smooth_fw_reference = rel_smooth_fw_reference.dropna(how="all")
                (
                    rel_smooth_fw_reference_order,
                    rel_smooth_fw_reference_freeze_actual,
                ) = _ordered_columns_from_reference(rel_smooth_fw_reference)
            except Exception as exc:
                print(f"[heatmap] failed to prepare smooth-relative FW ordering reference: {exc}")
                rel_smooth_fw_reference_order = []
                rel_smooth_fw_reference_freeze_actual = None

        def _save_relative_heatmap(
            rel_df: pd.DataFrame,
            *,
            fname: str,
            out_dir: Path,
            out_suffix: str,
            label_prefix: str,
            tau_used: float,
        ) -> None:
            rel_df = rel_df.dropna(how="all")
            if rel_df.empty:
                print(f"[relative_evolution] no valid rows for '{fname}' ({label_prefix}); skipping.")
                return
            rel_sorted = _sort_columns_by_freeze(rel_df)
            rel_array = rel_sorted.values
            finite = rel_array[np.isfinite(rel_array)]
            if finite.size == 0:
                print(f"[relative_evolution] no finite values for '{fname}' ({label_prefix}); skipping.")
                return
            plt.figure(figsize=(14, 6))
            ax_rel = _plot_relative_matrix_heatmap(
                rel_sorted,
                yticklabels=False,
                label_prefix=label_prefix,
                tau_used=tau_used,
            )
            ax_rel.set_xlabel("Ranked concept id")
            ax_rel.set_ylabel("Date")
            year_ticks(ax_rel, pd.DatetimeIndex(rel_sorted.index))
            node_ticks(ax_rel, rel_sorted.shape[1], cfg["XTICK_STEP"])
            plt.tight_layout()
            out_rel = out_dir / f"{tag}__heatmap_{out_suffix}_lag{rel_lag}_{fname}.pdf"
            plt.savefig(out_rel, dpi=cfg["DPI"], bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close()
            print(f"[relative_evolution] saved heatmap -> {out_rel} (tau={tau_used:.6g})")

        def _plot_relative_evolution_heatmap(
            df_base: pd.DataFrame,
            fname: str,
            *,
            force_topk: bool = False,
        ) -> None:
            if not rel_enabled:
                return
            rel_df, rel_tau_used = _compute_relative_lag_df(df_base)
            _save_relative_heatmap(
                rel_df,
                fname=fname,
                out_dir=rel_save_dir,
                out_suffix="relative",
                label_prefix="relative growth",
                tau_used=rel_tau_used,
            )
            if not rel_smooth_enabled:
                return
            df_smooth = _compute_trailing_mean_df(df_base, rel_smooth_window)
            rel_smooth_df, rel_smooth_tau = _compute_relative_lag_df(df_smooth)
            _save_relative_heatmap(
                rel_smooth_df,
                fname=fname,
                out_dir=rel_smooth_save_dir,
                out_suffix=f"relative_smooth_trailing{rel_smooth_window}",
                label_prefix="relative growth",
                tau_used=rel_smooth_tau,
            )
            process_topk = (
                topk_k > 0 and rel_smooth_fw_reference_freeze_actual is not None and
                rel_smooth_fw_reference_order and
                (force_topk or not topk_features or fname.lower() in topk_features)
            )
            if not process_topk:
                return
            selected_labels = [
                label for label in rel_smooth_fw_reference_order if label in rel_smooth_df.columns
            ][:topk_k]
            out_top = rel_smooth_save_dir / (
                f"{tag}__topk_heatmap_relative_smooth_trailing{rel_smooth_window}_lag{rel_lag}_{fname}_"
                "ordered_by_relative_smooth_fw_aggregate_"
                f"{rel_smooth_fw_reference_freeze_actual.strftime('%Y-%m-%d')}.pdf"
            )
            _save_selected_topk_heatmap(
                rel_smooth_df,
                selected_labels,
                out_top,
                lambda heat_df: _plot_relative_matrix_heatmap(
                    heat_df,
                    yticklabels=heat_df.index,
                    label_prefix="relative growth",
                    tau_used=rel_smooth_tau,
                ),
                context_label=f"'{fname}' smooth relative ordered by relative_smooth_fw_aggregate",
            )

        for idx, fname in zip(feat_idx, feat_names):
            data = Xt[:, :, idx]
            df = pd.DataFrame(data, index=ts)
            df.columns = column_labels

            mode, eps = cfg['EMERGENCE_MODE'], float(cfg['EPSILON'])
            if mode == 'raw':
                df_sorted = _sort_columns_by_freeze(df)
                Z = np.log1p(np.clip(df_sorted.values, a_min=0.0, a_max=None))
                vmin = 0.0; vmax = np.nanpercentile(Z, 99.5)
                norm = PowerNorm(gamma=1.8, vmin=vmin, vmax=vmax)
                plt.figure(figsize=(14, 6))
                ax = sns.heatmap(Z, cmap=cmap_name, norm=norm, yticklabels=False,
                                 cbar_kws={"label": "log(1 + emergence)"})
            elif mode == 'ratio':
                df_sorted = _sort_columns_by_freeze(df)
                v  = df_sorted.values[np.isfinite(df_sorted.values)]
                vmax = np.nanpercentile(v, cfg['CAP_PCT']) if v.size else 1.0
                vmax = max(vmax, 1.0)
                vmin = np.nanmin(v) if v.size else 0.0
                plt.figure(figsize=(14, 6))
                ax = sns.heatmap(np.clip(df_sorted.values, a_min=vmin, a_max=vmax),
                                 cmap=cmap_name, vmin=vmin, vmax=vmax, yticklabels=False,
                                 cbar_kws={"label": f"ratio = curr/(prev{' + ε' if eps>0 else ''}), ε={eps:g}"})
            elif mode == 'pct':
                df_sorted = _sort_columns_by_freeze(df)
                v  = df_sorted.values[np.isfinite(df_sorted.values)]
                vmax = np.nanpercentile(v, cfg['CAP_PCT']) if v.size else 1.0
                vmax = max(vmax, 1.0)
                vmin = np.nanmin(v) if v.size else 0.0
                plt.figure(figsize=(14, 6))
                ax = sns.heatmap(np.clip(df_sorted.values, a_min=vmin, a_max=vmax),
                                 cmap=cmap_name, vmin=vmin, vmax=vmax, yticklabels=False,
                                 cbar_kws={"label": f"pct = curr/(prev{' + ε' if eps>0 else ''}) - 1, ε={eps:g}"})
            else:
                raise ValueError("EMERGENCE_MODE must be 'raw', 'ratio', or 'pct'.")

            ax.set_xlabel("Ranked concept id"); ax.set_ylabel("Date")
            year_ticks(ax, ts); node_ticks(ax, N, cfg['XTICK_STEP'])
            plt.tight_layout()
            out = heatmap_dir / f"{tag}__heatmap_{mode}_{fname}.pdf"
            plt.savefig(out, dpi=cfg['DPI'], bbox_inches="tight");
            if show: plt.show()
            else: plt.close()
            _generate_topk_outputs(df, fname, allow_bubble=True)
            _plot_relative_evolution_heatmap(df, fname)
            if fw_reference_order and fw_reference_freeze_actual is not None:
                try:
                    ordered_labels = [label for label in fw_reference_order if label in df.columns]
                    df_fw_ordered = df.loc[:, ordered_labels]
                    out_fw_ordered = heatmap_dir / (
                        f"{tag}__heatmap_{mode}_{fname}_ordered_by_fw_aggregate_"
                        f"{fw_reference_freeze_actual.strftime('%Y-%m-%d')}.pdf"
                    )
                    _save_global_heatmap(df_fw_ordered, out_fw_ordered)
                    _save_reference_topk_heatmap(
                        df,
                        fname,
                        ordered_labels,
                        "fw_aggregate",
                        fw_reference_freeze_actual,
                    )
                except Exception as exc:
                    print(f"[heatmap] failed to build ordered-by-FW view for '{fname}': {exc}")

        # FW-weighted aggregate heatmap and top-k
        try:
            if FW is None:
                raise ValueError("FW not available for aggregate heatmap")
            if df_fw_reference is None:
                agg_vals = np.tensordot(Xt, FW, axes=(2, 0))
                df_fw = pd.DataFrame(agg_vals, index=ts, columns=column_labels)
            else:
                df_fw = df_fw_reference
            df_fw_sorted = _sort_columns_by_freeze(df_fw)
            Z = np.log1p(np.clip(df_fw_sorted.values, a_min=0.0, a_max=None))
            vmin = 0.0; vmax = np.nanpercentile(Z, 99.5)
            norm = PowerNorm(gamma=1.8, vmin=vmin, vmax=vmax)
            plt.figure(figsize=(14, 6))
            ax = sns.heatmap(Z, cmap=cmap_name, norm=norm, yticklabels=False,
                             cbar_kws={"label": "log(1 + emergence)"})
            ax.set_xlabel("Ranked concept id"); ax.set_ylabel("Date")
            year_ticks(ax, ts); node_ticks(ax, N, cfg['XTICK_STEP'])
            plt.tight_layout()
            out_fw = heatmap_dir / f"{tag}__heatmap_fw_aggregate.pdf"
            plt.savefig(out_fw, dpi=cfg['DPI'], bbox_inches="tight");
            if show: plt.show()
            else: plt.close()
            _generate_topk_outputs(df_fw, "fw_aggregate", allow_bubble=True, force=True)
            _plot_relative_evolution_heatmap(df_fw, "fw_aggregate", force_topk=True)
            if fw_reference_order and fw_reference_freeze_actual is not None:
                ordered_labels = [label for label in fw_reference_order if label in df_fw.columns]
                df_fw_ordered = df_fw.loc[:, ordered_labels]
                out_fw_ordered = heatmap_dir / (
                    f"{tag}__heatmap_fw_aggregate_balanced_fw_ordered_by_fw_aggregate_"
                    f"{fw_reference_freeze_actual.strftime('%Y-%m-%d')}.pdf"
                )
                _save_global_heatmap(df_fw_ordered, out_fw_ordered)
                _save_reference_topk_heatmap(
                    df_fw,
                    "fw_aggregate",
                    ordered_labels,
                    "fw_aggregate",
                    fw_reference_freeze_actual,
                    force=True,
                )
        except Exception as exc:
            print(f"[heatmap] failed to build FW aggregate view: {exc}")

            if topk_wordcloud > 0 and wordcloud_dir is not None:
                wc_values = np.asarray(df.values, dtype=float)
                wc_values = np.where(np.isfinite(wc_values), wc_values, 0.0)

                weights = None
                if topk_freeze_ts is not None:
                    try:
                        wc_idx = df.index.get_indexer([topk_freeze_ts], method="nearest")[0]
                    except Exception:
                        wc_idx = -1
                    if 0 <= wc_idx < wc_values.shape[0]:
                        weights = np.clip(wc_values[wc_idx, :], a_min=0.0, a_max=None)
                    else:
                        print(f"[wordcloud] freeze date {topk_freeze_ts.date()} outside range, falling back to totals.")
                if weights is None:
                    weights = np.clip(wc_values.sum(axis=0), a_min=0.0, a_max=None)

                if np.any(weights > 0):
                    order = np.argsort(weights)[::-1]
                    freq = {}
                    for i in order[:topk_wordcloud]:
                        w = float(weights[i])
                        if w > 0:
                            freq[column_labels[i]] = w
                    if freq:
                        try:
                            wc = WordCloud(width=1600, height=900, background_color="white")
                            wc.generate_from_frequencies(freq)
                            safe_name = fname.replace(" ", "_")
                            out_wc = wordcloud_dir / f"{tag}__wordcloud_{safe_name}.pdf"
                            fig_wc, ax_wc = plt.subplots(figsize=(8, 4.5))
                            ax_wc.imshow(wc, interpolation="bilinear")
                            ax_wc.axis("off")
                            fig_wc.tight_layout()
                            fig_wc.savefig(out_wc, bbox_inches="tight")
                            plt.close(fig_wc)
                            print(f"[wordcloud] saved: {out_wc}")
                        except Exception as exc:  # pragma: no cover
                            print(f"[wordcloud] failed for '{fname}': {exc}")
                    else:
                        print(f"[wordcloud] {fname}: all weights filtered to zero, skipped.")
                else:
                    print(f"[wordcloud] {fname}: total weight is zero, skipped.")

            _generate_topk_outputs(df, fname, allow_bubble=True)

        # --- Additional Top-K heatmap using summed co-occurrence weights ---
        if mats is not None:
            try:
                mats_float = np.clip(np.asarray(mats, dtype=float), a_min=0.0, a_max=None)
                coocc_values = mats_float.sum(axis=1).transpose(1, 0)  # (T, N)
                coocc_df = pd.DataFrame(coocc_values, index=ts, columns=column_labels)
                _generate_topk_outputs(coocc_df, "coocc_sum", allow_bubble=False)
            except Exception as exc:
                print(f"[topk_heatmap] unable to build co-occurrence summary: {exc}")

                if topk_corr_k > 0 and corr_dir is not None:
                    corr_series = top_series.head(topk_corr_k)
                    if not corr_series.empty:
                        selected_infos = []
                        for nm in corr_series.index:
                            infos = column_lookup.get(nm)
                            if infos:
                                selected_infos.append(infos[0])
                        if selected_infos:
                            active_idx = [info["active_idx"] for info in selected_infos]
                            labels = [info["label"] for info in selected_infos]
                            adj_subset = None
                            try:
                                adj_slice = mats[:, :, freeze_idx]
                                adj_subset = adj_slice[np.ix_(active_idx, active_idx)].astype(float)
                            except Exception as exc:  # pragma: no cover
                                print(f"[adjacency] failed for {fname}: {exc}")
                            if adj_subset is not None:
                                fig_adj, ax_adj = plt.subplots(figsize=(8, 6))
                                sns.heatmap(np.nan_to_num(adj_subset), ax=ax_adj, cmap="viridis", square=True,
                                            xticklabels=labels, yticklabels=labels,
                                            cbar_kws={"label": "Co-occurrence weight"})
                                plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
                                plt.tight_layout()
                                safe_name = fname.replace(" ", "_")
                                out_adj = corr_dir / f"{tag}__adjacency_{safe_name}_{freeze_actual.strftime('%Y-%m')}.pdf"
                                fig_adj.savefig(out_adj, dpi=cfg['DPI'], bbox_inches="tight")
                                plt.close(fig_adj)
                                print(f"[correlation] saved adjacency heatmap: {out_adj}")
                            if topk_corr_from_ts is not None:
                                data_window = df.loc[(df.index >= topk_corr_from_ts) & (df.index <= freeze_actual), labels]
                            else:
                                start_idx = 0
                                if topk_corr_window > 0:
                                    start_idx = max(0, freeze_idx - topk_corr_window + 1)
                                data_window = df.iloc[start_idx:freeze_idx + 1, :][labels]

                            data_window = data_window.astype(float)
                            safe_name = fname.replace(" ", "_")

                            need_log_window = topk_corr_use_logdiff or topk_corr_linfits
                            log_window = None
                            if need_log_window:
                                log_window = np.log1p(np.clip(data_window, a_min=0.0, a_max=None))

                            if topk_corr_linfits and log_window is not None and log_window.shape[0] >= 2:
                                t_idx = np.arange(log_window.shape[0], dtype=float)
                                lin_scores = {}
                                for nm in labels:
                                    series = log_window[nm].to_numpy(dtype=float)
                                    if np.allclose(series, series[0]):
                                        corr_val = 0.0
                                    else:
                                        corr_val = float(np.nan_to_num(np.corrcoef(t_idx, series)[0, 1]))
                                    lin_scores[nm] = corr_val
                                lin_df = pd.DataFrame.from_dict(lin_scores, orient="index", columns=["corr_time_log"])
                                lin_df = lin_df.sort_values("corr_time_log", ascending=False)
                                lin_path = corr_dir / f"{tag}__linfit_{safe_name}_{freeze_actual.strftime('%Y-%m')}.csv"
                                if not lin_path.exists():
                                    lin_df.to_csv(lin_path)
                                    print(f"[correlation] saved log-linear fit scores: {lin_path}")

                            for mode_name in corr_modes:
                                if mode_name == "log":
                                    if log_window is None:
                                        corr_base = pd.DataFrame(columns=data_window.columns)
                                    else:
                                        corr_base = log_window.diff().dropna()
                                    suffix = "_dlog"
                                    label = "Pearson correlation of Δlog(1 + value)"
                                elif mode_name == "raw":
                                    corr_base = data_window.diff().dropna()
                                    suffix = "_diff"
                                    label = "Pearson correlation of Δvalue"
                                else:
                                    corr_base = data_window
                                    suffix = ""
                                    label = "Pearson correlation"

                                if corr_base.shape[0] >= 2:
                                    corr_matrix = np.corrcoef(corr_base.to_numpy(dtype=float).T)
                                    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                                    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                                    sns.heatmap(corr_matrix, ax=ax_corr, cmap="coolwarm", square=True, center=0.0,
                                                vmin=-1.0, vmax=1.0,
                                                xticklabels=labels, yticklabels=labels,
                                                cbar_kws={"label": label})
                                    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
                                    plt.tight_layout()
                                    out_corr = corr_dir / f"{tag}__correlation{suffix}_{safe_name}_{freeze_actual.strftime('%Y-%m')}.pdf"
                                    fig_corr.savefig(out_corr, dpi=cfg['DPI'], bbox_inches="tight")
                                    plt.close(fig_corr)
                                    print(f"[correlation] saved correlation heatmap: {out_corr}")
                                else:
                                    if mode_name == "log":
                                        mode_desc = "Δlog(1+value)"
                                    elif mode_name == "raw":
                                        mode_desc = "Δvalue"
                                    else:
                                        mode_desc = "levels"
                                    print(f"[correlation] insufficient history ({mode_desc}) for {fname}, skipped correlation heatmap.")

                if topk_corr_k > 0 and corr_dir is not None:
                    corr_series = top_series.head(topk_corr_k)
                    if not corr_series.empty:
                        selected_infos = []
                        for nm in corr_series.index:
                            infos = column_lookup.get(nm)
                            if infos:
                                selected_infos.append(infos[0])
                        if selected_infos:
                            active_idx = [info["active_idx"] for info in selected_infos]
                            labels = [info["label"] for info in selected_infos]
                            adj_subset = None
                            try:
                                adj_slice = mats[:, :, freeze_idx]
                                adj_subset = adj_slice[np.ix_(active_idx, active_idx)].astype(float)
                            except Exception as exc:  # pragma: no cover
                                print(f"[adjacency] failed for {fname}: {exc}")
                            if adj_subset is not None:
                                fig_adj, ax_adj = plt.subplots(figsize=(8, 6))
                                sns.heatmap(np.nan_to_num(adj_subset), ax=ax_adj, cmap="viridis", square=True,
                                            xticklabels=labels, yticklabels=labels,
                                            cbar_kws={"label": "Co-occurrence weight"})
                                plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
                                plt.tight_layout()
                                safe_name = fname.replace(" ", "_")
                                out_adj = corr_dir / f"{tag}__adjacency_{safe_name}_{freeze_actual.strftime('%Y-%m')}.pdf"
                                fig_adj.savefig(out_adj, dpi=cfg['DPI'], bbox_inches="tight")
                                plt.close(fig_adj)
                                print(f"[correlation] saved adjacency heatmap: {out_adj}")
                            data_window = df.iloc[:freeze_idx + 1, :][labels].astype(float)
                            need_log_window = topk_corr_use_logdiff or topk_corr_linfits
                            log_window = None
                            if need_log_window:
                                log_window = np.log1p(np.clip(data_window, a_min=0.0, a_max=None))
                            if topk_corr_linfits and log_window is not None and log_window.shape[0] >= 2:
                                t_idx = np.arange(log_window.shape[0], dtype=float)
                                lin_scores = {}
                                for nm in labels:
                                    series = log_window[nm].to_numpy(dtype=float)
                                    if np.allclose(series, series[0]):
                                        corr_val = 0.0
                                    else:
                                        corr_val = float(np.nan_to_num(np.corrcoef(t_idx, series)[0, 1]))
                                    lin_scores[nm] = corr_val
                                lin_df = pd.DataFrame.from_dict(lin_scores, orient="index", columns=["corr_time_log"])
                                lin_df = lin_df.sort_values("corr_time_log", ascending=False)
                                lin_path = corr_dir / f"{tag}__linfit_{safe_name}_{freeze_actual.strftime('%Y-%m')}.csv"
                                if not lin_path.exists():
                                    lin_df.to_csv(lin_path)
                                    print(f"[correlation] saved log-linear fit scores: {lin_path}")
                            for mode_name in corr_modes:
                                if mode_name == "log":
                                    if log_window is None:
                                        corr_base = pd.DataFrame(columns=data_window.columns)
                                    else:
                                        corr_base = log_window.diff().dropna()
                                    suffix = "_dlog"
                                    label = "Pearson correlation of Δlog(1 + value)"
                                elif mode_name == "raw":
                                    corr_base = data_window.diff().dropna()
                                    suffix = "_diff"
                                    label = "Pearson correlation of Δvalue"
                                else:
                                    corr_base = data_window
                                    suffix = ""
                                    label = "Pearson correlation"
                                if corr_base.shape[0] >= 2:
                                    corr_matrix = np.corrcoef(corr_base.to_numpy(dtype=float).T)
                                    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                                    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                                    sns.heatmap(corr_matrix, ax=ax_corr, cmap="coolwarm", square=True, center=0.0,
                                                vmin=-1.0, vmax=1.0,
                                                xticklabels=labels, yticklabels=labels,
                                                cbar_kws={"label": label})
                                    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
                                    plt.tight_layout()
                                    out_corr = corr_dir / f"{tag}__correlation{suffix}_{safe_name}_{freeze_actual.strftime('%Y-%m')}.pdf"
                                    fig_corr.savefig(out_corr, dpi=cfg['DPI'], bbox_inches="tight")
                                    plt.close(fig_corr)
                                    print(f"[correlation] saved correlation heatmap: {out_corr}")
                                else:
                                    if mode_name == "log":
                                        mode_desc = "Δlog(1+value)"
                                    elif mode_name == "raw":
                                        mode_desc = "Δvalue"
                                    else:
                                        mode_desc = "levels"
                                    print(f"[correlation] insufficient history ({mode_desc}) for {fname}, skipped correlation heatmap.")

    # ================================================================
    # Training bits (operate on *transformed* Xt)
    # ================================================================

    def make_criterion(name: str, delta: float):
        nm = (name or "").lower()
        if nm in ("l1", "mae"):   return (lambda pred, y: torch.abs(pred - y)), "mae"
        if nm == "huber":            return nn.HuberLoss(delta=delta, reduction='none'), "huber"
        if nm in ("mse", "rmse"):   return (lambda pred, y: (pred - y) ** 2), nm
        return nn.HuberLoss(delta=delta, reduction='none'), "huber"


    def fw_of_frame(fr, FW):
        return fr @ FW


    def _compute_smoothed_relative_level_series(values: np.ndarray, cfg_like: dict[str, Any] | None) -> np.ndarray:
        cfg_src = cfg_like if isinstance(cfg_like, dict) else {}
        smooth_window = int(cfg_src.get("TARGET_RELATIVE_SMOOTH_WINDOW", 12) or 12)
        rel_lag = int(cfg_src.get("TARGET_RELATIVE_LAG", 12) or 12)
        rel_eps = float(cfg_src.get("TARGET_RELATIVE_EPS", 1e-8) or 0.0)
        rel_tau = float(cfg_src.get("TARGET_RELATIVE_TAU", 0.0) or 0.0)
        rel_tau_quantile = float(cfg_src.get("TARGET_RELATIVE_TAU_QUANTILE", 0.25) or 0.0)
        rel_denom_min_abs = float(cfg_src.get("TARGET_RELATIVE_DENOM_MIN_ABS", 0.0) or 0.0)
        rel_denom_min_quantile = float(cfg_src.get("TARGET_RELATIVE_DENOM_MIN_QUANTILE", 0.0) or 0.0)
        smooth_window = max(1, smooth_window)
        rel_lag = max(1, rel_lag)

        vals = np.asarray(values, dtype=float)
        if vals.ndim != 2:
            raise ValueError("Smoothed relative target expects a 2D (T, N) array.")

        if smooth_window > 1:
            smooth_vals = pd.DataFrame(vals).rolling(smooth_window, min_periods=smooth_window).mean().to_numpy(dtype=float)
        else:
            smooth_vals = vals.copy()

        rel_vals = np.full_like(smooth_vals, np.nan, dtype=float)
        if smooth_vals.shape[0] <= rel_lag:
            return rel_vals

        prev = smooth_vals[:-rel_lag]
        curr = smooth_vals[rel_lag:]
        tau_used = rel_tau
        if rel_tau_quantile > 0.0:
            finite_prev = prev[np.isfinite(prev)]
            pos_prev = finite_prev[finite_prev > 0.0]
            if pos_prev.size:
                q_tau = float(np.nanquantile(pos_prev, rel_tau_quantile))
                if np.isfinite(q_tau):
                    tau_used = max(tau_used, q_tau)

        denom = np.asarray(prev, dtype=float).copy() + float(max(tau_used, 0.0))
        denom_floor = max(rel_eps, rel_denom_min_abs)
        if rel_denom_min_quantile > 0.0:
            finite_prev = prev[np.isfinite(prev)]
            pos_prev = np.abs(finite_prev[np.abs(finite_prev) > 0.0])
            if pos_prev.size:
                q_floor = float(np.nanquantile(pos_prev, rel_denom_min_quantile))
                if np.isfinite(q_floor):
                    denom_floor = max(denom_floor, q_floor)
        if denom_floor > 0.0:
            denom = np.where(np.isfinite(denom), np.maximum(denom, denom_floor), np.nan)

        mask = np.isfinite(curr) & np.isfinite(denom) & (np.abs(denom) > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_vals[rel_lag:] = np.where(mask, (curr - prev) / denom, np.nan)
        rel_vals[~np.isfinite(rel_vals)] = np.nan
        return rel_vals


    def _target_level_series(features: np.ndarray, fw: np.ndarray, mode: str, cfg_like: dict[str, Any] | None = None) -> np.ndarray:
        mode = (mode or "").strip().lower()
        if mode == "absolute":
            mode = "level"
        fw_vals = np.stack([fw_of_frame(features[t], fw) for t in range(features.shape[0])])
        if mode in {"level", "residual", "drift_residual", "log_change"}:
            return fw_vals
        if mode in {"smooth_relative", "smooth_relative_level"}:
            return _compute_smoothed_relative_level_series(fw_vals, cfg_like)
        raise ValueError("Unknown TARGET_MODE")


    def compute_targets(features, forecast, fw, mode, *, drift_lag=None, drift_damp=1.0, cfg_like=None):
        T, N, F = features.shape
        mode = (mode or "").strip().lower()
        if mode == "absolute":
            mode = "level"
        drift_vals = None
        level_vals = None
        if mode == "drift_residual":
            fw_vals = _target_level_series(features, fw, "level", cfg_like)
            lag = int(drift_lag) if drift_lag is not None else max(1, int(forecast))
            damp = float(drift_damp)
            drift_vals = _compute_drift_level(fw_vals, horizon=int(forecast), lag=lag, damp=damp)
        elif mode in {"level", "smooth_relative", "smooth_relative_level"}:
            level_vals = _target_level_series(features, fw, mode, cfg_like)
        out, masks = [], []
        for t in range(T - forecast):
            cur = features[t]
            fut = features[t + forecast]
            if mode == "level":
                y = level_vals[t + forecast]
                valid = np.isfinite(y)
            elif mode == "residual":
                valid = np.isfinite(fut).all(axis=1) & (np.abs(fut).sum(axis=1) > 0)
                y = fw_of_frame(fut, fw) - fw_of_frame(cur, fw)
            elif mode == "drift_residual":
                valid = np.isfinite(fut).all(axis=1) & (np.abs(fut).sum(axis=1) > 0)
                drift_fut = drift_vals[t] if drift_vals is not None else 0.0
                y = fw_of_frame(fut, fw) - drift_fut
            elif mode == "log_change":
                valid = np.isfinite(fut).all(axis=1) & (np.abs(fut).sum(axis=1) > 0)
                y = np.arcsinh(fw_of_frame(fut, fw)) - np.arcsinh(fw_of_frame(cur, fw))
            elif mode in {"smooth_relative", "smooth_relative_level"}:
                y = level_vals[t + forecast]
                valid = np.isfinite(y)
            else:
                raise ValueError("Unknown TARGET_MODE")
            out.append(y); masks.append(valid.astype(np.float32))
        return np.stack(out), np.stack(masks).astype(np.float32)


    def to_loss_space(arr, space):
        if space == "raw":   return arr
        if space == "log1p": return np.log1p(np.maximum(arr, 0.0))
        if space == "asinh": return np.arcsinh(arr)
        raise ValueError("LOSS_SPACE")


    def from_loss_space(arr, space):
        if space == "raw":   return arr
        if space == "log1p": return np.expm1(arr)
        if space == "asinh": return np.sinh(arr)
        raise ValueError("LOSS_SPACE")


    def fit_standardizer(train_targets, train_masks, per_node=False):
        if per_node:
            mu = np.zeros(train_targets.shape[1], dtype=np.float32)
            sd = np.ones(train_targets.shape[1], dtype=np.float32)
            m = (train_masks > 0).astype(bool)
            for n in range(train_targets.shape[1]):
                v = train_targets[:, n][m[:, n]]
                mu[n] = float(v.mean()) if v.size else 0.0
                sd[n] = float(v.std() + 1e-6) if v.size else 1.0
            return mu, sd
        v = train_targets[train_masks > 0]
        return float(v.mean()), float(v.std() + 1e-6)


    def apply_standardizer(targets, mu, sd, per_node=False):
        if per_node: return (targets - mu[None, :]) / sd[None, :]
        return (targets - mu) / sd


    def invert_std_and_space(arr_std, mu, sd, per_node, space):
        if per_node: arr = arr_std * sd[None, :] + mu[None, :]
        else:        arr = arr_std * sd + mu
        return from_loss_space(arr, space)


    def masked_metrics(y_true, y_pred, msk):
        diff = (y_pred - y_true)
        m = (msk > 0)
        if diff.ndim == 2: diff = diff[m]
        mae = float(np.abs(diff).mean()) if diff.size else float('nan')
        rmse = float(np.sqrt(np.mean(diff**2))) if diff.size else float('nan')
        return mae, rmse


    def normalize_edge_attr(edge_index, edge_attr, N, mode="deg"):
        if edge_index.numel() == 0 or mode == "none":
            return edge_attr
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        if mode == "deg":
            deg = torch.bincount(edge_index[0], minlength=N).clamp(min=1).float()
            norm = (deg[edge_index[0]].rsqrt() * deg[edge_index[1]].rsqrt()).unsqueeze(-1)
            return edge_attr * norm
        if mode == "row":
            feat_dim = edge_attr.size(1)
            wsum = torch.zeros(N, feat_dim, device=edge_attr.device)
            if edge_attr.numel():
                wsum.index_add_(0, edge_index[0], edge_attr)
            wsum = torch.clamp(wsum, min=1e-6)
            return edge_attr / wsum[edge_index[0]]
        if mode == "minmax":
            if edge_attr.numel() == 0:
                return edge_attr
            lo = edge_attr.min(dim=0).values
            hi = edge_attr.max(dim=0).values
            span = (hi - lo).clamp(min=1e-6)
            return (edge_attr - lo) / span
        raise ValueError("EDGE_NORM")


    def prepare_data(adj_matrices, node_features, edge_self_loops, edge_norm, edge_attr_mode="weight"):
        seq = []
        T = node_features.shape[0]; N = node_features.shape[1]
        for t in range(T):
            A = adj_matrices[:, :, t]
            r, c = np.nonzero(A)
            if r.size == 0:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr  = torch.zeros((0, 1), dtype=torch.float)
            else:
                edge_index = torch.tensor(np.vstack([r, c]), dtype=torch.long)
                edge_attr  = torch.tensor(A[r, c], dtype=torch.float).unsqueeze(-1)
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                if edge_self_loops:
                    edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, fill_value=1.0, num_nodes=N)
                if edge_attr_mode == "binary":
                    edge_attr = torch.ones_like(edge_attr)
                elif edge_attr_mode != "weight":
                    raise ValueError("EDGE_ATTR_MODE")
                edge_attr = normalize_edge_attr(edge_index, edge_attr, N, mode=edge_norm)
            x = torch.tensor(node_features[t], dtype=torch.float)
            seq.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
        return seq


    def _parse_int_list(raw):
        if raw is None:
            return []
        if isinstance(raw, (list, tuple, set)):
            vals = list(raw)
        elif isinstance(raw, str):
            parts = [p.strip() for p in raw.replace(";", ",").replace(" ", ",").split(",")]
            vals = [p for p in parts if p]
        else:
            vals = [raw]
        out = []
        for v in vals:
            try:
                out.append(int(v))
            except Exception:
                continue
        return out


    def _parse_lags(cfg):
        lags = _parse_int_list(cfg.get("GRAPH_MULTI_LAGS", [0]))
        lags = [l for l in lags if l >= 0]
        if 0 not in lags:
            lags = [0] + lags
        return sorted(set(lags))


    def prepare_data_multi(adj_matrices, node_features, edge_self_loops, edge_norm,
                           edge_attr_mode="weight", edge_lags=None):
        if edge_lags is None:
            edge_lags = [0]
        edge_lags = [int(l) for l in edge_lags if int(l) >= 0]
        if not edge_lags:
            edge_lags = [0]
        edge_lags = sorted(set(edge_lags))
        L = len(edge_lags)
        seq = []
        T = node_features.shape[0]; N = node_features.shape[1]
        for t in range(T):
            lag_data = []
            for lag in edge_lags:
                t_idx = t - lag
                if t_idx < 0:
                    lag_data.append((np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)))
                    continue
                A = adj_matrices[:, :, t_idx]
                r, c = np.nonzero(A)
                w = A[r, c] if r.size else np.array([], dtype=float)
                lag_data.append((r, c, w))

            if all(r.size == 0 for r, _, _ in lag_data):
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, L), dtype=torch.float)
            else:
                idx_all = np.concatenate([r * N + c for r, c, _ in lag_data if r.size])
                uniq_idx = np.unique(idx_all)
                edge_attr_np = np.zeros((uniq_idx.size, L), dtype=float)
                for li, (r, c, w) in enumerate(lag_data):
                    if r.size == 0:
                        continue
                    idx = r * N + c
                    pos = np.searchsorted(uniq_idx, idx)
                    edge_attr_np[pos, li] = w
                edge_index = torch.tensor(np.vstack([uniq_idx // N, uniq_idx % N]), dtype=torch.long)
                edge_attr = torch.tensor(edge_attr_np, dtype=torch.float)
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                if edge_self_loops:
                    edge_index, edge_attr = add_self_loops(
                        edge_index, edge_attr=edge_attr, fill_value=1.0, num_nodes=N
                    )
                if edge_attr_mode == "binary":
                    edge_attr = torch.ones_like(edge_attr)
                elif edge_attr_mode != "weight":
                    raise ValueError("EDGE_ATTR_MODE")
                edge_attr = normalize_edge_attr(edge_index, edge_attr, N, mode=edge_norm)

            x = torch.tensor(node_features[t], dtype=torch.float)
            seq.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
        return seq

    # ---- model ----
    class GTAN(nn.Module):
        def __init__(self, in_channels, hidden_channels, num_heads, dropout,
                     use_pos, temp_window, node_only, graph_gate, graph_mix,
                     edge_dim=1, copy_head=False, fw=None, aux_forecasts=None):
            super().__init__()
            self.hidden = hidden_channels
            self.node_only = node_only
            self.use_pos = use_pos
            self.copy_head = copy_head
            self.fw = fw
            self.graph_mix = graph_mix
            self.last_alpha = None
            self.aux_forecasts = [int(f) for f in (aux_forecasts or [])]
            self.aux_outs = nn.ModuleList([nn.Linear(hidden_channels, 1) for _ in self.aux_forecasts])
            self.last_aux = None

            self.node_mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            self.graph_attn = None
            if not self.node_only:
                self.graph_attn = TransformerConv(
                    in_channels, hidden_channels, heads=num_heads,
                    concat=False, dropout=dropout, edge_dim=edge_dim
                )

            self.temporal_attn = nn.MultiheadAttention(
                hidden_channels, num_heads, batch_first=True, dropout=dropout
            )

            if self.use_pos:
                self.pos = nn.Parameter(torch.zeros(1, temp_window, hidden_channels))

            self.dropout = nn.Dropout(dropout)
            self.out = nn.Linear(hidden_channels, 1)
            self.out_graph = None
            self.out_node = None
            self.mix_gate = None
            if self.graph_mix and not self.node_only:
                self.out_graph = nn.Linear(hidden_channels, 1)
                self.out_node = nn.Linear(hidden_channels, 1)
                self.mix_gate = nn.Sequential(
                    nn.Linear(hidden_channels * 2, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, 1),
                    nn.Sigmoid()
                )

            self.gate = None
            if (not self.graph_mix) and graph_gate and not self.node_only:
                self.gate = nn.Sequential(
                    nn.Linear(hidden_channels*2, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, 1),
                    nn.Sigmoid()
                )

            if self.copy_head:
                self.copy = nn.Linear(in_channels, 1, bias=True)
                with torch.no_grad():
                    self.copy.weight.zero_(); self.copy.bias.zero_()
                    if fw is not None:
                        self.copy.weight[0, :len(fw)] = torch.tensor(fw, dtype=torch.float)

            self._gate_vals = []

        def reset_gate_stats(self): self._gate_vals = []
        def gate_mean(self):  return float(np.mean(self._gate_vals)) if self._gate_vals else float('nan')
        def gate_median(self):return float(np.median(self._gate_vals)) if self._gate_vals else float('nan')

        def forward(self, sequence_data, use_causal_mask=True):
            node_embs = []
            node_embs_node = []
            node_embs_graph = []
            for data in sequence_data:
                x_node = self.node_mlp(data.x)
                if self.node_only or self.graph_attn is None:
                    x = x_node
                    node_embs.append(x)
                    continue

                x_graph = F.relu(self.graph_attn(data.x, data.edge_index, data.edge_attr))
                x_graph = self.dropout(x_graph)

                if self.graph_mix:
                    node_embs_node.append(x_node)
                    node_embs_graph.append(x_graph)
                else:
                    if self.gate is not None:
                        g = self.gate(torch.cat([x_node, x_graph], dim=-1))
                        with torch.no_grad():
                            self._gate_vals.append(float(g.mean().item()))
                        x = g * x_graph + (1 - g) * x_node
                    else:
                        x = x_graph
                    node_embs.append(x)

            if self.graph_mix and (not self.node_only):
                h_node = torch.stack(node_embs_node, dim=1)  # [N, W, H]
                h_graph = torch.stack(node_embs_graph, dim=1)
                if self.use_pos:
                    W = h_node.size(1)
                    pos = self.pos[:, :W, :]
                    h_node = h_node + pos
                    h_graph = h_graph + pos

                attn_mask = None
                if use_causal_mask:
                    W = h_node.size(1)
                    attn_mask = torch.triu(torch.ones(W, W, dtype=torch.bool, device=h_node.device), diagonal=1)

                h_node_out, _ = self.temporal_attn(h_node, h_node, h_node, attn_mask=attn_mask)
                h_graph_out, _ = self.temporal_attn(h_graph, h_graph, h_graph, attn_mask=attn_mask)
                h_node_out = self.dropout(h_node_out)
                h_graph_out = self.dropout(h_graph_out)

                h_node_last = h_node_out[:, -1, :]
                h_graph_last = h_graph_out[:, -1, :]
                pred_node = self.out_node(h_node_last).squeeze(-1)
                pred_graph = self.out_graph(h_graph_last).squeeze(-1)
                alpha = self.mix_gate(torch.cat([h_node_last, h_graph_last], dim=-1))
                self.last_alpha = alpha
                with torch.no_grad():
                    self._gate_vals.append(float(alpha.mean().item()))
                pred = alpha.squeeze(-1) * pred_graph + (1 - alpha.squeeze(-1)) * pred_node
                h_last = alpha * h_graph_last + (1 - alpha) * h_node_last
            else:
                self.last_alpha = None
                h = torch.stack(node_embs, dim=1)  # [N, W, H]
                if self.use_pos:
                    W = h.size(1)
                    h = h + self.pos[:, :W, :]

                attn_mask = None
                if use_causal_mask:
                    W = h.size(1)
                    attn_mask = torch.triu(torch.ones(W, W, dtype=torch.bool, device=h.device), diagonal=1)

                h_out, _ = self.temporal_attn(h, h, h, attn_mask=attn_mask)
                h_out = self.dropout(h_out)
                h_last = h_out[:, -1, :]
                pred = self.out(h_last).squeeze(-1)

            self.last_aux = None
            if self.aux_outs:
                self.last_aux = [head(h_last).squeeze(-1) for head in self.aux_outs]

            if self.copy_head:
                last_x = sequence_data[-1].x
                pred = pred + self.copy(last_x).squeeze(-1)
            return pred

    # ---- split helper ----

    def compute_splits(num_total, temp_window, split_fracs):
        tr_frac, va_frac, te_frac = split_fracs
        n_train = int(round(tr_frac * num_total))
        n_val   = int(round(va_frac * num_total))
        n_test  = num_total - n_train - n_val

        def step_range(start, count):
            s = temp_window - 1 + start
            e = s + count
            return np.fromiter(range(s, e), dtype=int)

        train_idx = step_range(0, n_train)
        val_idx   = step_range(n_train, n_val)
        test_idx  = step_range(n_train + n_val, n_test)
        return train_idx, val_idx, test_idx, n_train, n_val, n_test

    def _parse_dt(val):
        if val is None:
            return None
        dt = pd.to_datetime(str(val), errors="coerce")
        return None if pd.isna(dt) else dt

    def compute_splits_fixed(ts, temp_window, forecast, split_cfg):
        if not isinstance(split_cfg, dict) or not split_cfg.get("enabled", False):
            return None
        required = [
            "train_start", "train_end",
            "val_start", "val_end",
            "test_start", "test_end",
        ]
        missing = [k for k in required if split_cfg.get(k) in (None, "")]
        if missing:
            print(f"[split] fixed split enabled but missing {missing}; falling back to SPLIT_FRACS.")
            return None
        ts_all = pd.to_datetime(ts)
        Tprime = len(ts_all) - forecast
        if Tprime <= 0:
            print("[split] fixed split has no valid target months; falling back to SPLIT_FRACS.")
            return None
        ts_t = ts_all[:Tprime]

        def _range(start_key, end_key):
            s = _parse_dt(split_cfg.get(start_key))
            e = _parse_dt(split_cfg.get(end_key))
            mask = np.ones(Tprime, dtype=bool)
            if s is not None:
                mask &= ts_t >= s
            if e is not None:
                mask &= ts_t <= e
            idx = np.where(mask)[0]
            # ensure enough history for the temporal window
            idx = idx[idx >= (temp_window - 1)]
            return idx

        train_idx = _range("train_start", "train_end")
        val_idx   = _range("val_start", "val_end")
        test_idx  = _range("test_start", "test_end")
        if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
            print("[split] fixed split produced empty range(s); falling back to SPLIT_FRACS.")
            return None
        return train_idx, val_idx, test_idx

    def get_split_indices(ts, temp_window, forecast, split_fracs, split_cfg):
        Tprime = len(ts) - forecast
        num_total = Tprime - temp_window + 1
        if num_total <= 0:
            return None
        fixed = compute_splits_fixed(ts, temp_window, forecast, split_cfg)
        if fixed is not None:
            train_idx, val_idx, test_idx = fixed
            return train_idx, val_idx, test_idx, len(train_idx), len(val_idx), len(test_idx), True
        train_idx, val_idx, test_idx, n_train, n_val, n_test = compute_splits(
            num_total, temp_window, split_fracs
        )
        return train_idx, val_idx, test_idx, n_train, n_val, n_test, False

    # ---- diagnostics ----

    def masked_describe(arr, msk):
        m = (msk > 0); v = arr[m]
        if v.size == 0:
            return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan, count=0)
        return dict(mean=float(v.mean()), std=float(v.std()), min=float(v.min()), max=float(v.max()), count=int(v.size))


    def print_split_target_stats(name, targets_raw, masks, idx_range):
        vals = []; msks = []
        for t in idx_range:
            vals.append(targets_raw[t]); msks.append(masks[t])
        y = np.stack(vals); m = np.stack(msks)
        stats = masked_describe(y, m)
        print(f"[TargetStats:{name}] mean={stats['mean']:.3f} | std={stats['std']:.3f} | min={stats['min']:.3f} | max={stats['max']:.3f} | count={stats['count']}")


    def degree_over_time(adj_matrices):
        deg_t = adj_matrices.sum(axis=1)  # (N,T)
        deg_mean = deg_t.mean(axis=1)     # (N,)
        return deg_mean


    def _shuffle_edges_rowwise(adj_matrices, seed=42):
        rng = np.random.default_rng(seed)
        N = adj_matrices.shape[0]
        T = adj_matrices.shape[2]
        out = np.zeros_like(adj_matrices)
        for t in range(T):
            A = adj_matrices[:, :, t]
            B = np.zeros_like(A)
            for i in range(N):
                cols = np.nonzero(A[i])[0]
                if cols.size == 0:
                    continue
                if cols.size == 1:
                    B[i, cols[0]] = A[i, cols[0]]
                    continue
                perm = rng.permutation(cols)
                B[i, perm] = A[i, cols]
            out[:, :, t] = B
        return out


    def _time_shift_mats(adj_matrices, lag=6):
        if lag <= 0:
            return adj_matrices.copy()
        N = adj_matrices.shape[0]
        T = adj_matrices.shape[2]
        out = np.zeros((N, N, T), dtype=adj_matrices.dtype)
        for t in range(T):
            src = t - lag
            if src < 0:
                src = 0
            out[:, :, t] = adj_matrices[:, :, src]
        return out


    def error_vs_degree_buckets(y_true_test, y_pred_test, masks_test, deg_vec, quantiles=(0.33, 0.66), *, S=None, N=None):
        total = y_true_test.size
        if S is None and N is None:
            if len(deg_vec) > 0 and total % len(deg_vec) == 0:
                N = len(deg_vec); S = total // N
            else:
                raise AssertionError(f"Cannot infer S/N: total={total}, len(deg_vec)={len(deg_vec)}.")
        if S is None:
            if N is None or N <= 0 or total % N != 0:
                raise AssertionError(f"Cannot infer S with total={total}, N={N}.")
            S = total // N
        if N is None:
            if S <= 0 or total % S != 0:
                raise AssertionError(f"Cannot infer N with total={total}, S={S}.")
            N = total // S

        if len(deg_vec) != N:
            if len(deg_vec) > N:
                deg_vec = deg_vec[:N]
            elif len(deg_vec) > 0:
                pad = np.full(N - len(deg_vec), deg_vec[-1], dtype=float)
                deg_vec = np.concatenate([deg_vec, pad])
            else:
                deg_vec = np.zeros(N, dtype=float)

        try:
            y = y_true_test.reshape(S, N)
            p = y_pred_test.reshape(S, N)
            m = (masks_test.reshape(S, N) > 0)
        except Exception as e:
            raise AssertionError(f"Cannot reshape to [S={S}, N={N}]: total={total}") from e

        qvals = np.quantile(deg_vec, quantiles) if N > 0 else np.array([])
        bounds = []; last = -np.inf
        for q in qvals:
            bounds.append((last, q)); last = q
        bounds.append((last, np.inf))

        bucket_stats = []
        for (lo, hi) in bounds:
            idx = (deg_vec > lo) & (deg_vec <= hi)
            if not idx.any():
                bucket_stats.append(dict(lo=float(lo), hi=float(hi), mae=np.nan, rmse=np.nan, count=0))
                continue
            mb = m[:, idx]
            if mb.sum() == 0:
                bucket_stats.append(dict(lo=float(lo), hi=float(hi), mae=np.nan, rmse=np.nan, count=0))
                continue
            err = (p[:, idx] - y[:, idx])
            err = err[mb]
            mae = float(np.abs(err).mean()); rmse = float(np.sqrt((err**2).mean()))
            bucket_stats.append(dict(lo=float(lo), hi=float(hi), mae=mae, rmse=rmse, count=int(mb.sum())))
        return bucket_stats

    # ---- Top-K CSV ----

    def _emit_topk_csv(exp_name, save_dir, ts_series, forecast, t_idx_single,
                       y_pred_row, y_true_row, mask_row,
                       node_names, orig_idx, k):
        import csv
        out_path = Path(save_dir) / f"topk_{exp_name}_RECENT.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        valid_idx = np.where(mask_row > 0)[0]
        if valid_idx.size == 0:
            print("[TopK][recent] no valid nodes.")
            return None
        scores = y_pred_row[valid_idx]
        order  = np.argsort(scores)[::-1][:min(k, valid_idx.size)]
        top    = valid_idx[order]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["date","t_idx","rank","node","name","orig_index","pred","true"])
            for rank, n in enumerate(top, 1):
                nm = node_names[n] if 0 <= n < len(node_names) else f"n{n}"
                orig = int(orig_idx[n]) if (orig_idx is not None and n < len(orig_idx)) else ""
                w.writerow([
                    str(pd.to_datetime(ts_series[t_idx_single + forecast]).date()),
                    int(t_idx_single),
                    int(rank),
                    int(n),
                    nm,
                    orig,
                    float(y_pred_row[n]),
                    float(y_true_row[n]),
                ])
        return out_path

    # ================================================================
    # One experiment (works on *transformed* features Xt)
    # ================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_ranges_printed = False


    def run_one(exp_name, cfg, ts, Xt, mats, feature_names, return_split=False):
        import csv
        import copy
        FW = cfg['FW']
        cfg_keys = [
            "TEMP_WINDOW", "FORECAST",
            "HIDDEN_CHANNELS", "NUM_HEADS",
            "DROPOUT", "LR", "WD",
            "EDGE_NORM", "EDGE_SELF_LOOPS",
            "NODE_ONLY", "GRAPH_GATE", "GRAPH_MIX", "GRAPH_MULTI",
            "GRAPH_MIX_LAMBDA", "GRAPH_MULTI_LAGS",
            "GRAPH_MULTI_AUX_FORECASTS", "GRAPH_MULTI_AUX_WEIGHT",
            "DRIFT_LAG", "DRIFT_DAMP",
        ]
        cfg_summary = ", ".join([f"{k}={cfg.get(k)}" for k in cfg_keys if k in cfg])
        print(f"[train] starting model '{exp_name}' ({cfg_summary})")

        # Ensure enough data relative to FORECAST/WINDOW
        needed = max(cfg['TEMP_WINDOW'], 1) + cfg['FORECAST'] + 1
        if Xt.shape[0] <= needed:
            print(f"Skipping {exp_name}, not enough data.")
            ret = (dict(val_min=float('inf')), None, None) if return_split else dict(val_min=float('inf'))
            return ret

        # Names
        N_nodes = Xt.shape[1]
        node_names, orig_idx, name_source = resolve_active_names(N_nodes, cfg['FEATURE_NAMES_DEFAULT'])
        print(f"[names] source={name_source} | N_nodes={N_nodes}")

        # Targets from *transformed* features
        drift_lag_cfg = int(cfg.get("DRIFT_LAG") or cfg["TEMP_WINDOW"])
        drift_damp_cfg = float(cfg.get("DRIFT_DAMP", 1.0) or 1.0)
        targets_raw, masks = compute_targets(
            Xt,
            cfg['FORECAST'],
            FW,
            cfg['TARGET_MODE'],
            drift_lag=drift_lag_cfg,
            drift_damp=drift_damp_cfg,
            cfg_like=cfg,
        )

        graph_multi = bool(cfg.get("GRAPH_MULTI", False))
        edge_lags = _parse_lags(cfg) if graph_multi else [0]
        aux_forecasts = _parse_int_list(cfg.get("GRAPH_MULTI_AUX_FORECASTS", [])) if graph_multi else []
        aux_forecasts = sorted({f for f in aux_forecasts if f > 0 and f <= cfg["FORECAST"] and f != cfg["FORECAST"]})
        aux_weight = float(cfg.get("GRAPH_MULTI_AUX_WEIGHT", 0.0) or 0.0)
        if aux_weight <= 0 or not aux_forecasts:
            aux_forecasts = []
        if graph_multi:
            print(f"[{exp_name}] graph_multi lags={edge_lags} aux_forecasts={aux_forecasts} aux_weight={aux_weight}")

        # Build temporal sequence (exclude last FORECAST steps for causality)
        if graph_multi and len(edge_lags) > 1:
            data_sequence = prepare_data_multi(
                mats[:, :, :-cfg['FORECAST']],
                Xt[:-cfg['FORECAST']],
                cfg['EDGE_SELF_LOOPS'],
                cfg['EDGE_NORM'],
                cfg.get('EDGE_ATTR_MODE', 'weight'),
                edge_lags=edge_lags,
            )
            edge_dim = len(edge_lags)
        else:
            data_sequence = prepare_data(
                mats[:, :, :-cfg['FORECAST']],
                Xt[:-cfg['FORECAST']],
                cfg['EDGE_SELF_LOOPS'],
                cfg['EDGE_NORM'],
                cfg.get('EDGE_ATTR_MODE', 'weight')
            )
            edge_dim = 1

        Tprime = len(targets_raw)
        num_total = Tprime - cfg['TEMP_WINDOW'] + 1
        if num_total <= 0:
            print(f"Skipping {exp_name}, not enough data for temporal window.")
            ret = (dict(val_min=float('inf')), None, None) if return_split else dict(val_min=float('inf'))
            return ret

        split_cfg = cfg.get("SPLIT_DATES")
        split_res = get_split_indices(ts, cfg['TEMP_WINDOW'], cfg['FORECAST'], cfg['SPLIT_FRACS'], split_cfg)
        if split_res is None:
            print(f"Skipping {exp_name}, not enough data for temporal window.")
            ret = (dict(val_min=float('inf')), None, None) if return_split else dict(val_min=float('inf'))
            return ret
        train_idx, va_idx, test_idx, n_train, n_val, n_test, fixed_split = split_res

        nonlocal split_ranges_printed
        if not split_ranges_printed:
            def _range(idx):
                if idx.size == 0:
                    return ("n/a", "n/a")
                return (str(pd.to_datetime(ts[idx[0]]).date()), str(pd.to_datetime(ts[idx[-1]]).date()))
            tr = _range(train_idx)
            va = _range(va_idx)
            te = _range(test_idx)
            split_label = "FixedSplit" if fixed_split else "SplitDates"
            print(f"[{split_label}] TRAIN {tr[0]} -> {tr[1]} | VAL {va[0]} -> {va[1]} | TEST {te[0]} -> {te[1]}")
            split_ranges_printed = True

        print_split_target_stats("TRAIN", targets_raw, masks, train_idx)
        print_split_target_stats("VAL",   targets_raw, masks, va_idx)
        print_split_target_stats("TEST",  targets_raw, masks, test_idx)

        # standardize (train only) in LOSS_SPACE
        t_loss = to_loss_space(targets_raw.copy(), cfg['LOSS_SPACE'])
        mu, sd = fit_standardizer(t_loss[train_idx], masks[train_idx], per_node=cfg['PER_NODE_STD'])
        targets_proc = apply_standardizer(t_loss, mu, sd, per_node=cfg['PER_NODE_STD']).astype(np.float32)
        targets_t = torch.tensor(targets_proc)
        masks_t   = torch.tensor(masks.astype(np.float32))

        aux_targets_t = {}
        aux_masks_t = {}
        if aux_forecasts:
            for f in aux_forecasts:
                aux_raw, aux_m = compute_targets(
                    Xt,
                    f,
                    FW,
                    cfg['TARGET_MODE'],
                    drift_lag=drift_lag_cfg,
                    drift_damp=drift_damp_cfg,
                    cfg_like=cfg,
                )
                aux_T = aux_raw.shape[0]
                aux_train_idx = train_idx[train_idx < aux_T]
                if aux_train_idx.size == 0:
                    continue
                aux_loss = to_loss_space(aux_raw.copy(), cfg['LOSS_SPACE'])
                mu_aux, sd_aux = fit_standardizer(aux_loss[aux_train_idx], aux_m[aux_train_idx], per_node=cfg['PER_NODE_STD'])
                aux_proc = apply_standardizer(aux_loss, mu_aux, sd_aux, per_node=cfg['PER_NODE_STD']).astype(np.float32)
                aux_targets_t[f] = torch.tensor(aux_proc)
                aux_masks_t[f] = torch.tensor(aux_m.astype(np.float32))

        mix_lambda = float(cfg.get("GRAPH_MIX_LAMBDA", 0.0) or 0.0)
        mix_weights = None
        if cfg.get("GRAPH_MIX", False) and mix_lambda > 0:
            eps = float(cfg.get("GRAPH_MIX_EPS", 1e-6) or 1e-6)
            tr = targets_raw[train_idx].copy()
            tm = masks[train_idx].copy()
            tr[tm <= 0] = np.nan
            node_std = np.nanstd(tr, axis=0)
            weights = 1.0 / (node_std + eps)
            weights = np.where(np.isfinite(weights), weights, 0.0)
            mean_w = float(np.nanmean(weights)) if np.isfinite(weights).any() else 0.0
            if mean_w > 0:
                weights = weights / mean_w
            mix_weights = torch.tensor(weights, dtype=torch.float32, device=device)

        print(f"[{exp_name}] T'={Xt.shape[0]} | num_total={num_total} | n_train={n_train} | n_val={n_val} | n_test={n_test}")

        model = GTAN(
            in_channels=Xt.shape[2],
            hidden_channels=cfg['HIDDEN_CHANNELS'],
            num_heads=cfg['NUM_HEADS'],
            dropout=cfg['DROPOUT'],
            use_pos=cfg['USE_POSENC'],
            temp_window=cfg['TEMP_WINDOW'],
            node_only=cfg['NODE_ONLY'],
            graph_gate=cfg['GRAPH_GATE'],
            graph_mix=cfg.get('GRAPH_MIX', False),
            edge_dim=edge_dim,
            copy_head=cfg['COPY_HEAD'],
            fw=cfg['FW'],
            aux_forecasts=aux_forecasts,
        ).to(device)

        def _resolve_checkpoint_path() -> Path:
            explicit = cfg.get("CHECKPOINT_PATHS")
            if isinstance(explicit, dict):
                candidate = explicit.get(exp_name)
                if isinstance(candidate, str) and candidate.strip():
                    p = Path(candidate).expanduser()
                    if not p.is_absolute():
                        p = (BASE_DIR / p).resolve()
                    return p
            ckpt_dir_raw = cfg.get("CHECKPOINT_DIR", "checkpoints")
            ckpt_dir = Path(ckpt_dir_raw).expanduser()
            if not ckpt_dir.is_absolute():
                ckpt_dir = (BASE_DIR / ckpt_dir).resolve()
            return ckpt_dir / f"{exp_name}.pt"

        ckpt_path = _resolve_checkpoint_path()
        load_checkpoint = bool(cfg.get("LOAD_CHECKPOINT", False))
        save_checkpoint = bool(cfg.get("SAVE_CHECKPOINT", True))
        checkpoint_strict = bool(cfg.get("CHECKPOINT_STRICT", True))
        checkpoint_loaded = False
        best_state = None

        if load_checkpoint:
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"[{exp_name}] checkpoint not found at {ckpt_path}. "
                    "Run a training pass with SAVE_CHECKPOINT=true first."
                )
            payload = torch.load(ckpt_path, map_location=device)
            state_dict = payload.get("state_dict") if isinstance(payload, dict) and "state_dict" in payload else payload
            model.load_state_dict(state_dict, strict=checkpoint_strict)
            checkpoint_loaded = True
            best_state = copy.deepcopy(model.state_dict())
            print(f"[{exp_name}] loaded checkpoint: {ckpt_path}")

        criterion, crit_kind = make_criterion(cfg.get('LOSS_FN', 'huber'), cfg.get('HUBER_DELTA', 1.0))
        optim = torch.optim.AdamW(model.parameters(), lr=cfg['LR'], weight_decay=cfg['WD'])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', patience=4, factor=0.5)

        early_stop_metric = str(cfg.get("EARLY_STOP_METRIC", "auto") or "auto").lower()
        if early_stop_metric == "auto":
            loss_name = str(cfg.get("LOSS_FN", "rmse") or "rmse").lower()
            early_stop_metric = "rmse" if loss_name == "rmse" else "mae"
        if early_stop_metric not in {"mae", "rmse"}:
            early_stop_metric = "mae"

        history = dict(
            crit_kind=crit_kind,
            train_mae=[], val_mae=[],
            train_rmse=[], val_rmse=[],
            train_crit=[], val_crit=[],
            train_gate_mean=[], train_gate_median=[],
            val_gate_mean=[], val_gate_median=[],
        )

        def _masked_loss_value(pred, target, mask):
            target_safe = torch.where(mask > 0, target, torch.zeros_like(target))
            target_safe = torch.nan_to_num(target_safe, nan=0.0, posinf=0.0, neginf=0.0)
            lv = criterion(pred, target_safe)
            num = (lv * mask).sum()
            den = mask.sum() + 1e-6
            loss_val = num / den if crit_kind != "rmse" else torch.sqrt(num / den + 1e-12)
            return loss_val, num, den

        def eval_split_raw(split_idx, track_gate=False):
            model.eval()
            if track_gate: model.reset_gate_stats()
            maes, rmses = [], []
            with torch.no_grad():
                for t in split_idx:
                    seq = [d.to(device) for d in data_sequence[t - cfg['TEMP_WINDOW'] + 1:t + 1]]
                    y_std = targets_t[t].to(device)
                    msk   = masks_t[t].to(device)
                    pred_std = model(seq, use_causal_mask=cfg['USE_CAUSAL_MASK'])
                    y_raw_np    = invert_std_and_space(y_std.cpu().numpy(),  mu, sd, cfg['PER_NODE_STD'], cfg['LOSS_SPACE'])
                    pred_raw_np = invert_std_and_space(pred_std.cpu().numpy(), mu, sd, cfg['PER_NODE_STD'], cfg['LOSS_SPACE'])
                    mae, rmse = masked_metrics(y_raw_np, pred_raw_np, msk.cpu().numpy())
                    maes.append(mae); rmses.append(rmse)
            g_mean = model.gate_mean() if track_gate else float('nan')
            g_med  = model.gate_median() if track_gate else float('nan')
            return float(np.nanmean(maes)), float(np.nanmean(rmses)), g_mean, g_med

        if not checkpoint_loaded:
            best_val = float('inf'); noimp = 0
            for epoch in range(1, cfg['EPOCHS'] + 1):
                model.train(); model.reset_gate_stats()
                tr_num = 0.0; tr_den = 0.0
                for t in train_idx:
                    seq = [d.to(device) for d in data_sequence[t - cfg['TEMP_WINDOW'] + 1:t + 1]]
                    y   = targets_t[t].to(device)
                    msk = masks_t[t].to(device)
                    pred = model(seq, use_causal_mask=cfg['USE_CAUSAL_MASK'])

                    loss, num, den = _masked_loss_value(pred, y, msk)
                    if mix_weights is not None and model.last_alpha is not None:
                        alpha = model.last_alpha.squeeze(-1)
                        mix_pen = (alpha * mix_weights).mean()
                        loss = loss + mix_lambda * mix_pen
                    if aux_targets_t and model.last_aux:
                        for i, f in enumerate(aux_forecasts):
                            if i >= len(model.last_aux):
                                break
                            if t >= aux_targets_t[f].shape[0]:
                                continue
                            y_aux = aux_targets_t[f][t].to(device)
                            msk_aux = aux_masks_t[f][t].to(device)
                            aux_loss, _num_aux, _den_aux = _masked_loss_value(model.last_aux[i], y_aux, msk_aux)
                            loss = loss + aux_weight * aux_loss

                    optim.zero_grad(); loss.backward(); optim.step()
                    tr_num += num.item(); tr_den += den.item()

                train_loss_z = tr_den and (tr_num / tr_den) or 0.0
                if crit_kind == "rmse": train_loss_z = math.sqrt(train_loss_z + 1e-12)

                train_mae_raw, train_rmse_raw, gtr_mean, gtr_med = eval_split_raw(train_idx, track_gate=True)
                val_mae_raw,   val_rmse_raw,   gval_mean, gval_med = eval_split_raw(va_idx,   track_gate=True)

                model.eval(); model.reset_gate_stats()
                va_num = 0.0; va_den = 0.0
                with torch.no_grad():
                    for t in va_idx:
                        seq = [d.to(device) for d in data_sequence[t - cfg['TEMP_WINDOW'] + 1:t + 1]]
                        y_std = targets_t[t].to(device)
                        msk   = masks_t[t].to(device)
                        _loss_val, num_val, den_val = _masked_loss_value(
                            model(seq, use_causal_mask=cfg['USE_CAUSAL_MASK']),
                            y_std,
                            msk,
                        )
                        va_num += num_val.item()
                        va_den += den_val.item()
                val_loss_z  = va_num / va_den if va_den else 0.0
                if crit_kind == "rmse": val_loss_z = math.sqrt(val_loss_z + 1e-12)

                history['train_mae'].append(train_mae_raw)
                history['val_mae'].append(val_mae_raw)
                history['train_rmse'].append(train_rmse_raw)
                history['val_rmse'].append(val_rmse_raw)
                history['train_crit'].append(train_loss_z)
                history['val_crit'].append(val_loss_z)
                history['train_gate_mean'].append(gtr_mean)
                history['train_gate_median'].append(gtr_med)
                history['val_gate_mean'].append(gval_mean)
                history['val_gate_median'].append(gval_med)

                val_monitor = val_rmse_raw if early_stop_metric == "rmse" else val_mae_raw
                sched.step(val_monitor)
                if (best_val - val_monitor) > CFG['MIN_DELTA']:
                    best_val = val_monitor; best_state = copy.deepcopy(model.state_dict()); noimp = 0
                else:
                    noimp += 1

                base = (f"[{exp_name}] epoch {epoch:02d} | train_{crit_kind}={train_loss_z:.6f} "
                        f"| val_{crit_kind}={val_loss_z:.6f} | monitor_{early_stop_metric}={val_monitor:.3f} | val_MAE_raw={val_mae_raw:.3f} "
                        f"| val_RMSE_raw={val_rmse_raw:.3f} | g_train~mean={gtr_mean:.3f},med={gtr_med:.3f} "
                        f"| g_val~mean={gval_mean:.3f},med={gval_med:.3f}")
                if CFG.get("LOG_TRAIN_RAW", False):
                    base = (f"[{exp_name}] epoch {epoch:02d} | train_{crit_kind}={train_loss_z:.6f} "
                            f"| val_{crit_kind}={val_loss_z:.6f} | train_MAE_raw={train_mae_raw:.3f} "
                            f"| val_MAE_raw={val_mae_raw:.3f} | train_RMSE_raw={train_rmse_raw:.3f} "
                            f"| val_RMSE_raw={val_rmse_raw:.3f} | g_train~mean={gtr_mean:.3f},med={gtr_med:.3f} "
                            f"| g_val~mean={gval_mean:.3f},med={gval_med:.3f}")
                print(base)

                if noimp >= cfg['PATIENCE']:
                    print(f"[{exp_name}] early stop.")
                    break

            if save_checkpoint and best_state is not None:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state_dict": best_state,
                        "exp_name": exp_name,
                        "forecast": int(cfg.get("FORECAST", 0) or 0),
                        "temp_window": int(cfg.get("TEMP_WINDOW", 0) or 0),
                        "target_mode": cfg.get("TARGET_MODE"),
                        "loss_space": cfg.get("LOSS_SPACE"),
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                    },
                    ckpt_path,
                )
                print(f"[{exp_name}] saved checkpoint: {ckpt_path}")

        # Optional: dump training history (used by diagnostics)
        diag_dir = cfg.get("DIAG_SAVE_DIR")
        diag_name = cfg.get("DIAG_RUN_NAME")
        if diag_dir and diag_name and history:
            try:
                diag_dir = Path(diag_dir)
                diag_dir.mkdir(parents=True, exist_ok=True)
                hist_path = diag_dir / f"history_{diag_name}.csv"
                with hist_path.open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        "epoch",
                        "train_mae", "val_mae",
                        "train_rmse", "val_rmse",
                        "train_crit", "val_crit",
                        "train_gate_mean", "train_gate_median",
                        "val_gate_mean", "val_gate_median",
                    ])
                    epochs = range(1, len(history.get("train_mae", [])) + 1)
                    for i, ep in enumerate(epochs):
                        w.writerow([
                            ep,
                            history["train_mae"][i] if i < len(history["train_mae"]) else None,
                            history["val_mae"][i] if i < len(history["val_mae"]) else None,
                            history["train_rmse"][i] if i < len(history["train_rmse"]) else None,
                            history["val_rmse"][i] if i < len(history["val_rmse"]) else None,
                            history["train_crit"][i] if i < len(history["train_crit"]) else None,
                            history["val_crit"][i] if i < len(history["val_crit"]) else None,
                            history["train_gate_mean"][i] if i < len(history["train_gate_mean"]) else None,
                            history["train_gate_median"][i] if i < len(history["train_gate_median"]) else None,
                            history["val_gate_mean"][i] if i < len(history["val_gate_mean"]) else None,
                            history["val_gate_median"][i] if i < len(history["val_gate_median"]) else None,
                        ])

                # Save a simple train/val learning-curve figure for diagnostics runs.
                metric = str(cfg.get("PLOT_METRIC", "rmse") or "rmse").lower()
                key_map = {
                    "mae": ("train_mae", "val_mae", "MAE"),
                    "rmse": ("train_rmse", "val_rmse", "RMSE"),
                    "crit": ("train_crit", "val_crit", "Criterion"),
                }
                train_key, val_key, ylab = key_map.get(metric, key_map["rmse"])
                if history.get(train_key) and history.get(val_key):
                    epochs = np.arange(1, len(history[train_key]) + 1)
                    fig, ax = plt.subplots(figsize=(6.2, 3.6))
                    ax.plot(epochs, history[train_key], label=f"train_{ylab.lower()}")
                    ax.plot(epochs, history[val_key], label=f"val_{ylab.lower()}")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel(ylab)
                    ax.set_title(f"Learning curve - {diag_name}")
                    ax.grid(alpha=0.25)
                    ax.legend(loc="best", fontsize=8)
                    fig.tight_layout()
                    dpi = int(cfg.get("DPI", 180) or 180)
                    fig.savefig(diag_dir / f"learning_curve_{diag_name}.png", dpi=dpi)
                    plt.close(fig)
            except Exception:
                pass

        # ---------------- Test ----------------
        if best_state is None:
            results = {'mae': float('nan'), 'rmse': float('nan')}
            payload = (ts, Xt, mats, targets_raw, masks, [], None, None)
            return (results, payload, history) if return_split else results

        model.load_state_dict(best_state)
        model.eval()
        all_preds, all_ys, all_masks = [], [], []
        alpha_sum = None; alpha_cnt = None
        alpha_out_path = cfg.get("ALPHA_OUT_PATH")
        if alpha_out_path:
            alpha_sum = np.zeros(Xt.shape[1], dtype=np.float64)
            alpha_cnt = np.zeros(Xt.shape[1], dtype=np.float64)

        with torch.no_grad():
            for t in test_idx:
                seq = [d.to(device) for d in data_sequence[t - cfg['TEMP_WINDOW'] + 1:t + 1]]
                y_std = targets_t[t].cpu().numpy()
                msk   = masks_t[t].cpu().numpy()
                pred_std = model(seq, use_causal_mask=cfg['USE_CAUSAL_MASK']).cpu().numpy()
                if alpha_sum is not None and model.last_alpha is not None:
                    try:
                        alpha = model.last_alpha.squeeze(-1).detach().cpu().numpy()
                        alpha_sum += alpha * msk
                        alpha_cnt += msk
                    except Exception:
                        pass
                pred_raw = invert_std_and_space(pred_std, mu, sd, cfg['PER_NODE_STD'], cfg['LOSS_SPACE'])
                y_raw    = invert_std_and_space(y_std,  mu, sd, cfg['PER_NODE_STD'], cfg['LOSS_SPACE'])
                all_preds.append(pred_raw); all_ys.append(y_raw); all_masks.append(msk)

        y_pred_test = np.concatenate(all_preds)
        y_true_test = np.concatenate(all_ys)
        masks_test  = np.concatenate(all_masks)
        mae, rmse = masked_metrics(y_true_test, y_pred_test, masks_test)
        results = {'mae': mae, 'rmse': rmse}

        if alpha_sum is not None:
            try:
                alpha_mean = np.where(alpha_cnt > 0, alpha_sum / alpha_cnt, np.nan)
                out_path = Path(alpha_out_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                import csv
                with out_path.open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["node", "alpha_mean", "count"])
                    for n in range(Xt.shape[1]):
                        name = node_names[n] if n < len(node_names) else f"n{n}"
                        w.writerow([name, float(alpha_mean[n]), int(alpha_cnt[n])])
            except Exception:
                pass

        # Optional: per-node test metrics
        per_node_out = cfg.get("PER_NODE_OUT_PATH")
        if per_node_out:
            try:
                S_steps = len(test_idx)
                N_nodes = Xt.shape[1]
                y_t = y_true_test.reshape(S_steps, N_nodes)
                y_p = y_pred_test.reshape(S_steps, N_nodes)
                m_t = masks_test.reshape(S_steps, N_nodes) > 0
                import csv
                out_path = Path(per_node_out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["node", "mae", "rmse", "count"])
                    for n in range(N_nodes):
                        idx = m_t[:, n]
                        if idx.sum() == 0:
                            mae_n = float("nan"); rmse_n = float("nan"); cnt = 0
                        else:
                            diff = (y_p[:, n] - y_t[:, n])[idx]
                            mae_n = float(np.abs(diff).mean())
                            rmse_n = float(np.sqrt((diff ** 2).mean()))
                            cnt = int(idx.sum())
                        name = node_names[n] if n < len(node_names) else f"n{n}"
                        w.writerow([name, mae_n, rmse_n, cnt])
            except Exception:
                pass

        # Error vs degree
        S_steps = len(test_idx)
        deg_vec = degree_over_time(mats[:, :, :max(1, mats.shape[2] - cfg['FORECAST'])])
        bucket_stats = error_vs_degree_buckets(
            y_true_test, y_pred_test, masks_test, deg_vec,
            S=S_steps, N=N_nodes
        )
        print("\n[Test Error vs Degree]")
        for i, bs in enumerate(bucket_stats, 1):
            print(f"  Bucket {i}: deg in ({bs['lo']:.3f}, {bs['hi']:.3f}] | MAE={bs['mae']:.3f} | RMSE={bs['rmse']:.3f} | count={bs['count']}")

        def _coerce_cutoff_dates(raw) -> list[str]:
            if raw is None:
                return []
            vals = raw if isinstance(raw, (list, tuple, set)) else [raw]
            out: list[str] = []
            for v in vals:
                if isinstance(v, str):
                    token = v.strip()
                    if token:
                        out.append(token)
            return out

        # MOST-RECENT Top-K (single cutoff or many forced cutoffs)
        cutoff_dates = _coerce_cutoff_dates(cfg.get("RECENT_CUTOFF_DATES"))
        if cutoff_dates:
            for cutoff_date in cutoff_dates:
                cfg_cut = dict(cfg)
                cfg_cut["RECENT_CUTOFF_DATE"] = cutoff_date
                if cfg_cut.get("TOPK_RECENT", True):
                    recent_forecast_and_topk(
                        exp_name=exp_name,
                        cfg=cfg_cut,
                        ts_full=ts,
                        feats_full=Xt,
                        mats_full=mats,
                        model=model,
                        mu=mu, sd=sd,
                        per_node_std=cfg_cut['PER_NODE_STD'],
                        loss_space=cfg_cut['LOSS_SPACE'],
                        node_names=node_names,
                        orig_idx=orig_idx,
                        save_dir=PLOT_DIR,
                        mode="eval"
                    )
                    if cfg_cut.get("TOPK_RECENT_INFERENCE", True):
                        recent_forecast_and_topk(
                            exp_name=exp_name,
                            cfg=cfg_cut,
                            ts_full=ts,
                            feats_full=Xt,
                            mats_full=mats,
                            model=model,
                            mu=mu, sd=sd,
                            per_node_std=cfg_cut['PER_NODE_STD'],
                            loss_space=cfg_cut['LOSS_SPACE'],
                            node_names=node_names,
                            orig_idx=orig_idx,
                            save_dir=PLOT_DIR,
                            mode="inference"
                        )
        elif cfg.get("TOPK_RECENT", True):
            recent_forecast_and_topk(
                exp_name=exp_name,
                cfg=cfg,
                ts_full=ts,
                feats_full=Xt,
                mats_full=mats,
                model=model,
                mu=mu, sd=sd,
                per_node_std=cfg['PER_NODE_STD'],
                loss_space=cfg['LOSS_SPACE'],
                node_names=node_names,
                orig_idx=orig_idx,
                save_dir=PLOT_DIR,
                mode="eval"
            )
            if cfg.get("TOPK_RECENT_INFERENCE", True):
                recent_forecast_and_topk(
                    exp_name=exp_name,
                    cfg=cfg,
                    ts_full=ts,
                    feats_full=Xt,
                    mats_full=mats,
                    model=model,
                    mu=mu, sd=sd,
                    per_node_std=cfg['PER_NODE_STD'],
                    loss_space=cfg['LOSS_SPACE'],
                    node_names=node_names,
                    orig_idx=orig_idx,
                    save_dir=PLOT_DIR,
                    mode="inference"
                )

        payload = (ts, Xt, mats, targets_raw, masks, test_idx, mu, sd)
        return (results, payload, history) if return_split else results
    # --------------------------
    # Recent forecast (latest window) -> FULL ranking CSV + console Top-K
    # --------------------------
    def recent_forecast_and_topk(exp_name, cfg, ts_full, feats_full, mats_full,
                                 model, mu, sd, per_node_std, loss_space,
                                 node_names, orig_idx, save_dir, mode="eval"):
        """
        Generates a recent forecast ranking.
        mode="eval": matches historical evaluation (excludes last FORECAST months from inputs)
        mode="inference": uses the final observed month as context to emit a forward-only forecast.
        """
        import numpy as np, pandas as pd, torch
        from pathlib import Path

        assert mode in {"eval", "inference"}, f"unknown mode '{mode}'"
        inference_only = (mode == "inference")

        model.eval()
        FCAST = int(cfg['FORECAST']); W = int(cfg['TEMP_WINDOW'])
        T, N = feats_full.shape[0], feats_full.shape[1]
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)

        graph_multi = bool(cfg.get("GRAPH_MULTI", False))
        edge_lags = _parse_lags(cfg) if graph_multi else [0]

        forced_cutoff_raw = cfg.get("RECENT_CUTOFF_DATE")
        forced_cutoff_idx = None
        forced_cutoff_tag = None
        if isinstance(forced_cutoff_raw, str) and forced_cutoff_raw.strip():
            try:
                cutoff_ts = pd.to_datetime(forced_cutoff_raw).to_period("M").to_timestamp(how="start")
                ts_month = pd.to_datetime(ts_full).to_period("M").to_timestamp(how="start")
                match = np.where(ts_month == cutoff_ts)[0]
                if match.size > 0:
                    forced_cutoff_idx = int(match[-1])
                    forced_cutoff_tag = cutoff_ts.strftime("%Y-%m")
                else:
                    print(f"[recent] RECENT_CUTOFF_DATE={forced_cutoff_raw} not found in timeline; using default recent index.")
            except Exception:
                print(f"[recent] invalid RECENT_CUTOFF_DATE={forced_cutoff_raw}; using default recent index.")

        def _prepare_recent_seq(mats_arr, feats_arr):
            if graph_multi and len(edge_lags) > 1:
                return prepare_data_multi(
                    mats_arr,
                    feats_arr,
                    cfg['EDGE_SELF_LOOPS'],
                    cfg['EDGE_NORM'],
                    cfg.get('EDGE_ATTR_MODE', 'weight'),
                    edge_lags=edge_lags,
                )
            return prepare_data(
                mats_arr,
                feats_arr,
                cfg['EDGE_SELF_LOOPS'],
                cfg['EDGE_NORM'],
                cfg.get('EDGE_ATTR_MODE', 'weight')
            )

        if inference_only:
            t_in = T - 1
            if forced_cutoff_idx is not None:
                t_in = max(0, min(forced_cutoff_idx, T - 1))
            data_seq_full = _prepare_recent_seq(mats_full, feats_full)
        else:
            t_in = T - 1 - FCAST
            if forced_cutoff_idx is not None:
                max_eval_idx = T - 1 - FCAST
                t_in = max(0, min(forced_cutoff_idx, max_eval_idx))
            if t_in < W - 1:
                print("[recent] Not enough data for a recent forecast.")
                return None
            data_seq_full = _prepare_recent_seq(mats_full[:, :, :-FCAST], feats_full[:-FCAST])

        if t_in < W - 1:
            print("[recent] Not enough history for recent forecasting.")
            return None

        device = next(model.parameters()).device
        seq = [d.to(device) for d in data_seq_full[t_in - W + 1 : t_in + 1]]

        with torch.no_grad():
            pred_std = model(seq, use_causal_mask=cfg['USE_CAUSAL_MASK']).detach().cpu().numpy()
        pred_raw = invert_std_and_space(pred_std, mu, sd, per_node_std, loss_space)

        if inference_only:
            cur = feats_full[t_in]
            valid = np.isfinite(cur).all(axis=1)
            context_ts = pd.to_datetime(ts_full[min(t_in, len(ts_full) - 1)])
            context_period = pd.Period(context_ts.strftime("%Y-%m"), freq="M")
            tgt_date = (context_period + FCAST).to_timestamp(how="end").date()
        else:
            cur = feats_full[t_in]
            fut = feats_full[t_in + FCAST]
            valid = np.isfinite(fut).all(axis=1) & (np.abs(fut).sum(axis=1) > 0)
            tgt_date = pd.to_datetime(ts_full[t_in + FCAST]).date()

        valid_idx = np.where(valid)[0]
        if valid_idx.size == 0:
            print("[Recent Ranking] No valid nodes at target date.")
            return None

        scores = pred_raw[valid_idx]
        order_all = np.argsort(scores)[::-1]
        idx_sorted = valid_idx[order_all]
        names_sorted = [node_names[i] if 0 <= i < len(node_names) else f"n{i}" for i in idx_sorted]
        preds_sorted = scores[order_all].astype(float)

        df_rank = pd.DataFrame({
            "rank": np.arange(1, len(idx_sorted) + 1, dtype=int),
            "name": names_sorted,
            "prediction": preds_sorted,
        })

        suffix = "_RECENT.csv" if not inference_only else "_RECENT_infer.csv"
        if forced_cutoff_tag is not None:
            suffix = suffix.replace(".csv", f"_{forced_cutoff_tag}.csv")
        out_rank = save_dir / f"ranking_{exp_name}{suffix}"
        df_rank.to_csv(out_rank, index=False, encoding="utf-8")

        K = int(cfg.get("TOPK_K", 10))
        K = min(K, len(df_rank))
        label = "Inference" if inference_only else "Recent"
        print(f"\n[{label} Ranking @ {tgt_date}] Top-{K} (console only) - full CSV: {out_rank}")
        print(df_rank.head(K).to_string(index=False))

        return {"ranking_csv": out_rank}

    # ---- plot training curves ----

    def plot_histories(hist_graph, hist_nograph, persist_value, metric, save_path):
        metric = metric.lower()
        assert metric in ("mae", "rmse")
        if metric == "mae":
            g_tr, g_va = hist_graph['train_mae'],   hist_graph['val_mae']
            ng_tr, ng_va = hist_nograph['train_mae'], hist_nograph['val_mae']
            ylabel = "MAE (raw)"
        else:
            g_tr, g_va = hist_graph['train_rmse'],   hist_graph['val_rmse']
            ng_tr, ng_va = hist_nograph['train_rmse'], hist_nograph['val_rmse']
            ylabel = "RMSE (raw)"

        epochs_g  = np.arange(1, len(g_va)+1)
        epochs_ng = np.arange(1, len(ng_va)+1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs_g, g_va, label="Graph: Val", linewidth=2)
        plt.plot(epochs_g, g_tr, label="Graph: Train", linestyle="--", alpha=0.7)
        plt.plot(epochs_ng, ng_va, label="No-Graph: Val", linewidth=2)
        plt.plot(epochs_ng, ng_tr, label="No-Graph: Train", linestyle="--", alpha=0.7)
        plt.axhline(persist_value, color="k", linestyle=":", linewidth=2, label=f"Persistence (test {metric.upper()})")

        plt.xlabel("Epoch"); plt.ylabel(ylabel)
        plt.title(f"Train/Val {metric.upper()} — Graph vs No-Graph (+ Persistence)")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(save_path, dpi=CFG['DPI']); plt.close()
        print(f"[plot] saved: {save_path}")

    # ================================================================
    # Baselines (operate on *transformed* features Xt)
    # ================================================================

    def _compute_drift_level(values: np.ndarray, horizon: int, lag: int, damp: float = 1.0) -> np.ndarray:
        """Linear drift forecast level at t+h from a (T, N) level series."""
        if values.size == 0:
            return values
        T = values.shape[0]
        out = np.empty_like(values, dtype=float)
        for t in range(T):
            cur = values[t]
            k = min(max(int(lag), 1), t)
            if k <= 0:
                slope = np.zeros_like(cur)
            else:
                past = values[t - k]
                slope = (cur - past) / float(k)
            fut = cur + float(damp) * float(horizon) * slope
            if np.isnan(fut).any():
                fut = np.where(np.isfinite(fut), fut, cur)
            out[t] = fut
        return out

    def _fw_series(Xt, fw):
        return np.stack([fw_of_frame(Xt[t], fw) for t in range(Xt.shape[0])])

    def baselines_on_test(ts, Xt, mats, cfg, targets, masks, test_idx, eval_idx=None):
        fw = cfg['FW']; W = cfg['TEMP_WINDOW']; F = int(cfg['FORECAST']); mode = (cfg['TARGET_MODE'] or "").strip().lower()
        if mode == "absolute":
            mode = "level"
        drift_lag = int(cfg.get("DRIFT_LAG") or W)
        drift_damp = float(cfg.get("DRIFT_DAMP", 1.0) or 1.0)
        idx_eval = test_idx if eval_idx is None else eval_idx

        base_vals = _target_level_series(Xt, fw, mode, cfg)
        drift_vals = _compute_drift_level(base_vals, horizon=F, lag=drift_lag, damp=drift_damp)

        def fw_level(t_idx):
            return base_vals[t_idx]

        def drift_level(t_idx):
            return drift_vals[t_idx]

        preds_pers, preds_drift, ys, msks = [], [], [], []
        for t in idx_eval:
            y_true_raw = targets[t]; msk = masks[t]
            if mode in {"level", "smooth_relative", "smooth_relative_level"}:
                pred_pers = fw_level(t); pred_drift = drift_level(t)
            elif mode == "residual":
                cur = fw_level(t); drift_fut = drift_level(t)
                pred_pers = np.zeros_like(cur); pred_drift = drift_fut - cur
            elif mode == "drift_residual":
                cur = fw_level(t); drift_fut = drift_level(t)
                pred_pers = cur - drift_fut
                pred_drift = np.zeros_like(cur)
            elif mode == "log_change":
                cur = fw_level(t); drift_fut = drift_level(t)
                pred_pers = np.zeros_like(cur)
                pred_drift  = np.arcsinh(drift_fut) - np.arcsinh(cur)
            else:
                raise ValueError("Unknown TARGET_MODE")
            preds_pers.append(pred_pers); preds_drift.append(pred_drift)
            ys.append(y_true_raw); msks.append(msk)

        y = np.concatenate(ys); m = np.concatenate(msks)
        p1 = np.concatenate(preds_pers); p2 = np.concatenate(preds_drift)
        mae_p, rmse_p = masked_metrics(y, p1, m)
        mae_d, rmse_d = masked_metrics(y, p2, m)
        return (mae_p, rmse_p), (mae_d, rmse_d)

    # ================================================================
    # Main pipeline
    # ================================================================

    def main():
        _announce("Cell 9: training GTAN models (graph vs no-graph)")
        # Repro
        SEED = 42
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Load + transform once
        ts, Xt, mats, _feats_raw = load_and_transform(CFG)

        # Feature names
        F = Xt.shape[2]
        feature_names = CFG['FEATURE_NAMES_DEFAULT'] if F == 3 else [f"f{i}" for i in range(F)]

        # Visualize the exact inputs the model sees (Xt)
        if not sweep_enabled and CFG.get("PLOT_INPUT_PREVIEWS", True):
            visualize_inputs(ts, Xt, mats, CFG, feature_names=feature_names, save_dir=INPUTS_DIR, show=CFG['SHOW_FIGS'])

        if skip_training:
            _announce("Skipping training per configuration")
            return

        if sweep_enabled:
            _announce("Running hyperparameter sweep (graph model)")

            try:
                import optuna  # type: ignore
            except Exception:
                print("[sweep] Optuna is not installed. Install with: pip install optuna")
                return

            sweep_metric = str(sweep_cfg.get("metric", "rmse")).lower()
            if sweep_metric not in {"rmse", "mae"}:
                sweep_metric = "rmse"
            n_trials = int(sweep_cfg.get("n_trials", 30))
            env_n_trials = os.environ.get("GNN_SWEEP_N_TRIALS")
            if env_n_trials not in (None, ""):
                try:
                    n_trials = int(env_n_trials)
                except Exception:
                    pass
            max_total_trials = sweep_cfg.get("max_total_trials")
            env_max_total = os.environ.get("GNN_SWEEP_MAX_TOTAL_TRIALS")
            if env_max_total not in (None, ""):
                try:
                    max_total_trials = int(env_max_total)
                except Exception:
                    pass
            if max_total_trials in ("", None):
                max_total_trials = None
            else:
                max_total_trials = int(max_total_trials)
            model_family = str(os.environ.get("GNN_SWEEP_MODEL_FAMILY") or sweep_cfg.get("model_family", "graph_base")).strip().lower()
            if model_family not in {"graph_base", "graph_mix", "graph_multi", "graph_multi_mix", "drift"}:
                print(f"[sweep] unknown model_family={model_family}; expected graph_base|graph_mix|graph_multi|graph_multi_mix|drift")
                return

            split_cfg_fixed = CFG.get("SPLIT_DATES")
            if not (isinstance(split_cfg_fixed, dict) and split_cfg_fixed.get("enabled", False)):
                print("[sweep] enable cfg_defaults.SPLIT_DATES for fixed evaluation protocol before running sweeps.")
                return

            variance_penalty = float(sweep_cfg.get("variance_penalty", 0.05) or 0.0)
            rolling_cfg = sweep_cfg.get("rolling_validation") if isinstance(sweep_cfg.get("rolling_validation"), dict) else {}
            rolling_enabled = bool(rolling_cfg.get("enabled", True))
            rolling_folds = max(1, int(rolling_cfg.get("n_folds", 3) or 1))
            search_space = sweep_cfg.get("search_space") if isinstance(sweep_cfg.get("search_space"), dict) else {}
            common_space = search_space.get("common") if isinstance(search_space.get("common"), dict) else {}
            family_space = search_space.get(model_family) if isinstance(search_space.get(model_family), dict) else {}

            results_path = BASE_DIR / f"sweep_results_{model_family}.csv"
            best_path = BASE_DIR / f"sweep_best_{model_family}.json"

            def _as_choices(raw, default_vals):
                if isinstance(raw, (list, tuple)) and raw:
                    return list(raw)
                return list(default_vals)

            def _range_pair(raw, lo, hi):
                if isinstance(raw, (list, tuple)) and len(raw) == 2:
                    try:
                        a = float(raw[0]); b = float(raw[1])
                        if a < b:
                            return a, b
                    except Exception:
                        pass
                return float(lo), float(hi)

            def _period(v):
                dt = pd.to_datetime(v, errors="coerce")
                if pd.isna(dt):
                    return None
                return pd.Period(dt, freq="M")

            def _period_start_str(p):
                return str(p.to_timestamp(how="start").date())

            def _build_fold_cfgs(cfg_trial):
                if not rolling_enabled or rolling_folds <= 1:
                    return [cfg_trial]
                split_cfg = cfg_trial.get("SPLIT_DATES")
                if not isinstance(split_cfg, dict) or not split_cfg.get("enabled", False):
                    return [cfg_trial]
                tr_start = _period(split_cfg.get("train_start"))
                va_start = _period(split_cfg.get("val_start"))
                va_end = _period(split_cfg.get("val_end"))
                if tr_start is None or va_start is None or va_end is None or va_end < va_start:
                    return [cfg_trial]
                vals = list(pd.period_range(va_start, va_end, freq="M"))
                if len(vals) < 2:
                    return [cfg_trial]
                chunks = [c for c in np.array_split(vals, min(rolling_folds, len(vals))) if len(c) > 0]
                out = []
                for c in chunks:
                    vs = c[0]; ve = c[-1]; te = vs - 1
                    if te < tr_start:
                        continue
                    s = dict(split_cfg)
                    s.update({
                        "enabled": True,
                        "train_start": _period_start_str(tr_start),
                        "train_end": _period_start_str(te),
                        "val_start": _period_start_str(vs),
                        "val_end": _period_start_str(ve),
                        "test_start": _period_start_str(vs),
                        "test_end": _period_start_str(ve),
                    })
                    cfold = cfg_trial.copy()
                    cfold["SPLIT_DATES"] = s
                    out.append(cfold)
                return out if out else [cfg_trial]

            def _apply_family_flags(cfg_trial):
                cfg_trial.update({
                    "NODE_ONLY": False,
                    "TOPK_RECENT": False,
                    "TOPK_RECENT_INFERENCE": False,
                    "SHOW_FIGS": False,
                    "GRAPH_MIX": False,
                    "GRAPH_MULTI": False,
                })
                if model_family == "graph_mix":
                    cfg_trial["GRAPH_MIX"] = True
                    cfg_trial["GRAPH_GATE"] = False
                elif model_family == "graph_multi":
                    cfg_trial["GRAPH_MULTI"] = True
                elif model_family == "graph_multi_mix":
                    cfg_trial["GRAPH_MULTI"] = True
                    cfg_trial["GRAPH_MIX"] = True
                    cfg_trial["GRAPH_GATE"] = False

            def _eval_family(cfg_trial, use_val=True):
                if model_family == "drift":
                    drift_lag_trial = int(cfg_trial.get("DRIFT_LAG") or cfg_trial.get("TEMP_WINDOW", 1))
                    drift_damp_trial = float(cfg_trial.get("DRIFT_DAMP", 1.0) or 1.0)
                    targets_raw, masks = compute_targets(
                        Xt,
                        cfg_trial["FORECAST"],
                        cfg_trial["FW"],
                        cfg_trial["TARGET_MODE"],
                        drift_lag=drift_lag_trial,
                        drift_damp=drift_damp_trial,
                    )
                    split_res = get_split_indices(ts, cfg_trial["TEMP_WINDOW"], cfg_trial["FORECAST"], cfg_trial["SPLIT_FRACS"], cfg_trial.get("SPLIT_DATES"))
                    if split_res is None:
                        return float("inf"), {"mae": float("inf"), "rmse": float("inf")}, None, {}
                    _tr, va_idx, test_idx, _ntr, _nva, _nte, _fixed = split_res
                    eval_idx = va_idx if use_val else test_idx
                    (mae_p, rmse_p), (mae_d, rmse_d) = baselines_on_test(ts, Xt, mats, cfg_trial, targets_raw, masks, test_idx, eval_idx=eval_idx)
                    metric_val = float(rmse_d if sweep_metric == "rmse" else mae_d)
                    payload = (ts, Xt, mats, targets_raw, masks, test_idx, None, None)
                    return metric_val, {"mae": float(mae_d), "rmse": float(rmse_d), "persist_mae": float(mae_p), "persist_rmse": float(rmse_p)}, payload, {}

                exp_name = {
                    "graph_base": "SweepGraphBase",
                    "graph_mix": "SweepGraphMix",
                    "graph_multi": "SweepGraphMulti",
                    "graph_multi_mix": "SweepGraphMultiMix",
                }.get(model_family, "SweepGraph")
                results, payload, hist = run_one(exp_name, cfg_trial, ts, Xt, mats, feature_names, return_split=True)
                if not hist or not hist.get(f"val_{sweep_metric}"):
                    return float("inf"), results, payload, hist
                series = [v for v in hist[f"val_{sweep_metric}"] if np.isfinite(v)]
                val_best = float(np.min(series)) if series else float("inf")
                return val_best, results, payload, hist

            def _score_cfg(cfg_trial, trial=None):
                cfg_trial = cfg_trial.copy()
                cfg_trial["EARLY_STOP_METRIC"] = sweep_metric
                fold_cfgs = _build_fold_cfgs(cfg_trial)
                vals = []
                for fi, cfg_fold in enumerate(fold_cfgs):
                    vbest, _res, _payload, _hist = _eval_family(cfg_fold, use_val=True)
                    if np.isfinite(vbest):
                        vals.append(float(vbest))
                        if trial is not None:
                            trial.report(float(np.mean(vals)), step=fi)
                            if trial.should_prune():
                                raise optuna.TrialPruned()
                if not vals:
                    return float("inf"), float("inf"), float("inf"), 0
                vmean = float(np.mean(vals))
                vstd = float(np.std(vals))
                return float(vmean + variance_penalty * vstd), vmean, vstd, len(vals)

            if model_family == "drift" and bool(sweep_cfg.get("drift_grid", True)):
                lag_grid = _as_choices(family_space.get("DRIFT_LAG"), [6, 12, 18, 24, 36, 48])
                damp_grid = _as_choices(family_space.get("DRIFT_DAMP"), [1.0])
                rows = []
                best_row = None
                for lag in lag_grid:
                    for damp in damp_grid:
                        print(f"[sweep] drift grid candidate: DRIFT_LAG={int(lag)}, DRIFT_DAMP={float(damp)}")
                        cfg_trial = CFG.copy()
                        cfg_trial.update({"DRIFT_LAG": int(lag), "DRIFT_DAMP": float(damp)})
                        obj, vmean, vstd, nf = _score_cfg(cfg_trial, trial=None)
                        _v, test_res, payload, _h = _eval_family(cfg_trial, use_val=False)
                        row = {
                            "DRIFT_LAG": int(lag),
                            "DRIFT_DAMP": float(damp),
                            "objective": float(obj),
                            "val_mean": float(vmean),
                            "val_std": float(vstd),
                            "folds": int(nf),
                            "test_mae": float(test_res.get("mae", float("nan"))),
                            "test_rmse": float(test_res.get("rmse", float("nan"))),
                            "persist_test_mae": float(test_res.get("persist_mae", float("nan"))),
                            "persist_test_rmse": float(test_res.get("persist_rmse", float("nan"))),
                        }
                        rows.append(row)
                        if best_row is None or row["objective"] < best_row["objective"]:
                            best_row = row
                import csv
                with results_path.open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
                best_path.write_text(json.dumps(best_row, indent=2), encoding="utf-8")
                print(f"[sweep] drift best objective={best_row['objective']:.6f}")
                print(f"[sweep] results -> {results_path}")
                print(f"[sweep] best cfg -> {best_path}")
                return

            def objective(trial):
                cfg_trial = CFG.copy()
                cfg_trial["TEMP_WINDOW"] = trial.suggest_categorical("TEMP_WINDOW", _as_choices(common_space.get("TEMP_WINDOW"), [6, 12, 18, 24]))
                cfg_trial["FORECAST"] = trial.suggest_categorical("FORECAST", _as_choices(common_space.get("FORECAST"), [6, 12, 24]))
                cfg_trial["HIDDEN_CHANNELS"] = trial.suggest_categorical("HIDDEN_CHANNELS", _as_choices(common_space.get("HIDDEN_CHANNELS"), [32, 64, 96, 128]))
                cfg_trial["NUM_HEADS"] = trial.suggest_categorical("NUM_HEADS", _as_choices(common_space.get("NUM_HEADS"), [2, 4, 8]))
                d_lo, d_hi = _range_pair(common_space.get("DROPOUT"), 0.1, 0.5)
                lr_lo, lr_hi = _range_pair(common_space.get("LR"), 1e-4, 3e-3)
                wd_lo, wd_hi = _range_pair(common_space.get("WD"), 1e-6, 1e-3)
                cfg_trial["DROPOUT"] = trial.suggest_float("DROPOUT", d_lo, d_hi)
                cfg_trial["LR"] = trial.suggest_float("LR", lr_lo, lr_hi, log=True)
                cfg_trial["WD"] = trial.suggest_float("WD", wd_lo, wd_hi, log=True)
                if int(cfg_trial["HIDDEN_CHANNELS"]) % int(cfg_trial["NUM_HEADS"]) != 0:
                    raise optuna.TrialPruned("HIDDEN_CHANNELS % NUM_HEADS != 0")

                if model_family == "graph_base":
                    cfg_trial["EDGE_NORM"] = trial.suggest_categorical("EDGE_NORM", _as_choices(family_space.get("EDGE_NORM"), ["none", "deg", "row", "minmax"]))
                    cfg_trial["EDGE_SELF_LOOPS"] = trial.suggest_categorical("EDGE_SELF_LOOPS", _as_choices(family_space.get("EDGE_SELF_LOOPS"), [False, True]))
                    cfg_trial["GRAPH_GATE"] = trial.suggest_categorical("GRAPH_GATE", _as_choices(family_space.get("GRAPH_GATE"), [True, False]))
                elif model_family in {"graph_mix", "graph_multi_mix"}:
                    gm_lo, gm_hi = _range_pair(family_space.get("GRAPH_MIX_LAMBDA"), 0.0, 0.5)
                    cfg_trial["GRAPH_MIX_LAMBDA"] = trial.suggest_float("GRAPH_MIX_LAMBDA", gm_lo, gm_hi)
                if model_family in {"graph_multi", "graph_multi_mix"}:
                    lag_opts = family_space.get("GRAPH_MULTI_LAGS")
                    lag_choices = []
                    if isinstance(lag_opts, (list, tuple)) and lag_opts:
                        for raw in lag_opts:
                            vals = _parse_int_list(raw)
                            if 0 not in vals:
                                vals = [0] + vals
                            if vals:
                                lag_choices.append(tuple(sorted(set(vals))))
                    if not lag_choices:
                        lag_choices = [(0, 6), (0, 6, 12), (0, 12, 24)]
                    cfg_trial["GRAPH_MULTI_LAGS"] = list(trial.suggest_categorical("GRAPH_MULTI_LAGS", lag_choices))
                    aux_opts = family_space.get("GRAPH_MULTI_AUX_FORECASTS")
                    aux_choices = []
                    if isinstance(aux_opts, (list, tuple)) and aux_opts:
                        for raw in aux_opts:
                            vals = [v for v in _parse_int_list(raw) if v > 0]
                            aux_choices.append(tuple(sorted(set(vals))))
                    if not aux_choices:
                        aux_choices = [tuple(), (6,), (6, 12)]
                    aux = list(trial.suggest_categorical("GRAPH_MULTI_AUX_FORECASTS", aux_choices))
                    cfg_trial["GRAPH_MULTI_AUX_FORECASTS"] = aux
                    aw_lo, aw_hi = _range_pair(family_space.get("GRAPH_MULTI_AUX_WEIGHT"), 0.0, 0.2)
                    cfg_trial["GRAPH_MULTI_AUX_WEIGHT"] = 0.0 if not aux else trial.suggest_float("GRAPH_MULTI_AUX_WEIGHT", aw_lo, aw_hi)

                _apply_family_flags(cfg_trial)
                shown_keys = [
                    "TEMP_WINDOW", "FORECAST",
                    "HIDDEN_CHANNELS", "NUM_HEADS",
                    "DROPOUT", "LR", "WD",
                    "EDGE_NORM", "EDGE_SELF_LOOPS", "GRAPH_GATE",
                    "GRAPH_MIX", "GRAPH_MIX_LAMBDA",
                    "GRAPH_MULTI", "GRAPH_MULTI_LAGS",
                    "GRAPH_MULTI_AUX_FORECASTS", "GRAPH_MULTI_AUX_WEIGHT",
                ]
                shown_cfg = {k: cfg_trial.get(k) for k in shown_keys if k in cfg_trial}
                print(f"[sweep] trial={trial.number} family={model_family} cfg={shown_cfg}")
                obj, vmean, vstd, nf = _score_cfg(cfg_trial, trial=trial)
                trial.set_user_attr("val_mean", float(vmean))
                trial.set_user_attr("val_std", float(vstd))
                trial.set_user_attr("folds", int(nf))
                return obj

            pruner_cfg = sweep_cfg.get("pruner") if isinstance(sweep_cfg.get("pruner"), dict) else {}
            ptype = str(pruner_cfg.get("type", "median") or "median").lower()
            if ptype == "hyperband":
                pruner = optuna.pruners.HyperbandPruner(
                    min_resource=int(pruner_cfg.get("min_resource", 1) or 1),
                    max_resource=int(pruner_cfg.get("max_resource", n_trials) or n_trials),
                    reduction_factor=int(pruner_cfg.get("reduction_factor", 3) or 3),
                )
            elif ptype == "none":
                pruner = optuna.pruners.NopPruner()
            else:
                pruner = optuna.pruners.MedianPruner(
                    n_startup_trials=int(pruner_cfg.get("n_startup_trials", 8) or 8),
                    n_warmup_steps=int(pruner_cfg.get("n_warmup_steps", 0) or 0),
                )

            study_cfg = sweep_cfg.get("study") if isinstance(sweep_cfg.get("study"), dict) else {}
            storage = study_cfg.get("storage")
            if isinstance(storage, str) and "{family}" in storage:
                storage = storage.format(family=model_family)
            study_name = str(study_cfg.get("name", f"usecase_cyberspace_{model_family}") or f"usecase_cyberspace_{model_family}")
            if "{family}" in study_name:
                study_name = study_name.format(family=model_family)
            load_if_exists = bool(study_cfg.get("load_if_exists", True))

            study = optuna.create_study(
                direction="minimize",
                study_name=study_name,
                storage=storage,
                load_if_exists=load_if_exists,
                pruner=pruner,
            )
            existing_trials = len(study.trials)
            n_trials_run = n_trials
            if isinstance(max_total_trials, int):
                n_trials_run = max(0, max_total_trials - existing_trials)
                print(f"[sweep] existing_trials={existing_trials} | max_total_trials={max_total_trials} | running_now={n_trials_run}")
            else:
                print(f"[sweep] existing_trials={existing_trials} | adding_trials={n_trials_run}")

            if n_trials_run > 0:
                study.optimize(objective, n_trials=n_trials_run)
            else:
                print("[sweep] trial budget reached; skipping optimize().")

            complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not complete_trials:
                print("[sweep] no completed trials in study; nothing to report.")
                return

            best_cfg = CFG.copy()
            for k, v in study.best_params.items():
                best_cfg[k] = list(v) if isinstance(v, tuple) else v
            _apply_family_flags(best_cfg)
            best_cfg["EARLY_STOP_METRIC"] = sweep_metric
            val_best, family_results, payload, _hist = _eval_family(best_cfg, use_val=False)
            best_trial = study.best_trial
            val_mean = float(best_trial.user_attrs.get("val_mean", float("nan")))
            val_std = float(best_trial.user_attrs.get("val_std", float("nan")))

            mae_p = rmse_p = mae_d = rmse_d = float("nan")
            if payload is not None:
                ts_c, Xt_c, mats_c, targets_raw, masks, test_idx, _mu, _sd = payload
                (mae_p, rmse_p), (mae_d, rmse_d) = baselines_on_test(ts_c, Xt_c, mats_c, best_cfg, targets_raw, masks, test_idx)

            import csv
            param_cols = sorted({k for t in study.trials for k in t.params.keys()})
            with results_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["trial", "objective", "val_mean", "val_std", "folds", "family"] + param_cols)
                for t in study.trials:
                    row = [t.number, t.value, t.user_attrs.get("val_mean"), t.user_attrs.get("val_std"), t.user_attrs.get("folds"), model_family]
                    row.extend([t.params.get(c) for c in param_cols])
                    w.writerow(row)

            best_path.write_text(
                json.dumps(
                    {
                        "family": model_family,
                        "metric": sweep_metric,
                        "objective": float(study.best_value),
                        "val_mean": val_mean,
                        "val_std": val_std,
                        "best_params": study.best_params,
                        "test": {
                            "family_mae": float(family_results.get("mae", float("nan"))),
                            "family_rmse": float(family_results.get("rmse", float("nan"))),
                            "persist_mae": float(mae_p),
                            "persist_rmse": float(rmse_p),
                            "drift_mae": float(mae_d),
                            "drift_rmse": float(rmse_d),
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            print("\n--- Sweep Summary ---")
            print(f"Family: {model_family}")
            print(f"Best objective: {study.best_value:.6f} | val_mean={val_mean:.6f} | val_std={val_std:.6f}")
            print(f"Family  MAE: {family_results.get('mae', float('nan')):.6f} | RMSE: {family_results.get('rmse', float('nan')):.6f}")
            print(f"Persist MAE: {mae_p:.6f}            | RMSE: {rmse_p:.6f}")
            print(f"Drift   MAE: {mae_d:.6f}            | RMSE: {rmse_d:.6f}")
            print(f"[sweep] results -> {results_path}")
            print(f"[sweep] best cfg -> {best_path}")
            return

        if diag_enabled:
            _announce("Running diagnostics")

            diag_overrides = diag_cfg.get("cfg_overrides") if isinstance(diag_cfg, dict) else None
            cfg_base = CFG.copy()
            if isinstance(diag_overrides, dict):
                cfg_base.update(diag_overrides)
            cfg_base.update({
                "TOPK_RECENT": False,
                "TOPK_RECENT_INFERENCE": False,
                "SHOW_FIGS": False,
                "DIAG_SAVE_DIR": str(BASE_DIR / "diagnostics_histories"),
            })

            # Edge + activity stats
            T, N = mats.shape[2], mats.shape[0]
            edge_rows = []
            active_rows = []
            for t in range(T):
                A = mats[:, :, t]
                nz = A[A > 0]
                cnt = int(nz.size)
                density = float(cnt) / float(N * N) if N > 0 else 0.0
                if cnt > 0:
                    edge_rows.append({
                        "t": t,
                        "edges": cnt,
                        "density": density,
                        "mean": float(nz.mean()),
                        "std": float(nz.std()),
                        "min": float(nz.min()),
                        "max": float(nz.max()),
                    })
                else:
                    edge_rows.append({
                        "t": t,
                        "edges": 0,
                        "density": density,
                        "mean": float("nan"),
                        "std": float("nan"),
                        "min": float("nan"),
                        "max": float("nan"),
                    })

                X = Xt[t]
                active = int((np.abs(X).sum(axis=1) > 0).sum())
                active_rows.append({"t": t, "active_nodes": active})

            import csv
            edge_path = BASE_DIR / "diagnostics_edge_stats.csv"
            act_path = BASE_DIR / "diagnostics_active_nodes.csv"
            with edge_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(edge_rows[0].keys()))
                w.writeheader()
                w.writerows(edge_rows)
            with act_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(active_rows[0].keys()))
                w.writeheader()
                w.writerows(active_rows)

            # Ablations on same split (fixed TEMP_WINDOW/FORECAST)
            diag_mode = str(diag_cfg.get("mode", "full") if isinstance(diag_cfg, dict) else "full").lower()
            graph_value_tests = bool(diag_cfg.get("graph_value_tests", True)) if isinstance(diag_cfg, dict) else True
            graph_value_lag = int(diag_cfg.get("graph_value_lag", 6) or 6) if isinstance(diag_cfg, dict) else 6
            mats_edge_shuffle = None
            mats_time_shift = None
            if graph_value_tests:
                mats_edge_shuffle = _shuffle_edges_rowwise(mats, seed=42)
                mats_time_shift = _time_shift_mats(mats, lag=graph_value_lag)
            if diag_mode in {"basic", "core", "minimal"}:
                runs = [
                    ("graph_base", {}),
                    ("graph_mix", {"GRAPH_MIX": True, "GRAPH_GATE": False}),
                    ("graph_multi", {"GRAPH_MULTI": True, "GRAPH_MIX": False}),
                    ("graph_multi_mix", {"GRAPH_MULTI": True, "GRAPH_MIX": True, "GRAPH_GATE": False}),
                    ("nograph", {"NODE_ONLY": True}),
                ]
            else:
                runs = [
                    ("graph_base", {}),
                    ("graph_mix", {"GRAPH_MIX": True, "GRAPH_GATE": False}),
                    ("graph_multi", {"GRAPH_MULTI": True, "GRAPH_MIX": False}),
                    ("graph_multi_mix", {"GRAPH_MULTI": True, "GRAPH_MIX": True, "GRAPH_GATE": False}),
                    ("nograph", {"NODE_ONLY": True}),
                    ("graph_deg_norm", {"EDGE_NORM": "deg"}),
                    ("graph_row_norm", {"EDGE_NORM": "row"}),
                    ("graph_self_loops", {"EDGE_SELF_LOOPS": True}),
                    ("graph_binary_edges", {"EDGE_ATTR_MODE": "binary"}),
                    ("graph_no_gate", {"GRAPH_GATE": False}),
                ]
            if isinstance(diag_cfg, dict) and diag_cfg.get("graph_multi_tests"):
                runs.extend([
                    ("graph_multi_aux0", {"GRAPH_MULTI": True, "GRAPH_MULTI_AUX_WEIGHT": 0.0}),
                    ("graph_multi_lags_0_12_24", {"GRAPH_MULTI": True, "GRAPH_MULTI_LAGS": [0, 12, 24]}),
                    ("graph_multi_small", {"GRAPH_MULTI": True, "HIDDEN_CHANNELS": 32, "NUM_HEADS": 2}),
                ])
            if graph_value_tests:
                runs.extend([
                    ("graph_edge_shuffle", {}),
                    ("graph_time_shift", {}),
                ])
            def _normalize_block(block_raw, idx):
                if not isinstance(block_raw, dict):
                    return {"name": f"block_{idx}", "overrides": {}}
                name = block_raw.get("name") or block_raw.get("label") or f"block_{idx}"
                overrides: dict[str, Any] = {}
                raw_overrides = block_raw.get("overrides")
                if isinstance(raw_overrides, dict):
                    overrides.update(raw_overrides)
                key_map = {
                    "hidden_channels": "HIDDEN_CHANNELS",
                    "num_heads": "NUM_HEADS",
                    "temp_window": "TEMP_WINDOW",
                    "forecast": "FORECAST",
                    "dropout": "DROPOUT",
                    "lr": "LR",
                    "wd": "WD",
                }
                for short_key, cfg_key in key_map.items():
                    if short_key in block_raw:
                        overrides[cfg_key] = block_raw[short_key]
                    if cfg_key in block_raw:
                        overrides[cfg_key] = block_raw[cfg_key]
                return {"name": str(name), "overrides": overrides}

            raw_blocks = None
            block_cfg = diag_cfg.get("block_experiments") if isinstance(diag_cfg, dict) else None
            if isinstance(block_cfg, dict):
                if bool(block_cfg.get("enabled", False)):
                    raw_blocks = block_cfg.get("blocks")
                    if raw_blocks is None and isinstance(diag_cfg, dict):
                        raw_blocks = diag_cfg.get("blocks")
                else:
                    raw_blocks = None
            elif isinstance(diag_cfg, dict):
                raw_blocks = diag_cfg.get("blocks")

            blocks: list[dict[str, Any]] = []
            if raw_blocks:
                for i, blk in enumerate(raw_blocks, start=1):
                    blocks.append(_normalize_block(blk, i))
            if not blocks:
                blocks = [{"name": "default", "overrides": {}}]

            diag_rows: list[dict[str, Any]] = []
            baselines_by_block: dict[str, dict[str, dict[str, float]]] = {}
            last_block_name = None
            last_payload = None
            last_cfg = None

            total_blocks = len(blocks)
            for block_idx, block in enumerate(blocks, start=1):
                block_name = block.get("name") or f"block_{block_idx}"
                block_overrides = block.get("overrides") or {}
                print(f"[diagnostics] Block {block_idx}/{total_blocks}: {block_name} overrides={block_overrides}")
                cfg_block = cfg_base.copy()
                cfg_block.update(block.get("overrides") or {})

                base_payload = None
                for name, overrides in runs:
                    cfg_run = cfg_block.copy()
                    cfg_run.update(overrides)
                    cfg_run["DIAG_RUN_NAME"] = name
                    if name == "graph_base":
                        cfg_run["PER_NODE_OUT_PATH"] = str(BASE_DIR / "diagnostics_per_node_graph.csv")
                    if name == "graph_mix":
                        cfg_run["PER_NODE_OUT_PATH"] = str(BASE_DIR / "diagnostics_per_node_graph_mix.csv")
                        if isinstance(diag_cfg, dict) and "graph_mix_lambda" in diag_cfg:
                            cfg_run["GRAPH_MIX_LAMBDA"] = float(diag_cfg.get("graph_mix_lambda") or 0.0)
                        cfg_run["ALPHA_OUT_PATH"] = str(BASE_DIR / "diagnostics_alpha_graph_mix.csv")
                    if name == "graph_multi_mix":
                        cfg_run["PER_NODE_OUT_PATH"] = str(BASE_DIR / "diagnostics_per_node_graph_multi_mix.csv")
                        if isinstance(diag_cfg, dict) and "graph_mix_lambda" in diag_cfg:
                            cfg_run["GRAPH_MIX_LAMBDA"] = float(diag_cfg.get("graph_mix_lambda") or 0.0)
                        cfg_run["ALPHA_OUT_PATH"] = str(BASE_DIR / "diagnostics_alpha_graph_multi_mix.csv")
                    if name == "graph_multi":
                        cfg_run["PER_NODE_OUT_PATH"] = str(BASE_DIR / "diagnostics_per_node_graph_multi.csv")
                        if isinstance(diag_cfg, dict):
                            if "graph_multi_lags" in diag_cfg:
                                cfg_run["GRAPH_MULTI_LAGS"] = diag_cfg.get("graph_multi_lags")
                            if "graph_multi_aux_forecasts" in diag_cfg:
                                cfg_run["GRAPH_MULTI_AUX_FORECASTS"] = diag_cfg.get("graph_multi_aux_forecasts")
                            if "graph_multi_aux_weight" in diag_cfg:
                                cfg_run["GRAPH_MULTI_AUX_WEIGHT"] = float(diag_cfg.get("graph_multi_aux_weight") or 0.0)
                    if name == "graph_multi_mix":
                        if isinstance(diag_cfg, dict):
                            if "graph_multi_lags" in diag_cfg:
                                cfg_run["GRAPH_MULTI_LAGS"] = diag_cfg.get("graph_multi_lags")
                            if "graph_multi_aux_forecasts" in diag_cfg:
                                cfg_run["GRAPH_MULTI_AUX_FORECASTS"] = diag_cfg.get("graph_multi_aux_forecasts")
                            if "graph_multi_aux_weight" in diag_cfg:
                                cfg_run["GRAPH_MULTI_AUX_WEIGHT"] = float(diag_cfg.get("graph_multi_aux_weight") or 0.0)
                    if name == "nograph":
                        cfg_run["PER_NODE_OUT_PATH"] = str(BASE_DIR / "diagnostics_per_node_nograph.csv")
                    mats_run = mats
                    if name == "graph_edge_shuffle" and mats_edge_shuffle is not None:
                        mats_run = mats_edge_shuffle
                    if name == "graph_time_shift" and mats_time_shift is not None:
                        mats_run = mats_time_shift
                    results, payload, hist = run_one(name, cfg_run, ts, Xt, mats_run, feature_names, return_split=True)
                    if base_payload is None:
                        base_payload = payload
                    diag_rows.append({
                        "block": block_name,
                        "name": name,
                        "mae": results.get("mae"),
                        "rmse": results.get("rmse"),
                        "hidden_channels": cfg_run.get("HIDDEN_CHANNELS"),
                        "num_heads": cfg_run.get("NUM_HEADS"),
                        "dropout": cfg_run.get("DROPOUT"),
                        "lr": cfg_run.get("LR"),
                        "wd": cfg_run.get("WD"),
                        "temp_window": cfg_run.get("TEMP_WINDOW"),
                        "forecast": cfg_run.get("FORECAST"),
                    })

                mae_p = rmse_p = mae_d = rmse_d = float("nan")
                if base_payload is not None:
                    ts_c, Xt_c, mats_c, targets_raw, masks, test_idx, mu, sd = base_payload
                    (mae_p, rmse_p), (mae_d, rmse_d) = baselines_on_test(
                        ts_c, Xt_c, mats_c, cfg_block, targets_raw, masks, test_idx
                    )
                    # Per-node baseline metrics
                    try:
                        fw = cfg_block['FW']; W = cfg_block['TEMP_WINDOW']
                        mode = (cfg_block['TARGET_MODE'] or "").strip().lower()
                        if mode == "absolute":
                            mode = "level"
                        drift_lag = int(cfg_block.get("DRIFT_LAG") or W)
                        drift_damp = float(cfg_block.get("DRIFT_DAMP", 1.0) or 1.0)

                        fw_vals = _target_level_series(Xt_c, fw, mode, cfg_block)
                        drift_vals = _compute_drift_level(
                            fw_vals,
                            horizon=int(cfg_block['FORECAST']),
                            lag=drift_lag,
                            damp=drift_damp,
                        )

                        def fw_level(t_idx):
                            return fw_vals[t_idx]

                        def drift_level(t_idx):
                            return drift_vals[t_idx]

                        preds_pers, preds_drift, ys, msks = [], [], [], []
                        for t in test_idx:
                            y_true_raw = targets_raw[t]; msk = masks[t]
                            if mode in {"level", "smooth_relative", "smooth_relative_level"}:
                                pred_pers = fw_level(t); pred_drift = drift_level(t)
                            elif mode == "residual":
                                cur = fw_level(t); drift_fut = drift_level(t)
                                pred_pers = np.zeros_like(cur); pred_drift = drift_fut - cur
                            elif mode == "drift_residual":
                                cur = fw_level(t); drift_fut = drift_level(t)
                                pred_pers = cur - drift_fut
                                pred_drift = np.zeros_like(cur)
                            elif mode == "log_change":
                                cur = fw_level(t); drift_fut = drift_level(t)
                                pred_pers = np.zeros_like(cur)
                                pred_drift  = np.arcsinh(drift_fut) - np.arcsinh(cur)
                            else:
                                raise ValueError("Unknown TARGET_MODE")
                            preds_pers.append(pred_pers); preds_drift.append(pred_drift)
                            ys.append(y_true_raw); msks.append(msk)

                        y = np.stack(ys)
                        m = (np.stack(msks) > 0)
                        p1 = np.stack(preds_pers)
                        p2 = np.stack(preds_drift)

                        node_names, _orig_idx, _src = resolve_active_names(y.shape[1], cfg_block['FEATURE_NAMES_DEFAULT'])
                        import csv
                        def _write_per_node(path, pred):
                            out_path = Path(path)
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            with out_path.open("w", newline="", encoding="utf-8") as f:
                                w = csv.writer(f)
                                w.writerow(["node", "mae", "rmse", "count"])
                                for n in range(y.shape[1]):
                                    idx = m[:, n]
                                    if idx.sum() == 0:
                                        mae_n = float("nan"); rmse_n = float("nan"); cnt = 0
                                    else:
                                        diff = (pred[:, n] - y[:, n])[idx]
                                        mae_n = float(np.abs(diff).mean())
                                        rmse_n = float(np.sqrt((diff ** 2).mean()))
                                        cnt = int(idx.sum())
                                    name = node_names[n] if n < len(node_names) else f"n{n}"
                                    w.writerow([name, mae_n, rmse_n, cnt])

                        _write_per_node(BASE_DIR / "diagnostics_per_node_persist.csv", p1)
                        _write_per_node(BASE_DIR / "diagnostics_per_node_drift.csv", p2)
                    except Exception:
                        pass

                    # Append baselines into diagnostics_runs.csv for convenience
                    baseline_rows = [
                        {
                            "block": block_name,
                            "name": "persist",
                            "mae": mae_p,
                            "rmse": rmse_p,
                            "hidden_channels": "",
                            "num_heads": "",
                            "dropout": "",
                            "lr": "",
                            "wd": "",
                            "temp_window": cfg_block.get("TEMP_WINDOW"),
                            "forecast": cfg_block.get("FORECAST"),
                        },
                        {
                            "block": block_name,
                            "name": "drift",
                            "mae": mae_d,
                            "rmse": rmse_d,
                            "hidden_channels": "",
                            "num_heads": "",
                            "dropout": "",
                            "lr": "",
                            "wd": "",
                            "temp_window": cfg_block.get("TEMP_WINDOW"),
                            "forecast": cfg_block.get("FORECAST"),
                        },
                    ]
                    diag_rows.extend(baseline_rows)

                baselines_by_block[str(block_name)] = {
                    "persist": {"mae": float(mae_p), "rmse": float(rmse_p)},
                    "drift": {"mae": float(mae_d), "rmse": float(rmse_d)},
                }

                if base_payload is not None:
                    last_payload = base_payload
                    last_cfg = cfg_block
                    last_block_name = block_name

            diag_path = BASE_DIR / "diagnostics_runs.csv"
            if diag_rows:
                with diag_path.open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
                    w.writeheader()
                    w.writerows(diag_rows)
            else:
                print("[diagnostics] no runs to write.")

            cfg_for_extra = last_cfg or cfg_base
            base_payload = last_payload
            if base_payload is not None:
                ts_c, Xt_c, mats_c, targets_raw, masks, test_idx, mu, sd = base_payload

            # ---- Additional diagnostics: per-node gains + lag correlations ----
            if isinstance(diag_cfg, dict) and diag_cfg.get("node_gain_tests", True) and base_payload is not None:
                try:
                    base_dir = Path(BASE_DIR)
                    paths = {
                        "graph": base_dir / "diagnostics_per_node_graph.csv",
                        "graph_multi": base_dir / "diagnostics_per_node_graph_multi.csv",
                        "graph_multi_mix": base_dir / "diagnostics_per_node_graph_multi_mix.csv",
                        "nograph": base_dir / "diagnostics_per_node_nograph.csv",
                        "persist": base_dir / "diagnostics_per_node_persist.csv",
                        "drift": base_dir / "diagnostics_per_node_drift.csv",
                    }
                    dfs = {}
                    for k, p in paths.items():
                        if p.exists():
                            df = pd.read_csv(p)
                            dfs[k] = df.rename(columns={"mae": f"mae_{k}", "rmse": f"rmse_{k}", "count": f"count_{k}"})
                    merged = None
                    for df in dfs.values():
                        merged = df if merged is None else merged.merge(df, on="node", how="outer")

                    if merged is not None:
                        # reconstruct train_idx to compute volatility
                        split_cfg = cfg_for_extra.get("SPLIT_DATES")
                        split_res = get_split_indices(
                            ts_c,
                            cfg_for_extra['TEMP_WINDOW'],
                            cfg_for_extra['FORECAST'],
                            cfg_for_extra['SPLIT_FRACS'],
                            split_cfg,
                        )
                        if split_res is None:
                            raise RuntimeError("Fixed split produced no valid indices for diagnostics.")
                        train_idx, _va_idx, _test_idx, _ntr, _nva, _nte, _fixed = split_res
                        tr_train = targets_raw[train_idx].copy()
                        tm_train = masks[train_idx].copy()
                        tr_train[tm_train <= 0] = np.nan
                        vol = np.nanstd(tr_train, axis=0)
                        deg = degree_over_time(mats_c[:, :, :max(1, mats_c.shape[2] - cfg_for_extra['FORECAST'])])
                        node_names, _orig, _src = resolve_active_names(len(vol), cfg_for_extra['FEATURE_NAMES_DEFAULT'])
                        meta = pd.DataFrame({
                            "node": node_names,
                            "volatility_train": vol,
                            "degree_mean": deg,
                        })
                        merged = merged.merge(meta, on="node", how="left")

                        if "mae_persist" in merged.columns:
                            for k in ["graph", "graph_multi", "graph_multi_mix", "nograph"]:
                                if f"mae_{k}" in merged.columns:
                                    merged[f"delta_{k}"] = merged[f"mae_{k}"] - merged["mae_persist"]
                        out_gain = base_dir / "diagnostics_node_gain.csv"
                        merged.to_csv(out_gain, index=False)

                        buckets = []
                        for col, label in [("volatility_train", "volatility"), ("degree_mean", "degree")]:
                            if col not in merged.columns:
                                continue
                            series = pd.to_numeric(merged[col], errors="coerce")
                            if series.notna().sum() < 3:
                                continue
                            try:
                                q = pd.qcut(series, 3, labels=["low", "mid", "high"])
                            except Exception:
                                q = pd.cut(series, 3, labels=["low", "mid", "high"])
                            merged["_bucket"] = q
                            grp = merged.groupby("_bucket", dropna=True)
                            for b, g in grp:
                                row = {"bucket_type": label, "bucket": str(b), "count_nodes": int(len(g))}
                                for k in ["graph", "graph_multi", "graph_multi_mix", "nograph"]:
                                    dk = f"delta_{k}"
                                    if dk in g.columns:
                                        row[f"mean_{dk}"] = float(np.nanmean(g[dk].astype(float)))
                                buckets.append(row)
                        if buckets:
                            pd.DataFrame(buckets).to_csv(base_dir / "diagnostics_node_gain_buckets.csv", index=False)
                except Exception as exc:
                    print(f"[diagnostics] node gain diagnostics failed: {exc}")

            if isinstance(diag_cfg, dict) and diag_cfg.get("lag_corr_tests", True) and base_payload is not None:
                try:
                    fw = np.asarray(cfg_for_extra['FW'], dtype=float)
                    T = Xt_c.shape[0]
                    S = T - cfg_for_extra['FORECAST']
                    if S > 0:
                        node_vals = np.zeros((T, Xt_c.shape[1]), dtype=float)
                        for t in range(T):
                            node_vals[t] = fw_of_frame(Xt_c[t], fw)
                        lags = diag_cfg.get("lag_corr_lags", [0, 6, 12, 24])
                        rows = []
                        node_names, _orig, _src = resolve_active_names(Xt_c.shape[1], cfg_for_extra['FEATURE_NAMES_DEFAULT'])
                        for lag in lags:
                            lag = int(lag)
                            for n in range(Xt_c.shape[1]):
                                xs = []
                                xn = []
                                ys = []
                                for t in range(S):
                                    t0 = t - lag
                                    if t0 < 0:
                                        continue
                                    y = targets_raw[t, n]
                                    if not np.isfinite(y):
                                        continue
                                    x_self = node_vals[t0, n]
                                    A = mats_c[:, :, t0]
                                    wsum = A[n].sum()
                                    if wsum > 0:
                                        x_nei = (A[n] @ node_vals[t0]) / wsum
                                    else:
                                        x_nei = np.nan
                                    xs.append(x_self); xn.append(x_nei); ys.append(y)
                                xs = np.asarray(xs, dtype=float)
                                xn = np.asarray(xn, dtype=float)
                                ys = np.asarray(ys, dtype=float)
                                def _corr(a, b):
                                    mask = np.isfinite(a) & np.isfinite(b)
                                    if mask.sum() < 5:
                                        return float("nan"), int(mask.sum())
                                    aa = a[mask] - a[mask].mean()
                                    bb = b[mask] - b[mask].mean()
                                    denom = np.sqrt((aa**2).sum() * (bb**2).sum())
                                    if denom <= 0:
                                        return float("nan"), int(mask.sum())
                                    return float((aa*bb).sum() / denom), int(mask.sum())
                                cs, cnts = _corr(xs, ys)
                                cn, cntn = _corr(xn, ys)
                                name = node_names[n] if n < len(node_names) else f"n{n}"
                                rows.append({
                                    "node": name,
                                    "lag": lag,
                                    "corr_self": cs,
                                    "corr_neighbor": cn,
                                    "count": min(cnts, cntn),
                                })
                        if rows:
                            pd.DataFrame(rows).to_csv(base_dir / "diagnostics_lag_corr.csv", index=False)
                except Exception as exc:
                    print(f"[diagnostics] lag correlation failed: {exc}")

            summary_cfg = cfg_for_extra if cfg_for_extra is not None else cfg_base
            default_baselines = {
                "persist": {"mae": float("nan"), "rmse": float("nan")},
                "drift": {"mae": float("nan"), "rmse": float("nan")},
            }
            summary = {
                "target_mode": summary_cfg.get("TARGET_MODE"),
                "loss_space": summary_cfg.get("LOSS_SPACE"),
                "forecast": summary_cfg.get("FORECAST"),
                "temp_window": summary_cfg.get("TEMP_WINDOW"),
                "edge_stats": {
                    "edges_mean": float(np.nanmean([r["edges"] for r in edge_rows])),
                    "density_mean": float(np.nanmean([r["density"] for r in edge_rows])),
                    "weight_mean": float(np.nanmean([r["mean"] for r in edge_rows])),
                    "weight_std": float(np.nanmean([r["std"] for r in edge_rows])),
                },
                "active_nodes_mean": float(np.nanmean([r["active_nodes"] for r in active_rows])),
                "baselines": baselines_by_block.get(str(last_block_name), default_baselines),
                "baselines_by_block": baselines_by_block,
                "blocks": blocks,
                "runs": diag_rows,
            }
            (BASE_DIR / "diagnostics_summary.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )

            print("\n--- Diagnostics Summary ---")
            print(f"Edge stats -> {edge_path}")
            print(f"Active nodes -> {act_path}")
            print(f"Runs -> {diag_path}")
            print(f"Summary -> {BASE_DIR / 'diagnostics_summary.json'}")
            if isinstance(diag_cfg, dict) and diag_cfg.get("merge_per_node_all", True):
                try:
                    import subprocess
                    root_dir = Path(__file__).resolve().parents[2]
                    merge_script = root_dir / "scripts" / "merge_per_node_diagnostics.py"
                    subprocess.run(
                        [sys.executable, str(merge_script), "--base-dir", str(BASE_DIR)],
                        check=True,
                    )
                except Exception as exc:
                    print(f"[diagnostics] failed to merge per-node diagnostics: {exc}")
            return

        # --- Graph model ---
        cfg_graph = CFG.copy()
        cfg_graph.update({
            "NODE_ONLY": False,
        })
        print("\n--- Training and Evaluating Graph Model (with edges) ---")
        graph_results, payload, hist_graph = run_one("GraphModel", cfg_graph, ts, Xt, mats, feature_names, return_split=True)
        print(f"Graph Model Test MAE:  {graph_results['mae']:.6f}, RMSE: {graph_results['rmse']:.6f}")

        print("\n" + "="*60 + "\n")

        # --- No-Graph model ---
        cfg_nograph = cfg_graph.copy(); cfg_nograph.update({"NODE_ONLY": True})
        print("\n--- Training and Evaluating No-Graph Model (no message passing) ---")
        nograph_results, _, hist_nograph = run_one("NoGraphModel", cfg_nograph, ts, Xt, mats, feature_names, return_split=True)
        print(f"No-Graph Model Test MAE: {nograph_results['mae']:.6f}, RMSE: {nograph_results['rmse']:.6f}")

        # --- Baselines on SAME split ---
        ts_c, Xt_c, mats_c, targets_raw, masks, test_idx, mu, sd = payload
        print("\n--- Baselines (same split / masks) ---")
        (mae_p, rmse_p), (mae_d, rmse_d) = baselines_on_test(ts_c, Xt_c, mats_c, CFG, targets_raw, masks, test_idx)
        print(f"Persistence baseline   MAE: {mae_p:.6f}, RMSE: {rmse_p:.6f}")
        print(f"Drift baseline         MAE: {mae_d:.6f}, RMSE: {rmse_d:.6f}")

        print("\n" + "="*60 + "\n")
        print("--- Comparison Summary ---")
        print(f"Graph    MAE: {graph_results['mae']:.6f} | RMSE: {graph_results['rmse']:.6f}")
        print(f"NoGraph  MAE: {nograph_results['mae']:.6f} | RMSE: {nograph_results['rmse']:.6f}")
        print(f"Persist  MAE: {mae_p:.6f}            | RMSE: {rmse_p:.6f}")
        print(f"Drift    MAE: {mae_d:.6f}            | RMSE: {rmse_d:.6f}")

        # --- Plot train/val curves + persistence line ---
        metric = CFG.get("PLOT_METRIC", "mae").lower()
        persist_value = mae_p if metric == "mae" else rmse_p
        fig_name = f"train_val_{metric}_graph_vs_nograph_with_persist__EM_{CFG['EMERGENCE_MODE']}.pdf"
        save_path = PLOT_DIR / fig_name
        plot_histories(hist_graph, hist_nograph, persist_value, metric, save_path)

    main()


def run_graph_pipeline(ctx: Dict[str, Any]) -> None:
    """Public entry point so other steps can reuse the notebook logic."""
    _run_graph_notebook(ctx)


# Cell 11
def external_inputs(cfg, usecase_dir: Path, prev_dir: Path | None) -> list[Path]:
    assert prev_dir is not None, "step_04_05_graph requires step_03 outputs"
    inputs = [prev_dir / "outputs"]
    extract_dir = usecase_dir / "01_extract" / "outputs"
    inputs.append(extract_dir)
    extra = cfg.get("paths", {}).get("graph_extra_inputs", []) if isinstance(cfg, dict) else []
    for rel in extra:
        p = Path(rel)
        if not p.is_absolute():
            p = (usecase_dir / rel).resolve()
        inputs.append(p)
    return inputs

# Cell 12
def relevant_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    params = cfg.get("params", {}) if isinstance(cfg, dict) else {}
    if not isinstance(params, dict):
        return {}
    graph_params = params.get("graph")
    if not isinstance(graph_params, dict):
        graph_params = params.get("graph_train")
    if not isinstance(graph_params, dict):
        graph_params = params.get("graph_build")
    return graph_params if isinstance(graph_params, dict) else {}

# Cell 13
def run(cfg: Dict[str, Any], usecase_dir: Path, prev_dir: Path, step_dir: Path) -> Dict[str, Any]:
    params = cfg.get("params", {}) if isinstance(cfg, dict) else {}
    graph_params = params.get("graph", {}) if isinstance(params, dict) else {}
    predict_params = params.get("predict", {}) if isinstance(params, dict) else {}
    skip_training = bool(graph_params.get("skip_training", False)) if isinstance(graph_params, dict) else False
    out_dir, outputs_root, run_id = _resolve_training_output_dir(
        step_dir,
        graph_params if isinstance(graph_params, dict) else {},
        skip_training=skip_training,
    )

    preprocess_cfg = graph_params.get("preprocess", {}) if isinstance(graph_params, dict) else {}
    tail_correction_cfg = graph_params.get("tail_correction", {}) if isinstance(graph_params, dict) else {}
    preview_cfg = graph_params.get("preview", {}) if isinstance(graph_params, dict) else {}
    plot_cfg = graph_params.get("plot", {}) if isinstance(graph_params, dict) else {}
    if not plot_cfg and isinstance(predict_params, dict):
        plot_cfg = predict_params.get("plot", {})
    cfg_defaults = graph_params.get("cfg_defaults") if isinstance(graph_params, dict) else None
    if not isinstance(cfg_defaults, dict) and isinstance(predict_params, dict) and predict_params.get("cfg_defaults") is not None:
        print("[graph] Note: predict.cfg_defaults is ignored for training; set params.graph.cfg_defaults explicitly.")
    cfg_overrides = graph_params.get("cfg_overrides", {}) if isinstance(graph_params, dict) else {}

    forced_global = _coerce_list_any(params.get("forced_keywords"), []) if isinstance(params, dict) else []
    forced_graph = _coerce_list_any(graph_params.get("forced_keywords"), []) if isinstance(graph_params, dict) else []
    combined_extra: list[str] = []
    for kw in forced_global + forced_graph:
        if isinstance(kw, str):
            token = kw.strip()
            if token and token not in combined_extra:
                combined_extra.append(token)
    additional_keywords = combined_extra
    remove_keywords = graph_params.get("remove_keywords") if isinstance(graph_params, dict) else None
    keyword_aliases_cfg = graph_params.get("keyword_aliases") if isinstance(graph_params, dict) else None
    drop_keywords_cfg = graph_params.get("drop_keywords", "drop_fully_inactive") if isinstance(graph_params, dict) else "drop_fully_inactive"
    if remove_keywords is None and isinstance(params, dict):
        remove_keywords = params.get("remove_keywords")
    if keyword_aliases_cfg is None and isinstance(params, dict):
        keyword_aliases_cfg = params.get("keyword_aliases")

    refined_collection_cfg = params.get("refined_collection", {}) if isinstance(params, dict) else {}
    refined_enabled = bool(refined_collection_cfg.get("enabled"))
    reuse_graph = bool(graph_params.get("reuse_existing_graph", False)) if isinstance(graph_params, dict) else False

    keywords_name = graph_params.get("keywords_csv", "cleaned_keywords_to_build_graphs.csv")
    papers_name = graph_params.get("papers_csv", "papers.csv")

    refined_outputs = prev_dir / "outputs"
    extract_dir = usecase_dir / "01_extract" / "outputs"
    if reuse_graph:
        data_base_dir = refined_outputs
        tensors_ok = (
            (data_base_dir / "3_corrected_data" / "stacked_features_active_corrected.npy").exists()
            and (data_base_dir / "3_corrected_data" / "stacked_matrices_corrected.npy").exists()
        )
        if not tensors_ok:
            raise FileNotFoundError(
                "reuse_existing_graph is true but corrected tensors were not found in "
                f"{data_base_dir / '3_corrected_data'}. Run step_04_build_graph first."
            )
        keywords_path = data_base_dir / "1_raw_data" / "top_keywords.csv"
        papers_path = extract_dir / papers_name
    else:
        data_base_dir = None
        keywords_path = refined_outputs / keywords_name
        if not keywords_path.stem.endswith("_refined"):
            refined_candidate = keywords_path.with_name(f"{keywords_path.stem}_refined{keywords_path.suffix}")
            if refined_candidate.exists():
                print(f"[graph] Using refined keyword CSV: {refined_candidate}")
                keywords_path = refined_candidate
        if not keywords_path.exists():
            raise FileNotFoundError(f"Missing keywords file: {keywords_path}")
        _apply_refined_removals_to_counts(keywords_path, refined_outputs)

        if refined_enabled:
            refined_papers_name = refined_collection_cfg.get("output_papers_csv") or "refined_papers.csv"
            refined_candidate = Path(refined_papers_name)
            if not refined_candidate.is_absolute():
                refined_candidate = refined_outputs / refined_papers_name
            if not refined_candidate.exists():
                raise FileNotFoundError(
                    f"Refined collection enabled but papers file not found at {refined_candidate}. "
                    "Rerun step_03_refined or update refined_collection.papers_csv."
                )
            papers_path = refined_candidate
            print(f"[graph] Using refined papers file: {papers_path}")
        else:
            papers_path = extract_dir / papers_name
        if not papers_path.exists():
            raise FileNotFoundError(f"Missing papers file: {papers_path}")

    default_overrides = {"SHOW_FIGS": False}
    if isinstance(cfg_overrides, dict):
        merged_overrides = {**default_overrides, **cfg_overrides}
    else:
        merged_overrides = default_overrides

    scan_filter_cfg = preprocess_cfg.get("scan_filter") if isinstance(preprocess_cfg, dict) else None
    if not isinstance(scan_filter_cfg, dict):
        scan_filter_cfg = {}

    params_refined_topics = params.get("refined_topics_csv") if isinstance(params, dict) else None
    scan_topics_csv = scan_filter_cfg.get("topics_csv") or params_refined_topics
    resolved_topics_path = None
    if isinstance(scan_topics_csv, str) and scan_topics_csv.strip():
        topics_path = Path(scan_topics_csv).expanduser()
        if not topics_path.is_absolute():
            candidate = (extract_dir / topics_path).resolve()
            if candidate.exists():
                resolved_topics_path = candidate
            elif topics_path.exists():
                resolved_topics_path = topics_path
        else:
            resolved_topics_path = topics_path
    scan_filter_cfg = {
        "enabled": bool(scan_filter_cfg.get("enabled")),
        "column": scan_filter_cfg.get("column") or "scan",
        "accepted_values": scan_filter_cfg.get("accepted_values") or ["x"],
        "fallback_to_all": bool(scan_filter_cfg.get("fallback_to_all", True)),
        "topics_path": resolved_topics_path,
    }

    train_time_filter = graph_params.get("train_time_filter") if isinstance(graph_params, dict) else None

    ctx = {
        "path_keyword_counts": keywords_path,
        "path_papers": papers_path,
        "base_dir": out_dir,
        "data_base_dir": data_base_dir,
        "cfg_overrides": merged_overrides,
        "cfg_defaults": cfg_defaults,
        "preprocess_cfg": preprocess_cfg,
        "tail_correction_cfg": tail_correction_cfg,
        "preview_cfg": preview_cfg,
        "plot_cfg": plot_cfg,
        "additional_keywords": additional_keywords,
        "remove_keywords": remove_keywords,
        "keyword_aliases": keyword_aliases_cfg,
        "drop_keywords": drop_keywords_cfg,
        "reuse_existing_graph": reuse_graph,
        "skip_training": skip_training,
        "train_time_filter": train_time_filter,
        "paper_volume_reweight": graph_params.get("paper_volume_reweight"),
        "counts_base_dir": refined_outputs,
        "scan_filter_cfg": scan_filter_cfg,
        "sweep_cfg": graph_params.get("sweep") if isinstance(graph_params, dict) else None,
        "diag_cfg": graph_params.get("diagnostics") if isinstance(graph_params, dict) else None,
    }

    log_path = out_dir / "cell_outputs.txt"
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    with log_path.open("w", encoding="utf-8") as log_file:
        console_stdout = _FilteredStream(orig_stdout, lambda line: not line.startswith("[graph]"))
        console_stderr = _FilteredStream(orig_stderr, lambda line: not line.startswith("[graph]"))
        tee_stdout = _TeeStream(console_stdout, log_file)
        tee_stderr = _TeeStream(console_stderr, log_file)
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            _run_graph_notebook(ctx)
        console_stdout.flush()
        console_stderr.flush()

    summary = {
        "keywords_csv": str(keywords_path),
        "papers_csv": str(papers_path),
        "log_path": str(log_path),
        "base_dir": str(out_dir),
        "outputs_root": str(outputs_root),
        "run_id": run_id,
        "graph_params": graph_params,
        "refined_collection_enabled": refined_enabled,
    }
    (out_dir / "used_config.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    if out_dir != outputs_root:
        try:
            rel_run_dir = str(out_dir.relative_to(outputs_root))
        except ValueError:
            rel_run_dir = str(out_dir)
        latest_payload = {
            "run_id": run_id,
            "run_dir": str(out_dir),
            "run_dir_relative": rel_run_dir,
            "log_path": str(log_path),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        (outputs_root / "latest_run.json").write_text(
            json.dumps(latest_payload, indent=2, default=str), encoding="utf-8"
        )
        root_summary = dict(summary)
        root_summary["latest_run"] = latest_payload
        (outputs_root / "used_config.json").write_text(
            json.dumps(root_summary, indent=2, default=str), encoding="utf-8"
        )

    print(f"[graph] Completed notebook replication. Outputs stored in {out_dir}")
    if run_id:
        print(f"[graph] Run ID: {run_id}")
    print(f"[graph] Log captured at {log_path}")

    return {
        "state": "DONE",
        "outputs": str(out_dir),
        "outputs_root": str(outputs_root),
        "run_id": run_id,
        "log": str(log_path),
    }
