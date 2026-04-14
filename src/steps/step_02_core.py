# src/steps/step_02_core.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Sequence, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import unicodedata

import pandas as pd
from tqdm import tqdm


# -------------------------- shared defaults --------------------------

DEFAULT_MODEL = "sentence-transformers/allenai-specter"
DEFAULT_TOP_N = 20
DEFAULT_RANGE = (2, 3)
DEFAULT_NR_CANDIDATES = 80
DEFAULT_DIVERSITY = 0.7
DEFAULT_STOP_WORDS = "english"
DEFAULT_STRICT_LITERAL = True
DEFAULT_JACCARD_THRESHOLD = 0.5
DEFAULT_REMOVE_KEYWORDS = [
    "abstract",
    "graphical abstract",
    "email share",
    "state",
    "deep",
    "content",
    "smart",
    "fauna",
    "forest",
]

IRRELEVANT_BASE = ["state", "email share", "email share share", "abstract"]


# -------------------------- helper utilities --------------------------

def load_abstracts(in_dir: Path, candidates: Sequence[str]) -> Tuple[pd.DataFrame, Path]:
    for name in candidates:
        p = in_dir / name
        if p.exists():
            if p.suffix == ".parquet":
                return pd.read_parquet(p), p
            if p.suffix == ".jsonl":
                return pd.read_json(p, lines=True), p
            if p.suffix == ".csv":
                return pd.read_csv(p), p
    raise FileNotFoundError(f"No abstracts file found in {in_dir}. Tried: {candidates}")


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s*\n+\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_span_check(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.replace("\r\n", "\n")
    t = re.sub(r"\s+", " ", t)
    return t.lower()


def has_only_whitespace_separators(original: str, tokens: Sequence[str]) -> bool:
    if len(tokens) <= 1:
        return True
    pattern = r"\b" + r"\s+".join(re.escape(tok) for tok in tokens) + r"\b"
    return re.search(pattern, original) is not None


def token_set(phrase: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]+", phrase.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a | b)
    return inter / union if union else 0.0


def ensure_cols(df: pd.DataFrame, text_col_pref: str, id_candidates: Sequence[str]) -> pd.DataFrame:
    if "abstract" not in df.columns:
        if text_col_pref in df.columns:
            df = df.rename(columns={text_col_pref: "abstract"})
        else:
            for c in ("abstract", "abstract_text", "abstract_en", "ABSTRACT", "Abstract"):
                if c in df.columns:
                    df = df.rename(columns={c: "abstract"})
                    break
    if "abstract" not in df.columns:
        raise RuntimeError("No abstract column found; set params.abstracts_text_col accordingly.")
    if "paper_id" not in df.columns:
        for c in id_candidates:
            if c in df.columns:
                df = df.rename(columns={c: "paper_id"})
                break
    if "paper_id" not in df.columns:
        df["paper_id"] = df.index
    return df


# -------------------------- extractor helpers --------------------------

def make_extractor(params: Dict[str, Any]) -> Tuple[str, Any]:
    import os
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer

    model_override = params.get("keybert_model_path")
    candidate = model_override or params.get("keybert_model") or DEFAULT_MODEL
    candidate_path = Path(candidate)
    if candidate_path.exists():
        load_target = str(candidate_path)
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    else:
        load_target = candidate

    device_pref = params.get("device", "auto")
    if device_pref == "cuda":
        device = "cuda"
    elif device_pref == "cpu":
        device = "cpu"
    else:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    st = SentenceTransformer(load_target, device=device)
    return ("keybert", KeyBERT(st))


@dataclass
class ExtractionOutput:
    keywords_per_doc: Optional[List[List[str]]]
    flat_keywords: List[str]


def extract_keywords_batch(
    texts: Sequence[str],
    params: Dict[str, Any],
    *,
    extractor: Any | None = None,
    kind: str | None = None,
    progress_desc: str = "[02_keywords] extracting",
    progress_enabled: bool = False,
    threads: int = 1,
    return_per_doc: bool = False,
) -> ExtractionOutput:
    if kind is None or extractor is None:
        kind, extractor = make_extractor(params)

    if isinstance(params.get("keyword_ngram"), (list, tuple)) and len(params["keyword_ngram"]) >= 2:
        ngram_vals = params["keyword_ngram"]
        ngram = (int(min(ngram_vals)), int(max(ngram_vals)))
    else:
        ngram = tuple(DEFAULT_RANGE)

    topn = int(params.get("keyword_topn", DEFAULT_TOP_N))
    nr_candidates = int(params.get("keyword_nr_candidates", DEFAULT_NR_CANDIDATES))
    diversity = float(params.get("keyword_diversity", DEFAULT_DIVERSITY))
    stop_words = params.get("keyword_stop_words", DEFAULT_STOP_WORDS)
    strict_literal = bool(params.get("keyword_strict_literal", DEFAULT_STRICT_LITERAL))
    jaccard_threshold = float(params.get("keyword_jaccard_threshold", DEFAULT_JACCARD_THRESHOLD))

    def _extract_single(text: str) -> List[str]:
        text = text if isinstance(text, str) else ""
        orig_norm = normalize_for_span_check(text)

        if kind == "keybert":
            cleaned = clean_text(text)
            if not cleaned:
                return []
            tuples = extractor.extract_keywords(
                cleaned,
                keyphrase_ngram_range=ngram,
                nr_candidates=nr_candidates,
                top_n=nr_candidates,
                stop_words=stop_words,
                use_maxsum=False,
                use_mmr=True,
                diversity=diversity,
            )
            if strict_literal:
                text_lc = cleaned.lower()
                tuples = [(kw, score) for kw, score in tuples if kw.lower() in text_lc]
            kept: List[str] = []
            seen_sets: List[set[str]] = []
            for phrase, _score in tuples:
                tokens = token_set(phrase)
                if not tokens:
                    continue
                words = phrase.lower().split()
                if len(words) > 1 and not has_only_whitespace_separators(orig_norm, words):
                    continue
                if any(jaccard(tokens, existing) > jaccard_threshold for existing in seen_sets):
                    continue
                kept.append(phrase)
                seen_sets.append(tokens)
                if len(kept) >= topn:
                    break
            return kept

        if hasattr(extractor, "extract_keywords"):
            tuples = extractor.extract_keywords(
                text,
                keyphrase_ngram_range=ngram,
                top_n=topn,
                use_maxsum=False,
                use_mmr=True,
                diversity=diversity,
                stop_words=stop_words,
            )
            phrases: List[str] = []
            if tuples:
                for k, _s in tuples:
                    words = k.lower().split()
                    if len(words) > 1 and not has_only_whitespace_separators(orig_norm, words):
                        continue
                    phrases.append(k)
            return phrases

        return extractor(text, ngram=ngram, top_n=topn)

    total = len(texts)
    per_doc: List[List[str]] = [[] for _ in range(total)]

    if total == 0:
        return ExtractionOutput(per_doc if return_per_doc else None, [])

    if threads and threads > 1:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            futures = {ex.submit(_extract_single, text): idx for idx, text in enumerate(texts)}
            with tqdm(total=total, desc=progress_desc, unit="doc", mininterval=0.3, disable=not progress_enabled) as pbar:
                for fut in as_completed(futures):
                    idx = futures[fut]
                    per_doc[idx] = fut.result()
                    pbar.update(1)
    else:
        iterator = enumerate(texts)
        bar = tqdm(total=total, desc=progress_desc, unit="doc", mininterval=0.3, disable=not progress_enabled)
        for idx, text in iterator:
            per_doc[idx] = _extract_single(text)
            bar.update(1)
        bar.close()

    flat_keywords = [kw for doc in per_doc for kw in doc]
    return ExtractionOutput(per_doc if return_per_doc else None, flat_keywords)


@dataclass
class KeywordTables:
    df_keys: pd.DataFrame
    df_hits: pd.DataFrame
    presence_summary: pd.DataFrame
    cleaned_list: List[str]


def build_keyword_tables(flat_keywords: Sequence[str], params: Dict[str, Any]) -> KeywordTables:
    df_kw = pd.DataFrame({"Keyword_raw": flat_keywords})
    df_kw["Keyword_norm"] = df_kw["Keyword_raw"].map(normalize_text)
    df_kw = df_kw[df_kw["Keyword_norm"].ne("")]

    irrelevant_norm = {normalize_text(x) for x in IRRELEVANT_BASE}
    df_kw = df_kw[~df_kw["Keyword_norm"].isin(irrelevant_norm)]

    counts = df_kw.groupby("Keyword_norm").size().reset_index(name="Count")
    rep = (
        df_kw.groupby("Keyword_norm")["Keyword_raw"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index(name="Keyword")
    )
    df_keys = (
        counts.merge(rep, on="Keyword_norm")
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )

    remove_norm = {normalize_text(x) for x in params.get("remove_keywords", DEFAULT_REMOVE_KEYWORDS)}
    df_keys = df_keys[~df_keys["Keyword_norm"].isin(remove_norm)].reset_index(drop=True)

    raw_patterns = params.get("remove_patterns", []) or []
    normalized_token_patterns: list[str] = []
    normalized_substring_patterns: list[str] = []
    if raw_patterns:
        if not isinstance(raw_patterns, (list, tuple, set)):
            raw_patterns = [raw_patterns]
        for pat in raw_patterns:
            if pat is None:
                continue
            pat_str = str(pat)
            if not pat_str:
                continue
            pat_norm = normalize_text(pat_str)
            if not pat_norm:
                continue
            if " " in pat_norm:
                normalized_substring_patterns.append(pat_norm)
            else:
                normalized_token_patterns.append(pat_norm)
        if normalized_substring_patterns or normalized_token_patterns:
            def _pattern_hit(name: str) -> bool:
                if normalized_substring_patterns:
                    for sub in normalized_substring_patterns:
                        if sub in name:
                            return True
                if normalized_token_patterns:
                    tokens = name.split()
                    for token in tokens:
                        if token in normalized_token_patterns:
                            return True
                return False
            mask = ~df_keys["Keyword_norm"].apply(_pattern_hit)
            df_keys = df_keys.loc[mask].reset_index(drop=True)

    additional_norm_map = {kw: normalize_text(kw) for kw in params.get("additional_keywords", [])}
    existing_norms = set(df_keys["Keyword_norm"])
    extras = [
        {"Keyword": kw, "Keyword_norm": norm_kw, "Count": 0}
        for kw, norm_kw in additional_norm_map.items()
        if norm_kw and norm_kw not in existing_norms
    ]
    if extras:
        df_keys = pd.concat([df_keys, pd.DataFrame(extras)], ignore_index=True)
        df_keys = df_keys.sort_values("Count", ascending=False).reset_index(drop=True)

    targets = params.get("ARE_THERE_KEYWORDS") or []
    targets_norm_map = {orig: normalize_text(orig) for orig in targets}

    def _matched_targets(norm_kw: str) -> List[str]:
        hits = [
            orig
            for orig, t in targets_norm_map.items()
            if t and (t == norm_kw or t in norm_kw or norm_kw in t)
        ]
        return sorted(set(hits))

    df_keys["Matched_Targets"] = df_keys["Keyword_norm"].map(_matched_targets)
    df_hits = df_keys[df_keys["Matched_Targets"].map(bool)].copy()
    if not df_hits.empty:
        df_hits["Matched_Targets"] = df_hits["Matched_Targets"].apply(lambda lst: ", ".join(lst))

    presence = {
        orig: any((tn == nk) or (tn in nk) or (nk in tn) for nk in df_keys["Keyword_norm"])
        for orig, tn in targets_norm_map.items()
        if tn
    }
    presence_summary = pd.DataFrame([{"Target": k, "Present": v} for k, v in presence.items()])

    df_keys = df_keys.drop(columns=["Matched_Targets"])
    cleaned_list = df_keys["Keyword"].tolist()

    return KeywordTables(df_keys=df_keys, df_hits=df_hits, presence_summary=presence_summary, cleaned_list=cleaned_list)


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_TOP_N",
    "DEFAULT_RANGE",
    "DEFAULT_NR_CANDIDATES",
    "DEFAULT_DIVERSITY",
    "DEFAULT_STOP_WORDS",
    "DEFAULT_STRICT_LITERAL",
    "DEFAULT_JACCARD_THRESHOLD",
    "DEFAULT_REMOVE_KEYWORDS",
    "load_abstracts",
    "normalize_text",
    "clean_text",
    "normalize_for_span_check",
    "has_only_whitespace_separators",
    "token_set",
    "jaccard",
    "ensure_cols",
    "make_extractor",
    "extract_keywords_batch",
    "build_keyword_tables",
    "ExtractionOutput",
    "KeywordTables",
]
