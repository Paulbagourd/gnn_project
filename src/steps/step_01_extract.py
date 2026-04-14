# src/steps/step_01_extract.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Sequence
import os, re, glob, json, time, math, hashlib, threading, logging
import requests, pandas as pd, numpy as np
from difflib import SequenceMatcher
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

STEP_NAME = "extract"
STEP_CODE_VERSION = "13"   # bump when logic/outputs change
inputs_from_prev = False
ROOT = Path(__file__).resolve().parents[2]

# ------------------- tiny utils (same spirit as notebook) -------------------

def _json_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

def _session_with_retries():
    s = requests.Session()
    try:
        from urllib3.util.retry import Retry
        from requests.adapters import HTTPAdapter
        retry = Retry(
            total=8, backoff_factor=1.2,
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True,
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
    except Exception:
        pass
    s.headers.update({"Accept-Encoding": "gzip, deflate"})
    return s

def _normalize_topic_id(x: str) -> str:
    s = str(x)
    if s.startswith("https://openalex.org/"):
        s = s.rsplit("/", 1)[-1]
    if not s.startswith("T"):
        try: s = f"T{int(s)}"
        except Exception: pass
    return s

# ------------------- SELECT builder (identical semantics) -------------------

REQUIRED_FOR_PIPELINE = ["id", "language"]
NEEDED_FOR_ABSTRACT   = ["abstract_inverted_index"]

def _root_fields(cols: List[str]) -> List[str]:
    roots = []
    for c in cols:
        if c == "Abstract":
            continue
        roots.append(c.split(".", 1)[0])
    return roots

def build_select_from_relevant(cols: List[str]) -> str:
    base_roots    = _root_fields(cols)
    require_roots = _root_fields(REQUIRED_FOR_PIPELINE)
    needed = set(base_roots) | set(require_roots) | set(NEEDED_FOR_ABSTRACT)
    return ",".join(sorted(needed))

# ---------------- runner integration (pipeline contract) --------------------

def external_inputs(cfg: Dict[str, Any], usecase_dir: Path, prev_dir: Path | None) -> List[Path]:
    # network-only step
    return []

def relevant_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    p = (cfg or {}).get("params", {})
    topic_seeds = p.get("topics_search_seed", p.get("topic_names", []))
    # expose the knobs you had in your notebook (defaults match the notebook)
    return {
        "topic_names_hash": _json_hash(topic_seeds),
        "refined_topics_csv": p.get("refined_topics_csv", None),
        "per_page": int(p.get("per_page", 200)),
        "max_workers": int(p.get("max_workers", 3)),
        "request_sleep": float(p.get("request_sleep", 0.25)),
        "max_pages_per_topic": int(p.get("max_pages_per_topic", 0)),
        "max_total_pages": int(p.get("max_total_pages", 0)),
        "stop_after_seconds": int(p.get("stop_after_seconds", 0)),
        "type_filter": p.get("type_filter", "journal-article|proceedings-article|preprint|book-chapter"),
        "relevant_columns": p.get("relevant_columns", [
            "id","title","publication_date","language","doi","cited_by_count",
            "referenced_works",
            "primary_topic.id","primary_topic.display_name","Abstract","type"
        ]),
        "openalex_mailto": p.get("openalex_mailto", None),
        "fetch_pages": bool(p.get("fetch_pages", True)),
        "filter_types": bool(p.get("filter_types", True)),
        "filter_gbif": bool(p.get("filter_gbif", True)),
        "allowed_types": list(p.get("allowed_types", [
            "journal-article","proceedings-article","preprint","book-chapter"
        ])),
        "gbif_doi_prefixes": list(p.get("gbif_doi_prefixes", ["10.15468"])),
        "gbif_phrases": list(p.get("gbif_phrases", [
            "species occurrences","taxonkey","hasgeospatialissue","gbif.org",
            "naturalis biodiversity center","biodiversity center nl",
            "dataset includes records","constituent datasets","datasets records",
            "dataset containing species","containing species occurrences"
        ])),
        # balancing / normalization constants (defaults = notebook)
        "start_year": int(p.get("start_year", 1987)),
        "chunk_size": int(p.get("chunk_size", 200_000)),
        "random_state": int(p.get("random_state", 42)),
        "spike_win": int(p.get("spike_win", 6)),
        "spike_cap": float(p.get("spike_cap", 0.80)),
        "sigma_mult": float(p.get("sigma_mult", 3.0)),
        "min_abs_bump": int(p.get("min_abs_bump", 200)),
        "weight_alpha": float(p.get("weight_alpha", 0.5)),
        "weight_tau": float(p.get("weight_tau", 0.85)),
    }

# --------------------- topic resolution (same logic) ------------------------

def _best_topic_match(session: requests.Session, query_name: str, mailto: str | None):
    BASE = "https://api.openalex.org/topics"
    params = {"search": query_name, "per-page": 5, "select": "id,display_name,description,keywords"}
    if mailto: params["mailto"] = mailto
    r = session.get(BASE, params=params, timeout=30)
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results: return None
    exact = [t for t in results if t.get("display_name","").strip().lower() == query_name.strip().lower()]
    if exact:
        return exact[0]
    # fuzzy fallback
    return max(results, key=lambda t: SequenceMatcher(None, query_name.lower(),
                                                     t.get("display_name","").lower()).ratio())

def _suggest_topics_from_names(p: Dict[str, Any], cache_dir: Path) -> pd.DataFrame:
    names = list(dict.fromkeys(p.get("topics_search_seed", p.get("topic_names", []))))
    cache = cache_dir / "topics_by_name.json"
    if cache.exists():
        try:
            cached = pd.DataFrame(json.loads(cache.read_text(encoding="utf-8")))
            if not cached.empty:
                needed = ["display_name", "approval", "scan", "description", "keywords", "id", "Tid"]
                for col in needed:
                    if col not in cached.columns:
                        cached[col] = ""
                return cached[needed]
        except Exception:
            pass
    mail = p.get("openalex_mailto", None)
    sess = _session_with_retries()
    rows = []
    for name in names:
        t = _best_topic_match(sess, name, mail)
        if not t:
            rows.append({
                "display_name": name,
                "approval": "",
                "scan": "",
                "description": "",
                "keywords": "",
                "id": "",
                "Tid": None,
            })
        else:
            tid = (t.get("id") or "").rsplit("/", 1)[-1] or None
            desc = t.get("description") or ""
            kws = t.get("keywords")
            if isinstance(kws, list):
                kws = json.dumps(kws, ensure_ascii=False)
            elif kws is None:
                kws = ""
            else:
                kws = json.dumps([kws], ensure_ascii=False)
            rows.append({
                "display_name": t.get("display_name"),
                "approval": "",
                "scan": "",
                "description": desc,
                "keywords": kws,
                "id": t.get("id") or "",
                "Tid": tid,
            })
        time.sleep(1.0)  # ≥ 1 req/s
    cache.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = ["display_name", "approval", "scan", "description", "keywords", "id", "Tid"]
    df_rows = pd.DataFrame(rows)
    for col in columns:
        if col not in df_rows.columns:
            df_rows[col] = ""
    return df_rows[columns]

def _resolve_topics_table(p: Dict[str, Any], cache_dir: Path, out_dir: Path) -> pd.DataFrame:
    refined_cfg = p.get("refined_topics_csv")
    if not refined_cfg:
        raise RuntimeError(
            "params.refined_topics_csv must be set. Provide the CSV path where approved topics will be stored." )
    refined_path = Path(refined_cfg).expanduser()
    if refined_path.is_absolute():
        target_path = refined_path
    else:
        target_path = (out_dir / refined_path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    suggestions = _suggest_topics_from_names(p, cache_dir)
    if suggestions.empty:
        raise RuntimeError("Unable to resolve any topics from params.topics_search_seed (or legacy params.topic_names).")
    openalex_csv = out_dir / "openalex_selected_topics.csv"
    suggestions.to_csv(openalex_csv, index=False, encoding="utf-8")
    template_cols = ["display_name", "approval", "scan", "description", "keywords", "id"]
    full_cols = template_cols + ["Tid"]
    if target_path.exists():
        df_ref = pd.read_csv(target_path)
    else:
        df_ref = suggestions[template_cols].copy()
    for col in template_cols:
        if col not in df_ref.columns:
            df_ref[col] = ""
    if "Tid" not in df_ref.columns:
        df_ref["Tid"] = ""
    df_ref = df_ref.reindex(columns=full_cols, fill_value="")
    df_ref = df_ref.drop_duplicates(subset=["display_name"], keep="last")
    df_ref["Tid"] = df_ref["Tid"].where(
        df_ref["Tid"].astype(str).str.strip().astype(bool),
        df_ref["id"].astype(str).str.split("/").str[-1]
    )
    df_ref["approval_flag"] = df_ref["approval"].astype(str).str.strip().str.lower().eq("x")
    df_ref["scan_flag"] = df_ref["scan"].astype(str).str.strip().str.lower().eq("x")
    mask_fetch = df_ref["approval_flag"] | df_ref["scan_flag"]
    flagged = df_ref.loc[mask_fetch].dropna(subset=["Tid"]).drop_duplicates(subset=["display_name"])
    if not flagged.empty:
        keep_cols = ["display_name", "id", "Tid", "approval", "scan"]
        return flagged[keep_cols]
    template = suggestions.merge(
        df_ref[template_cols + ["Tid"]],
        on="display_name",
        how="left",
        suffixes=("", "_ref")
    )
    for col in template_cols:
        ref_col = f"{col}_ref"
        if ref_col in template.columns:
            template[col] = template[col].where(
                template[col].astype(str).str.strip().astype(bool),
                template[ref_col]
            )
            template.drop(columns=[ref_col], inplace=True)
        template[col] = template[col].fillna("")
    if "Tid_ref" in template.columns:
        template["Tid"] = template["Tid"].where(
            template["Tid"].astype(str).str.strip().astype(bool),
            template["Tid_ref"]
        )
        template.drop(columns=["Tid_ref"], inplace=True)
    template.to_csv(target_path, index=False, columns=template_cols, encoding="utf-8")
    instructions = (
        "Need topic refinement before proceeding. "
        f"Please open {target_path} and mark desired rows with 'x' in the 'approval' and/or 'scan' columns. "
        "Saved template columns: display_name, approval, scan, description, keywords, id."
    )
    raise RuntimeError(instructions)


def _annotate_topic_flags(papers_path: Path, df_topics: pd.DataFrame) -> None:
    if papers_path is None or not papers_path.exists():
        return
    if df_topics.empty or "primary_topic.id" not in pd.read_csv(papers_path, nrows=0).columns:
        return
    flags = df_topics.copy()
    for col in ("approval", "scan"):
        if col not in flags.columns:
            flags[col] = ""
    flags["approved_flag"] = flags["approval"].astype(str).str.strip().str.lower().eq("x")
    flags["scan_flag"] = flags["scan"].astype(str).str.strip().str.lower().eq("x")
    approved_map = flags.set_index("Tid")["approved_flag"].to_dict()
    scan_map = flags.set_index("Tid")["scan_flag"].to_dict()
    tmp_path = papers_path.with_suffix(".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    header = True
    for chunk in pd.read_csv(papers_path, chunksize=200000):
        chunk["topic_is_approved"] = chunk["primary_topic.id"].map(approved_map).fillna(False).astype(bool)
        chunk["topic_is_scan"] = chunk["primary_topic.id"].map(scan_map).fillna(False).astype(bool)
        chunk.to_csv(tmp_path, mode="a", index=False, header=header)
        header = False
    tmp_path.replace(papers_path)

# ---------------------- fetch pages (same semantics) ------------------------

GLOBAL_SEEN_IDS: set[str] = set()
GLOBAL_SEEN_LOCK = threading.Lock()
PAGES_FETCHED = 0
PAGES_LOCK = threading.Lock()
STOP_EVENT = threading.Event()
START_TIME = [time.time()]

def _should_stop(stop_after: int) -> bool:
    if STOP_EVENT.is_set():
        return True
    if stop_after and (time.time() - START_TIME[0]) >= stop_after:
        STOP_EVENT.set()
        return True
    return False

def _bump_global_pages_and_maybe_stop(max_total_pages: int):
    global PAGES_FETCHED
    with PAGES_LOCK:
        PAGES_FETCHED += 1
        if max_total_pages and PAGES_FETCHED >= max_total_pages:
            STOP_EVENT.set()

def _fetch_page(session: requests.Session, url: str, sleep_s: float, logger=None) -> dict | None:
    for attempt in range(3):
        try:
            time.sleep(sleep_s)
            r = session.get(url, timeout=60)
            if r.status_code == 200:
                return r.json()
            if logger:
                body = (r.text or "")[:500].replace("\n"," ")
                logger.warning(f"HTTP {r.status_code} — attempt {attempt+1} — {url[:200]} ... body: {body}")
            if r.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            return None
        except requests.exceptions.RequestException as e:
            if logger: logger.exception(f"Request exception: {e}")
            time.sleep(5)
    return None

def _rebuild_abstracts(df: pd.DataFrame) -> pd.DataFrame:
    word_cols = [c for c in df.columns if c.startswith("abstract_inverted_index.")]
    if not word_cols:
        df["Abstract"] = ""
        return df
    words = [c.split(".", 1)[1] for c in word_cols]
    vals = df[word_cols].to_numpy()

    def row_to_abstract(row_vals):
        pos2w = {}
        for word, positions in zip(words, row_vals):
            if isinstance(positions, list):
                for p in positions:
                    pos2w[p] = word
        if not pos2w:
            return ""
        return " ".join(pos2w[k] for k in sorted(pos2w))
    df["Abstract"] = [row_to_abstract(row) for row in vals]
    df.drop(columns=word_cols, inplace=True, errors="ignore")
    return df

def _count_pages_for_topic(session, tid: str, type_filter: str | None, per_page: int, mail: str | None) -> int:
    filters = f"topics.id:{_normalize_topic_id(tid)}"
    if type_filter:
        filters += f",type:{type_filter}"
    url = ("https://api.openalex.org/works"
           f"?filter={filters}"
           f"&per-page=1&select=id"
           f"{'&mailto=' + mail if mail else ''}")
    r = session.get(url, timeout=30)
    if r.status_code == 429:
        time.sleep(int(r.headers.get("Retry-After", "2")))
        r = session.get(url, timeout=30)
    r.raise_for_status()
    total = r.json().get("meta", {}).get("count", 0)
    return int(math.ceil(total / max(1, per_page)))

def _setup_logger(log_file: Path):
    log = logging.getLogger(str(log_file))
    log.setLevel(logging.INFO)
    for h in list(log.handlers):  # avoid dup handlers if re-run
        log.removeHandler(h)
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
    log.addHandler(fh)
    return log

def _store_topic(topic: str, index_topic: int, p: Dict[str, Any], store_dir: Path, logs_dir: Path, pbar: tqdm):
    if _should_stop(int(p.get("stop_after_seconds", 0))):
        return
    log_path = logs_dir / f"topic_{index_topic}_{_normalize_topic_id(topic)}.log"
    logger = _setup_logger(log_path)
    per_page      = int(p["per_page"])
    request_sleep = float(p["request_sleep"])
    max_pages_pt  = int(p["max_pages_per_topic"])
    max_total     = int(p["max_total_pages"])
    type_filter   = p["type_filter"]
    mail          = p.get("openalex_mailto")
    select_fields = build_select_from_relevant(p["relevant_columns"])
    session = requests.Session()
    session.headers.update({"Accept-Encoding": "gzip, deflate"})
    base_url = "https://api.openalex.org/works"
    next_cursor = "*"
    page_count = 0
    seen_cursors = set()
    seen_ids_local = set()
    buffer_frames: List[pd.DataFrame] = []
    buffer_ids: set[str] = set()
    temp_dir = store_dir / f"temp_{_normalize_topic_id(topic)}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    def flush_buffer(final=False):
        nonlocal buffer_frames, buffer_ids
        if not buffer_frames:
            return None
        batch = pd.concat(buffer_frames, ignore_index=True) if len(buffer_frames) > 1 else buffer_frames[0]
        if "id" in batch.columns:
            batch.drop_duplicates(subset=["id"], inplace=True)
        # global dedup
        with GLOBAL_SEEN_LOCK:
            dup_ids = buffer_ids & GLOBAL_SEEN_IDS
            if dup_ids:
                batch = batch[~batch["id"].isin(dup_ids)]
                buffer_ids.difference_update(dup_ids)
            GLOBAL_SEEN_IDS.update(buffer_ids)
        if not final:
            fn = temp_dir / f"batch_{int(time.time()*1000)}.csv"
            batch.to_csv(fn, index=False)
            buffer_frames = []; buffer_ids.clear()
            return fn
        else:
            buffer_frames = []; buffer_ids.clear()
            return batch
    try:
        while next_cursor and not _should_stop(int(p.get("stop_after_seconds", 0))):
            if max_pages_pt and page_count >= max_pages_pt:
                logger.info(f"Reached MAX_PAGES_PER_TOPIC={max_pages_pt}")
                break
            if next_cursor in seen_cursors:
                logger.warning(f"Repeat cursor detected, breaking: {next_cursor}")
                break
            seen_cursors.add(next_cursor)
            filters = f"topics.id:{_normalize_topic_id(topic)}"
            if type_filter:
                filters += f",type:{type_filter}"
            url = (f"{base_url}?filter={filters}"
                   f"&per-page={per_page}"
                   f"&cursor={next_cursor}"
                   f"&select={select_fields}"
                   f"{'&mailto=' + mail if mail else ''}")
            data = _fetch_page(session, url, logger=logger, sleep_s=request_sleep)
            if not data:
                logger.warning(f"No data for url (None/HTTP error): {url}")
                break
            results = data.get("results", [])
            if not results:
                logger.info("Empty results page; stopping.")
                break
            df = pd.json_normalize(results)
            # rebuild abstract (identical)
            df = _rebuild_abstracts(df)
            # content filtering
            if "Abstract" in df.columns:
                bad1  = df["Abstract"].astype("string").str.contains('ADVERTISEMENT RETURN TO ISSUEPREV', na=False)
                empty = df["Abstract"].astype("string").fillna("").str.strip().eq("")
                recv  = df["Abstract"].astype("string").str.startswith("Received ", na=False)
                df = df[~(bad1 | empty | recv)]
            # language
            if "language" in df.columns:
                df = df[df["language"].astype("string").eq("en")]
            # drop per-page + local dups
            if "id" in df.columns:
                df.drop_duplicates(subset=["id"], inplace=True)
                if seen_ids_local:
                    df = df[~df["id"].isin(seen_ids_local)]
                seen_ids_local.update(df["id"].astype(str).tolist())
            # keep only desired columns
            keep = [c for c in p["relevant_columns"] if c in df.columns]
            df = df[keep] if keep else df
            # buffer
            if not df.empty:
                buffer_frames.append(df)
                if "id" in df.columns:
                    buffer_ids.update(df["id"].astype(str).tolist())
            # periodic flush
            if page_count > 0 and page_count % 20 == 0:
                flush_buffer(final=False)
            pbar.update(1); page_count += 1
            _bump_global_pages_and_maybe_stop(max_total)
            next_cursor = data.get("meta", {}).get("next_cursor", None)
        final_df_or_path = flush_buffer(final=True)
        # consolidate
        batch_files = [temp_dir / f for f in os.listdir(temp_dir) if f.endswith(".csv")]
        if batch_files:
            frames = [pd.read_csv(f, low_memory=False) for f in batch_files]
            if isinstance(final_df_or_path, pd.DataFrame) and not final_df_or_path.empty:
                frames.append(final_df_or_path)
            all_pages_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if "id" in all_pages_df.columns:
                all_pages_df.drop_duplicates(subset=["id"], inplace=True)
        else:
            all_pages_df = final_df_or_path if isinstance(final_df_or_path, pd.DataFrame) else pd.DataFrame()
        out = store_dir / f"df_{_normalize_topic_id(topic)}_n{index_topic}.csv"
        all_pages_df.to_csv(out, index=False)
        logger.info(f"Saved consolidated file: {out}")
    except Exception as e:
        logger.exception(f"Aborted topic {topic}: {e}")
    finally:
        # cleanup temp
        try:
            for f in list(os.listdir(temp_dir)):
                os.remove(temp_dir / f)
            os.rmdir(temp_dir)
        except Exception:
            pass
        logger.info(f"Completed topic {topic} (pages kept: {page_count})")

# ----------------------- duckdb merge (same queries) ------------------------

def _duckdb_merge_to_combined(store_dir: Path, out_dir: Path) -> Path:
    import duckdb as d
    pattern = os.path.join(str(store_dir), "df_*_n*.csv")
    out_path = out_dir / "all_papers_combined.csv"
    db_path  = out_dir / "merge.duckdb"
    tmp_dir  = out_dir / "duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files match: {pattern}")

    def to_duck(p: str) -> str: return os.fspath(p).replace("\\", "/")

    def esc(p: str) -> str:     return p.replace("'", "''")
    pattern_duck = esc(to_duck(pattern))
    out_duck     = esc(to_duck(str(out_path)))
    db_duck      = to_duck(str(db_path))
    tmp_duck     = esc(to_duck(str(tmp_dir)))
    con = d.connect(database=db_duck)
    con.execute(f"SET temp_directory='{tmp_duck}';")
    con.execute("PRAGMA threads=2;")
    con.execute("DROP TABLE IF EXISTS raw;")
    con.execute("""
    CREATE TABLE raw (
      id VARCHAR,
      publication_date VARCHAR,
      abstract VARCHAR,
      title VARCHAR,
      language VARCHAR,
      doi VARCHAR,
      cited_by_count VARCHAR,
      referenced_works VARCHAR,
      type VARCHAR,
      is_paratext VARCHAR,
      "primary_topic.id" VARCHAR,
      "primary_topic.display_name" VARCHAR
    );
    """)
    for i, f in enumerate(matches, 1):
        f_esc = esc(to_duck(f))
        con.execute(f"""
        CREATE OR REPLACE TEMP VIEW vf AS
        SELECT * FROM read_csv_auto('{f_esc}',
          header=true, all_varchar=true, union_by_name=true,
          delim=',', quote='"', escape='"',
          ignore_errors=true, null_padding=true,
          maximum_line_size=100000000
        );
        """)
        cols = set(con.execute("PRAGMA table_info('vf');").df()["name"].tolist())

        def pick(opts):
            for o in opts:
                if o in cols: return f'"{o}"'
            return "NULL"
        id_expr   = pick(["id","ID","work_id"])
        if id_expr == "NULL":
            con.execute("DROP VIEW vf;")
            continue
        date_expr = pick(["publication_date","published_date","date"])
        abs_expr  = pick(["Abstract","abstract","paper_abstract"])
        ttl_expr  = pick(["title","Title"])
        lang_expr = pick(["language","Language"])
        doi_expr  = pick(["doi","DOI"])
        cited_expr = pick(["cited_by_count","cited_by","citedbycount","cited-by-count",
                           "cited_by_count__int","cited_by_count_x","cited_by_count_y"])
        refs_expr = pick(["referenced_works","referenced_works_x","referenced_works_y"])
        type_expr  = pick(["type","document_type"])
        ispt_expr  = pick(["is_paratext","isParatext","is-paratext","paratext"])
        ptid_expr  = pick(["primary_topic.id","primary_topic_id","topic.id"])
        ptnm_expr  = pick(["primary_topic.display_name","primary_topic_display_name","topic.name","topic"])
        con.execute(f"""
        INSERT INTO raw
        SELECT
          {id_expr}   AS id,
          {date_expr} AS publication_date,
          {abs_expr}  AS abstract,
          {ttl_expr}  AS title,
          {lang_expr} AS language,
          {doi_expr}  AS doi,
          {cited_expr} AS cited_by_count,
          {refs_expr}  AS referenced_works,
          {type_expr}  AS type,
          {ispt_expr}  AS is_paratext,
          {ptid_expr}  AS "primary_topic.id",
          {ptnm_expr}  AS "primary_topic.display_name"
        FROM vf
        WHERE COALESCE({id_expr}, '') <> '';
        """)
        con.execute("DROP VIEW vf;")
    con.execute("CREATE INDEX IF NOT EXISTS raw_idx ON raw(id);")
    con.execute(f"""
    COPY (
      WITH typed AS (
        SELECT
          id,
          publication_date                      AS publication_date_raw,
          try_strptime(publication_date, '%Y-%m-%d') AS publication_date_dt,
          abstract,
          title,
          language,
          doi,
          try_cast(casted.cited_by_count AS BIGINT) AS cited_by_count,
          referenced_works,
          type,
          try_cast(is_paratext AS BOOLEAN) AS is_paratext,
          "primary_topic.id",
          "primary_topic.display_name"
        FROM (
          SELECT
            id, publication_date, abstract, title, language, doi,
            NULLIF(TRIM(cited_by_count), '') AS cited_by_count,
            referenced_works,
            type, is_paratext,
            "primary_topic.id", "primary_topic.display_name"
          FROM raw
        ) AS casted
      ),
      ranked AS (
        SELECT
          *,
          ROW_NUMBER() OVER (
            PARTITION BY id
            ORDER BY
              publication_date_dt NULLS LAST,
              (NULLIF(abstract, '') IS NOT NULL) DESC,
              length(abstract) DESC,
              id
          ) AS rn
        FROM typed
        WHERE (language IS NULL OR language='en')
      )
      SELECT
        id,
        publication_date_dt           AS publication_date,
        publication_date_raw,
        abstract AS "Abstract",
        title,
        language,
        doi,
        cited_by_count,
        referenced_works,
        "primary_topic.id",
        "primary_topic.display_name",
        type,
        is_paratext
      FROM ranked
      WHERE rn=1
    ) TO '{out_duck}' (HEADER, DELIMITER ',');
    """)
    return out_path

# ---------------- final parquet + quick plots (same ideas) ------------------

def _write_parquet_from_csv(csv_path: Path, out_parquet: Path) -> Tuple[int,int]:
    df = pd.read_csv(csv_path, low_memory=False)
    if "Abstract" not in df.columns: df["Abstract"] = ""
    s = df["Abstract"].astype("string")
    n_with_abs = int((s.notna() & s.str.strip().ne("")).sum())
    if "id" not in df.columns:
        df["id"] = pd.RangeIndex(len(df)).astype(str)
    out = df.rename(columns={"id":"paper_id","Abstract":"abstract"})[["paper_id","abstract"]]
    out.to_parquet(out_parquet, index=False)
    return len(df), n_with_abs

def _plot_monthly_from_final(csv_path: Path, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty or "publication_date" not in df.columns: return
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    s = df["Abstract"].astype("string")
    df["has_abs"] = s.notna() & s.str.strip().ne("")
    monthly = (df.dropna(subset=["publication_date"])
                 .groupby(pd.Grouper(key="publication_date", freq="MS"))
                 .agg(abstracts=("has_abs","sum"), total=("id","size"))
                 .reset_index())
    monthly["abstracts_ma6"] = monthly["abstracts"].rolling(6, min_periods=1).mean()
    # figure 1
    fig = plt.figure(figsize=(9,4))
    plt.plot(monthly["publication_date"], monthly["abstracts"], label="Abstracts / month")
    plt.plot(monthly["publication_date"], monthly["abstracts_ma6"], label="6-month MA")
    plt.title("Extracted abstracts over time (since 1987)")
    plt.xlabel("Month"); plt.ylabel("# abstracts"); plt.legend(); plt.tight_layout()
    fig.savefig(out_dir / "abstracts_monthly.pdf", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------------- balancing pipeline (verbatim logic) -----------------------

def _balancing_pipeline(out_dir: Path, params: Dict[str, Any]) -> Tuple[Path, Dict[str, int]]:
    """Replicate the notebook balancing logic (Jan relocation, spike smoothing, downsampling)."""
    import heapq
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from collections import Counter
    from pandas.api.types import is_datetime64_ns_dtype
    INPUT_CSV = str(out_dir / "all_papers_combined.csv")
    FINAL_CSV = str(out_dir / "papers.csv")
    START_YEAR = int(params["start_year"])
    CHUNK_SIZE = int(params["chunk_size"])
    RANDOM_STATE = int(params["random_state"])
    SPIKE_WIN = int(params["spike_win"])
    SPIKE_CAP = float(params["spike_cap"])
    SIGMA_MULT = float(params["sigma_mult"])
    MIN_ABS_BUMP = int(params["min_abs_bump"])
    WEIGHT_ALPHA = float(params["weight_alpha"])
    WEIGHT_TAU = float(params["weight_tau"])
    DESIRED_COLS = [
        "id", "title", "publication_date", "publication_date_raw", "language", "doi", "cited_by_count",
        "referenced_works",
        "primary_topic.id", "primary_topic.display_name", "Abstract", "type", "is_paratext",
    ]
    filter_types_flag = bool(params.get("filter_types", True))
    allowed_types_param = params.get("allowed_types") or [
        "journal-article", "proceedings-article", "preprint", "book-chapter"
    ]
    allowed_types_set = {str(x).strip().lower() for x in allowed_types_param if str(x).strip()}

    filter_gbif_flag = bool(params.get("filter_gbif", True))
    gbif_doi_prefixes_param = params.get("gbif_doi_prefixes") or ["10.15468"]
    GBIF_DOI_PREFIXES = tuple(str(x).strip().lower() for x in gbif_doi_prefixes_param if str(x).strip())
    gbif_phrases_param = params.get("gbif_phrases") or [
        "species occurrences", "taxonkey", "hasgeospatialissue", "gbif.org",
        "naturalis biodiversity center", "biodiversity center nl",
        "dataset includes records", "constituent datasets", "datasets records",
        "dataset containing species", "containing species occurrences",
    ]
    GBIF_RE = (
        re.compile("|".join(map(re.escape, gbif_phrases_param)), flags=re.I)
        if filter_gbif_flag and gbif_phrases_param
        else None
    )

    def _series_from_counter(counter: Counter):
        if not counter:
            return None
        pairs = sorted(counter.keys())
        periods = pd.PeriodIndex([f"{y:04d}-{m:02d}" for (y, m) in pairs], freq="M")
        vals = [counter[(y, m)] for (y, m) in pairs]
        s = pd.Series(vals, index=periods)
        s = s.groupby(level=0).sum()
        s = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"), fill_value=0)
        return s.to_timestamp()

    def _plot_counts(counter: Counter, title: str, outfile: Path | Sequence[Path] | None = None):
        s = _series_from_counter(counter)
        if s is None:
            return
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(s.index, s.values)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Count")
        ax.margins(x=0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if outfile is not None:
            paths = outfile if isinstance(outfile, Sequence) and not isinstance(outfile, (str, Path)) else [outfile]
            for path in paths:
                fig.savefig(str(path), dpi=200)
        plt.close(fig)

    def _purge_non_papers_and_gbif(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        if "is_paratext" in out.columns:
            is_pt = (
                out["is_paratext"].astype("string").str.strip().str.lower()
                .isin(("true", "t", "1", "yes")).astype("boolean")
            )
            out = out.loc[~is_pt]
        if "language" in out.columns:
            lang = out["language"].astype("string").str.lower()
            out = out.loc[lang.eq("en") | lang.isna()]
        if filter_types_flag and allowed_types_set and "type" in out.columns:
            t = out["type"].astype("string").str.lower()
            out = out.loc[t.isna() | t.isin(allowed_types_set)]
        if not filter_gbif_flag:
            return out.copy()
        gbif = pd.Series(False, index=out.index)
        if GBIF_RE is not None and "Abstract" in out.columns:
            gbif |= out["Abstract"].astype("string").str.contains(GBIF_RE, na=False)
        if GBIF_RE is not None and "title" in out.columns:
            gbif |= out["title"].astype("string").str.contains(r"\bGBIF\b", case=False, na=False)
        if GBIF_DOI_PREFIXES and "doi" in out.columns:
            sdoi = out["doi"].astype("string").str.lower()
            gbif |= sdoi.str.startswith(GBIF_DOI_PREFIXES)
        return out.loc[~gbif].copy()

    def _usecols(path):
        cols = pd.read_csv(path, nrows=0).columns.tolist()
        use = [c for c in DESIRED_COLS if c in cols]
        if "publication_date" not in use:
            use.append("publication_date")
        return use

    def _ensure_dt(df: pd.DataFrame, col: str = "publication_date") -> pd.DataFrame:
        if col not in df.columns or df.empty:
            return df
        s = pd.to_datetime(df[col].astype("string"), errors="coerce", utc=True)
        mask = s.notna()
        if not mask.all():
            df = df.loc[mask].copy()
            s = s.loc[mask]
        if not is_datetime64_ns_dtype(df[col].dtype):
            df[col] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        df.loc[:, col] = s.dt.tz_localize(None).astype("datetime64[ns]")
        return df

    def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        raw_src = "publication_date_raw" if "publication_date_raw" in df.columns else "publication_date"
        raw = (
            df[raw_src].astype("string")
            if raw_src in df.columns
            else pd.Series("", index=df.index, dtype="string")
        )
        df = df.copy()
        df["_date_str"] = raw
        raw0 = df["_date_str"].astype("string").str.strip()
        raw_clean = raw0.str.replace(
            r"[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$",
            "",
            regex=True,
        )
        is_year_only = raw_clean.str.fullmatch(r"\d{4}", na=False)
        is_year_month = raw_clean.str.fullmatch(r"\d{4}-\d{2}", na=False)
        is_jan_month = is_year_month & raw_clean.str.endswith("-01")
        is_exact_jan01 = raw_clean.str.fullmatch(r"\d{4}-01-01", na=False)
        df["is_placeholder_jan"] = is_exact_jan01 | is_jan_month | is_year_only
        s = pd.to_datetime(df.get("publication_date", pd.NaT), errors="coerce", utc=True)
        mask_na = s.isna()
        if mask_na.any():
            filler = raw_clean.copy()
            filler = filler.where(~is_year_only, raw_clean.str.cat(pd.Series(["-01-01"] * len(df), index=df.index)))
            filler = filler.where(~is_year_month, raw_clean.str.cat(pd.Series(["-01"] * len(df), index=df.index)))
            s2 = pd.to_datetime(filler, errors="coerce", utc=True)
            s = s.where(~mask_na, s2)
        mask = s.notna()
        if not mask.all():
            df = df.loc[mask].copy()
            s = s.loc[mask]
            df["is_placeholder_jan"] = df["is_placeholder_jan"]
        df["publication_date"] = s.dt.tz_localize(None).astype("datetime64[ns]")
        dt = df["publication_date"]
        df["year"] = dt.dt.year.astype("Int16")
        df["month"] = dt.dt.month.astype("Int8")
        df["year_month"] = dt.dt.strftime("%Y-%m")
        return df

    def _to_wide(counter: Counter):
        if not counter:
            return None
        years = sorted({yr for (yr, _) in counter})
        data = {yr: [counter.get((yr, m), 0) for m in range(1, 13)] for yr in years}
        df = pd.DataFrame(data, index=range(1, 13))
        df.index.name = "month"
        return df

    def _hash64_series(df: pd.DataFrame):
        key = (
            df["id"].astype("string")
            if "id" in df.columns
            else df["publication_date"].astype("string")
        ).fillna("")
        return pd.util.hash_pandas_object(key, index=False).astype("uint64")

    def _u01_from_hash(h):
        return (h / np.float64(2 ** 64)).astype(np.float64)

    def _dest_month_day_from_hash(h):
        dest_m = (h % 11) + 2
        dest_d = ((h // 11) % 28) + 1
        return dest_m.astype(np.int16), dest_d.astype(np.int16)

    def _offset_from_hash(h, W=SPIKE_WIN):
        r = (h % (2 * W)) - W
        off = r + (r >= 0)
        return off.astype(np.int16)

    def _ym_add_months(y, m, off):
        base = (y * 12 + (m - 1)).astype(np.int64)
        new = base + off.astype(np.int64)
        ny = (new // 12).astype(int)
        nm = (new % 12 + 1).astype(int)
        return ny, nm
    TODAY = pd.Timestamp("today").normalize()
    usecols = _usecols(INPUT_CSV)

    # -------- PASS 1: counts, placeholder tallies, spike candidates --------

    raw_counts = Counter()
    orig_counts = Counter()
    per_year_counts = Counter()
    placeholder_jan_counts = Counter()
    ym_counts = Counter()
    for chunk in pd.read_csv(
        INPUT_CSV,
        usecols=usecols,
        dtype={"publication_date": "string"},
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        if chunk.empty:
            continue
        parsed = _parse_dates(chunk)
        parsed = parsed[(parsed["year"] >= START_YEAR) & (parsed["publication_date"] <= TODAY)]
        if parsed.empty:
            continue
        vc_raw = parsed.groupby(["year", "month"], sort=False).size()
        for (yr, mo), cnt in vc_raw.items():
            raw_counts[(int(yr), int(mo))] += int(cnt)
        a = _purge_non_papers_and_gbif(parsed)
        if a.empty:
            continue
        vc = a.groupby(["year", "month"], sort=False).size()
        for (yr, mo), cnt in vc.items():
            c = int(cnt)
            orig_counts[(int(yr), int(mo))] += c
            per_year_counts[(int(yr), int(mo))] += c
        for ym, cnt in a["year_month"].value_counts().items():
            ym_counts[str(ym)] += int(cnt)
        pj = a[(a["month"] == 1) & (a["is_placeholder_jan"])]
        if not pj.empty:
            for yr, cnt in pj["year"].value_counts().items():
                placeholder_jan_counts[int(yr)] += int(cnt)
    yms_sorted = sorted(ym_counts.keys())
    counts = np.array([ym_counts[ym] for ym in yms_sorted], dtype=np.int64)

    def _running_median_excl_center(arr, win):
        out = np.full(arr.shape, np.nan, dtype=float)
        n = len(arr)
        for i in range(n):
            lo, hi = max(0, i - win), min(n, i + win + 1)
            if hi - lo <= 1:
                continue
            neigh = np.concatenate([arr[lo:i], arr[i + 1 : hi]])
            if neigh.size:
                out[i] = np.median(neigh)
        return out
    baseline = _running_median_excl_center(counts, SPIKE_WIN)
    cushion = SIGMA_MULT * np.sqrt(np.maximum(baseline, 1.0))
    target = np.ceil(baseline + np.maximum(cushion, MIN_ABS_BUMP))
    excess = np.maximum(0, counts - target.astype(int))
    cap = (SPIKE_CAP * counts).astype(int)
    move_k = np.minimum(excess, cap)
    spike_months = {ym: int(k) for ym, k in zip(yms_sorted, move_k) if int(k) > 0}
    years = sorted({yr for (yr, _) in per_year_counts})
    target_jan = {
        yr: int(round(sum(per_year_counts.get((yr, m), 0) for m in range(1, 13)) / 12.0))
        for yr in years
    }
    jan_totals = {yr: per_year_counts.get((yr, 1), 0) for yr in years}
    move_needed = {yr: max(0, jan_totals.get(yr, 0) - target_jan.get(yr, 0)) for yr in years}
    move_from_placeholder = {
        yr: min(move_needed.get(yr, 0), placeholder_jan_counts.get(yr, 0))
        for yr in years
    }

    # -------- PASS 2A: select placeholder-Jan moves --------

    heaps_jan = {yr: [] for yr in years}
    for chunk in pd.read_csv(
        INPUT_CSV,
        usecols=usecols,
        dtype={"publication_date": "string"},
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        if chunk.empty:
            continue
        a = _parse_dates(chunk)
        a = a[(a["year"] >= START_YEAR) & (a["publication_date"] <= TODAY)]
        a = _purge_non_papers_and_gbif(a)
        if a.empty:
            continue
        cand = a[(a["month"] == 1) & (a["is_placeholder_jan"])]
        if cand.empty:
            continue
        h = _hash64_series(cand)
        u = _u01_from_hash(h).to_numpy()
        yrs = cand["year"].astype(int).to_numpy()
        for ui, yy in zip(u, yrs):
            K = move_from_placeholder.get(int(yy), 0)
            if K <= 0:
                continue
            heap = heaps_jan[int(yy)]
            if len(heap) < K:
                heapq.heappush(heap, -float(ui))
            else:
                top = -heap[0]
                if ui < top:
                    heapq.heapreplace(heap, -float(ui))
    move_threshold_jan = {}
    for yr in years:
        K = move_from_placeholder.get(yr, 0)
        if K <= 0:
            move_threshold_jan[yr] = -1.0
        elif len(heaps_jan[yr]) < K:
            move_threshold_jan[yr] = 1.0
        else:
            move_threshold_jan[yr] = -heaps_jan[yr][0]

    # -------- PASS 2B: select spike-month moves --------

    heaps_spike = {ym: [] for ym in spike_months}
    for chunk in pd.read_csv(
        INPUT_CSV,
        usecols=usecols,
        dtype={"publication_date": "string"},
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        if chunk.empty or not spike_months:
            continue
        a = _parse_dates(chunk)
        a = a[(a["year"] >= START_YEAR) & (a["publication_date"] <= TODAY)]
        a = _purge_non_papers_and_gbif(a)
        if a.empty:
            continue
        sub = a[a["year_month"].isin(spike_months.keys())]
        if sub.empty:
            continue
        h = _hash64_series(sub)
        u = _u01_from_hash(h)
        ym_arr = sub["year_month"].astype(str).to_numpy()
        for ui, ym in zip(u, ym_arr):
            K = spike_months.get(ym, 0)
            if K <= 0:
                continue
            heap = heaps_spike[ym]
            if len(heap) < K:
                heapq.heappush(heap, -float(ui))
            else:
                top = -heap[0]
                if ui < top:
                    heapq.heapreplace(heap, -float(ui))
    move_threshold_spike = {}
    for ym, heap in heaps_spike.items():
        K = spike_months.get(ym, 0)
        if K <= 0:
            move_threshold_spike[ym] = -1.0
        elif len(heap) < K:
            move_threshold_spike[ym] = 1.0
        else:
            move_threshold_spike[ym] = -heap[0]

    # -------- PASS 2.5: diagnostic redistribution counts --------

    redist_counts = Counter()
    for chunk in pd.read_csv(
        INPUT_CSV,
        usecols=usecols,
        dtype={"publication_date": "string"},
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        if chunk.empty:
            continue
        a = _parse_dates(chunk)
        a = a[(a["year"] >= START_YEAR) & (a["publication_date"] <= TODAY)]
        a = _purge_non_papers_and_gbif(a)
        if a.empty:
            continue
        b = a.copy()
        is_jan_pl = (b["month"] == 1) & (b["is_placeholder_jan"])
        if is_jan_pl.any():
            h = _hash64_series(b.loc[is_jan_pl])
            u = _u01_from_hash(h)
            dm, _ = _dest_month_day_from_hash(h)
            yrs = b.loc[is_jan_pl, "year"].astype(int)
            move = u <= yrs.map(move_threshold_jan).astype(float)
            idx = b.loc[is_jan_pl].index[move.values]
            if len(idx) > 0:
                b.loc[idx, "month"] = pd.Series(dm[move.values], index=idx, dtype="Int8")
        vc2 = b.groupby(["year", "month"], sort=False).size()
        for (yr, mo), cnt in vc2.items():
            redist_counts[(int(yr), int(mo))] += int(cnt)

    # -------- PASS 2.6: build per-year weights after applying both moves --------

    def _apply_moves_no_sample(df: pd.DataFrame) -> pd.DataFrame:
        df = _parse_dates(df)
        df = df[(df["year"] >= START_YEAR) & (df["publication_date"] <= TODAY)]
        df = _purge_non_papers_and_gbif(df)
        if df.empty:
            return df
        is_jan_pl = (df["month"] == 1) & (df["is_placeholder_jan"])
        if is_jan_pl.any():
            cand = df.loc[is_jan_pl]
            h = _hash64_series(cand)
            u = _u01_from_hash(h)
            dm, dd = _dest_month_day_from_hash(h)
            yrs = cand["year"].astype(int)
            move = u <= yrs.map(move_threshold_jan).astype(float)
            idx = cand.index[move.values]
            if len(idx) > 0:
                y = df.loc[idx, "year"].astype("Int16").astype(str)
                m = pd.Series(dm[move.values], index=idx, dtype="Int16").astype(str).str.zfill(2)
                d = pd.Series(dd[move.values], index=idx, dtype="Int16").astype(str).str.zfill(2)
                df.loc[idx, "publication_date"] = pd.to_datetime(y + "-" + m + "-" + d, utc=True).dt.tz_localize(None).astype("datetime64[ns]")
                df = _ensure_dt(df, "publication_date")
                dt = df["publication_date"]
                df.loc[idx, "year"] = dt.loc[idx].dt.year.astype("Int16")
                df.loc[idx, "month"] = dt.loc[idx].dt.month.astype("Int8")
        if move_threshold_spike:
            df = _ensure_dt(df, "publication_date")
            dt = df["publication_date"]
            df["_ym"] = dt.dt.to_period("M").astype(str)
            cand = df[df["_ym"].isin(move_threshold_spike.keys())]
            if not cand.empty:
                h = _hash64_series(cand)
                u = _u01_from_hash(h)
                ym_arr = cand["_ym"].astype(str).to_numpy()
                thr = pd.Series(move_threshold_spike)
                move = u <= thr.loc[ym_arr].astype(float).to_numpy()
                idx = cand.index[move]
                if len(idx) > 0:
                    off = _offset_from_hash(h[move], W=SPIKE_WIN)
                    y = dt.loc[idx].dt.year.to_numpy().astype(int)
                    m = dt.loc[idx].dt.month.to_numpy().astype(int)
                    ny, nm = _ym_add_months(y, m, off)
                    dd = ((h[move] // 13) % 28 + 1).astype(int)
                    y = pd.Series(ny, index=idx, dtype="Int32").astype(str)
                    m = pd.Series(nm, index=idx, dtype="Int16").astype(str).str.zfill(2)
                    d = pd.Series(dd, index=idx, dtype="Int16").astype(str).str.zfill(2)
                    df.loc[idx, "publication_date"] = pd.to_datetime(y + "-" + m + "-" + d, utc=True).dt.tz_localize(None).astype("datetime64[ns]")
                    df = _ensure_dt(df, "publication_date")
                    dt2 = df["publication_date"]
                    df.loc[idx, "year"] = dt2.loc[idx].dt.year.astype("Int16")
                    df.loc[idx, "month"] = dt2.loc[idx].dt.month.astype("Int8")
            df.drop(columns=["_ym"], inplace=True, errors="ignore")
        return df
    counts_for_w = Counter()
    for chunk in pd.read_csv(
        INPUT_CSV,
        usecols=usecols,
        dtype={"publication_date": "string"},
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        if chunk.empty:
            continue
        b = _apply_moves_no_sample(chunk)
        if b.empty:
            continue
        vc = b.groupby(["year", "month"], sort=False).size()
        for (yr, mo), cnt in vc.items():
            counts_for_w[(int(yr), int(mo))] += int(cnt)
    weights_by_ym: Dict[Tuple[int, int], float] = {}
    eps = 1e-12
    for y in sorted({yr for (yr, _) in counts_for_w}):
        vec = np.array([counts_for_w.get((y, m), 0) for m in range(1, 13)], dtype=float)
        tot = vec.sum()
        if tot <= 0:
            for m in range(1, 13):
                weights_by_ym[(y, m)] = 1.0
            continue
        share = vec / max(tot, eps)
        u = 1.0 / 12.0
        target = (1.0 - WEIGHT_ALPHA) * share + WEIGHT_ALPHA * u
        w = np.minimum(target / np.maximum(share, eps), 1.0)
        scale = min(WEIGHT_TAU / max(w.mean(), eps), 1.0)
        w = np.minimum(w * scale, 1.0)
        for m in range(1, 13):
            weights_by_ym[(y, m)] = float(w[m - 1])

    # -------- PASS 3: apply moves and downsample --------

    post_counts = Counter()
    header_final = True
    open(FINAL_CSV, "w", encoding="utf-8").close()
    rng = np.random.default_rng(RANDOM_STATE)
    for chunk in pd.read_csv(
        INPUT_CSV,
        usecols=usecols,
        dtype={"publication_date": "string"},
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        if chunk.empty:
            continue
        df = _parse_dates(chunk)
        df = df[(df["year"] >= START_YEAR) & (df["publication_date"] <= TODAY)].copy()
        df = _purge_non_papers_and_gbif(df)
        if df.empty:
            continue
        is_jan_pl = (df["month"] == 1) & (df["is_placeholder_jan"])
        if is_jan_pl.any():
            cand = df.loc[is_jan_pl]
            h_all = _hash64_series(cand)
            u_all = _u01_from_hash(h_all)
            dm, dd = _dest_month_day_from_hash(h_all)
            yrs = cand["year"].astype(int)
            move = u_all <= yrs.map(move_threshold_jan).astype(float)
            idx_move = cand.index[move.values]
            if len(idx_move) > 0:
                y_str = df.loc[idx_move, "year"].astype("Int16").astype(str)
                m_str = pd.Series(dm[move.values], index=idx_move, dtype="Int16").astype(str).str.zfill(2)
                d_str = pd.Series(dd[move.values], index=idx_move, dtype="Int16").astype(str).str.zfill(2)
                new_dt = pd.to_datetime(y_str + "-" + m_str + "-" + d_str, errors="coerce", utc=True).dt.tz_localize(None)
                df.loc[idx_move, "publication_date"] = new_dt.astype("datetime64[ns]")
                df = _ensure_dt(df, "publication_date")
                dt = df["publication_date"]
                df.loc[idx_move, "year"] = dt.loc[idx_move].dt.year.astype("Int16")
                df.loc[idx_move, "month"] = dt.loc[idx_move].dt.month.astype("Int8")
                df.loc[idx_move, "year_month"] = dt.loc[idx_move].dt.strftime("%Y-%m")
        if move_threshold_spike:
            df = _ensure_dt(df, "publication_date")
            dt = df["publication_date"]
            df["_ym"] = dt.dt.to_period("M").astype(str)
            cand = df[df["_ym"].isin(move_threshold_spike.keys())]
            if not cand.empty:
                h = _hash64_series(cand)
                u = _u01_from_hash(h)
                ym_arr = cand["_ym"].astype(str).to_numpy()
                thr = pd.Series(move_threshold_spike)
                move = u <= thr.loc[ym_arr].astype(float).to_numpy()
                idx_move = cand.index[move]
                if len(idx_move) > 0:
                    off = _offset_from_hash(h[move], W=SPIKE_WIN)
                    y = dt.loc[idx_move].dt.year.to_numpy().astype(int)
                    m = dt.loc[idx_move].dt.month.to_numpy().astype(int)
                    ny, nm = _ym_add_months(y, m, off)
                    dd = ((h[move] // 13) % 28 + 1).astype(int)
                    y_str = pd.Series(ny, index=idx_move, dtype="Int32").astype(str)
                    m_str = pd.Series(nm, index=idx_move, dtype="Int16").astype(str).str.zfill(2)
                    d_str = pd.Series(dd, index=idx_move, dtype="Int16").astype(str).str.zfill(2)
                    new_dt = pd.to_datetime(y_str + "-" + m_str + "-" + d_str, errors="coerce", utc=True).dt.tz_localize(None)
                    df.loc[idx_move, "publication_date"] = new_dt.astype("datetime64[ns]")
                    df = _ensure_dt(df, "publication_date")
                    dt2 = df["publication_date"]
                    df.loc[idx_move, "year"] = dt2.loc[idx_move].dt.year.astype("Int16")
                    df.loc[idx_move, "month"] = dt2.loc[idx_move].dt.month.astype("Int8")
                    df.loc[idx_move, "year_month"] = dt2.loc[idx_move].dt.strftime("%Y-%m")
            df.drop(columns=["_ym"], inplace=True, errors="ignore")
        yrs = df["year"].astype(int).to_numpy()
        mos = df["month"].astype(int).to_numpy()
        ym_keys = list(zip(yrs, mos))
        wvals = np.fromiter((weights_by_ym.get(k, 1.0) for k in ym_keys), dtype=float, count=len(ym_keys))
        keep_mask = rng.random(len(df)) < wvals
        sampled = df.loc[keep_mask].copy()
        if sampled.empty:
            continue
        if "Abstract" in sampled.columns:
            sampled = sampled.dropna(subset=["Abstract"]).copy()
            if sampled.empty:
                continue
            sampled["Abstract"] = (
                sampled["Abstract"].astype("string")
                .str.lower().str.replace(r"[^a-z]+", " ", regex=True)
                .str.strip().str.replace(r"\s+", " ", regex=True)
            )
        vc3 = sampled.groupby(["year", "month"], sort=False).size()
        for (yr, mo), cnt in vc3.items():
            post_counts[(int(yr), int(mo))] += int(cnt)
        sampled.to_csv(FINAL_CSV, index=False, mode="a", header=header_final)
        header_final = False
    plot0_pdf = out_dir / "counts_stage0_raw.pdf"
    plot1_pdf = out_dir / "counts_stage1_purge.pdf"
    plot2_pdf = out_dir / "counts_stage2_redist.pdf"
    plot3_pdf = out_dir / "counts_stage3_final.pdf"
    _plot_counts(raw_counts, "Before purge (raw monthly counts)", [plot0_pdf])
    _plot_counts(orig_counts, "After purge (no redistribution yet)", [plot1_pdf])
    _plot_counts(redist_counts, "After moving placeholder-Jan (for weights)", [plot2_pdf])
    _plot_counts(post_counts, "After downsampling (final, written to papers.csv)", [plot3_pdf])
    return Path(FINAL_CSV), post_counts

# ------------------------------- RUN ---------------------------------------

def run(cfg: Dict[str, Any], usecase_dir: Path, prev_dir: Path | None, step_dir: Path):
    p = (cfg or {}).get("params", {})
    out_dir = step_dir / "outputs"; out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = step_dir / "cache"; cache_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"; logs_dir.mkdir(parents=True, exist_ok=True)
    store_dir = out_dir / "df_store_with_abstract"; store_dir.mkdir(parents=True, exist_ok=True)
    fetch_pages_flag = bool(p.get("fetch_pages", True))
    # Save the exact params used
    (out_dir / "used_config.json").write_text(
        json.dumps(p, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # ---- Topics (exact notebook semantics)

    df_topics = _resolve_topics_table(p, cache_dir, out_dir)
    if df_topics.empty or df_topics["Tid"].isna().all():
        raise RuntimeError("No topics resolved (check params.topics_search_seed/topic_names or refined_topics_csv).")
    df_topics.to_csv(out_dir / "topics_resolved.csv", index=False, encoding="utf-8")
    tids = [t for t in df_topics["Tid"].dropna().astype(str).tolist() if t]

    # ---- Pre-count for tqdm

    est_pages = 0
    if fetch_pages_flag:
        sess = _session_with_retries()
        per_page = int(p.get("per_page", 200))
        type_filter = p.get("type_filter", "journal-article|proceedings-article|preprint|book-chapter")
        mail = p.get("openalex_mailto")
        max_total = int(p.get("max_total_pages", 0))
        topic_page_counts: list[tuple[str, int, bool, bool]] = []
        topic_info: list[tuple[str, str, bool, bool]] = []
        for _, row in df_topics.iterrows():
            tid = str(row.get("Tid", "")).strip()
            if not tid:
                continue
            approved_flag = str(row.get("approval", "")).strip().lower() == "x"
            scan_flag = str(row.get("scan", "")).strip().lower() == "x"
            topic_info.append((tid, row.get("display_name") or row.get("id") or tid, approved_flag, scan_flag))
        for tid, label, approved_flag, scan_flag in topic_info:
            try:
                pages = _count_pages_for_topic(sess, tid, type_filter, per_page, mail)
            except Exception:
                pages = 1
            est_pages += pages
            topic_page_counts.append((label, pages, approved_flag, scan_flag))
            if max_total and est_pages >= max_total:
                est_pages = max_total
                break
        est_pages = max(est_pages, 1)
        if topic_page_counts:
            print("[01_extract] Estimated pages per selected topic (approval or scan):")
            for name, pages, approved_flag, scan_flag in topic_page_counts:
                flags = []
                if approved_flag:
                    flags.append("approved")
                if scan_flag:
                    flags.append("scan")
                suffix = f" [{' & '.join(flags)}]" if flags else ""
                print(f"  - {name}: {pages}{suffix}")

        # ---- Fetch (buffers, per-topic logs, global dedup) - identical logic

        with tqdm(total=est_pages, desc="All Pages", mininterval=0.5, smoothing=0.1) as pbar:
            with ThreadPoolExecutor(max_workers=int(p.get("max_workers", 3))) as ex:
                futures = [ex.submit(_store_topic, tid, idx, p, store_dir, logs_dir, pbar)
                           for idx, tid in enumerate(tids)]
                for fut in as_completed(futures):
                    fut.result()
    else:
        print("[01_extract] fetch_pages=false -> skipping OpenAlex fetch and using existing store")

    # ---- DuckDB merge + dedup with same window ordering

    print("[01_extract] Merging fetched batches via DuckDB ...")
    combined_csv = _duckdb_merge_to_combined(store_dir, out_dir)
    print(f"[01_extract] Merge done -> {combined_csv}")

    # ---- Balancing pipeline (January deflation + spikes + downsample) -> papers.csv

    print("[01_extract] Running balancing pipeline ...")
    final_csv, post_counts = _balancing_pipeline(out_dir, p)
    print(f"[01_extract] Balancing done -> {final_csv}")
    print("[01_extract] Annotating topic flags ...")
    _annotate_topic_flags(final_csv, df_topics)

    # ---- Parquet for step_02 (from final set, same columns expected)

    print("[01_extract] Building abstracts parquet ...")
    abstracts_parquet = out_dir / "abstracts.parquet"
    nrows_final, n_with_abs_final = _write_parquet_from_csv(final_csv, abstracts_parquet)

    # ---- Quick monthly plot from final

    print("[01_extract] Rendering monthly plot ...")
    _plot_monthly_from_final(final_csv, out_dir)

    # ---- Summary

    counts_stage0_pdf = out_dir / "counts_stage0_raw.pdf"
    counts_stage1_pdf = out_dir / "counts_stage1_purge.pdf"
    counts_stage2_pdf = out_dir / "counts_stage2_redist.pdf"
    counts_stage3_pdf = out_dir / "counts_stage3_final.pdf"

    summary = pd.DataFrame([{
        "topics_resolved": int(len(tids)),
        "estimated_pages": int(est_pages),
        "combined_csv": str(combined_csv),
        "final_csv": str(final_csv),
        "abstracts_parquet": str(abstracts_parquet),
        "rows_final": int(nrows_final),
        "rows_with_abstract_final": int(n_with_abs_final),
        "logs_dir": str(logs_dir),
    }])
    summary.to_csv(out_dir / "extract_summary.csv", index=False)
    print("[01_extract] summary")
    print(summary.to_string(index=False))
    return {
        "state": "DONE",
        "outputs": str(out_dir),
        "artifacts": {
        "topics_table": str(out_dir / "topics_resolved.csv"),
        "combined_csv": str(combined_csv),
        "final_csv": str(final_csv),
        "abstracts_parquet": str(abstracts_parquet),
        "counts_stage0": str(counts_stage0_pdf),
        "counts_stage1": str(counts_stage1_pdf),
        "counts_stage2": str(counts_stage2_pdf),
        "counts_stage3": str(counts_stage3_pdf),
        "abstracts_monthly_plot": str(out_dir / "abstracts_monthly.pdf"),
        "logs_dir": str(logs_dir),
    }
    }
