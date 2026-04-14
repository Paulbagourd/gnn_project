from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

STEP_NAME = "sample"
STEP_CODE_VERSION = "1"
inputs_from_prev = False

ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def external_inputs(cfg: Dict[str, Any], usecase_dir: Path, prev_dir: Path | None) -> List[Path]:
    params = (cfg or {}).get("params", {}).get("sample", {})
    input_csv = params.get("input_csv", "data/sample/abstracts.csv")
    return [_resolve_path(input_csv)]


def relevant_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    params = (cfg or {}).get("params", {}).get("sample", {})
    return {
        "input_csv": params.get("input_csv", "data/sample/abstracts.csv"),
        "date_column": params.get("date_column", "publication_date"),
        "text_column": params.get("text_column", "abstract"),
        "keywords": list(params.get("keywords", [])),
    }


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def _parse_month(value: str) -> str | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y/%m/%d", "%Y/%m"):
        try:
            dt = datetime.strptime(value, fmt)
            return f"{dt.year:04d}-{dt.month:02d}"
        except ValueError:
            continue
    return None


def run(cfg: Dict[str, Any], usecase_dir: Path, prev_dir: Path | None, step_dir: Path) -> Dict[str, Any]:
    params = relevant_params(cfg)
    input_path = _resolve_path(params["input_csv"])
    if not input_path.exists():
        raise FileNotFoundError(f"Sample input not found: {input_path}")

    keywords = [kw for kw in params["keywords"] if str(kw).strip()]
    if not keywords:
        raise ValueError("params.sample.keywords must contain at least one keyword.")

    patterns = {
        kw: re.compile(r"\\b" + re.escape(_normalize(kw)) + r"\\b")
        for kw in keywords
    }

    text_col = params["text_column"]
    date_col = params["date_column"]
    total_counts: Counter[str] = Counter()
    monthly_counts: dict[str, Counter[str]] = defaultdict(Counter)
    doc_count = 0

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            doc_count += 1
            text = _normalize(row.get(text_col, ""))
            month = _parse_month(row.get(date_col, ""))
            for kw, pattern in patterns.items():
                if text and pattern.search(text):
                    total_counts[kw] += 1
                    if month:
                        monthly_counts[month][kw] += 1

    outputs_dir = step_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    keyword_counts_path = outputs_dir / "sample_keyword_counts.csv"
    with keyword_counts_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["keyword", "doc_count"])
        for kw in keywords:
            writer.writerow([kw, total_counts.get(kw, 0)])

    monthly_counts_path = outputs_dir / "sample_monthly_counts.csv"
    with monthly_counts_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["month", "keyword", "doc_count"])
        for month in sorted(monthly_counts.keys()):
            for kw in keywords:
                writer.writerow([month, kw, monthly_counts[month].get(kw, 0)])

    summary_path = outputs_dir / "sample_summary.json"
    summary = {
        "documents": doc_count,
        "keywords": keywords,
        "months": sorted(monthly_counts.keys()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "state": "DONE",
        "outputs": str(outputs_dir),
        "artifacts": {
            "keyword_counts": str(keyword_counts_path),
            "monthly_counts": str(monthly_counts_path),
            "summary": str(summary_path),
        },
    }
