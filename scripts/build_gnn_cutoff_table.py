#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a single GNN CSV with cutoff_date column for multi-cutoff conversion."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-csv", help="Single source CSV.")
    src.add_argument("--input-dir", help="Directory of source CSV files.")
    p.add_argument("--glob", default="*.csv", help="Glob pattern for --input-dir.")

    p.add_argument("--output-csv", required=True, help="Output consolidated CSV path.")
    p.add_argument("--keyword-col", default="name", help="Keyword column in source CSVs.")
    p.add_argument("--score-col", default="prediction", help="Score column in source CSVs.")
    p.add_argument(
        "--cutoff-date",
        default="",
        help="Forced cutoff date for --input-csv (YYYY-MM-DD). If empty, infer from filename.",
    )
    p.add_argument(
        "--cutoff-regex",
        default=r"(20\d{2})[-_](\d{2})",
        help="Regex used to infer YYYY-MM from filename when cutoff not provided.",
    )
    p.add_argument(
        "--frozen-cutoffs-csv",
        default="",
        help="Optional cutoff list CSV to filter rows (expects column cutoff_date).",
    )
    return p.parse_args()


def _infer_cutoff(path: Path, cutoff_regex: str) -> str:
    m = re.search(cutoff_regex, path.stem)
    if not m:
        raise ValueError(f"Could not infer cutoff from filename: {path.name}")
    return f"{m.group(1)}-{m.group(2)}-01"


def _normalize_date(s: str) -> str:
    m = re.match(r"^(20\d{2})-(\d{2})(?:-(\d{2}))?$", s.strip())
    if not m:
        raise ValueError(f"Invalid date format: {s!r}")
    return f"{m.group(1)}-{m.group(2)}-01"


def _load_paths(root: Path, args: argparse.Namespace) -> list[Path]:
    if args.input_csv:
        p = (root / args.input_csv).resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return [p]
    d = (root / args.input_dir).resolve()
    if not d.exists():
        raise FileNotFoundError(d)
    paths = sorted(d.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"No files with pattern {args.glob} in {d}")
    return [p.resolve() for p in paths]


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = _load_paths(root, args)

    fixed_cutoff = _normalize_date(args.cutoff_date) if args.cutoff_date.strip() else ""

    rows = []
    for p in paths:
        cutoff_date = fixed_cutoff if fixed_cutoff else _infer_cutoff(p, args.cutoff_regex)
        df = pd.read_csv(p)
        if args.keyword_col not in df.columns:
            raise ValueError(f"{p.name}: missing keyword col '{args.keyword_col}'")
        if args.score_col not in df.columns:
            raise ValueError(f"{p.name}: missing score col '{args.score_col}'")

        sub = df[[args.keyword_col, args.score_col]].copy()
        sub.columns = ["keyword", "prediction"]
        sub["cutoff_date"] = cutoff_date
        rows.append(sub)

    out = pd.concat(rows, ignore_index=True)
    out = out[["cutoff_date", "keyword", "prediction"]]

    if args.frozen_cutoffs_csv.strip():
        cpath = (root / args.frozen_cutoffs_csv).resolve()
        cdf = pd.read_csv(cpath)
        if "cutoff_date" not in cdf.columns:
            raise ValueError(f"{cpath}: missing column cutoff_date")
        keep = set(cdf["cutoff_date"].astype(str).tolist())
        out = out[out["cutoff_date"].isin(keep)].copy()

    out_path = (root / args.output_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[ok] wrote {out_path}")
    print(f"[info] rows={len(out)} cutoffs={out['cutoff_date'].nunique()} keywords={out['keyword'].nunique()}")


if __name__ == "__main__":
    main()

