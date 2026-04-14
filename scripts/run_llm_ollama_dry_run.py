#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests


@dataclass
class RunConfig:
    model: str
    ollama_url: str
    temperature: float
    top_p: float
    timeout_sec: int
    max_retries: int
    retry_sleep_sec: float
    history_months: int
    max_cutoffs: int
    max_keywords: int
    batch_size: int
    run_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local Ollama forecasts for LLM dry-run inputs."
    )
    parser.add_argument(
        "--inputs-dir",
        default="data/usecase_cyberspace/gnn_llm_comparison/outputs/inputs_by_cutoff",
    )
    parser.add_argument(
        "--frozen-setup",
        default="data/usecase_cyberspace/gnn_llm_comparison/outputs/frozen_setup.json",
    )
    parser.add_argument(
        "--out-dir",
        default="data/usecase_cyberspace/gnn_llm_comparison/ollama_runs",
    )
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep-sec", type=float, default=2.0)
    parser.add_argument(
        "--history-months",
        type=int,
        default=120,
        help="Keep only last N history months per keyword. 0=full history.",
    )
    parser.add_argument(
        "--max-cutoffs",
        type=int,
        default=0,
        help="0=all cutoffs; otherwise first N cutoffs only.",
    )
    parser.add_argument(
        "--max-keywords",
        type=int,
        default=0,
        help="0=all keywords; otherwise first N keywords only.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run identifier (default: timestamp-based).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of keywords predicted per LLM call.",
    )
    return parser.parse_args()


def _now_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _collect_cutoff_files(inputs_dir: Path, max_cutoffs: int) -> list[Path]:
    files = sorted(inputs_dir.glob("llm_input_cutoff_*.json"))
    if max_cutoffs > 0:
        files = files[:max_cutoffs]
    return files


def _trim_history(values: list[list[float]], months: list[str], history_months: int) -> tuple[list[list[float]], list[str]]:
    if history_months <= 0 or history_months >= len(months):
        return values, months
    return values[-history_months:], months[-history_months:]


def _build_global_context(
    keywords: list[dict[str, Any]],
    feature_names: list[str],
) -> dict[str, Any]:
    if not keywords:
        return {"n_keywords": 0, "feature_names": feature_names}
    F = len(feature_names)
    last_rows = []
    deltas = []
    trends = []
    for rec in keywords:
        vals = np.asarray(rec["history"]["values"], dtype=np.float64)
        if vals.ndim != 2 or vals.shape[1] != F or vals.shape[0] == 0:
            continue
        last = vals[-1]
        first = vals[0]
        delta = last - first
        last_rows.append(last)
        deltas.append(delta)
        trends.append((str(rec["keyword"]), float(np.sum(delta))))
    if not last_rows:
        return {"n_keywords": len(keywords), "feature_names": feature_names}
    mean_last = np.mean(np.vstack(last_rows), axis=0).round(6).tolist()
    mean_delta = np.mean(np.vstack(deltas), axis=0).round(6).tolist()
    trends_sorted = sorted(trends, key=lambda x: x[1], reverse=True)
    top_up = [{"keyword": k, "delta_sum": round(v, 6)} for k, v in trends_sorted[:10]]
    top_down = [{"keyword": k, "delta_sum": round(v, 6)} for k, v in trends_sorted[-10:]]
    return {
        "n_keywords": len(keywords),
        "feature_names": feature_names,
        "mean_last": mean_last,
        "mean_delta": mean_delta,
        "top_trending": top_up,
        "top_declining": top_down,
    }


def _build_batch_prompt(
    batch_keywords: list[dict[str, Any]],
    cutoff_date: str,
    epsilon: float,
    horizons: list[int],
    feature_names: list[str],
    global_context: dict[str, Any],
) -> str:
    series_payload = []
    for rec in batch_keywords:
        series_payload.append(
            {
                "keyword": str(rec["keyword"]),
                "history": rec["history"],
            }
        )
    payload = {
        "cutoff_date": cutoff_date,
        "epsilon": epsilon,
        "horizons_months": horizons,
        "feature_names": feature_names,
        "global_context": global_context,
        "keywords": series_payload,
    }
    return (
        "You are a numeric forecaster.\n"
        "Use ONLY provided history and global context. No external knowledge.\n"
        "Return ONLY one valid JSON object with key 'predictions' (array).\n"
        "Each item must include: keyword, abs_12, abs_24, abs_36, abs_48, rel_12, rel_24, rel_36, rel_48.\n"
        "Predict exactly for all input keywords in this batch.\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=True)}"
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Fallback: find first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start : end + 1]
        obj = json.loads(snippet)
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"Could not parse JSON object from response: {text[:300]!r}")


def _validate_prediction(obj: dict[str, Any]) -> dict[str, float]:
    keys = ["abs_12", "abs_24", "abs_36", "abs_48", "rel_12", "rel_24", "rel_36", "rel_48"]
    out: dict[str, float] = {}
    for k in keys:
        if k not in obj:
            raise ValueError(f"Missing key {k} in model response.")
        out[k] = float(obj[k])
    return out


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    size = max(1, int(size))
    return [items[i : i + size] for i in range(0, len(items), size)]


def _validate_batch_predictions(
    parsed: dict[str, Any],
    expected_keywords: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    preds_raw = parsed.get("predictions")
    if not isinstance(preds_raw, list):
        raise ValueError("Missing 'predictions' array in model response.")
    by_kw: dict[str, dict[str, Any]] = {}
    for rec in preds_raw:
        if not isinstance(rec, dict):
            continue
        kw = str(rec.get("keyword", "")).strip()
        if kw:
            by_kw[kw] = rec

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for kw in expected_keywords:
        if kw not in by_kw:
            failures.append({"keyword": kw, "error": "missing_prediction_in_batch_response"})
            continue
        try:
            pred = _validate_prediction(by_kw[kw])
            rows.append({"keyword": kw, **pred})
        except Exception as exc:  # noqa: BLE001
            failures.append({"keyword": kw, "error": f"invalid_prediction: {exc}"})
    return rows, failures


def _ollama_generate(cfg: RunConfig, prompt: str) -> str:
    url = cfg.ollama_url.rstrip("/") + "/api/generate"
    payload = {
        "model": cfg.model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
        },
    }
    resp = requests.post(url, json=payload, timeout=cfg.timeout_sec)
    resp.raise_for_status()
    data = resp.json()
    return str(data.get("response", "")).strip()


def _check_ollama_alive(base_url: str, timeout_sec: int) -> None:
    tags_url = base_url.rstrip("/") + "/api/tags"
    resp = requests.get(tags_url, timeout=timeout_sec)
    resp.raise_for_status()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    inputs_dir = (project_root / args.inputs_dir).resolve()
    out_root = (project_root / args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id.strip() or _now_run_id()
    run_dir = out_root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    per_cutoff_dir = run_dir / "predictions_by_cutoff"
    raw_dir = run_dir / "raw_responses"
    per_cutoff_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    cfg = RunConfig(
        model=args.model,
        ollama_url=args.ollama_url,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout_sec=args.timeout_sec,
        max_retries=args.max_retries,
        retry_sleep_sec=args.retry_sleep_sec,
        history_months=args.history_months,
        max_cutoffs=args.max_cutoffs,
        max_keywords=args.max_keywords,
        batch_size=max(1, int(args.batch_size)),
        run_id=run_id,
    )

    frozen_setup_path = (project_root / args.frozen_setup).resolve()
    frozen_setup = json.loads(frozen_setup_path.read_text(encoding="utf-8"))
    cutoff_files = _collect_cutoff_files(inputs_dir, cfg.max_cutoffs)
    if not cutoff_files:
        raise FileNotFoundError(f"No llm_input_cutoff_*.json found in {inputs_dir}")

    _check_ollama_alive(cfg.ollama_url, cfg.timeout_sec)

    meta = {
        "run_id": cfg.run_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": cfg.model,
        "ollama_url": cfg.ollama_url,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "history_months": cfg.history_months,
        "max_cutoffs": cfg.max_cutoffs,
        "max_keywords": cfg.max_keywords,
        "batch_size": cfg.batch_size,
        "frozen_setup": frozen_setup,
        "inputs_dir": str(inputs_dir),
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    total_calls = 0
    total_failures = 0

    for cutoff_file in cutoff_files:
        inp = json.loads(cutoff_file.read_text(encoding="utf-8"))
        cutoff_date = str(inp["cutoff_date"])
        epsilon = float(inp["epsilon"])
        horizons = [int(x) for x in inp["horizons_months"]]
        feature_names = [str(x) for x in inp["feature_names"]]
        keywords = list(inp["keywords"])
        if cfg.max_keywords > 0:
            keywords = keywords[: cfg.max_keywords]

        trimmed_keywords: list[dict[str, Any]] = []
        for rec in keywords:
            months = list(rec["history"]["months"])
            values = list(rec["history"]["values"])
            values, months = _trim_history(values, months, cfg.history_months)
            trimmed_keywords.append(
                {
                    "keyword": str(rec["keyword"]),
                    "history": {"months": months, "values": values},
                }
            )

        global_context = _build_global_context(trimmed_keywords, feature_names)
        rows: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []

        batches = _chunks(trimmed_keywords, cfg.batch_size)
        for bidx, batch in enumerate(batches):
            expected_keywords = [str(x["keyword"]) for x in batch]
            prompt = _build_batch_prompt(
                batch_keywords=batch,
                cutoff_date=cutoff_date,
                epsilon=epsilon,
                horizons=horizons,
                feature_names=feature_names,
                global_context=global_context,
            )

            last_error = None
            raw_text = ""
            parsed: dict[str, Any] | None = None
            for _attempt in range(cfg.max_retries + 1):
                try:
                    raw_text = _ollama_generate(cfg, prompt)
                    parsed = _extract_json_object(raw_text)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    time.sleep(cfg.retry_sleep_sec)

            if parsed is None:
                for kw in expected_keywords:
                    failures.append({"keyword": kw, "error": last_error or "unknown_error"})
                    total_calls += 1
                    total_failures += 1
                raw_path = raw_dir / f"{cutoff_date}_batch_{bidx:04d}.txt"
                raw_path.write_text(raw_text, encoding="utf-8")
                continue

            try:
                good_rows, bad_rows = _validate_batch_predictions(parsed, expected_keywords)
                rows.extend(good_rows)
                failures.extend(bad_rows)
                total_calls += len(expected_keywords)
                total_failures += len(bad_rows)
            except Exception as exc:  # noqa: BLE001
                err = str(exc)
                for kw in expected_keywords:
                    failures.append({"keyword": kw, "error": err})
                    total_calls += 1
                    total_failures += 1

            raw_path = raw_dir / f"{cutoff_date}_batch_{bidx:04d}.txt"
            raw_path.write_text(raw_text, encoding="utf-8")

        cutoff_key = cutoff_date[:7]
        out_pred = per_cutoff_dir / f"predictions_{cutoff_key}.json"
        out_fail = per_cutoff_dir / f"failures_{cutoff_key}.json"
        out_pred.write_text(
            json.dumps(
                {
                    "run_id": cfg.run_id,
                    "model_id": cfg.model,
                    "cutoff_date": cutoff_date,
                    "predictions": rows,
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        out_fail.write_text(json.dumps(failures, ensure_ascii=True, indent=2), encoding="utf-8")
        print(
            f"[cutoff {cutoff_key}] predictions={len(rows)} failures={len(failures)}"
        )

    summary = {
        "run_id": cfg.run_id,
        "n_cutoffs": len(cutoff_files),
        "total_calls": total_calls,
        "total_failures": total_failures,
        "success_rate": 0.0 if total_calls == 0 else (1.0 - total_failures / total_calls),
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(f"[done] run_dir={run_dir}")
    print(f"[done] success_rate={summary['success_rate']:.4f}")


if __name__ == "__main__":
    main()

