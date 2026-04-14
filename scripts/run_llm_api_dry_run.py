#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class RunConfig:
    model: str
    api_base: str
    api_key_env: str
    temperature: float
    timeout_sec: int
    max_retries: int
    retry_sleep_sec: float
    history_months: int
    max_cutoffs: int
    max_keywords: int
    run_id: str


PRED_KEYS = [
    "abs_12",
    "abs_24",
    "abs_36",
    "abs_48",
    "rel_12",
    "rel_24",
    "rel_36",
    "rel_48",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cloud/API forecasts for LLM dry-run inputs."
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
        default="data/usecase_cyberspace/gnn_llm_comparison/api_runs",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep-sec", type=float, default=2.0)
    parser.add_argument("--history-months", type=int, default=120)
    parser.add_argument("--max-cutoffs", type=int, default=0)
    parser.add_argument("--max-keywords", type=int, default=0)
    parser.add_argument("--run-id", default="")
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


def _build_prompt(
    keyword: str,
    cutoff_date: str,
    epsilon: float,
    horizons: list[int],
    feature_names: list[str],
    months: list[str],
    values: list[list[float]],
) -> str:
    payload = {
        "keyword": keyword,
        "cutoff_date": cutoff_date,
        "epsilon": epsilon,
        "horizons_months": horizons,
        "feature_names": feature_names,
        "history": {
            "months": months,
            "values": values,
        },
    }
    return (
        "Forecast keyword emergence from provided history only. "
        "No external knowledge, no tools, no web.\n"
        "Return ONLY a JSON object with numeric keys: "
        "abs_12,abs_24,abs_36,abs_48,rel_12,rel_24,rel_36,rel_48.\n"
        f"Input={json.dumps(payload, ensure_ascii=True)}"
    )


def _prediction_schema() -> dict[str, Any]:
    props = {k: {"type": "number"} for k in PRED_KEYS}
    return {
        "name": "keyword_prediction",
        "schema": {
            "type": "object",
            "properties": props,
            "required": PRED_KEYS,
            "additionalProperties": False,
        },
        "strict": True,
    }


def _extract_json_from_content(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return json.loads(content)
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str):
                    texts.append(txt)
        if texts:
            return json.loads("".join(texts))
    raise ValueError(f"Unsupported content format: {type(content)}")


def _validate_prediction(obj: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in PRED_KEYS:
        if k not in obj:
            raise ValueError(f"Missing key {k} in response.")
        out[k] = float(obj[k])
    return out


def _chat_completion(cfg: RunConfig, api_key: str, prompt: str) -> tuple[dict[str, Any], str]:
    url = cfg.api_base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "messages": [
            {"role": "system", "content": "You are a strict JSON forecasting assistant."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": _prediction_schema(),
        },
    }
    r = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout_sec)
    r.raise_for_status()
    data = r.json()
    msg = data["choices"][0]["message"]
    parsed = _extract_json_from_content(msg.get("content"))
    return parsed, json.dumps(data, ensure_ascii=True)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    inputs_dir = (project_root / args.inputs_dir).resolve()
    frozen_setup_path = (project_root / args.frozen_setup).resolve()
    out_root = (project_root / args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key in env var: {args.api_key_env}")

    run_id = args.run_id.strip() or _now_run_id()
    run_dir = out_root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    per_cutoff_dir = run_dir / "predictions_by_cutoff"
    raw_dir = run_dir / "raw_responses"
    per_cutoff_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    cfg = RunConfig(
        model=args.model,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        timeout_sec=args.timeout_sec,
        max_retries=args.max_retries,
        retry_sleep_sec=args.retry_sleep_sec,
        history_months=args.history_months,
        max_cutoffs=args.max_cutoffs,
        max_keywords=args.max_keywords,
        run_id=run_id,
    )

    frozen_setup = json.loads(frozen_setup_path.read_text(encoding="utf-8"))
    cutoff_files = _collect_cutoff_files(inputs_dir, cfg.max_cutoffs)
    if not cutoff_files:
        raise FileNotFoundError(f"No llm_input_cutoff_*.json found in {inputs_dir}")

    meta = {
        "run_id": run_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provider": "api",
        "api_base": cfg.api_base,
        "api_key_env": cfg.api_key_env,
        "model": cfg.model,
        "temperature": cfg.temperature,
        "history_months": cfg.history_months,
        "max_cutoffs": cfg.max_cutoffs,
        "max_keywords": cfg.max_keywords,
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

        rows: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []

        for idx, rec in enumerate(keywords):
            keyword = str(rec["keyword"])
            months = list(rec["history"]["months"])
            values = list(rec["history"]["values"])
            values, months = _trim_history(values, months, cfg.history_months)
            prompt = _build_prompt(
                keyword=keyword,
                cutoff_date=cutoff_date,
                epsilon=epsilon,
                horizons=horizons,
                feature_names=feature_names,
                months=months,
                values=values,
            )

            pred = None
            raw_blob = ""
            last_error = None
            for _attempt in range(cfg.max_retries + 1):
                try:
                    parsed, raw_blob = _chat_completion(cfg, api_key, prompt)
                    pred = _validate_prediction(parsed)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    time.sleep(cfg.retry_sleep_sec)

            total_calls += 1
            if pred is None:
                total_failures += 1
                failures.append({"keyword": keyword, "error": last_error or "unknown_error"})
                continue

            rows.append({"keyword": keyword, **pred})
            (raw_dir / f"{cutoff_date}_{idx:04d}.json").write_text(raw_blob, encoding="utf-8")

        cutoff_key = cutoff_date[:7]
        (per_cutoff_dir / f"predictions_{cutoff_key}.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "model_id": cfg.model,
                    "cutoff_date": cutoff_date,
                    "predictions": rows,
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        (per_cutoff_dir / f"failures_{cutoff_key}.json").write_text(
            json.dumps(failures, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        print(f"[cutoff {cutoff_key}] predictions={len(rows)} failures={len(failures)}")

    summary = {
        "run_id": run_id,
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


