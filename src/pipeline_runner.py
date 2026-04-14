from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]   # project root
DATA = ROOT / "data"


def load_yaml(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = deep_merge(out.get(k), v)
        return out
    return b if b is not None else a


def json_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def collect_inputs(paths: list[Path]) -> Dict[str, str]:
    d = {}
    for p in paths:
        if p.is_file():
            d[str(p)] = file_hash(p)
        elif p.is_dir():
            inner = sorted([pp for pp in p.rglob("*") if pp.is_file()])
            d[str(p)] = json_hash({str(pp): file_hash(pp) for pp in inner})
    return d


def should_run(step_dir: Path, sig: Dict[str, Any], force: bool) -> bool:
    meta = step_dir / ".meta.json"
    if force or not meta.exists():
        return True
    try:
        prev = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return True
    return json_hash(prev.get("signature")) != json_hash(sig)


def write_meta(step_dir: Path, sig: Dict[str, Any], extras: Dict[str, Any] | None = None):
    step_dir.mkdir(parents=True, exist_ok=True)
    meta = {"signature": sig}
    if extras:
        meta["extras"] = extras
    (step_dir / ".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def import_step(module_name: str):
    # e.g., "steps.step_01_extract"
    return importlib.import_module(f"src.steps.{module_name}")


def step_dirname(sid: str, step_mod) -> str:
    """
    Build the on-disk folder name for a step. If `sid` already ends with
    `_<STEP_NAME>`, return `sid` as-is; otherwise return `"{sid}_{STEP_NAME}"`.
    This prevents duplicates like 01_extract_extract.
    """
    step_name = getattr(step_mod, "STEP_NAME", sid)
    return sid if sid.endswith(f"_{step_name}") else f"{sid}_{step_name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usecase", required=True)               # e.g., usecase_quantum
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--override", nargs="*", default=[],      # key=value pairs
                    help="e.g. run.force=true params.keyword_topn=50")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--from-step", default=None)
    ap.add_argument("--up-to", default=None)
    args = ap.parse_args()

    base = load_yaml(ROOT / args.config)
    uc = load_yaml(ROOT / "config" / "usecases" / f"{args.usecase}.yaml")
    cfg = deep_merge(base, uc)

    # CLI overrides (very simple dotted-path assigner)
    for kv in args.override:
        k, v = kv.split("=", 1)
        cur = cfg
        keys = k.split(".")
        for kk in keys[:-1]:
            cur = cur.setdefault(kk, {})
        # try parse bool/int/float/json
        vv = v
        if isinstance(v, str) and v.lower() in ("true", "false"):
            vv = v.lower() == "true"
        else:
            try:
                vv = int(v)
            except Exception:
                try:
                    vv = float(v)
                except Exception:
                    try:
                        vv = json.loads(v)
                    except Exception:
                        vv = v
        cur[keys[-1]] = vv

    usecase_dir = DATA / args.usecase
    usecase_dir.mkdir(parents=True, exist_ok=True)

    all_steps = list(cfg["run"]["steps"])
    steps = list(all_steps)
    if args.from_step:
        if args.from_step not in steps:
            raise SystemExit(f"Step '{args.from_step}' is not listed in run.steps")
        start = steps.index(args.from_step)
        steps = steps[start:]
    if args.up_to:
        if args.up_to not in steps:
            raise SystemExit(f"Step '{args.up_to}' is not listed in run.steps")
        end = steps.index(args.up_to)
        steps = steps[: end + 1]

    # Import step modules dynamically & run
    for sid in steps:
        mod = import_step(f"step_{sid}")

        # Build current step dir name consistently
        step_label = step_dirname(sid, mod)
        step_dir = usecase_dir / step_label

        # Resolve previous step dir (if this step depends on previous)
        prev_dir = None
        if hasattr(mod, "inputs_from_prev") and mod.inputs_from_prev:
            idx_all = all_steps.index(sid)
            if idx_all > 0:
                prev_sid = all_steps[idx_all - 1]
                prev_mod = import_step(f"step_{prev_sid}")
                prev_label = step_dirname(prev_sid, prev_mod)
                prev_dir = usecase_dir / prev_label

        # Allow the step to declare its external inputs
        ext_inputs = mod.external_inputs(cfg, usecase_dir, prev_dir)

        input_hashes = collect_inputs(ext_inputs)
        step_sig = {
            "code_version": getattr(mod, "STEP_CODE_VERSION", "0"),
            "params": mod.relevant_params(cfg),
            "inputs": input_hashes,
        }

        cfg_force = bool(cfg.get("run", {}).get("force", False))
        if should_run(step_dir, step_sig, force=args.force or cfg_force):
            artifacts = mod.run(cfg, usecase_dir, prev_dir, step_dir)
            write_meta(step_dir, step_sig, {"artifacts": artifacts})

            state = artifacts.get("state") if isinstance(artifacts, dict) else None
            if state == "WAITING":
                message = artifacts.get("message") if isinstance(artifacts, dict) else None
                if message:
                    print(f"[{sid}] {message}")
                else:
                    print(f"[{sid}] waiting for external processing.")
                request_path = artifacts.get("request_path") if isinstance(artifacts, dict) else None
                if request_path:
                    print(f"[{sid}] Request file: {request_path}")
                wait_for = artifacts.get("wait_for") if isinstance(artifacts, dict) else None
                if wait_for:
                    print(f"[{sid}] Expecting outputs under: {wait_for}")
                print(f"[{sid}] Pausing pipeline until the external step completes.")
                return

            print(f"[{sid}] ran -> {step_dir}")
        else:
            print(f"[{sid}] up-to-date, skipped.")


if __name__ == "__main__":
    main()
