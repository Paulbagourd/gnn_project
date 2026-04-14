from pathlib import Path
import argparse
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _resolve_out_dir(raw: str | None) -> Path:
    if not raw:
        return ROOT / "data" / "usecase_cyberspace" / "04_graph" / "outputs"
    path = Path(raw)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _load_overall_metrics(out_dir: Path) -> dict[str, dict[str, float]] | None:
    summary_path = out_dir / "diagnostics_summary.json"
    if not summary_path.exists():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    overall: dict[str, dict[str, float]] = {}
    runs = data.get("runs") or []
    for row in runs:
        name = row.get("name")
        if name == "graph_base":
            overall["graph"] = {"mae": row.get("mae"), "rmse": row.get("rmse")}
        elif name == "graph_mix":
            overall["graph_mix"] = {"mae": row.get("mae"), "rmse": row.get("rmse")}
        elif name == "graph_multi":
            overall["graph_multi"] = {"mae": row.get("mae"), "rmse": row.get("rmse")}
        elif name == "graph_multi_mix":
            overall["graph_multi_mix"] = {"mae": row.get("mae"), "rmse": row.get("rmse")}
        elif name == "nograph":
            overall["nograph"] = {"mae": row.get("mae"), "rmse": row.get("rmse")}
    baselines = data.get("baselines") or {}
    if "persist" in baselines:
        overall["persist"] = {
            "mae": baselines["persist"].get("mae"),
            "rmse": baselines["persist"].get("rmse"),
        }
    if "drift" in baselines:
        overall["drift"] = {
            "mae": baselines["drift"].get("mae"),
            "rmse": baselines["drift"].get("rmse"),
        }
    elif "ema" in baselines:
        # Backward compatibility with older diagnostics summaries.
        overall["drift"] = {
            "mae": baselines["ema"].get("mae"),
            "rmse": baselines["ema"].get("rmse"),
        }
    return overall


def _find_fw_aggregate_ranking(out_dir: Path) -> Path | None:
    pattern = "*ranking_fw_aggregate*2025-10-01*.csv"
    matches = list(out_dir.rglob(pattern))
    if matches:
        return sorted(matches)[0]
    fallback = ROOT / "data" / "usecase_cyberspace" / "04_build_graph" / "outputs"
    matches = list(fallback.rglob(pattern))
    if matches:
        return sorted(matches)[0]
    return None


def _load_rank_from_fw_aggregate(out_dir: Path) -> dict[str, float] | None:
    path = _find_fw_aggregate_ranking(out_dir)
    if path is None:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    name_col = "keyword" if "keyword" in df.columns else ("name" if "name" in df.columns else df.columns[0])
    value_col = "value" if "value" in df.columns else None
    if value_col is None:
        # If no explicit value column, fall back to rank order.
        return {str(k): float(i) for i, k in enumerate(df[name_col].astype(str).tolist())}
    return dict(zip(df[name_col].astype(str), pd.to_numeric(df[value_col], errors="coerce")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-node diagnostics CSVs into a single file.")
    parser.add_argument("--base-dir", dest="base_dir", default=None, help="Directory containing diagnostics_per_node_*.csv")
    args = parser.parse_args()

    out_dir = _resolve_out_dir(args.base_dir)

    files = {
        "graph": out_dir / "diagnostics_per_node_graph.csv",
        "graph_mix": out_dir / "diagnostics_per_node_graph_mix.csv",
        "graph_multi": out_dir / "diagnostics_per_node_graph_multi.csv",
        "graph_multi_mix": out_dir / "diagnostics_per_node_graph_multi_mix.csv",
        "nograph": out_dir / "diagnostics_per_node_nograph.csv",
        "persist": out_dir / "diagnostics_per_node_persist.csv",
        "drift": out_dir / "diagnostics_per_node_drift.csv",
    }
    if not files["drift"].exists():
        ema_path = out_dir / "diagnostics_per_node_ema.csv"
        if ema_path.exists():
            # Backward compatibility with older diagnostics outputs.
            files["drift"] = ema_path

    dfs = []
    for key, path in files.items():
        if not path.exists():
            if key in {"graph_mix", "graph_multi", "graph_multi_mix"}:
                continue
            raise SystemExit(f"Missing: {path}")
        df = pd.read_csv(path)
        df = df.rename(columns={
            "mae": f"mae_{key}",
            "rmse": f"rmse_{key}",
            "count": f"count_{key}",
        })
        dfs.append(df)

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="node", how="outer")

    overall = _load_overall_metrics(out_dir)
    overall_row = None
    if overall:
        overall_row = {"node": "__overall__", "count": ""}
        for key in ["graph", "graph_mix", "graph_multi", "graph_multi_mix", "nograph", "persist", "drift"]:
            metrics = overall.get(key, {})
            overall_row[f"mae_{key}"] = metrics.get("mae")
            overall_row[f"rmse_{key}"] = metrics.get("rmse")
        merged = pd.concat([pd.DataFrame([overall_row]), merged], ignore_index=True)

    # Keep the overall row (if present) at the top, then follow the per-node graph order.
    head = merged.iloc[:1] if (overall_row is not None) else merged.iloc[:0]
    body = merged.iloc[1:] if (overall_row is not None) else merged
    rank = _load_rank_from_fw_aggregate(out_dir)
    if rank:
        body = body.copy()
        body["node"] = body["node"].astype(str)
        body["_rank"] = body["node"].map(rank)
        body = body.sort_values(["_rank", "node"], ascending=[False, True], kind="mergesort")
        body = body.drop(columns=["_rank"])
    merged = pd.concat([head, body], ignore_index=True)

    # Collapse count columns into a single "count" (prefer graph, else nograph, else persist, else drift).
    count_cols = [c for c in merged.columns if c.startswith("count_")]
    if count_cols:
        merged["count"] = merged[count_cols].bfill(axis=1).iloc[:, 0]
        merged = merged.drop(columns=count_cols)

    # Reorder: node, mae_* in preferred order, rmse_* in preferred order, count
    preferred = ["graph", "graph_mix", "graph_multi", "graph_multi_mix", "nograph", "persist", "drift"]
    mae_cols = [f"mae_{k}" for k in preferred if f"mae_{k}" in merged.columns]
    rmse_cols = [f"rmse_{k}" for k in preferred if f"rmse_{k}" in merged.columns]
    cols = ["node"] + mae_cols + rmse_cols
    if "count" in merged.columns:
        cols.append("count")
    merged = merged.loc[:, cols]

    out_path = out_dir / "diagnostics_per_node_all.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
