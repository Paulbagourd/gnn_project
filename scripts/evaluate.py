#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM dry-run predictions (RMSE/MAE abs+rel by horizon)."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="LLM run directory containing predictions_by_cutoff/.",
    )
    parser.add_argument(
        "--frozen-setup",
        default="data/usecase_cyberspace/gnn_llm_comparison/outputs/frozen_setup.json",
    )
    parser.add_argument(
        "--used-config",
        default="data/usecase_cyberspace/04_build_graph/outputs/used_config.json",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Override epsilon for relative target; default uses frozen_setup epsilon.",
    )
    parser.add_argument(
        "--out-name",
        default="evaluation_summary",
        help="Prefix for output files in <run-dir>/evaluation/.",
    )
    parser.add_argument(
        "--k-values",
        default="10,20,50",
        help="Comma-separated k values for Precision@k/Recall@k/NDCG@k.",
    )
    parser.add_argument(
        "--relevance-ratio",
        type=float,
        default=0.10,
        help="Top true fraction considered relevant for Recall@k (per cutoff, per horizon).",
    )
    parser.add_argument(
        "--min-relevance-k",
        type=int,
        default=10,
        help="Minimum size of the relevant set for Recall@k.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    if y_true.size == 0:
        return float("nan"), float("nan")
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    return mae, rmse


def _load_weights(used_config: dict[str, Any], n_features: int) -> np.ndarray:
    gp = used_config.get("graph_params", {})
    preview = gp.get("preview", {})
    raw = preview.get("feature_weights")
    if not isinstance(raw, list) or len(raw) != n_features:
        # fallback uniform
        return np.ones(n_features, dtype=np.float64) / float(n_features)
    w = np.asarray([float(x) for x in raw], dtype=np.float64)
    s = float(np.sum(w))
    if not math.isfinite(s) or abs(s) < 1e-12:
        return np.ones(n_features, dtype=np.float64) / float(n_features)
    return w / s


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    s1 = pd.Series(y_true)
    s2 = pd.Series(y_pred)
    return float(s1.corr(s2, method="spearman"))


def _safe_kendall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    s1 = pd.Series(y_true)
    s2 = pd.Series(y_pred)
    return float(s1.corr(s2, method="kendall"))


def _topk_indices_desc(vals: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), vals.size))
    idx = np.argsort(vals)[::-1]
    return idx[:k]


def _ranking_metrics_by_group(
    df_h: pd.DataFrame,
    y_col: str,
    p_col: str,
    k_values: list[int],
    relevance_ratio: float,
    min_relevance_k: int,
) -> dict[str, float]:
    per_cut = []
    for _cutoff, g in df_h.groupby("cutoff_month"):
        gp = g[[y_col, p_col]].copy()
        gp = gp.replace([np.inf, -np.inf], np.nan).dropna()
        y = gp[y_col].to_numpy(dtype=np.float64)
        p = gp[p_col].to_numpy(dtype=np.float64)
        n = y.size
        if n < 2:
            continue

        rec: dict[str, float] = {
            "spearman": _safe_spearman(y, p),
            "kendall": _safe_kendall(y, p),
        }

        # Relevant set for recall: top true items.
        rel_k = max(int(round(relevance_ratio * n)), int(min_relevance_k))
        rel_k = max(1, min(rel_k, n))
        rel_idx = set(_topk_indices_desc(y, rel_k).tolist())

        y_shift = y - float(np.min(y)) + 1e-12
        for k in k_values:
            kk = max(1, min(int(k), n))
            pred_top = set(_topk_indices_desc(p, kk).tolist())
            hits = len(pred_top.intersection(rel_idx))
            rec[f"precision_at_{k}"] = float(hits / kk)
            rec[f"recall_at_{k}"] = float(hits / rel_k)
            try:
                rec[f"ndcg_at_{k}"] = float(
                    ndcg_score(
                        y_true=y_shift.reshape(1, -1),
                        y_score=p.reshape(1, -1),
                        k=kk,
                    )
                )
            except Exception:  # noqa: BLE001
                rec[f"ndcg_at_{k}"] = float("nan")

        per_cut.append(rec)

    if not per_cut:
        out = {"spearman": float("nan"), "kendall": float("nan")}
        for k in k_values:
            out[f"precision_at_{k}"] = float("nan")
            out[f"recall_at_{k}"] = float("nan")
            out[f"ndcg_at_{k}"] = float("nan")
        return out

    d = pd.DataFrame(per_cut)
    return {c: float(d[c].mean()) for c in d.columns}


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    run_dir = (project_root / args.run_dir).resolve()
    pred_dir = run_dir / "predictions_by_cutoff"
    if not pred_dir.exists():
        raise FileNotFoundError(f"Missing predictions_by_cutoff in: {run_dir}")

    frozen_setup_path = (project_root / args.frozen_setup).resolve()
    frozen = _read_json(frozen_setup_path)
    used_cfg_path = (project_root / args.used_config).resolve()
    used_cfg = _read_json(used_cfg_path)

    horizons = [int(x) for x in frozen["horizons_months"]]
    k_values = [int(x.strip()) for x in str(args.k_values).split(",") if x.strip()]
    if not k_values:
        k_values = [10, 20, 50]
    epsilon = float(args.epsilon if args.epsilon is not None else frozen["epsilon"])
    feature_names = list(frozen["feature_names"])

    feat_path = (project_root / frozen["feature_tensor_path"]).resolve()
    ts_path = (project_root / frozen["timestamps_path"]).resolve()
    kw_path = (project_root / frozen["keywords_path"]).resolve()

    X = np.load(feat_path)  # (T,N,F)
    ts = np.load(ts_path, allow_pickle=True).tolist()
    kw = [ln.strip() for ln in kw_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    T, N, F = X.shape
    if len(ts) != T or len(kw) != N:
        raise ValueError("Inconsistent shapes between features/timestamps/keywords.")
    if len(feature_names) != F:
        feature_names = [f"feature_{i}" for i in range(F)]

    w = _load_weights(used_cfg, n_features=F)
    E = np.tensordot(X, w, axes=([2], [0]))  # (T,N)

    ts_to_idx = {str(t): i for i, t in enumerate(ts)}
    kw_to_idx = {k: i for i, k in enumerate(kw)}

    pred_files = sorted(pred_dir.glob("predictions_*.json"))
    if not pred_files:
        raise FileNotFoundError(f"No predictions_*.json in {pred_dir}")

    rows = []
    for pf in pred_files:
        payload = _read_json(pf)
        cutoff_date = str(payload.get("cutoff_date", "")).strip()
        cutoff_month = cutoff_date[:7]
        if cutoff_month not in ts_to_idx:
            continue
        t_idx = ts_to_idx[cutoff_month]

        preds = payload.get("predictions", [])
        for rec in preds:
            keyword = str(rec.get("keyword", "")).strip()
            if keyword not in kw_to_idx:
                continue
            k_idx = kw_to_idx[keyword]
            e_t = float(E[t_idx, k_idx])
            for h in horizons:
                if t_idx + h >= T:
                    continue
                e_f = float(E[t_idx + h, k_idx])
                y_abs = e_f - e_t
                y_rel = (e_f - e_t) / (e_t + epsilon)

                p_abs_key = f"abs_{h}"
                p_rel_key = f"rel_{h}"
                p_abs = float("nan")
                p_rel = float("nan")
                if p_abs_key in rec:
                    try:
                        if rec[p_abs_key] is not None:
                            p_abs = float(rec[p_abs_key])
                    except Exception:  # noqa: BLE001
                        p_abs = float("nan")
                if p_rel_key in rec:
                    try:
                        if rec[p_rel_key] is not None:
                            p_rel = float(rec[p_rel_key])
                    except Exception:  # noqa: BLE001
                        p_rel = float("nan")
                if not (math.isfinite(p_abs) or math.isfinite(p_rel)):
                    continue

                rows.append(
                    {
                        "cutoff_month": cutoff_month,
                        "keyword": keyword,
                        "horizon": h,
                        "y_abs": y_abs,
                        "p_abs": p_abs,
                        "y_rel": y_rel,
                        "p_rel": p_rel,
                    }
                )

    df = pd.DataFrame(rows)
    out_eval_dir = run_dir / "evaluation"
    out_eval_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        (out_eval_dir / f"{args.out_name}.json").write_text(
            json.dumps({"error": "No comparable prediction rows found."}, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        print("[error] No comparable prediction rows found.")
        return

    metrics_rows: list[dict[str, Any]] = []
    for h in horizons:
        sub = df[df["horizon"] == h]
        sub_abs = sub[np.isfinite(sub["p_abs"].to_numpy(dtype=np.float64))]
        sub_rel = sub[np.isfinite(sub["p_rel"].to_numpy(dtype=np.float64))]

        mae_abs, rmse_abs = _compute_metrics(sub_abs["y_abs"].to_numpy(), sub_abs["p_abs"].to_numpy())
        mae_rel, rmse_rel = _compute_metrics(sub_rel["y_rel"].to_numpy(), sub_rel["p_rel"].to_numpy())
        rank_abs = _ranking_metrics_by_group(
            df_h=sub_abs,
            y_col="y_abs",
            p_col="p_abs",
            k_values=k_values,
            relevance_ratio=float(args.relevance_ratio),
            min_relevance_k=int(args.min_relevance_k),
        )
        rank_rel = _ranking_metrics_by_group(
            df_h=sub_rel,
            y_col="y_rel",
            p_col="p_rel",
            k_values=k_values,
            relevance_ratio=float(args.relevance_ratio),
            min_relevance_k=int(args.min_relevance_k),
        )
        row = {
            "horizon_months": h,
            "n_samples": int(len(sub)),
            "n_samples_abs": int(len(sub_abs)),
            "n_samples_rel": int(len(sub_rel)),
            "mae_abs": mae_abs,
            "rmse_abs": rmse_abs,
            "mae_rel": mae_rel,
            "rmse_rel": rmse_rel,
            "spearman_abs": rank_abs["spearman"],
            "kendall_abs": rank_abs["kendall"],
            "spearman_rel": rank_rel["spearman"],
            "kendall_rel": rank_rel["kendall"],
        }
        for k in k_values:
            row[f"precision_abs_at_{k}"] = rank_abs[f"precision_at_{k}"]
            row[f"recall_abs_at_{k}"] = rank_abs[f"recall_at_{k}"]
            row[f"ndcg_abs_at_{k}"] = rank_abs[f"ndcg_at_{k}"]
            row[f"precision_rel_at_{k}"] = rank_rel[f"precision_at_{k}"]
            row[f"recall_rel_at_{k}"] = rank_rel[f"recall_at_{k}"]
            row[f"ndcg_rel_at_{k}"] = rank_rel[f"ndcg_at_{k}"]
        metrics_rows.append(
            row
        )

    mdf = pd.DataFrame(metrics_rows).sort_values("horizon_months")
    macro = {
        "n_rows_total": int(len(df)),
        "n_cutoffs_predicted": int(df["cutoff_month"].nunique()),
        "n_keywords_predicted": int(df["keyword"].nunique()),
        "mae_abs_macro": float(mdf["mae_abs"].mean()),
        "rmse_abs_macro": float(mdf["rmse_abs"].mean()),
        "mae_rel_macro": float(mdf["mae_rel"].mean()),
        "rmse_rel_macro": float(mdf["rmse_rel"].mean()),
        "spearman_abs_macro": float(mdf["spearman_abs"].mean()),
        "kendall_abs_macro": float(mdf["kendall_abs"].mean()),
        "spearman_rel_macro": float(mdf["spearman_rel"].mean()),
        "kendall_rel_macro": float(mdf["kendall_rel"].mean()),
        "horizons_months": horizons,
        "k_values": k_values,
        "relevance_ratio": float(args.relevance_ratio),
        "min_relevance_k": int(args.min_relevance_k),
        "epsilon": epsilon,
        "feature_names": feature_names,
        "feature_weights_normalized": w.tolist(),
    }
    for k in k_values:
        macro[f"precision_abs_at_{k}_macro"] = float(mdf[f"precision_abs_at_{k}"].mean())
        macro[f"recall_abs_at_{k}_macro"] = float(mdf[f"recall_abs_at_{k}"].mean())
        macro[f"ndcg_abs_at_{k}_macro"] = float(mdf[f"ndcg_abs_at_{k}"].mean())
        macro[f"precision_rel_at_{k}_macro"] = float(mdf[f"precision_rel_at_{k}"].mean())
        macro[f"recall_rel_at_{k}_macro"] = float(mdf[f"recall_rel_at_{k}"].mean())
        macro[f"ndcg_rel_at_{k}_macro"] = float(mdf[f"ndcg_rel_at_{k}"].mean())

    out_json = {
        "macro": macro,
        "per_horizon": metrics_rows,
    }

    (out_eval_dir / f"{args.out_name}.json").write_text(
        json.dumps(out_json, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    mdf.to_csv(out_eval_dir / f"{args.out_name}_per_horizon.csv", index=False)
    df.to_csv(out_eval_dir / f"{args.out_name}_rows.csv", index=False)

    print(f"[ok] evaluation json: {out_eval_dir / (args.out_name + '.json')}")
    print(f"[ok] per-horizon csv: {out_eval_dir / (args.out_name + '_per_horizon.csv')}")
    print(
        "[macro] "
        f"RMSE abs={macro['rmse_abs_macro']:.6f} rel={macro['rmse_rel_macro']:.6f} | "
        f"MAE abs={macro['mae_abs_macro']:.6f} rel={macro['mae_rel_macro']:.6f} | "
        f"Spearman abs={macro['spearman_abs_macro']:.6f} rel={macro['spearman_rel_macro']:.6f}"
    )


if __name__ == "__main__":
    main()

