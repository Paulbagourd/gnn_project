from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import PowerNorm

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover - scipy is optional at runtime
    minimize = None


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != 3:
        raise ValueError(f"Expected 3 weights, got {w.size}.")
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError(f"Invalid weight sum: {s}")
    return w / s


def _load_names(path: Path, n: int) -> list[str]:
    if path.exists():
        try:
            names = (
                pd.read_csv(path, header=None, names=["kw"], dtype=str, engine="python")["kw"]
                .astype(str)
                .tolist()
            )
            if len(names) == n:
                return names
        except Exception:
            pass
    return [f"n{i}" for i in range(n)]


def _set_year_ticks(ax, ts: pd.DatetimeIndex, every: int = 5) -> None:
    years = pd.Series(ts.year)
    if years.empty:
        return
    y0 = int(years.min())
    y1 = int(years.max())
    ticks = []
    labels = []
    for y in range(y0, y1 + 1, every):
        idx = np.where((ts.year == y) & (ts.month == 1))[0]
        if idx.size > 0:
            ticks.append(float(idx[0]) + 0.5)
            labels.append(str(y))
    if ticks:
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, rotation=0)


def _balanced_curves(avg_per_feat: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = _normalize_weights(weights)
    return np.log1p(np.clip(avg_per_feat * w[None, :], a_min=0.0, a_max=None))


def _balanced_loss(avg_per_feat: np.ndarray, weights: np.ndarray) -> float:
    Y = _balanced_curves(avg_per_feat, weights)
    center = np.nanmean(Y, axis=1, keepdims=True)
    return float(np.nansum((Y - center) ** 2))


def _grid_fit_weights(avg_per_feat: np.ndarray, step: float = 0.002) -> np.ndarray:
    if step <= 0.0 or step >= 1.0:
        raise ValueError(f"fit grid step must be in (0, 1), got {step}")
    best_w = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
    best_loss = float("inf")
    for a in np.arange(0.0, 1.0 + 1e-12, step):
        rem = 1.0 - float(a)
        for b in np.arange(0.0, rem + 1e-12, step):
            c = rem - float(b)
            w = np.array([a, b, c], dtype=float)
            loss = _balanced_loss(avg_per_feat, w)
            if loss < best_loss:
                best_loss = loss
                best_w = w
    return _normalize_weights(best_w)


def _fit_balanced_weights(
    avg_per_feat: np.ndarray,
    initial_weights: np.ndarray,
    grid_step: float = 0.002,
) -> np.ndarray:
    grid_best = _grid_fit_weights(avg_per_feat, step=grid_step)
    if minimize is None:
        return grid_best
    starts = [
        _normalize_weights(initial_weights),
        grid_best,
        np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float),
        np.array([0.2, 0.6, 0.2], dtype=float),
        np.array([0.5, 0.4, 0.1], dtype=float),
    ]
    bounds = [(1e-9, 1.0)] * 3
    constraints = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    best_w = grid_best
    best_loss = _balanced_loss(avg_per_feat, best_w)
    for start in starts:
        try:
            res = minimize(
                lambda w: _balanced_loss(avg_per_feat, w),
                start,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-12},
            )
        except Exception:
            continue
        if not getattr(res, "success", False):
            continue
        w = _normalize_weights(np.asarray(res.x, dtype=float))
        loss = _balanced_loss(avg_per_feat, w)
        if loss < best_loss:
            best_loss = loss
            best_w = w
    return best_w


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate balanced FW aggregate plot and FW heatmap.")
    ap.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root.",
    )
    ap.add_argument(
        "--usecase",
        default="usecase_cyberspace",
        help="Usecase name under data/.",
    )
    ap.add_argument(
        "--weights",
        default="0.3831,0.5189,0.0980",
        help="Comma-separated FW weights in feature order: citation, oc_freq, edge_weight.",
    )
    ap.add_argument(
        "--feature-names",
        default="xcum_frac_split,oc_freq,edge_weight",
        help="Comma-separated feature names for labels.",
    )
    ap.add_argument(
        "--line-start",
        default="2005-01-01",
        help="Start date for the balanced aggregate line plot.",
    )
    ap.add_argument(
        "--line-end",
        default="2025-12-31",
        help="End date for the balanced aggregate line plot.",
    )
    ap.add_argument(
        "--freeze-date",
        default="2025-10-01",
        help="Freeze date used to sort columns for the FW heatmap.",
    )
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument(
        "--fit-weights",
        action="store_true",
        help="Fit FW to minimize the spread between the three plotted feature curves on the line window.",
    )
    ap.add_argument(
        "--fit-grid-step",
        type=float,
        default=0.002,
        help="Simplex grid step used before continuous refinement when --fit-weights is enabled.",
    )
    ap.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix appended to generated filenames, for example '_refit'.",
    )
    args = ap.parse_args()

    root = args.project_root
    suffix = str(args.output_suffix or "")
    out_base = root / "data" / args.usecase / "04_build_graph" / "outputs"
    feats_path = out_base / "3_corrected_data" / "stacked_features_active_corrected.npy"
    time_path = out_base / "1_raw_data" / "feature_timestamps.npy"
    names_path = out_base / "2_active_data" / "keywords_active.txt"
    save_dir = out_base / "plots" / "inputs_seen"
    heatmap_dir = save_dir / "heatmaps"
    ranking_dir = save_dir / "ranking"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    ranking_dir.mkdir(parents=True, exist_ok=True)

    if not feats_path.exists():
        raise FileNotFoundError(f"Missing features file: {feats_path}")
    if not time_path.exists():
        raise FileNotFoundError(f"Missing timestamps file: {time_path}")

    initial_weights = _normalize_weights(np.array([float(x.strip()) for x in args.weights.split(",")], dtype=float))
    weights = initial_weights.copy()
    feature_names = [x.strip() for x in args.feature_names.split(",")]
    if len(feature_names) != 3:
        raise ValueError("feature_names must contain exactly 3 labels.")

    X = np.load(feats_path)  # (T, N, F)
    ts_raw = np.load(time_path, allow_pickle=True).astype(str)
    ts = pd.to_datetime(ts_raw, errors="coerce").to_period("M").to_timestamp(how="start")
    if X.ndim != 3 or X.shape[2] < 3:
        raise ValueError(f"Unexpected feature shape: {X.shape}")

    T = min(len(ts), X.shape[0])
    ts = pd.DatetimeIndex(ts[:T])
    X = X[:T, :, :3].astype(float)
    X = np.where(np.isfinite(X), X, np.nan)

    t0 = pd.to_datetime(args.line_start, errors="coerce")
    t1 = pd.to_datetime(args.line_end, errors="coerce")
    if pd.isna(t0) or pd.isna(t1):
        raise ValueError("Invalid line-start or line-end date.")
    mask = (ts >= t0) & (ts <= t1)
    if not np.any(mask):
        raise RuntimeError("Date window for line plot removed all rows.")

    avg_per_feat = np.nanmean(X[mask], axis=1)
    if args.fit_weights:
        weights = _fit_balanced_weights(avg_per_feat, initial_weights=weights, grid_step=args.fit_grid_step)

    # Balanced aggregate line plot over the selected window
    Xw = X * weights[None, None, :]
    avg_per_feat_w = np.nanmean(Xw, axis=1)  # (T, 3)
    weighted_sum = np.nanmean(np.nansum(Xw, axis=-1), axis=1)  # (T,)
    Y = np.log1p(np.clip(avg_per_feat_w, a_min=0.0, a_max=None))
    Yw = np.log1p(np.clip(weighted_sum, a_min=0.0, a_max=None))

    ts_line = ts[mask]
    Y_line = Y[mask]
    Yw_line = Yw[mask]

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, nm in enumerate(feature_names):
        ax.plot(ts_line, Y_line[:, i], label=f"{nm} × {weights[i]:.2f} (avg over nodes, log1p)")
    ax.plot(ts_line, Yw_line, ls="--", label="Σ (weighted features)")
    ax.set_xlabel("Year")
    ax.set_ylabel("log1p(value)")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    line_pdf = heatmap_dir / f"EM_raw_Eps1e-08_F8_W12__aggregate_raw_balanced_fw_2005_2025{suffix}.pdf"
    fig.savefig(line_pdf, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    line_csv = ranking_dir / f"EM_raw_Eps1e-08_F8_W12__aggregate_raw_balanced_fw_2005_2025{suffix}.csv"
    line_df = pd.DataFrame(
        {
            "date": ts_line.strftime("%Y-%m-%d"),
            feature_names[0]: Y_line[:, 0],
            feature_names[1]: Y_line[:, 1],
            feature_names[2]: Y_line[:, 2],
            "fw_sum": Yw_line,
        }
    )
    line_df.to_csv(line_csv, index=False, encoding="utf-8")

    # FW aggregate heatmap over full timeline, sorted by freeze date
    agg_vals = np.tensordot(X, weights, axes=(2, 0))  # (T, N)
    names = _load_names(names_path, agg_vals.shape[1])
    df_fw = pd.DataFrame(agg_vals, index=ts, columns=names)

    freeze_ts = pd.to_datetime(args.freeze_date, errors="coerce")
    if pd.isna(freeze_ts):
        freeze_idx = len(df_fw) - 1
    else:
        freeze_idx = df_fw.index.get_indexer([freeze_ts], method="nearest")[0]
        if freeze_idx < 0:
            freeze_idx = len(df_fw) - 1

    order = df_fw.iloc[freeze_idx].fillna(-np.inf).sort_values(ascending=False).index
    df_fw_sorted = df_fw.loc[:, order]

    Z = np.log1p(np.clip(df_fw_sorted.values, a_min=0.0, a_max=None))
    finite = Z[np.isfinite(Z)]
    vmax = float(np.nanpercentile(finite, 99.5)) if finite.size else 1.0
    norm = PowerNorm(gamma=1.8, vmin=0.0, vmax=max(vmax, 1e-6))

    plt.figure(figsize=(14, 6))
    ax = sns.heatmap(
        Z,
        cmap="viridis",
        norm=norm,
        yticklabels=False,
        cbar_kws={"label": "log(1 + FW-weighted score)"},
    )
    ax.set_xlabel("Node")
    ax.set_ylabel("Date")
    _set_year_ticks(ax, ts, every=5)
    xtick_step = 100
    n_nodes = df_fw_sorted.shape[1]
    if n_nodes > 0:
        xt = np.arange(0, n_nodes, xtick_step)
        if xt.size == 0 or xt[-1] != (n_nodes - 1):
            xt = np.append(xt, n_nodes - 1)
        ax.set_xticks(xt + 0.5)
        ax.set_xticklabels([str(int(i)) for i in xt], rotation=0)
    plt.tight_layout()
    hm_pdf = heatmap_dir / f"EM_raw_Eps1e-08_F8_W12__heatmap_fw_aggregate_balanced_fw{suffix}.pdf"
    plt.savefig(hm_pdf, dpi=args.dpi, bbox_inches="tight")
    plt.close()

    rank = df_fw.iloc[freeze_idx].sort_values(ascending=False)
    rank_df = pd.DataFrame(
        {
            "rank": np.arange(1, len(rank) + 1, dtype=int),
            "keyword": rank.index,
            "value": rank.values,
        }
    )
    rank_csv = ranking_dir / f"EM_raw_Eps1e-08_F8_W12__ranking_fw_aggregate_balanced_fw_2025-10-01{suffix}.csv"
    rank_df.to_csv(rank_csv, index=False, encoding="utf-8")

    print(f"[ok] line plot: {line_pdf}")
    print(f"[ok] line csv : {line_csv}")
    print(f"[ok] heatmap  : {hm_pdf}")
    print(f"[ok] ranking  : {rank_csv}")
    if args.fit_weights:
        print(f"[ok] initial weights (normalized): {initial_weights.tolist()}")
        print(f"[ok] fitted  weights (normalized): {weights.tolist()}")
        print(f"[ok] initial balanced loss: {_balanced_loss(avg_per_feat, initial_weights):.12f}")
        print(f"[ok] fitted  balanced loss: {_balanced_loss(avg_per_feat, weights):.12f}")
    else:
        print(f"[ok] weights (normalized): {weights.tolist()}")


if __name__ == "__main__":
    main()
