from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != 3:
        raise ValueError(f"Expected 3 weights, got {w.size}.")
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError(f"Invalid weight sum: {s}")
    return w / s


def _transform_by_mode_3d(values: np.ndarray, mode: str, epsilon: float) -> np.ndarray:
    mode = (mode or "raw").lower()
    if mode == "raw":
        return values.astype(float, copy=False)

    if values.ndim != 3:
        raise ValueError(f"Expected 3D array (T,N,F), got shape={values.shape}")

    def ratio_curr_prev(arr_2d: np.ndarray) -> np.ndarray:
        out = np.zeros_like(arr_2d, dtype=float)
        if arr_2d.shape[0] <= 1:
            return out
        prev, curr = arr_2d[:-1], arr_2d[1:]
        denom = np.maximum(prev, epsilon) if epsilon > 0.0 else prev
        mask = denom > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            out[1:] = np.where(mask, curr / denom, 0.0)
        return out

    def pct_curr_prev(arr_2d: np.ndarray) -> np.ndarray:
        out = np.zeros_like(arr_2d, dtype=float)
        if arr_2d.shape[0] <= 1:
            return out
        prev, curr = arr_2d[:-1], arr_2d[1:]
        denom = np.maximum(prev, epsilon) if epsilon > 0.0 else prev
        mask = denom > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            out[1:] = np.where(mask, curr / denom - 1.0, 0.0)
        return out

    out = np.zeros_like(values, dtype=float)
    for fi in range(values.shape[2]):
        arr = values[:, :, fi]
        out[:, :, fi] = ratio_curr_prev(arr) if mode == "ratio" else pct_curr_prev(arr)
    return out


def _vis(arr: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(arr, a_min=0.0, a_max=None))


def _fit_line_with_huber_weights(idx: np.ndarray, y_log: np.ndarray) -> tuple[float, float]:
    b1, b0 = np.polyfit(idx, y_log, 1)
    resid = y_log - (b1 * idx + b0)
    mad = np.median(np.abs(resid)) + 1e-9
    w = np.clip(1.0 / np.maximum(1.0, np.abs(resid) / (1.345 * mad)), 0.2, 1.0)
    W = np.sqrt(w)
    b1, b0 = np.polyfit(idx * W, y_log * W, 1)
    return float(b1), float(b0)


def _first_correction_index(coeff: np.ndarray) -> int:
    nz = np.flatnonzero(np.asarray(coeff, dtype=float) > 1.0 + 1e-3)
    return int(nz[0]) if nz.size else len(coeff)


def _trend_preserving_tail(
    y_avg: np.ndarray,
    t_break: int,
    *,
    cumulative: bool,
    fit_months_min: int = 24,
    fit_months_max: int = 84,
    epsilon: float = 1e-8,
) -> np.ndarray:
    y_avg = np.asarray(y_avg, dtype=float).reshape(-1)
    Tn = y_avg.shape[0]
    y_out = y_avg.copy()
    if Tn == 0 or t_break <= 1 or t_break >= Tn:
        return y_out

    if cumulative:
        y_src = np.diff(y_avg, prepend=y_avg[:1])
        y_src = np.clip(y_src, a_min=0.0, a_max=None)
    else:
        y_src = np.clip(y_avg, a_min=0.0, a_max=None)

    if not np.isfinite(y_src[:t_break]).any() or np.nanmax(y_src[:t_break]) <= 0.0:
        return y_out

    y_log = np.log1p(y_src + epsilon)
    m = min(max(fit_months_min, t_break), fit_months_max)
    t0 = max(0, (t_break - 1) - m + 1)
    if t_break - t0 < 2:
        t0 = max(0, t_break - 2)

    idx_fit = np.arange(t0, t_break, dtype=float)
    y_fit = y_log[t0:t_break]
    if idx_fit.size < 2 or not np.all(np.isfinite(y_fit)):
        return y_out

    b1, b0 = _fit_line_with_huber_weights(idx_fit, y_fit)
    t_anchor = float(t_break - 1)
    y_anchor = float(y_log[t_break - 1])
    b0 = y_anchor - b1 * t_anchor

    idx_future = np.arange(t_break, Tn, dtype=float)
    target = np.expm1(b1 * idx_future + b0)
    target = np.clip(target, a_min=0.0, a_max=None)

    if cumulative:
        raw_increments = np.diff(y_avg, prepend=y_avg[:1])[t_break:]
        raw_increments = np.clip(raw_increments, a_min=0.0, a_max=None)
        adj_increments = np.maximum(raw_increments, target)
        y_out[t_break:] = y_out[t_break - 1] + np.cumsum(adj_increments)
    else:
        y_out[t_break:] = np.maximum(y_out[t_break:], target)

    return y_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate overlay plots: corrected signal vs former pre-correction signal (dashed)."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root.",
    )
    parser.add_argument("--usecase", default="usecase_cyberspace", help="Usecase name under data/.")
    parser.add_argument("--emergence-mode", default="raw", choices=["raw", "ratio", "pct"])
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--forecast", type=int, default=8)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--feature-names", default="xcum_frac_split,oc_freq,edge_weight")
    parser.add_argument("--weights", default="1,1,1")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--copy-to-stable",
        action="store_true",
        help="Also copy generated PDFs into data/<usecase>/ar24.",
    )
    args = parser.parse_args()

    out_base = args.project_root / "data" / args.usecase / "04_build_graph" / "outputs"
    raw_path = out_base / "2_active_data" / "stacked_features_active.npy"
    corr_path = out_base / "3_corrected_data" / "stacked_features_active_corrected.npy"
    time_path = out_base / "1_raw_data" / "feature_timestamps.npy"
    coeff_path = out_base / "3_corrected_data" / "month_coefficients_per_feature.csv"
    save_dir = out_base / "plots" / "inputs_seen"
    heatmap_dir = save_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw features: {raw_path}")
    if not corr_path.exists():
        raise FileNotFoundError(f"Missing corrected features: {corr_path}")
    if not time_path.exists():
        raise FileNotFoundError(f"Missing timestamps: {time_path}")

    feature_names = [x.strip() for x in args.feature_names.split(",") if x.strip()]
    if len(feature_names) != 3:
        raise ValueError("feature-names must contain exactly 3 comma-separated labels.")
    weights = _normalize_weights(np.array([float(x.strip()) for x in args.weights.split(",")], dtype=float))

    x_raw = np.load(raw_path).astype(float)
    x_cor = np.load(corr_path).astype(float)
    ts_raw = np.load(time_path, allow_pickle=True).astype(str)
    ts = pd.to_datetime(ts_raw, errors="coerce").to_period("M").to_timestamp(how="start")

    if x_raw.ndim != 3 or x_cor.ndim != 3:
        raise ValueError(f"Unexpected array dims: raw={x_raw.shape}, corr={x_cor.shape}")
    if x_raw.shape[2] < 3 or x_cor.shape[2] < 3:
        raise ValueError(f"Need at least 3 features: raw={x_raw.shape}, corr={x_cor.shape}")

    t = min(len(ts), x_raw.shape[0], x_cor.shape[0])
    ts = pd.DatetimeIndex(ts[:t])
    x_raw = x_raw[:t, :, :3]
    x_cor = x_cor[:t, :, :3]

    xt_raw = _transform_by_mode_3d(x_raw, args.emergence_mode, args.epsilon)
    xt_cor = _transform_by_mode_3d(x_cor, args.emergence_mode, args.epsilon)

    avg_raw = np.nanmean(xt_raw, axis=1)  # (T,F)
    avg_cor = np.nanmean(xt_cor, axis=1)
    agg_raw = np.nanmean(xt_raw @ weights, axis=1)
    agg_cor = np.nanmean(xt_cor @ weights, axis=1)

    plt.rcParams.update({"figure.dpi": args.dpi})
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    tag = f"EM_{args.emergence_mode}_Eps{args.epsilon}_F{args.forecast}_W{args.window}"

    coeff_df = None
    if coeff_path.exists():
        coeff_df = pd.read_csv(coeff_path)
        if "date" in coeff_df.columns:
            coeff_df["date"] = pd.to_datetime(coeff_df["date"], errors="coerce")
            coeff_df = coeff_df.set_index("date")

    avg_tail = avg_cor.copy()
    for i, name in enumerate(feature_names):
        coeff_series = None
        if coeff_df is not None and name in coeff_df.columns:
            coeff_series = (
                coeff_df.reindex(ts)[name]
                .fillna(1.0)
                .to_numpy(dtype=float)
            )
        if coeff_series is None:
            coeff_series = np.ones(len(ts), dtype=float)
        t_break = _first_correction_index(coeff_series)
        avg_tail[:, i] = _trend_preserving_tail(
            avg_raw[:, i],
            t_break,
            cumulative=("cum" in name.lower()),
            epsilon=args.epsilon,
        )

    fig_pf, ax_pf = plt.subplots(figsize=(10, 4))
    for i, name in enumerate(feature_names):
        color = colors[i % len(colors)]
        ax_pf.plot(
            ts,
            _vis(avg_raw[:, i]),
            color=color,
            lw=1.8,
            alpha=0.9,
            label=f"{name} (uncorrected)",
        )
        ax_pf.plot(
            ts,
            _vis(avg_tail[:, i]),
            color=color,
            ls="--",
            lw=1.4,
            label=f"{name} (tail-adjusted)",
        )
    ax_pf.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax_pf.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_pf.set_xlabel("Year")
    ax_pf.set_ylabel("log1p(value)")
    ax_pf.legend(ncol=2, fontsize=8)
    fig_pf.tight_layout()
    out_pf = heatmap_dir / f"{tag}__inputs_seen_per_feature_former_dashed.pdf"
    fig_pf.savefig(out_pf, bbox_inches="tight")
    plt.close(fig_pf)

    # Legacy-style aggregate plot (3 weighted feature lines + weighted sum),
    # with pre-correction lag shown as dashed weighted sum.
    y_feat_cor = _vis(avg_cor * weights[None, :])  # (T, F), weighted per-feature
    y_sum_cor = _vis(agg_cor)
    y_sum_raw = _vis(agg_raw)
    fig_legacy, ax_legacy = plt.subplots(figsize=(10, 4))
    for i, name in enumerate(feature_names):
        ax_legacy.plot(
            ts,
            y_feat_cor[:, i],
            lw=1.4,
            label=f"{name} x {weights[i]:.2f} (avg over nodes, log1p)",
        )
    ax_legacy.plot(
        ts,
        y_sum_cor,
        color="tab:red",
        lw=1.8,
        zorder=3,
        label="Σ (weighted features, corrected)",
    )
    ax_legacy.plot(
        ts,
        y_sum_raw,
        color="black",
        ls="--",
        lw=1.9,
        alpha=0.95,
        zorder=4,
        label="Σ (weighted features, former pre-correction / lag)",
    )
    if np.any(y_sum_cor > y_sum_raw):
        ax_legacy.fill_between(
            ts,
            y_sum_raw,
            y_sum_cor,
            where=(y_sum_cor > y_sum_raw),
            color="tab:red",
            alpha=0.10,
            zorder=2,
            label="correction gap",
        )
    ax_legacy.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax_legacy.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_legacy.set_xlabel("Year")
    ax_legacy.set_ylabel("log1p(value)")
    ax_legacy.legend(ncol=2, fontsize=8)
    fig_legacy.tight_layout()
    out_legacy = heatmap_dir / f"{tag}__aggregate_raw_with_lag_dashed.pdf"
    fig_legacy.savefig(out_legacy, bbox_inches="tight")
    plt.close(fig_legacy)

    diff_abs = agg_cor - agg_raw
    with np.errstate(divide="ignore", invalid="ignore"):
        diff_pct = np.where(np.abs(agg_cor) > 1e-12, 100.0 * diff_abs / agg_cor, 0.0)
    nz = np.where(np.abs(diff_abs) > 0)[0]
    tail_start = ts[int(nz[0])] if nz.size else ts[-1]

    fig_ag, (ax_ag, ax_delta) = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.4]},
    )
    ax_ag.plot(
        ts,
        _vis(agg_raw),
        ls="--",
        lw=1.3,
        alpha=0.9,
        label="sum(FW*features) (former, pre-correction)",
    )
    ax_ag.plot(
        ts,
        _vis(agg_cor),
        lw=1.9,
        label="sum(FW*features) (corrected)",
    )
    ax_ag.axvspan(tail_start, ts[-1], color="tab:red", alpha=0.08, label="tail corrected window")
    ax_ag.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax_ag.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_ag.set_ylabel("log1p(value)")
    ax_ag.legend(loc="upper left", fontsize=8)

    ax_delta.plot(ts, diff_abs, color="tab:red", lw=1.4, label="corrected - former (absolute)")
    ax_delta.axhline(0.0, color="black", lw=0.7, alpha=0.5)
    ax_delta.set_ylabel("delta")
    ax_delta.set_xlabel("Year")
    ax_delta.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax_delta.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2 = ax_delta.twinx()
    ax2.plot(ts, diff_pct, color="tab:purple", lw=1.0, alpha=0.8, label="delta (%)")
    ax2.set_ylabel("delta (%)")

    h1, l1 = ax_delta.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if h1 or h2:
        ax_delta.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

    fig_ag.tight_layout()
    out_ag = heatmap_dir / f"{tag}__inputs_seen_aggregate_former_dashed.pdf"
    fig_ag.savefig(out_ag, bbox_inches="tight")
    plt.close(fig_ag)

    out_cf = None
    if coeff_path.exists():
        df_coeff = pd.read_csv(coeff_path)
        if "date" in df_coeff.columns:
            df_coeff["date"] = pd.to_datetime(df_coeff["date"], errors="coerce")
            coeff_cols = [name for name in feature_names if name in df_coeff.columns]
            if coeff_cols:
                coeff_mask = np.zeros(len(df_coeff), dtype=bool)
                for name in coeff_cols:
                    values = pd.to_numeric(df_coeff[name], errors="coerce").fillna(1.0).to_numpy()
                    coeff_mask |= np.abs(values - 1.0) > 1e-9
                if coeff_mask.any():
                    first_idx = int(np.flatnonzero(coeff_mask)[0])
                    start_ts = df_coeff.loc[first_idx, "date"] - pd.DateOffset(years=2)
                    tail_start_coeff = df_coeff.loc[first_idx, "date"]
                else:
                    start_ts = df_coeff["date"].min()
                    tail_start_coeff = df_coeff["date"].max()
                df_plot = df_coeff[df_coeff["date"] >= start_ts].copy()
                fig_cf, ax_cf = plt.subplots(figsize=(10, 4))
                for i, name in enumerate(coeff_cols):
                    ax_cf.plot(
                        df_plot["date"],
                        pd.to_numeric(df_plot[name], errors="coerce").fillna(1.0),
                        lw=1.8,
                        color=colors[i % len(colors)],
                        label=name,
                    )
                ax_cf.axhline(1.0, color="black", lw=0.9, alpha=0.7, ls="--", label="no correction")
                if coeff_mask.any():
                    ax_cf.axvspan(tail_start_coeff, df_plot["date"].max(), color="tab:red", alpha=0.06)
                ax_cf.xaxis.set_major_locator(mdates.YearLocator(base=1))
                ax_cf.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                ax_cf.set_xlabel("Year")
                ax_cf.set_ylabel("correction multiplier")
                ax_cf.legend(ncol=2, fontsize=8)
                fig_cf.tight_layout()
                out_cf = heatmap_dir / f"{tag}__tail_correction_coefficients.pdf"
                fig_cf.savefig(out_cf, bbox_inches="tight")
                plt.close(fig_cf)

    print(f"[ok] per-feature overlay: {out_pf}")
    print(f"[ok] legacy aggregate: {out_legacy}")
    print(f"[ok] aggregate overlay : {out_ag}")
    if out_cf is not None:
        print(f"[ok] coefficient overlay: {out_cf}")
    print(f"[ok] normalized weights : {weights.tolist()}")

    if args.copy_to_stable:
        stable_dir = args.project_root / "data" / args.usecase / "ar24"
        stable_dir.mkdir(parents=True, exist_ok=True)
        outputs_to_copy = [out_pf, out_legacy, out_ag]
        if out_cf is not None:
            outputs_to_copy.append(out_cf)
        for pdf in outputs_to_copy:
            stable_pdf = stable_dir / pdf.name
            shutil.copy2(pdf, stable_pdf)
            print(f"[ok] copied to stable: {stable_pdf}")


if __name__ == "__main__":
    main()
