import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = deep_merge(out.get(k), v)
        return out
    return b if b is not None else a


def transform_by_mode_3d(X: np.ndarray, mode: str, eps: float) -> np.ndarray:
    mode = (mode or "").strip().lower()
    if mode in ("raw", "level", "none", ""):
        return X
    if mode == "log":
        return np.log1p(np.maximum(X, 0.0))
    if mode == "logdiff":
        out = np.zeros_like(X, dtype=float)
        if X.shape[0] <= 1:
            return out
        prev = np.maximum(X[:-1], 0.0)
        curr = np.maximum(X[1:], 0.0)
        out[1:] = np.log1p(curr) - np.log1p(prev)
        return out
    if mode == "ratio":
        out = np.zeros_like(X, dtype=float)
        if X.shape[0] <= 1:
            return out
        prev = X[:-1]
        curr = X[1:]
        denom = np.maximum(prev, eps) if eps > 0.0 else prev
        mask = denom > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            out[1:] = np.where(mask, curr / denom, 0.0)
        return out
    if mode == "pct":
        out = np.zeros_like(X, dtype=float)
        if X.shape[0] <= 1:
            return out
        prev = X[:-1]
        curr = X[1:]
        denom = np.maximum(prev, eps) if eps > 0.0 else prev
        mask = denom != 0
        with np.errstate(divide="ignore", invalid="ignore"):
            out[1:] = np.where(mask, (curr - prev) / denom, 0.0)
        return out
    raise ValueError(f"Unknown EMERGENCE_MODE '{mode}'")


def compute_targets(features: np.ndarray, forecast: int, fw: np.ndarray, mode: str):
    T, N, _ = features.shape
    mode = (mode or "").strip().lower()
    if mode == "absolute":
        mode = "level"
    out, masks = [], []
    for t in range(T - forecast):
        cur = features[t]
        fut = features[t + forecast]
        valid = np.isfinite(fut).all(axis=1) & (np.abs(fut).sum(axis=1) > 0)
        if mode == "level":
            y = fut @ fw
        elif mode == "residual":
            y = (fut @ fw) - (cur @ fw)
        elif mode == "log_change":
            y = np.arcsinh(fut @ fw) - np.arcsinh(cur @ fw)
        else:
            raise ValueError("Unknown TARGET_MODE")
        out.append(y)
        masks.append(valid.astype(np.float32))
    return np.stack(out), np.stack(masks).astype(np.float32)


def compute_splits(num_total: int, temp_window: int, split_fracs):
    tr_frac, va_frac, te_frac = split_fracs
    n_train = int(round(tr_frac * num_total))
    n_val = int(round(va_frac * num_total))
    n_test = num_total - n_train - n_val

    def step_range(start, count):
        s = temp_window - 1 + start
        e = s + count
        return np.fromiter(range(s, e), dtype=int)

    train_idx = step_range(0, n_train)
    val_idx = step_range(n_train, n_val)
    test_idx = step_range(n_train + n_val, n_test)
    return train_idx, val_idx, test_idx


def masked_metrics(y_true, y_pred, msk):
    diff = y_pred - y_true
    m = msk > 0
    if diff.ndim == 2:
        diff = diff[m]
    mae = float(np.abs(diff).mean()) if diff.size else float("nan")
    rmse = float(np.sqrt(np.mean(diff ** 2))) if diff.size else float("nan")
    return mae, rmse


def per_node_metrics(y_true, y_pred, msk):
    m = msk > 0
    mae = []
    rmse = []
    count = []
    for n in range(y_true.shape[1]):
        idx = m[:, n]
        if idx.sum() == 0:
            mae.append(float("nan"))
            rmse.append(float("nan"))
            count.append(0)
            continue
        diff = (y_pred[:, n] - y_true[:, n])[idx]
        mae.append(float(np.abs(diff).mean()))
        rmse.append(float(np.sqrt((diff ** 2).mean())))
        count.append(int(idx.sum()))
    return np.array(mae), np.array(rmse), np.array(count)


def main():
    base = load_yaml(ROOT / "config" / "base.yaml")
    uc = load_yaml(ROOT / "config" / "usecases" / "usecase_cyberspace.yaml")
    cfg = deep_merge(base, uc)

    params = cfg.get("params", {})
    graph_params = params.get("graph", {})
    predict_params = params.get("predict", {})
    cfg_defaults = graph_params.get("cfg_defaults") or predict_params.get("cfg_defaults") or {}

    emergence_mode = cfg_defaults.get("EMERGENCE_MODE", "raw")
    epsilon = float(cfg_defaults.get("EPSILON", 1e-8))
    target_mode = cfg_defaults.get("TARGET_MODE", "absolute")
    loss_space = cfg_defaults.get("LOSS_SPACE", "raw")
    temp_window = int(cfg_defaults.get("TEMP_WINDOW", 12))
    split_fracs = tuple(cfg_defaults.get("SPLIT_FRACS", [0.6, 0.2, 0.2]))
    fw = np.asarray(cfg_defaults.get("FW", [0.0, 0.5, 0.5]), dtype=float)

    out_dir = ROOT / "data" / "usecase_cyberspace" / "04_graph" / "outputs"
    mats_path = out_dir / "3_corrected_data" / "stacked_matrices_corrected.npy"
    feats_path = out_dir / "3_corrected_data" / "stacked_features_active_corrected.npy"
    time_path = out_dir / "1_raw_data" / "feature_timestamps.npy"
    names_path = out_dir / "2_active_data" / "keywords_active.txt"

    mats = np.load(mats_path)
    feats = np.load(feats_path)
    timestamps = np.load(time_path, allow_pickle=True)
    try:
        ts = pd.PeriodIndex(timestamps.astype(str), freq="M").to_timestamp(how="start")
    except Exception:
        ts = pd.to_datetime(timestamps, errors="coerce")

    T_aligned = min(len(ts), feats.shape[0], mats.shape[2])
    ts = ts[:T_aligned]
    feats = feats[:T_aligned]
    mats = mats[:, :, :T_aligned]

    Xt = transform_by_mode_3d(feats, emergence_mode, epsilon)

    if names_path.exists():
        node_names = pd.read_csv(names_path, header=None, names=["kw"], dtype=str)["kw"].tolist()
    else:
        node_names = [f"n{i}" for i in range(Xt.shape[1])]

    reports = {}
    for forecast in (12, 24):
        targets, masks = compute_targets(Xt, forecast, fw, target_mode)
        cur_vals = np.array([Xt[t] @ fw for t in range(Xt.shape[0] - forecast)])

        # Persistence baseline in target space
        if (target_mode or "").strip().lower() in ("residual", "log_change"):
            pred = np.zeros_like(targets)
        else:
            pred = cur_vals

        # Build split on windows
        num_total = targets.shape[0] - temp_window + 1
        train_idx, val_idx, test_idx = compute_splits(num_total, temp_window, split_fracs)

        # For consistency with model eval, use the same test indices
        y_test = np.stack([targets[t] for t in test_idx])
        m_test = np.stack([masks[t] for t in test_idx])
        p_test = np.stack([pred[t] for t in test_idx])

        mae, rmse = masked_metrics(y_test, p_test, m_test)
        mae_n, rmse_n, cnt_n = per_node_metrics(y_test, p_test, m_test)

        # Mean abs change per node (how much things move)
        if (target_mode or "").strip().lower() in ("residual", "log_change"):
            abs_change = np.abs(y_test)
        else:
            abs_change = np.abs(y_test - p_test)
        mean_abs_change = []
        for n in range(abs_change.shape[1]):
            idx = m_test[:, n] > 0
            if idx.sum() == 0:
                mean_abs_change.append(float("nan"))
            else:
                mean_abs_change.append(float(abs_change[:, n][idx].mean()))

        df = pd.DataFrame({
            "node": node_names,
            "mae_persist": mae_n,
            "rmse_persist": rmse_n,
            "mean_abs_change": mean_abs_change,
            "count": cnt_n,
        })
        df.sort_values("mae_persist", ascending=False, inplace=True)
        out_csv = out_dir / f"diagnostics_persistence_forecast{forecast}.csv"
        df.to_csv(out_csv, index=False)

        reports[str(forecast)] = {
            "forecast": forecast,
            "temp_window": temp_window,
            "target_mode": target_mode,
            "loss_space": loss_space,
            "overall_mae": mae,
            "overall_rmse": rmse,
            "csv": str(out_csv),
        }

    out_json = out_dir / "diagnostics_persistence_summary.json"
    out_json.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"Saved {out_json}")
    for k, v in reports.items():
        print(f"forecast={k} -> MAE={v['overall_mae']:.6f} RMSE={v['overall_rmse']:.6f}")


if __name__ == "__main__":
    main()
