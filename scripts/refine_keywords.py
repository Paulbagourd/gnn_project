import argparse
from pathlib import Path

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add a removal column to cleaned keyword CSVs and emit refined versions "
            "that exclude rows marked for removal."
        )
    )
    parser.add_argument(
        "--usecase",
        required=True,
        help="Use case name (e.g. usecase_cyberspace).",
    )
    parser.add_argument(
        "--keywords-dir",
        help=(
            "Optional override for the keywords output directory. "
            "Defaults to data/<usecase>/02_keywords/outputs."
        ),
    )
    parser.add_argument(
        "--removal-values",
        nargs="*",
        default=["x"],
        help="Values that count as removal markers (default: %(default)s).",
    )
    parser.add_argument(
        "--config-path",
        help="Optional override for the YAML config path (defaults to config/usecases/<usecase>.yaml).",
    )
    return parser.parse_args()


def normalise_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


def ensure_removal_column(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "removal" not in df.columns:
        df["removal"] = ""
        df.to_csv(path, index=False)
        print(f"[refine_keywords] Added empty 'removal' column to {path}")
    return pd.read_csv(path)


def seed_from_config(df: pd.DataFrame, keyword_col: str, config_path: Path) -> bool:
    if not config_path.exists():
        return False

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    params = cfg.get("params", {}) or {}
    remove_keywords = params.get("remove_keywords", []) or []
    remove_patterns = params.get("remove_patterns", []) or []

    if not remove_keywords and not remove_patterns:
        return False

    keyword_norm = normalise_series(df[keyword_col])
    removal_norm = normalise_series(df["removal"])
    changed = False
    exact_count = 0
    pattern_count = 0

    if remove_keywords:
        removal_set = {str(item).strip().lower() for item in remove_keywords if str(item).strip()}
        if removal_set:
            mask_exact = keyword_norm.isin(removal_set) & removal_norm.eq("")
            exact_count = int(mask_exact.sum())
            if exact_count:
                df.loc[mask_exact, "removal"] = "x"
                removal_norm = normalise_series(df["removal"])
                changed = True

    if remove_patterns:
        mask_pattern = pd.Series(False, index=df.index)
        for pattern in remove_patterns:
            pattern_norm = str(pattern).strip().lower()
            if not pattern_norm:
                continue
            mask_pattern |= keyword_norm.str.contains(pattern_norm, regex=False, na=False)
        mask_pattern = mask_pattern & removal_norm.eq("")
        if mask_pattern.any():
            pattern_count = int(mask_pattern.sum())
            if pattern_count:
                df.loc[mask_pattern, "removal"] = "x"
                changed = True

    if changed:
        print(
            f"[refine_keywords] Seeded removals from {config_path} "
            f"(exact={exact_count}, pattern={pattern_count})"
        )
    return changed


def main() -> None:
    args = parse_args()
    base_dir = Path(args.keywords_dir) if args.keywords_dir else Path("data") / args.usecase / "02_keywords" / "outputs"
    list_path = base_dir / "cleaned_keyword_list.csv"
    graph_path = base_dir / "cleaned_keywords_to_build_graphs.csv"

    if not list_path.exists():
        raise FileNotFoundError(f"Keyword list not found: {list_path}")
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph keyword list not found: {graph_path}")

    df_list = ensure_removal_column(list_path)
    config_path = Path(args.config_path) if args.config_path else Path("config") / "usecases" / f"{args.usecase}.yaml"
    if seed_from_config(df_list, df_list.columns[0], config_path):
        df_list.to_csv(list_path, index=False)
        df_list = pd.read_csv(list_path)

    removal_tokens = {token.strip().lower() for token in args.removal_values if token.strip()}
    removal_mask = normalise_series(df_list["removal"]).isin(removal_tokens)
    removed_keywords = set(normalise_series(df_list.loc[removal_mask, df_list.columns[0]]))

    refined_list_path = list_path.with_name(list_path.stem + "_refined.csv")
    df_list.loc[~removal_mask].to_csv(refined_list_path, index=False)

    df_graph = pd.read_csv(graph_path)
    keyword_col = "Keyword" if "Keyword" in df_graph.columns else df_graph.columns[0]
    refined_graph_path = graph_path.with_name(graph_path.stem + "_refined.csv")
    df_graph_norm = normalise_series(df_graph[keyword_col])
    df_graph.loc[~df_graph_norm.isin(removed_keywords)].to_csv(refined_graph_path, index=False)

    print(f"[refine_keywords] Wrote {refined_list_path.name} ({len(df_list) - removal_mask.sum()} rows kept)")
    print(f"[refine_keywords] Wrote {refined_graph_path.name} ({sum(~df_graph_norm.isin(removed_keywords))} rows kept)")


if __name__ == "__main__":
    main()
