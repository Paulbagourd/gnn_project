# Reproducibility Protocol (Scientometrics Submission)

This document records the minimal command sequence used to run the pipeline and regenerate core artifacts.

## 1) Environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Sanity check on synthetic sample

```powershell
python -m src.pipeline_runner --usecase usecase_sample --config config/base.yaml --force
```

Expected files:

- `data/usecase_sample/00_sample/outputs/sample_keyword_counts.csv`
- `data/usecase_sample/00_sample/outputs/sample_monthly_counts.csv`
- `data/usecase_sample/00_sample/outputs/sample_summary.json`

## 3) Main cyberspace workflow

Run steps with the curated use-case configuration:

```powershell
python -m src.pipeline_runner --usecase usecase_cyberspace --config config/base.yaml
```

Depending on configuration and data snapshot, this orchestrates extraction, refinement, graph construction, training, and forecasting stages.

## 4) Build manuscript

```powershell
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error gnn_scientometrics.tex
```

## Notes

- Results depend on the OpenAlex snapshot date and API state.
- The full OpenAlex-derived corpus is not redistributed in this repository.
- Randomness is controlled through config parameters (for example, `params.random_state`) but should be evaluated with multi-seed runs for statistical robustness.
