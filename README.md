# GNN for Technology Monitoring (GNN for TM)

This repository contains a graph-based pipeline for technology monitoring and
forecasting using OpenAlex-derived bibliometric data. It accompanies the
scientometric manuscript in `docs/gnn_scientometrics.tex`.

## Quickstart (synthetic sample)

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Run the sample pipeline step:

```powershell
python -m src.pipeline_runner --usecase usecase_sample --config config/base.yaml --force
```

Expected outputs:

- `data/usecase_sample/00_sample/outputs/sample_keyword_counts.csv`
- `data/usecase_sample/00_sample/outputs/sample_monthly_counts.csv`
- `data/usecase_sample/00_sample/outputs/sample_summary.json`

The sample uses `data/sample/abstracts.csv` and does not require network access.

## Full pipeline

The full pipeline (OpenAlex extraction -> keywording -> graph build -> GNN
training) is configured via files in `config/usecases/`. Real use cases expect
large datasets and may require GPU support for training. The OpenAlex-derived
dataset is not included in this repository.

Example:

```powershell
python -m src.pipeline_runner --usecase usecase_cyberspace --config config/base.yaml
```

Note: training requires PyTorch + torch-geometric and keyword extraction relies on
KeyBERT + sentence-transformers. Ensure compatible wheels for your platform.

## Reproduce the manuscript runs

See `reproduce.md` for a compact protocol with exact command order and expected
artifacts.

## Reproducibility notes

- Results depend on the OpenAlex snapshot and topic configuration in `config/usecases/`.
- Random seeds are controlled via `params.random_state` and related settings.
- Graph model training is resource-intensive; use the provided configs as a starting point.

## Repository contents

- `src/`: pipeline runner and steps.
- `config/`: base and usecase configurations.
- `data/sample/`: synthetic toy dataset for a quick run.
- `docs/gnn_article.tex`: paper manuscript.

## License

MIT, see `LICENSE`.

## Citation

If you use this code, please cite via `CITATION.cff`.

## Release and archival

For the manuscript release, create a tagged version and archive it on Zenodo:

1. Create GitHub release tag: `v1.0-scientometrics`
2. Publish release on GitHub
3. Archive release via Zenodo to obtain a DOI
4. Add DOI to `CITATION.cff` and manuscript availability statements
