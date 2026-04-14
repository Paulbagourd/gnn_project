# LLM Dry-Run Playbook (Before EPFL Meeting)

This playbook gives a concrete, reproducible procedure to test an LLM baseline against your GNN setup before the meeting.

## 1) Goal of the dry-run

Run an **offline LLM baseline** (no web/tools) on the same frozen split and targets as your GNN protocol, then compare:

- RMSE (primary), MAE (secondary)
- horizons: 1y / 2y / 3y / 4y
- both target views: absolute and relative

The objective is not to "win", but to validate that your comparison pipeline is fair and stable.

## 2) Ground rules (must keep)

- Use the **same test period** for all methods.
- Never expose test labels in prompts.
- Use the same target definitions as in `docs/protocol_gnn_llm_comparison.tex`.
- Keep a full run log: prompts, model/version, temperature, timestamps.

## 3) Minimal experiment matrix

Run these baselines in this order:

1. `Persistence` baseline (last value / drift proxy).
2. `Your best GNN` on frozen test.
3. `LLM-offline` (no web/tools), 3 to 5 repeated runs.
4. Optional: `LLM-open` with leakage audit logs.

If LLM is unstable, keep the offline result for the meeting and present open mode as exploratory.

## 4) Recommended LLM settings

- Temperature: `0.0` to `0.2`
- Top-p: `1.0`
- Max output tokens: enough for strict JSON only
- No chain-of-thought request in prompt
- Enforce structured output (JSON schema or strict format instruction)

## 5) Prompt template (offline)

Use this as your system/task prompt and fill placeholders.

```text
You are a forecasting model. Predict keyword emergence scores.
Important constraints:
- Use ONLY the provided historical data up to cutoff date {CUTOFF_DATE}.
- Do NOT use external knowledge, web, or tools.
- Return ONLY valid JSON matching the requested schema.

Target definitions:
- abs_{h} = E(t+{h}) - E(t)
- rel_{h} = (E(t+{h}) - E(t)) / (E(t) + epsilon), epsilon={EPSILON}

For each keyword, infer abs and rel targets for horizons h in {12,24,36,48} months.

Input payload:
{INPUT_JSON}
```

## 6) Input JSON template (one batch)

```json
{
  "cutoff_date": "2020-04-01",
  "epsilon": 1e-8,
  "horizons_months": [12, 24, 36, 48],
  "keywords": [
    {
      "keyword": "example_keyword",
      "history": {
        "months": ["2018-01", "2018-02", "..."],
        "fw_aggregate": [0.02, 0.03, "..."],
        "oc_freq": [12, 11, "..."],
        "edge_weight": [0.4, 0.39, "..."],
        "xcum_frac_split": [0.1, 0.11, "..."]
      }
    }
  ]
}
```

Keep feature names aligned with your graph pipeline (cyberspace config uses `xcum_frac_split`, `oc_freq`, `edge_weight` in several settings).

## 7) Required LLM output JSON schema

```json
{
  "run_id": "llm_offline_run_001",
  "model_id": "your-model-name",
  "cutoff_date": "2020-04-01",
  "predictions": [
    {
      "keyword": "example_keyword",
      "abs_12": 0.0,
      "abs_24": 0.0,
      "abs_36": 0.0,
      "abs_48": 0.0,
      "rel_12": 0.0,
      "rel_24": 0.0,
      "rel_36": 0.0,
      "rel_48": 0.0
    }
  ]
}
```

No text explanation in output. JSON only.

## 8) Step-by-step execution

1. Freeze your split and horizons.
2. Export the LLM input table from train/val-known history only (up to cutoff `t`).
3. Build one JSON batch per prompt call (small enough for context size).
4. Run LLM offline once to validate format parsing.
5. Run LLM offline 3-5 times (same prompt, same data) to measure variance.
6. Evaluate each run with your common script (RMSE/MAE per horizon, macro average, abs/rel separately).
7. Aggregate results across runs: mean and std of RMSE/MAE.
8. Compare against GNN and persistence baseline in one final table.

## 9) Leakage audit checklist (for optional LLM-open)

- Save all tool/web calls with timestamps.
- Save URLs and snippets retrieved.
- Flag any source containing future aggregates equivalent to labels.
- If leakage suspected: invalidate run, tighten source constraints, rerun.

## 10) Meeting-ready result table template

Use one row per method:

- Method
- Access mode (`offline` / `open`)
- RMSE abs macro
- RMSE rel macro
- MAE abs macro
- MAE rel macro
- RMSE by horizon (1y,2y,3y,4y)
- Validity flag (`valid` / `invalid-leakage`)
- Notes (stability, reproducibility)

## 11) What to say if asked "old local LLM vs modern API?"

Short answer:

- A local old model is useful as a reproducible baseline.
- But for scientific comparison quality, protocol control and leakage audit matter more than model age.
- So start with offline controlled runs, then optionally add open-mode runs with strict logging.

## 12) GNN checkpoints + fast multi-cutoff inference

If you want true "inference only" (no retraining for each cutoff), use:

- `scripts/run_gnn_checkpoint_workflow.py`

### A) Train checkpoints once (per horizon)

```powershell
python scripts/run_gnn_checkpoint_workflow.py --phase train --forecasts 12,24,36,48
```

This creates reusable checkpoints under:

- `data/usecase_cyberspace/gnn_llm_comparison/gnn_checkpoints/f12`
- `data/usecase_cyberspace/gnn_llm_comparison/gnn_checkpoints/f24`
- `data/usecase_cyberspace/gnn_llm_comparison/gnn_checkpoints/f36`
- `data/usecase_cyberspace/gnn_llm_comparison/gnn_checkpoints/f48`

By default, the script reuses precomputed graph tensors from:

- `data/usecase_cyberspace/04_build_graph/outputs`

so it avoids rebuilding the full graph tensors for each horizon.

### B) Run inference over frozen 19 cutoffs from checkpoints

```powershell
python scripts/run_gnn_checkpoint_workflow.py --phase infer --forecasts 12,24,36,48
```

Outputs are collected in:

- `data/usecase_cyberspace/gnn_llm_comparison/gnn_cutoff_csvs_from_ckpt/f12/GraphModel/predictions_YYYY-MM.csv`
- `data/usecase_cyberspace/gnn_llm_comparison/gnn_cutoff_csvs_from_ckpt/f12/NoGraphModel/predictions_YYYY-MM.csv`
- same for `f24`, `f36`, `f48`.

### C) Important: dry-run vs real run

- With `--dry-run`: prints commands only, executes nothing.
- Without `--dry-run`: executes training/inference and writes outputs/logs.

