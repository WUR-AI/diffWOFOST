# Pretrained hybrid-stress checkpoints

These weights ship with `hybrid_stress_correction.ipynb` so the tutorial can
run without a multi-hour training job.

They are trained on Ten Den et al. (2024) and are licensed **CC BY-NC-SA 4.0**.
See [`DATA_LICENSE.md`](../DATA_LICENSE.md).

| File | Model | Split |
|------|-------|-------|
| `stress_nn_year.pt` | StressNN inside WOFOST (273 parameters) | train 2019 / test 2020 |
| `pure_lstm_year.pt` | Pure LSTM baseline (no physics) | same |

The notebook loads these by default. Set `FORCE_RETRAIN = True` to train from
scratch; retrained copies are written to `data_temp/trained_models/` (gitignored)
and do not overwrite this folder.
