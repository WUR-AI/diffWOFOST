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

# Pretrained irrigation-control checkpoints

These weights ship with `differentiable_irrigation_control.ipynb` so the STE
and PPO sections can run without retraining (~minutes for STE, ~45–70 min for PPO).

| File | Model | Notes |
|------|-------|-------|
| `mlp_weekly_best.pt` | STE irrigation MLP (weekly train) | best hard $R$ on 2010 YAML |
| `ppo_weekly_stress.pt` | From-scratch PPO actor–critic | stress-shaped train; best hard $R$ on 2010 |

The notebook loads `pretrained/` when `data_temp/` has no matching checkpoint.
Set `PPO_LOAD_IF_AVAILABLE = False` (or delete the local `data_temp` copy) to
retrain; new weights are written under `data_temp/` (gitignored) and do not
overwrite this folder.

