# Data and model-weight licensing

The hybrid stress notebook downloads field data and crop-parameter files at
runtime. Those files are **not** part of the diffwofost package distribution
and are licensed separately from the EUPL-1.1 library code.

## Field-trial data — CC BY-NC-SA 4.0

The notebook downloads three files:

- `Plotspecific_processed.csv`
- `Weatherfile_lelystad.xlsx`
- `Weatherfile_vredepeel.xlsx`

These are part of the dataset:

> Ten Den, T., van de Wiel, I., van Evert, F., van Ittersum, M., de Wit, A.,
> & Reidsma, P. (2024). *Agronomic dataset on potato growth and yield in the
> Netherlands.* Harvard Dataverse.
> [https://doi.org/10.7910/DVN/1LC6W7](https://doi.org/10.7910/DVN/1LC6W7)

Licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

The notebook fetches a public GitHub mirror of those files rather than hitting
Dataverse from Colab (Dataverse can be flaky). The mirroring is permitted by
CC BY-NC-SA 4.0 provided attribution and the same licence are preserved. Any
further use must comply with the upstream licence (attribute the authors, no
commercial use, derivative works share alike).

## Pre-trained model weights — CC BY-NC-SA 4.0

Checkpoints in [`pretrained/`](pretrained/) (`stress_nn_year.pt` and
`pure_lstm_year.pt`) are trained on the field-trial dataset above and are
therefore derivative works under the same
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) licence.

## PCSE and WOFOST crop files — Apache-2.0

`Wofost72_PP.conf` is fetched from [`ajwdewit/pcse`](https://github.com/ajwdewit/pcse).
Per-cultivar potato parameters (`potato.yaml`) come from the `wofost72` branch of
[`ajwdewit/WOFOST_crop_parameters`](https://github.com/ajwdewit/WOFOST_crop_parameters).
Both are licensed under Apache License 2.0.

## Code (notebook + `hybrid_stress.py`) — EUPL-1.1

See the repository [LICENSE](../../LICENSE).
