"""Restart the AgMIP Adam and Nelder-Mead searches.

Adam uses the start counts from Wallach et al. (2024). Two or more major
parameters get 20 Latin-hypercube starts, one major parameter gets 5, and
each candidate gets 5. Already kept parameters start at their previous best
on every candidate start, and one of those starts is that previous best
itself. The weighted refit gets the group-wise result plus 19 further
Latin-hypercube starts. The learning rate begins at 0.25 for phenology and
0.12 for later groups. After three steps with no new best the search returns
to that best step, drops Adam's momentum, and halves the rate, up to four
times. At the smallest rate it stops after ten quiet steps.

Nelder-Mead keeps the budget that already converged for phenology: 20 starts
for TSUM1 and TSUM2, 5 for each candidate, ceiling 400 and 120 evaluations.
Growth uses 5 starts and stops when the simplex settles, with a ceiling of
120 evaluations (60 for one parameter).

Checkpoints are written next to this file so a reboot keeps them.
"""

import json
import math
import multiprocessing as mp
import os
import time
import traceback
import zlib
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch

torch.set_num_threads(1)

ROOT = Path(os.environ.get("DIFFWOFOST_ROOT", Path(__file__).resolve().parent))
CALIBRATION = ROOT / "agmip_calibration.py"
CHECKPOINT = Path(os.environ.get("ADAM_CHECKPOINT", ROOT / "agmip_retrain_checkpoint.json"))
ADAM_ONLY = os.environ.get("ADAM_ONLY") == "1"
NM_ONLY = os.environ.get("NM_ONLY") == "1"
WEIGHTED_ONLY = os.environ.get("WEIGHTED_ONLY") == "1"
NM_WEIGHTED_ONLY = os.environ.get("NM_WEIGHTED_ONLY") == "1"
WORKERS = int(os.environ.get("ADAM_WORKERS", "2" if ADAM_ONLY else "4"))
PHEN_SEED = 20241022
GROWTH_SEED = 20241024

ns = {"__name__": "__main__"}


def calibration_section(name):
    """One section of agmip_calibration.py. The notebook does not contain this search."""
    text = CALIBRATION.read_text()
    token = f"# %% {name}\n"
    start = text.index(token) + len(token)
    end = text.find("\n# %% ", start)
    return text[start:] if end < 0 else text[start:end]


def checkpoint(payload):
    payload["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = CHECKPOINT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(CHECKPOINT)


def as_active(values):
    return {name: torch.tensor(float(value), dtype=torch.float64) for name, value in values.items()}


def phenology_ss(values):
    with torch.no_grad():
        return float(ns["sum_of_squares"](as_active(values)))


def logit_of(names, values):
    z = []
    for name in names:
        low, high = ns["BOUNDS"][name]
        fraction = (float(values[name]) - low) / (high - low)
        fraction = min(max(fraction, 1.0e-6), 1.0 - 1.0e-6)
        z.append(float(np.log(fraction / (1.0 - fraction))))
    return np.asarray(z, dtype=float)


def values_of(names, z, fixed):
    merged = dict(fixed)
    for name, zi in zip(names, z):
        low, high = ns["BOUNDS"][name]
        fraction = 1.0 / (1.0 + np.exp(-float(zi)))
        merged[name] = low + (high - low) * fraction
    return merged


def latin_starts(names, center, n_extra, rng):
    starts = [{name: float(center[name]) for name in names}]
    if n_extra <= 0:
        return starts
    cube = np.zeros((n_extra, len(names)))
    for column, name in enumerate(names):
        slots = (np.arange(n_extra) + rng.random(n_extra)) / n_extra
        rng.shuffle(slots)
        low, high = ns["BOUNDS"][name]
        cube[:, column] = low + slots * (high - low)
    for row in cube:
        starts.append({name: float(value) for name, value in zip(names, row)})
    return starts


def nelder_job(job):
    torch.set_num_threads(1)
    try:
        return _nelder_job(job)
    except Exception:
        return {"index": job["index"], "error": traceback.format_exc()}


def _nelder_job(job):
    names = job["names"]
    fixed = job["fixed"]
    start = job["start"]
    kind = job["kind"]
    calls = {"n": 0, "best": None, "trace": [], "first": None}

    def objective(z):
        calls["n"] += 1
        values = values_of(names, z, fixed)
        if kind == "phenology":
            score = phenology_ss(values)
        elif kind == "weighted":
            score = fused_weighted_ss(values, job["variances"])
        else:
            score = float(ns["evaluate_kind"](kind, values))
        calls["trace"].append(score)
        if calls["first"] is None:
            calls["first"] = score
        if calls["best"] is None or score < calls["best"][0]:
            calls["best"] = (score, {name: float(values[name]) for name in names})
        if calls["n"] == 1 or calls["n"] % 10 == 0:
            print(
                f"  {job['label']} start {job['index']} eval {calls['n']} SS {score:.6g}",
                flush=True,
            )
        return score

    fatol = job["fatol"]
    if fatol is None:
        fatol = max(1.0e-4, 0.005 * abs(objective(logit_of(names, start))))
    t0 = time.perf_counter()
    result = ns["minimize"](
        objective,
        logit_of(names, start),
        method="Nelder-Mead",
        options={
            "maxfev": job["maxfev"],
            "xatol": job["xatol"],
            "fatol": fatol,
            "adaptive": True,
        },
    )
    score, found = calls["best"]
    return {
        "index": job["index"],
        "previous_best_start": job["index"] == 1,
        "start": start,
        "final": found,
        "ss": score,
        "evaluations": calls["n"],
        "seconds": time.perf_counter() - t0,
        "met_stopping_test": bool(result.success),
        "message": str(result.message),
        "fatol": fatol,
        "trace": calls["trace"],
    }


def round_rng(key):
    """One stream per round, so a resume does not redraw its starts."""
    seed = 20241025 + (zlib.crc32(key.encode("utf-8")) % 1_000_000)
    return np.random.default_rng(seed)


def n_protocol_starts(n_params, role):
    """Start counts from Wallach et al. (2024), section 2."""
    if role == "weighted":
        return 20
    if role == "candidate":
        return 5
    return 20 if n_params > 1 else 5


def candidate_starts(names, center, new_name, rng, n=5):
    """Five starts. Kept parameters stay at their previous best."""
    starts = [{name: float(center[name]) for name in names}]
    for extra in latin_starts([new_name], {new_name: center[new_name]}, n - 1, rng)[1:]:
        point = {name: float(center[name]) for name in names}
        point[new_name] = extra[new_name]
        starts.append(point)
    return starts


def kind_ss(season, output, kind):
    """Same residuals as the notebook, scored from one simulation."""
    if kind == "biomass":
        total = output["TAGP"].new_zeros(())
        for day, observed in season["biomass"]:
            if day < 0 or day >= output["TAGP"].shape[0]:
                continue
            total = total + (torch.log(output["TAGP"][day].clamp(min=1.0)) - math.log(observed)).square()
        return total
    if kind == "yield":
        if season["yield_gm2"] is None:
            return output["TWSO"].new_zeros(())
        return (output["TWSO"][-1] / 10.0 - season["yield_gm2"]).square()
    if kind == "nitrogen":
        if season["nitrogen"] is None:
            return output["TAGP"].new_zeros(())
        crop_n = output["NamountLV"][-1] + output["NamountST"][-1] + output["NamountSO"][-1]
        return (100.0 * crop_n / output["TAGP"][-1].clamp(min=1.0) - season["nitrogen"]).square()
    if season["protein"] is None:
        return output["TWSO"].new_zeros(())
    predicted = 100.0 * ns["PROTEIN_FACTOR"] * output["NamountSO"][-1] / output["TWSO"][-1].clamp(min=1.0)
    return (predicted - season["protein"]).square()


def phenology_from_run(season, output):
    """Heading and maturity errors from the full model's development stage."""
    if season.get("heading") is None or season.get("maturity") is None:
        return output["DVS"].new_zeros(())
    heading = ns["stage_error"](output["DVS"], 1.0, season["heading"])
    maturity = ns["stage_error"](output["DVS"], 2.0, season["maturity"])
    return heading.square() + maturity.square()


def fused_weighted_ss(values, variances):
    """One full-model run per season. Phenology is read from that same run."""
    ns["current"] = {name: float(value) for name, value in values.items()}
    active = {
        name: torch.tensor(float(values[name]), dtype=torch.float64)
        for name in ns["BOUNDS"]
        if name in values
    }
    merged = ns["merged"](active)
    total = 0.0
    for season in ns["seasons"]:
        output = ns["simulate"](season, merged)
        total += float(phenology_from_run(season, output)) / variances["phenology"]
        for spec in ns["GROUPS"]:
            total += float(kind_ss(season, output, spec["kind"])) / variances[spec["kind"]]
    return float(total)


def adam_loop(names, start, lr0, loss_of, max_steps, label=""):
    """Adam with the notebook learning-rate schedule. One start."""
    pack = ns["ParameterSet"](names, start)
    lr = lr0
    optimizer = torch.optim.Adam(pack.parameters(), lr=lr)
    best_ss = float("inf")
    best_state = {key: value.detach().clone() for key, value in pack.state_dict().items()}
    waited, cuts, rewound = 0, 0, False
    trace = []
    stopped = "ceiling"
    t0 = time.perf_counter()
    for step in range(max_steps):
        optimizer.zero_grad()
        loss = loss_of(pack())
        score = float(loss.detach())
        trace.append({"ss": score, "lr": lr})
        waited, cuts, rewound, action = ns["rate_decision"](score, best_ss, waited, cuts, rewound)
        if action == "improve":
            best_ss = score
            best_state = {key: value.detach().clone() for key, value in pack.state_dict().items()}
            mark = "best"
        elif action == "decay":
            pack.load_state_dict(best_state)
            lr *= ns["LR_FACTOR"]
            optimizer = torch.optim.Adam(pack.parameters(), lr=lr)
            print(f"  {label} learning rate {lr:.4g}. Back at the best step.", flush=True)
            continue
        elif action == "stop":
            stopped = "patience"
            print(f"  {label} early stopping at step {step}.", flush=True)
            break
        else:
            mark = f"no gain for {waited}"
        print(
            f"  {label} step {step + 1}/{max_steps} lr {lr:.4g} SS {score:.6g}. {mark}.",
            flush=True,
        )
        loss.backward()
        optimizer.step()
    else:
        print(f"  {label} reached the ceiling of {max_steps} steps.", flush=True)
    pack.load_state_dict(best_state)
    with torch.no_grad():
        values = {name: float(value) for name, value in pack().items()}
        final = float(loss_of(pack()))
    return {
        "parameters": list(names),
        "values": values,
        "ss": final,
        "trace": trace,
        "stopped": stopped,
        "learning_rate_cuts": cuts,
        "seconds": time.perf_counter() - t0,
    }


def adam_job(job):
    torch.set_num_threads(1)
    try:
        fixed = dict(job["fixed"])
        ns["current"] = dict(fixed)
        ns["accepted"] = {name: fixed[name] for name in phenology_parameters() if name in fixed}
        kind = job["kind"]
        if kind == "phenology":
            loss_of = phenology_loss
        elif kind == "weighted":
            variances = job["variances"]

            def loss_of(active, variances=variances):
                merged = ns["merged"](active)
                total = None
                for season in ns["seasons"]:
                    output = ns["simulate"](season, merged)
                    piece = phenology_from_run(season, output) / variances["phenology"]
                    for spec in ns["GROUPS"]:
                        piece = piece + kind_ss(season, output, spec["kind"]) / variances[spec["kind"]]
                    total = piece if total is None else total + piece
                return total

        else:
            loss_of = growth_loss(kind)
        fit = adam_loop(
            job["names"],
            {**fixed, **job["start"]},
            job["lr"],
            loss_of,
            job["max_steps"],
            label=job["label"],
        )
        fit["index"] = job["index"]
        fit["previous_best_start"] = job["index"] == 1
        fit["start"] = {name: float(job["start"][name]) for name in job["names"]}
        return fit
    except Exception:
        return {"index": job.get("index"), "error": traceback.format_exc()}


def run_adam_round(
    pool,
    payload,
    bucket,
    *,
    key,
    group,
    names,
    role,
    center,
    fixed,
    kind,
    lr,
    max_steps,
    slots,
    n_obs=None,
    new_name=None,
    variances=None,
    keep_below=None,
):
    """Run one Wallach round. A finished start is checkpointed on its own."""
    row = next((item for item in bucket if item.get("trial_key") == key), None)
    if row is None:
        count = n_protocol_starts(len(names), role)
        rng = round_rng(key)
        if role == "candidate":
            starts = candidate_starts(names, center, new_name, rng, count)
        else:
            starts = latin_starts(names, center, count - 1, rng)
        row = {
            "trial_key": key,
            "group": group,
            "role": role,
            "rule": "wallach",
            "names": list(names),
            "starts": starts,
            "runs": [],
            "complete": False,
        }
        bucket.append(row)
        checkpoint(payload)
    if row.get("complete"):
        return row
    done = {run["index"] for run in row["runs"]}
    jobs = []
    for index, start in enumerate(row["starts"], start=1):
        if index in done:
            continue
        jobs.append(
            {
                "index": index,
                "label": f"Adam {group} start {index}/{len(row['starts'])} {list(names)}",
                "names": list(names),
                "start": start,
                "fixed": dict(fixed),
                "kind": kind,
                "lr": lr,
                "max_steps": max_steps,
                "variances": variances,
            }
        )
    print(
        f"Adam {group} {list(names)} x {len(row['starts'])}, {len(jobs)} still to run",
        flush=True,
    )
    pending = []
    queue = list(jobs)
    while queue or pending:
        while queue and len(pending) < slots:
            pending.append(pool.apply_async(adam_job, (queue.pop(0),)))
        still = []
        for item in pending:
            if not item.ready():
                still.append(item)
                continue
            result = item.get()
            if result.get("error"):
                checkpoint(payload)
                raise RuntimeError(result["error"])
            row["runs"].append(result)
            checkpoint(payload)
            print(
                f"  Adam {group} start {result['index']} SS {result['ss']:.6g} via {result['stopped']}",
                flush=True,
            )
        pending = still
        if pending:
            time.sleep(2)
    best = min(row["runs"], key=lambda run: run["ss"])
    score = None if n_obs is None else aicc_of(best["ss"], n_obs, len(names))
    row.update(
        {
            "complete": True,
            "ss": best["ss"],
            "aicc": score,
            "values": best["values"],
            "best_was_previous_start": best["previous_best_start"],
            "n_starts": len(row["runs"]),
            "n_patience": sum(run["stopped"] == "patience" for run in row["runs"]),
            "stopped": best["stopped"],
            "parameters": list(names),
            "seconds": sum(run["seconds"] for run in row["runs"]),
            "kept": True if keep_below is None else score < keep_below,
        }
    )
    checkpoint(payload)
    print(
        f"  Adam {group} best start {best['index']} SS {best['ss']:.6g} "
        f"AICc {score if score is not None else float('nan'):.2f} kept {row['kept']}",
        flush=True,
    )
    return row


def restore_weighted_start(payload, defaults):
    """Kept, finished rounds only. An open candidate is left out."""
    adam_values = dict(defaults)
    adam_values.update(payload["phenology"]["adam"]["values"])
    adam_selected = list(payload["phenology"]["adam"]["selected"])
    finished = {
        row["trial_key"]: row
        for row in payload["growth"]["adam"]
        if row.get("complete") and row.get("rule") == "wallach"
    }
    adam_variance = {}
    for spec in ns["GROUPS"]:
        kind = spec["kind"]
        n_obs = ns["observation_count"](kind)
        major = [name for name in spec["major"] if name not in adam_selected]
        row = finished.get(spec["name"] + ":major:" + ",".join(major))
        if row is None:
            raise SystemExit(f"weighted refit is missing a finished {spec['name']} major round")
        adam_values.update(row["values"])
        chosen = list(major)
        best = row["ss"]
        for candidate in spec["candidates"]:
            if candidate in adam_selected or candidate in chosen:
                continue
            trial = chosen + [candidate]
            row = finished.get(spec["name"] + ":candidate:" + ",".join(trial))
            if row is None:
                break
            if row.get("kept"):
                adam_values.update(row["values"])
                chosen = list(trial)
                best = row["ss"]
        adam_selected.extend(name for name in chosen if name not in adam_selected)
        denom = n_obs - len(chosen)
        adam_variance[kind] = (best / denom) if denom > 0 else best / n_obs
        print(
            f"  {spec['name']} SS {best:.6g} kept {chosen}",
            flush=True,
        )
    print("  refit " + ", ".join(f"{name} {adam_values[name]:.4g}" for name in adam_selected), flush=True)
    return adam_selected, adam_values, adam_variance


def restore_nm_weighted_start(payload, defaults):
    """Nelder-Mead's kept rounds. Weights come from those rounds, not from Adam."""
    values = dict(defaults)
    values.update(payload["phenology"]["nelder_mead"]["values"])
    selected = list(payload["phenology"]["nelder_mead"]["selected"])
    rounds = {row["trial_key"]: row for row in payload["growth"]["nelder_mead"]}
    kept_pheno = [row for row in payload["phenology"]["nelder_mead"]["rounds"] if row.get("kept")]
    pheno = kept_pheno[-1]
    n_pheno = payload["phenology"]["n_observations"]
    variance = {
        "phenology": pheno["best_ss"] / max(n_pheno - len(pheno["names"]), 1),
    }
    print(f"  phenology SS {pheno['best_ss']:.6g} kept {pheno['names']}", flush=True)
    for spec in ns["GROUPS"]:
        kind = spec["kind"]
        n_obs = ns["observation_count"](kind)
        major = [name for name in spec["major"] if name not in selected]
        row = rounds.get(spec["name"] + ":major:" + ",".join(major))
        if row is None:
            raise SystemExit(f"Nelder-Mead weighted refit is missing a finished {spec['name']} major round")
        values.update(row["best_final"])
        chosen = list(major)
        best = row["best_ss"]
        for candidate in spec["candidates"]:
            if candidate in selected or candidate in chosen:
                continue
            trial = chosen + [candidate]
            row = rounds.get(spec["name"] + ":candidate:" + ",".join(trial))
            if row is None:
                break
            if row.get("kept"):
                values.update(row["best_final"])
                chosen = list(trial)
                best = row["best_ss"]
        selected.extend(name for name in chosen if name not in selected)
        denom = n_obs - len(chosen)
        variance[kind] = (best / denom) if denom > 0 else best / n_obs
        print(f"  {spec['name']} SS {best:.6g} kept {chosen}", flush=True)
    print("  refit " + ", ".join(f"{name} {values[name]:.4g}" for name in selected), flush=True)
    return selected, values, variance


def run_nm_weighted(pool, payload, names, center, fixed, variances):
    """Nelder-Mead on every selected parameter. The first start is the group-wise result."""
    bucket = payload.setdefault("weighted_nm_rounds", [])
    row = next((item for item in bucket if item.get("trial_key") == "weighted-nm"), None)
    if row is None:
        extra = int(os.environ.get("NM_WEIGHTED_EXTRA", "19"))
        starts = latin_starts(names, {name: center[name] for name in names}, extra, round_rng("weighted-nm"))
        row = {
            "trial_key": "weighted-nm",
            "names": list(names),
            "starts": starts,
            "runs": [],
            "complete": False,
        }
        bucket.append(row)
        checkpoint(payload)
    done = {run["index"] for run in row["runs"]}
    jobs = []
    for index, start in enumerate(row["starts"], start=1):
        if index in done:
            continue
        jobs.append(
            {
                "index": index,
                "label": "Nelder-Mead weighted",
                "names": list(names),
                "start": start,
                "fixed": dict(fixed),
                "kind": "weighted",
                "variances": variances,
                "maxfev": 200,
                "xatol": 0.03,
                "fatol": None,
            }
        )
    print(
        f"Nelder-Mead weighted {list(names)} x {len(row['starts'])}, {len(jobs)} still to run",
        flush=True,
    )
    pending = []
    queue = list(jobs)
    while queue or pending:
        while queue and len(pending) < WORKERS:
            pending.append(pool.apply_async(nelder_job, (queue.pop(0),)))
        still = []
        for item in pending:
            if not item.ready():
                still.append(item)
                continue
            result = item.get()
            if result.get("error"):
                checkpoint(payload)
                raise RuntimeError(result["error"])
            row["runs"].append(result)
            checkpoint(payload)
            print(
                f"  finished {result['index']} SS {result['ss']:.6g} "
                f"evals {result['evaluations']} stop {result['met_stopping_test']}",
                flush=True,
            )
        pending = still
        if pending:
            time.sleep(2)
    best = min(row["runs"], key=lambda run: run["ss"])
    row.update(
        {
            "complete": True,
            "best_ss": best["ss"],
            "best_final": best["final"],
            "best_was_previous_start": best["previous_best_start"],
            "n_starts": len(row["runs"]),
            "n_met_stopping_test": sum(bool(run["met_stopping_test"]) for run in row["runs"]),
            "evaluations": sum(run["evaluations"] for run in row["runs"]),
            "seconds": sum(run["seconds"] for run in row["runs"]),
        }
    )
    checkpoint(payload)
    print(f"  Nelder-Mead weighted best start {best['index']} SS {best['ss']:.6g}", flush=True)
    return row


def phenology_loss(active):
    return ns["sum_of_squares"](active)


def growth_loss(kind):
    def loss_of(active):
        total = None
        for season in ns["seasons"]:
            piece = ns["season_ss"](season, ns["merged"](active), kind)
            total = piece if total is None else total + piece
        return total

    return loss_of


def aicc_of(ss, n_obs, n_param):
    return float(ns["aicc"](torch.tensor(float(ss), dtype=torch.float64), n_obs, n_param))


def launch(pool, jobs):
    return [pool.apply_async(nelder_job, (job,)) for job in jobs]


def collect(pending, payload):
    runs = []
    while pending:
        still = []
        for item in pending:
            if item.ready():
                result = item.get()
                if result.get("error"):
                    checkpoint(payload)
                    raise RuntimeError(result["error"])
                runs.append(result)
                print(
                    f"  finished {result['index']} SS {result['ss']:.6g} "
                    f"evals {result['evaluations']} stop {result['met_stopping_test']}",
                    flush=True,
                )
            else:
                still.append(item)
        pending = still
        if pending:
            time.sleep(2)
    return runs


def summarise_round(runs, names, role, n_obs):
    best = min(runs, key=lambda run: run["ss"])
    score = aicc_of(best["ss"], n_obs, len(names))
    return {
        "names": list(names),
        "role": role,
        "trial": list(names),
        "aicc": score,
        "best_ss": best["ss"],
        "best_was_previous_start": best["previous_best_start"],
        "best_hit_evaluation_ceiling": not best["met_stopping_test"],
        "n_starts": len(runs),
        "n_met_stopping_test": sum(run["met_stopping_test"] for run in runs),
        "seconds": sum(run["seconds"] for run in runs),
        "evaluations": sum(run["evaluations"] for run in runs),
        "best_final": best["final"],
        "runs": runs,
    }


def phenology_parameters():
    return ("TSUM1", "TSUM2", "VERNSAT", "VERNBASE", "DLO")


def main():
    print("loading calibration model and seasons", flush=True)
    from scipy.optimize import minimize

    ns["minimize"] = minimize
    setup = calibration_section("setup")
    if os.environ.get("AGMIP_DATA"):
        cached = '''def download(name):
    import os
    from pathlib import Path
    path = Path(os.environ["AGMIP_DATA"]) / name
    if path.is_file():
        return path.read_bytes()
    with urllib.request.urlopen(f"{DATA_URL}/{name}") as response:
        return response.read()
'''
        needle = '''def download(name):
    with urllib.request.urlopen(f"{DATA_URL}/{name}") as response:
        return response.read()
'''
        if needle not in setup:
            raise SystemExit("download() changed; cannot use a local AgMIP cache")
        setup = setup.replace(needle, cached)
    exec(compile(setup, "setup", "exec"), ns)
    exec(compile(calibration_section("environments"), "environments", "exec"), ns)
    exec(compile(calibration_section("phenology"), "phenology", "exec"), ns)
    ns["accepted"] = {name: ns["defaults"][name] for name in phenology_parameters()}
    exec(compile(calibration_section("growth"), "growth", "exec"), ns)

    defaults = {name: float(ns["arminda_values"][name]) for name in ns["arminda_values"]}
    print(
        f"loaded. phenology lr {ns['PHENO_LR']} growth lr {ns['GROUP_LR']} "
        f"cuts {ns['LR_CUTS']} patience {ns['PATIENCE']}",
        flush=True,
    )

    print("smoke: one phenology evaluation", flush=True)
    t0 = time.perf_counter()
    base = phenology_ss(defaults)
    print(f"  phenology SS {base:.2f} in {time.perf_counter() - t0:.1f}s", flush=True)
    ctx = mp.get_context("fork")
    pool = ctx.Pool(WORKERS)
    smoke = pool.apply(
        nelder_job,
        (
            {
                "index": 1,
                "label": "smoke",
                "names": ["TSUM1", "TSUM2"],
                "start": {"TSUM1": defaults["TSUM1"], "TSUM2": defaults["TSUM2"]},
                "fixed": defaults,
                "kind": "phenology",
                "maxfev": 2,
                "xatol": 0.03,
                "fatol": 1.0,
            },
        ),
    )
    if smoke.get("error"):
        raise RuntimeError(smoke["error"])
    print("smoke: one biomass evaluation", flush=True)
    t0 = time.perf_counter()
    biomass = float(ns["evaluate_kind"]("biomass", defaults))
    print(f"  biomass SS {biomass:.4f} in {time.perf_counter() - t0:.1f}s", flush=True)
    dated = [
        season for season in ns["seasons"]
        if season.get("heading") is not None and season.get("maturity") is not None
    ]
    print(f"  seasons with heading and maturity: {len(dated)} / {len(ns['seasons'])}", flush=True)
    if os.environ.get("SMOKE_WEIGHTED") == "1":
        if not dated:
            raise SystemExit("full-model phenology needs heading and maturity on the seasons")
        season = dated[0]
        values = {
            name: torch.tensor(float(defaults[name]), dtype=torch.float64) for name in ns["BOUNDS"]
        }
        output = ns["simulate"](season, values)
        err = float(phenology_from_run(season, output))
        print(
            f"  one full run, site {season['number']}: phenology SS {err:.4f}, "
            f"final DVS {float(output['DVS'][-1]):.3f}",
            flush=True,
        )
        print("SMOKE WEIGHTED OK", flush=True)
        pool.close()
        pool.join()
        return
    print("SMOKE OK", flush=True)

    fresh = os.environ.get("FRESH") == "1" or not CHECKPOINT.exists()
    payload = {} if fresh else json.loads(CHECKPOINT.read_text())
    payload.setdefault(
        "budget",
        {
            "adam": "Wallach start counts: 20 if more than one major, 5 if one, 5 per candidate with earlier parameters started at their previous best, 20 for the weighted refit. Learning rate halves after 3 quiet steps, up to 4 times, then patience 10",
            "phenology_lr": ns["PHENO_LR"],
            "growth_lr": ns["GROUP_LR"],
            "nelder_phenology": "20 starts for the majors and 5 per candidate, ceiling 400 and 120, xatol 0.03, fatol 1",
            "nelder_growth": "5 starts, xatol 0.03, fatol 0.5% of the sum of squares at that start, ceiling 120 or 60",
            "workers": WORKERS,
        },
    )
    payload["budget"]["adam"] = (
        "Wallach start counts: 20 if more than one major, 5 if one, "
        "5 per candidate with earlier parameters started at their previous best, "
        "20 for the weighted refit. Learning rate halves after 3 quiet steps, "
        "up to 4 times, then patience 10"
    )
    payload["smoke"] = {"phenology_ss": base, "biomass_ss": biomass}
    n_pheno = 2 * len(ns["environments"])
    pheno_slots = int(os.environ.get("ADAM_PHENO_SLOTS", "2" if ADAM_ONLY else "4"))
    growth_slots = int(os.environ.get("ADAM_GROWTH_SLOTS", "1" if ADAM_ONLY else "2"))

    if ADAM_ONLY and "phenology" not in payload:
        print("ADAM ONLY, beside the Nelder-Mead search", flush=True)
        payload["phenology"] = {
            "n_observations": n_pheno,
            "adam": {"selected": [], "values": {}, "history": []},
            "nelder_mead": {
                "selected": [],
                "values": {name: defaults[name] for name in phenology_parameters()},
                "rounds": [],
            },
        }

    if "phenology" not in payload:
        print("PHENOLOGY START", flush=True)
        rng = np.random.default_rng(PHEN_SEED)
        fixed = dict(defaults)
        schedule = [
            (("TSUM1", "TSUM2"), 20, "majors", 400),
            (("VERNSAT",), 5, "candidate", 120),
            (("VERNBASE",), 5, "candidate", 120),
            (("DLO",), 5, "candidate", 120),
        ]
        nm_jobs = []
        nm_meta = []
        for names, n_starts, role, maxfev in schedule:
            center = {name: fixed[name] for name in names}
            starts = latin_starts(names, center, n_starts - 1, rng)
            # Candidates are launched after majors finish, because a kept
            # parameter changes the fixed vector. Majors can overlap Adam.
            nm_meta.append((names, role, maxfev, starts))

        names, role, maxfev, starts = nm_meta[0]
        print(f"Nelder-Mead phenology {names} x {len(starts)}", flush=True)
        pending = launch(
            pool,
            [
                {
                    "index": index,
                    "label": "phenology " + ",".join(names),
                    "names": list(names),
                    "start": start,
                    "fixed": dict(defaults),
                    "kind": "phenology",
                    "maxfev": maxfev,
                    "xatol": 0.03,
                    "fatol": 1.0,
                }
                for index, start in enumerate(starts, start=1)
            ],
        )
        accepted = dict(defaults)
        selected = []
        best_aicc = None
        adam_history = []
        for names_a, role_a in ((("TSUM1", "TSUM2"), "majors"),):
            fit = adam_loop(names_a, accepted, ns["PHENO_LR"], phenology_loss, 80)
            accepted.update(fit["values"])
            selected = list(names_a)
            best_aicc = aicc_of(fit["ss"], n_pheno, len(selected))
            adam_history.append({**fit, "aicc": best_aicc, "kept": True, "role": role_a})
            print(f"  Adam majors SS {fit['ss']:.2f} AICc {best_aicc:.2f} via {fit['stopped']}", flush=True)
        for candidate in ("VERNSAT", "VERNBASE", "DLO"):
            trial = selected + [candidate]
            fit = adam_loop(trial, accepted, ns["PHENO_LR"], phenology_loss, 80)
            score = aicc_of(fit["ss"], n_pheno, len(trial))
            keep = score < best_aicc
            adam_history.append({**fit, "aicc": score, "kept": keep, "role": "candidate"})
            print(f"  Adam {candidate} SS {fit['ss']:.2f} AICc {score:.2f} kept {keep} via {fit['stopped']}", flush=True)
            if keep:
                accepted.update(fit["values"])
                selected = trial
                best_aicc = score

        runs = collect(pending, payload)
        summary = summarise_round(runs, nm_meta[0][0], "majors", n_pheno)
        summary["kept"] = True
        nm_rounds = [summary]
        fixed.update(summary["best_final"])
        nm_selected = list(summary["trial"])
        best_nm = summary["aicc"]
        print(f"  Nelder-Mead majors SS {summary['best_ss']:.2f} AICc {summary['aicc']:.2f}", flush=True)
        for names, role, maxfev, _starts in nm_meta[1:]:
            center = {name: fixed[name] for name in names}
            starts = latin_starts(names, center, 4, rng)
            print(f"Nelder-Mead phenology {names} x {len(starts)}", flush=True)
            runs = collect(
                launch(
                    pool,
                    [
                        {
                            "index": index,
                            "label": "phenology " + ",".join(names),
                            "names": list(names),
                            "start": start,
                            "fixed": dict(fixed),
                            "kind": "phenology",
                            "maxfev": maxfev,
                            "xatol": 0.03,
                            "fatol": 1.0,
                        }
                        for index, start in enumerate(starts, start=1)
                    ],
                ),
                payload,
            )
            trial = nm_selected + list(names)
            summary = summarise_round(runs, trial, role, n_pheno)
            keep = summary["aicc"] < best_nm
            summary["kept"] = keep
            nm_rounds.append(summary)
            print(f"  best SS {summary['best_ss']:.2f} AICc {summary['aicc']:.2f} kept {keep}", flush=True)
            if keep:
                fixed.update(summary["best_final"])
                nm_selected = trial
                best_nm = summary["aicc"]
        payload["phenology"] = {
            "n_observations": n_pheno,
            "adam": {
                "selected": selected,
                "values": {name: accepted[name] for name in phenology_parameters()},
                "history": adam_history,
            },
            "nelder_mead": {
                "selected": nm_selected,
                "values": {name: fixed[name] for name in phenology_parameters()},
                "rounds": nm_rounds,
            },
        }
        checkpoint(payload)
        print("PHENOLOGY DONE", flush=True)

    if NM_WEIGHTED_ONLY:
        print("NELDER-MEAD WEIGHTED ONLY, using the rounds that have already finished", flush=True)
        nm_selected, nm_values, nm_variance = restore_nm_weighted_start(payload, defaults)
        if os.environ.get("NM_ADAM_VARIANCES") == "1":
            _, _, adam_variance = restore_weighted_start(payload, defaults)
            kept_pheno = [row for row in payload["phenology"]["adam"]["history"] if row["kept"]]
            pheno = kept_pheno[-1]
            n_pheno = payload["phenology"]["n_observations"]
            nm_variance = {
                "phenology": pheno["ss"] / max(n_pheno - len(pheno["parameters"]), 1),
                **adam_variance,
            }
            print(
                "  variances from Adam step 6: "
                + ", ".join(f"{key} {value:.6g}" for key, value in nm_variance.items()),
                flush=True,
            )
        names = [name for name in nm_selected if name in ns["BOUNDS"]]
        run_nm_weighted(
            pool,
            payload,
            names,
            {name: nm_values[name] for name in names},
            dict(nm_values),
            nm_variance,
        )
        payload["done"] = True
        checkpoint(payload)
        pool.close()
        pool.join()
        print("RETRAIN DONE", flush=True)
        return

    if not NM_ONLY and payload["phenology"]["adam"].get("rule") != "wallach":
        print("ADAM PHENOLOGY MULTI-START", flush=True)
        payload["phenology"].setdefault("adam_one_start", payload["phenology"]["adam"])
        bucket = payload["phenology"].setdefault("adam_rounds", [])
        accepted = dict(defaults)
        selected = []
        names = ["TSUM1", "TSUM2"]
        row = run_adam_round(
            pool,
            payload,
            bucket,
            key="phenology:major:TSUM1,TSUM2",
            group="phenology",
            names=names,
            role="major",
            center={name: accepted[name] for name in names},
            fixed=accepted,
            kind="phenology",
            lr=ns["PHENO_LR"],
            max_steps=80,
            slots=pheno_slots,
            n_obs=n_pheno,
        )
        accepted.update(row["values"])
        selected = list(names)
        best_aicc = row["aicc"]
        for candidate in ("VERNSAT", "VERNBASE", "DLO"):
            trial = selected + [candidate]
            row = run_adam_round(
                pool,
                payload,
                bucket,
                key="phenology:candidate:" + ",".join(trial),
                group="phenology",
                names=trial,
                role="candidate",
                new_name=candidate,
                center={name: accepted[name] for name in trial},
                fixed=dict(accepted),
                kind="phenology",
                lr=ns["PHENO_LR"],
                max_steps=80,
                slots=pheno_slots,
                n_obs=n_pheno,
                keep_below=best_aicc,
            )
            if row["kept"]:
                accepted.update(row["values"])
                selected = list(trial)
                best_aicc = row["aicc"]
        payload["phenology"]["adam"] = {
            "rule": "wallach",
            "selected": selected,
            "values": {name: accepted[name] for name in phenology_parameters()},
            "history": [item for item in bucket if item.get("complete")],
        }
        checkpoint(payload)
        print(f"ADAM PHENOLOGY DONE AICc {best_aicc:.2f} kept {selected}", flush=True)

    print("GROWTH START", flush=True)
    adam_values = dict(defaults)
    adam_values.update(payload["phenology"]["adam"]["values"])
    nm_values = dict(defaults)
    nm_values.update(payload["phenology"]["nelder_mead"]["values"])
    payload.setdefault("growth", {"adam": [], "nelder_mead": []})
    adam_selected = list(payload["phenology"]["adam"]["selected"])
    nm_selected = list(payload["phenology"]["nelder_mead"]["selected"])
    adam_variance = {}
    nm_variance = {}
    if not NM_ONLY and any(row.get("rule") != "wallach" for row in payload["growth"]["adam"]):
        payload["growth"]["adam_one_start"] = [
            row for row in payload["growth"]["adam"] if row.get("rule") != "wallach"
        ]
        payload["growth"]["adam"] = [row for row in payload["growth"]["adam"] if row.get("rule") == "wallach"]
        checkpoint(payload)
    rng = np.random.default_rng(GROWTH_SEED)
    done_nm = {row["trial_key"] for row in payload["growth"]["nelder_mead"]}

    if WEIGHTED_ONLY:
        print("WEIGHTED ONLY, using the rounds that have already finished", flush=True)
        adam_selected, adam_values, adam_variance = restore_weighted_start(payload, defaults)

    for spec in ns["GROUPS"]:
        if WEIGHTED_ONLY:
            break
        kind = spec["kind"]
        n_obs = ns["observation_count"](kind)
        print(f"\nGROUP {spec['name']} n={n_obs}", flush=True)
        adam_free_major = [name for name in spec["major"] if name not in adam_selected]
        nm_free_major = [name for name in spec["major"] if name not in nm_selected]
        adam_chosen, nm_chosen = [], []
        adam_best = nm_best = None
        adam_best_score = nm_best_score = None

        if adam_free_major and not NM_ONLY:
            key = spec["name"] + ":major:" + ",".join(adam_free_major)
            row = run_adam_round(
                pool,
                payload,
                payload["growth"]["adam"],
                key=key,
                group=spec["name"],
                names=adam_free_major,
                role="major",
                center={name: adam_values[name] for name in adam_free_major},
                fixed=dict(adam_values),
                kind=kind,
                lr=ns["GROUP_LR"],
                max_steps=ns["MAX_STEPS"],
                slots=growth_slots,
                n_obs=n_obs,
            )
            adam_values.update(row["values"])
            adam_chosen = list(adam_free_major)
            adam_best = row["ss"]
            adam_best_score = row["aicc"]
        elif not NM_ONLY:
            adam_best = float(ns["evaluate_kind"](kind, adam_values))

        for candidate in ([] if NM_ONLY else spec["candidates"]):
            if candidate in adam_selected or candidate in adam_chosen:
                continue
            trial = adam_chosen + [candidate]
            key = spec["name"] + ":candidate:" + ",".join(trial)
            row = run_adam_round(
                pool,
                payload,
                payload["growth"]["adam"],
                key=key,
                group=spec["name"],
                names=trial,
                role="candidate",
                new_name=candidate,
                center={name: adam_values[name] for name in trial},
                fixed=dict(adam_values),
                kind=kind,
                lr=ns["GROUP_LR"],
                max_steps=ns["MAX_STEPS"],
                slots=growth_slots,
                n_obs=n_obs,
                keep_below=adam_best_score,
            )
            if row["kept"]:
                adam_values.update(row["values"])
                adam_chosen = list(trial)
                adam_best = row["ss"]
                adam_best_score = row["aicc"]
        if adam_best is not None:
            if adam_chosen:
                adam_selected.extend(name for name in adam_chosen if name not in adam_selected)
            denom = n_obs - len(adam_chosen)
            adam_variance[kind] = (adam_best / denom) if denom > 0 else adam_best / n_obs
        if ADAM_ONLY:
            print(f"  {spec['name']} Adam SS {adam_best:.4g} kept {adam_chosen or 'none'}", flush=True)
            continue

        if nm_free_major:
            key = spec["name"] + ":major:" + ",".join(nm_free_major)
            starts = latin_starts(nm_free_major, {name: nm_values[name] for name in nm_free_major}, 4, rng)
            if key not in done_nm:
                print(f"Nelder-Mead {spec['name']} {list(nm_free_major)}", flush=True)
                runs = collect(
                    launch(
                        pool,
                        [
                            {
                                "index": index,
                                "label": spec["name"],
                                "names": list(nm_free_major),
                                "start": start,
                                "fixed": dict(nm_values),
                                "kind": kind,
                                "maxfev": 120 if len(nm_free_major) > 1 else 60,
                                "xatol": 0.03,
                                "fatol": None,
                            }
                            for index, start in enumerate(starts, start=1)
                        ],
                    ),
                    payload,
                )
                summary = summarise_round(runs, nm_free_major, "major", n_obs)
                summary.update({"trial_key": key, "group": spec["name"], "kept": True})
                payload["growth"]["nelder_mead"].append(summary)
                checkpoint(payload)
            else:
                summary = next(item for item in payload["growth"]["nelder_mead"] if item["trial_key"] == key)
            nm_values.update(summary["best_final"])
            nm_chosen = list(nm_free_major)
            nm_best = summary["best_ss"]
            nm_best_score = summary["aicc"]
        else:
            nm_best = float(ns["evaluate_kind"](kind, nm_values))

        for candidate in spec["candidates"]:
            if candidate in nm_selected or candidate in nm_chosen:
                continue
            trial = nm_chosen + [candidate]
            key = spec["name"] + ":candidate:" + ",".join(trial)
            starts = latin_starts(trial, {name: nm_values[name] for name in trial}, 4, rng)
            if key not in done_nm:
                print(f"Nelder-Mead {spec['name']} candidate {trial}", flush=True)
                runs = collect(
                    launch(
                        pool,
                        [
                            {
                                "index": index,
                                "label": spec["name"] + " " + candidate,
                                "names": list(trial),
                                "start": start,
                                "fixed": dict(nm_values),
                                "kind": kind,
                                "maxfev": 120 if len(trial) > 1 else 60,
                                "xatol": 0.03,
                                "fatol": None,
                            }
                            for index, start in enumerate(starts, start=1)
                        ],
                    ),
                    payload,
                )
                summary = summarise_round(runs, trial, "candidate", n_obs)
                keep = nm_best_score is None or summary["aicc"] < nm_best_score
                summary.update({"trial_key": key, "group": spec["name"], "kept": keep})
                payload["growth"]["nelder_mead"].append(summary)
                checkpoint(payload)
            else:
                summary = next(item for item in payload["growth"]["nelder_mead"] if item["trial_key"] == key)
                keep = summary["kept"]
            if keep:
                nm_values.update(summary["best_final"])
                nm_chosen = list(summary["trial"])
                nm_best = summary["best_ss"]
                nm_best_score = summary["aicc"]
        if nm_chosen:
            nm_selected.extend(name for name in nm_chosen if name not in nm_selected)
        denom = n_obs - len(nm_chosen)
        nm_variance[kind] = (nm_best / denom) if denom > 0 else nm_best / n_obs
        adam_txt = "not this job" if adam_best is None else f"{adam_best:.4g}"
        print(
            f"  {spec['name']} Adam SS {adam_txt} kept {adam_chosen or 'none'}; "
            f"Nelder-Mead SS {nm_best:.4g} kept {nm_chosen or 'none'}",
            flush=True,
        )

    payload["growth"]["adam_selected"] = adam_selected
    payload["growth"]["adam_values"] = adam_values
    payload["growth"]["adam_variance"] = adam_variance
    payload["growth"]["nelder_selected"] = nm_selected
    payload["growth"]["nelder_values"] = nm_values
    payload["growth"]["nelder_variance"] = nm_variance
    checkpoint(payload)

    if not NM_ONLY and payload.get("weighted", {}).get("rule") != "wallach":
        print("WEIGHTED REFIT", flush=True)
        kept_pheno = [row for row in payload["phenology"]["adam"]["history"] if row["kept"]]
        pheno_var = kept_pheno[-1]["ss"] / max(n_pheno - len(kept_pheno[-1]["parameters"]), 1)
        names = [name for name in adam_selected if name in ns["BOUNDS"]]
        variances = {"phenology": pheno_var, **adam_variance}
        bucket = payload.setdefault("weighted_rounds", [])
        row = run_adam_round(
            pool,
            payload,
            bucket,
            key="weighted",
            group="weighted",
            names=names,
            role="weighted",
            center={name: adam_values[name] for name in names},
            fixed=dict(adam_values),
            kind="weighted",
            lr=ns["GROUP_LR"],
            max_steps=ns["MAX_STEPS"],
            slots=growth_slots,
            variances=variances,
        )
        adam_values.update(row["values"])
        payload["weighted"] = {"rule": "wallach", "adam": {**row, "values_full": adam_values}}
        checkpoint(payload)

    payload["done"] = True
    checkpoint(payload)
    pool.close()
    pool.join()
    print("RETRAIN DONE", flush=True)


if __name__ == "__main__":
    main()
