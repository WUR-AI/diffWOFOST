"""Crop model and calibration objective for the AgMIP search.

agmip_retrain.py loads the sections below. The notebook does not run them.
"""
# %% setup

import datetime as dt
import json
import io
import tempfile
import urllib.request
import warnings
from pathlib import Path

import pandas as pd
import torch
from pcse.input import YAMLAgroManagementReader
from pcse.input import YAMLCropDataProvider
from pcse.models import Wofost81_NWLP_MLWB_SNOMIN

from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.config import Configuration
from diffwofost.physical_models.crop.phenology import DVS_Phenology
from diffwofost.physical_models.engine import Engine
from diffwofost.physical_models.parameter_providers import ParameterProvider

warnings.filterwarnings("ignore", message="Converting a tensor with requires_grad")
ComputeConfig.set_device("cpu")
ComputeConfig.set_dtype(torch.float64)

# True loads the stored search and skips the Adam fits. False repeats them
# and overwrites agmip_calibration_results.json.
DO_NOT_RERUN = True
PATIENCE = 10
LR_PATIENCE = 3
LR_FACTOR = 0.5
LR_CUTS = 4
MIN_DELTA = 1.0e-4
PHENO_LR = 0.25


def rate_decision(score, best_ss, waited, cuts, rewound):
    """Step, halve the learning rate, or stop.

    Three steps without a new best return to the best parameters and halve
    the rate, up to four times. At the smallest rate the search stops after
    PATIENCE steps with no new best. The evaluation just after a rewind is
    the best point itself, so it does not count as a stall.
    """
    if rewound:
        return 0, cuts, False, "step"
    if score < best_ss - MIN_DELTA:
        return 0, cuts, False, "improve"
    waited += 1
    limit = PATIENCE if cuts >= LR_CUTS else LR_PATIENCE
    if waited < limit:
        return waited, cuts, False, "step"
    if cuts < LR_CUTS:
        return 0, cuts + 1, True, "decay"
    return waited, cuts, False, "stop"
RESULTS_NAME = "agmip_calibration_results.json"
RESULTS_URL = (
    "https://raw.githubusercontent.com/WUR-AI/diffWOFOST/"
    "agmip-calibration-protocol/docs/notebooks/agmip_calibration_results.json"
)
NELDER_NAME = "agmip_nelder_mead_results.json"
NELDER_URL = (
    "https://raw.githubusercontent.com/WUR-AI/diffWOFOST/"
    "agmip-calibration-protocol/docs/notebooks/agmip_nelder_mead_results.json"
)

PAIRS_NAME = "agmip_calibration_pairs.json"
PAIRS_URL = (
    "https://raw.githubusercontent.com/WUR-AI/diffWOFOST/"
    "agmip-calibration-protocol/docs/notebooks/agmip_calibration_pairs.json"
)
TRAJECTORY_NAME = "agmip_trajectories.json"
TRAJECTORY_URL = (
    "https://raw.githubusercontent.com/WUR-AI/diffWOFOST/"
    "agmip-calibration-protocol/docs/notebooks/agmip_trajectories.json"
)

STEP7_NAME = "agmip_step7_scores.json"
STEP7_URL = (
    "https://raw.githubusercontent.com/WUR-AI/diffWOFOST/"
    "agmip-calibration-protocol/docs/notebooks/agmip_step7_scores.json"
)


def results_file():
    """Results beside this notebook, or in the working directory on Colab."""
    here = Path.cwd().resolve()
    for base in (here, *here.parents):
        notebook_dir = base / "docs" / "notebooks"
        if notebook_dir.is_dir():
            return notebook_dir / RESULTS_NAME
    return here / RESULTS_NAME


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    return float(value)


def load_stored(name, url):
    """JSON beside this notebook, downloaded once if it is missing."""
    path = results_file().with_name(name)
    if not path.exists():
        print(f"No results file at {path}. Downloading the stored search.")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    print(f"Loaded {path.name}.")
    return json.loads(path.read_text())


def load_results():
    """Stored parameters, sums of squares and AICc rows."""
    return load_stored(RESULTS_NAME, RESULTS_URL)


def load_nelder():
    """Stored Nelder-Mead comparison. Independent of DO_NOT_RERUN."""
    return load_stored(NELDER_NAME, NELDER_URL)


def save_results(payload):
    path = results_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    print(f"Stored calibration results in {path}.")


DATA_URL = "https://raw.githubusercontent.com/sbuis/AgMIP-Calibration-Phase-IV/main/data"


def crop_directory():
    """Winter-wheat parameter file shipped with the SNOMIN notebook."""
    here = Path.cwd().resolve()
    relative = (
        Path("docs/notebooks/snomin_calibration_data/crop"),
        Path("snomin_calibration_data/crop"),
        Path("crop"),
    )
    for base in (here, *here.parents):
        for name in relative:
            candidate = base / name
            if (candidate / "winterwheat.yaml").exists():
                return candidate
    destination = here / "snomin_calibration_data" / "crop"
    destination.mkdir(parents=True, exist_ok=True)
    names = ("winterwheat.yaml", "crops.yaml")
    for ref in ("agmip-calibration-protocol", "main"):
        base = (
            "https://raw.githubusercontent.com/WUR-AI/diffWOFOST/"
            f"{ref}/docs/notebooks/snomin_calibration_data/crop"
        )
        try:
            for name in names:
                urllib.request.urlretrieve(f"{base}/{name}", destination / name)
        except Exception:
            continue
        if (destination / "winterwheat.yaml").exists():
            return destination
    raise FileNotFoundError("winterwheat.yaml is not next to this notebook")


def download(name):
    with urllib.request.urlopen(f"{DATA_URL}/{name}") as response:
        return response.read()


observations = pd.read_csv(io.BytesIO(download("cal_4_obs_French_A.txt")), sep=r"\s+")
management = pd.read_csv(
    io.BytesIO(download("cal_4_input_data_French_A.txt")), sep=r"\s+", encoding="utf-8-sig"
)
units = pd.read_csv(io.BytesIO(download("cal_4_obs_French_units.csv")), sep=";", encoding="cp1252")
weather_book = pd.ExcelFile(io.BytesIO(download("cal_4_weather.xlsx")))
print(f"Calibration rows: {len(observations)}. Site-years in the input file: {len(management)}.")
print(f"Weather stations: {', '.join(weather_book.sheet_names)}.")
units

# %% environments
SHARPNESS = 40.0
BOUNDS = {
    "TSUM1": (400.0, 1400.0),
    "TSUM2": (400.0, 1400.0),
    "VERNSAT": (30.0, 90.0),
    "VERNBASE": (0.0, 20.0),
    "DLO": (12.0, 20.0),
}
MAJOR = ("TSUM1", "TSUM2")
CANDIDATES = ("VERNSAT", "VERNBASE", "DLO")
SCALARS = (
    "TSUMEM", "TBASEM", "TEFFMX", "TSUM1", "TSUM2", "IDSL", "DLO", "DLC",
    "DVSI", "DVSEND", "VERNSAT", "VERNBASE", "VERNDVS",
)


def parse_date(value):
    if pd.isna(value) or value == "NA":
        return None
    return dt.datetime.strptime(str(value), "%d/%m/%Y").date()


def station_weather(station):
    """Daily WOFOST weather. Radiation in the workbook is MJ m-2 day-1."""
    sheet = pd.read_excel(weather_book, f"station_{station}", header=None)
    latitude = float(str(sheet.iloc[0, -1]).split("Latitude =")[1].split("§")[0])
    altitude = float(str(sheet.iloc[0, -1]).split("Altitude =")[1])
    daily = sheet.iloc[5:].copy()
    daily.columns = [
        "DATECODE", "YYYY", "MM", "DD", "SRAD", "TMAX", "TMIN", "RAIN",
        "ET0", "WIND", "DEWP", "VPRS", "RHUM", "DAY",
    ]
    daily["DAY"] = pd.to_datetime(daily["DAY"]).dt.date
    return daily, latitude, altitude


def driving_rows(daily, latitude, altitude, start, end):
    window = daily[(daily["DAY"] >= start) & (daily["DAY"] <= end)]
    rows = []
    for record in window.itertuples(index=False):
        tmin = float(record.TMIN)
        tmax = float(record.TMAX)
        rows.append({
            "DAY": record.DAY,
            "LAT": latitude,
            "LON": 2.0,
            "ELEV": altitude,
            "IRRAD": float(record.SRAD) * 1.0e6,
            "TMIN": tmin,
            "TMAX": tmax,
            "TEMP": 0.5 * (tmin + tmax),
            "RAIN": float(record.RAIN) / 10.0,
            "VAP": 10.0,
            "WIND": 2.0,
        })
    return rows


def agro_calendar(start, end):
    text = f"""
AgroManagement:
- {start.isoformat()}:
    CropCalendar:
        crop_name: winterwheat
        variety_name: Arminda
        crop_start_date: {start.isoformat()}
        crop_start_type: sowing
        crop_end_date: {end.isoformat()}
        crop_end_type: harvest
        max_duration: 360
    TimedEvents: null
    StateEvents: null
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write(text)
        path = Path(handle.name)
    try:
        return YAMLAgroManagementReader(str(path))
    finally:
        path.unlink(missing_ok=True)


phenology_rows = (
    observations.dropna(subset=["Date_BBCH55", "Date_BBCH90"])
    .groupby("Number", as_index=False)
    .tail(1)
)
environments = []
for record in phenology_rows.itertuples(index=False):
    site = management.loc[management["Number"] == record.Number].iloc[0]
    sowing = parse_date(record.SowingDate)
    heading = parse_date(record.Date_BBCH55)
    maturity = parse_date(record.Date_BBCH90)
    harvest = parse_date(record.HarvestDate) or maturity
    daily, latitude, altitude = station_weather(int(site.Station))
    end = min(harvest + dt.timedelta(days=45), daily["DAY"].max())
    rows = driving_rows(daily, latitude, altitude, sowing, end)
    if not rows or rows[-1]["DAY"] < maturity:
        print(f"Skipping environment {record.Number}: weather does not cover maturity.")
        continue
    environments.append({
        "number": int(record.Number),
        "site": record.Site,
        "year": int(record.HarvestYear),
        "rows": rows,
        "agro": agro_calendar(sowing, rows[-1]["DAY"]),
        "heading": (heading - sowing).days,
        "maturity": (maturity - sowing).days,
    })

held_out = sorted(set(management["Number"]) - set(phenology_rows["Number"]))
print(f"Calibration environments with heading and maturity: {len(environments)}.")
print(f"Held-out environment numbers, observations not public: {held_out}.")

crop = YAMLCropDataProvider(
    Wofost81_NWLP_MLWB_SNOMIN, fpath=str(crop_directory()), force_reload=True
)
provider = ParameterProvider(cropdata=crop)
provider.set_active_crop("winterwheat", "Arminda", "sowing", "harvest")
for name in list(SCALARS) + ["DTSMTB", "VERNRTB"]:
    value = provider[name]
    if not torch.is_tensor(value):
        value = torch.tensor(value, dtype=torch.float64)
    provider.set_override(name, value, check=False)
defaults = {name: float(provider[name]) for name in list(BOUNDS) + ["TSUMEM", "IDSL"]}
print(
    "Arminda defaults: "
    + ", ".join(f"{name} {defaults[name]:.1f}" for name in ("TSUM1", "TSUM2", "VERNSAT", "VERNBASE", "DLO"))
)

config = Configuration(CROP=DVS_Phenology, OUTPUT_VARS=["DVS"])

# %% phenology
class BoundedParameter(torch.nn.Module):
    """A parameter kept inside its protocol bounds."""

    def __init__(self, low, high, init_value):
        super().__init__()
        self.low = low
        self.high = high
        fraction = (init_value - low) / (high - low)
        fraction = min(max(fraction, 1.0e-4), 1.0 - 1.0e-4)
        self.raw = torch.nn.Parameter(torch.logit(torch.tensor(fraction, dtype=torch.float64)))

    def forward(self):
        return self.low + (self.high - self.low) * torch.sigmoid(self.raw)


class ParameterSet(torch.nn.Module):
    def __init__(self, names, start):
        super().__init__()
        self.boxes = torch.nn.ModuleDict({
            name: BoundedParameter(*BOUNDS[name], start[name]) for name in names
        })

    def forward(self):
        return {name: box() for name, box in self.boxes.items()}


def apply(active):
    for name in SCALARS:
        if name in active:
            provider.set_override(name, active[name], check=False)
        elif name in accepted:
            provider.set_override(name, torch.tensor(accepted[name], dtype=torch.float64), check=False)


def development(environment, active):
    apply(active)
    engine = Engine(config)
    engine.setup(provider, iter(environment["rows"]), environment["agro"])
    engine.run_till(environment["rows"][-1]["DAY"])
    return torch.stack([day["DVS"].reshape(()) for day in engine.get_output()])


def stage_error(dvs, threshold, observed_day):
    """Error in days. A stage past the season end uses the remaining development."""
    past = dvs > threshold + 0.05
    capped = torch.where(past, dvs.detach(), dvs)
    previous = torch.cat((capped.new_zeros(1), capped[:-1]))
    entered = torch.sigmoid(SHARPNESS * (capped - threshold))
    entered = entered - torch.sigmoid(SHARPNESS * (previous - threshold))
    entered = entered.clamp(min=0.0)
    days = torch.arange(capped.shape[0], dtype=capped.dtype)
    weight = entered.sum()
    day = (entered * days).sum() / weight.clamp(min=1.0e-6)
    calendar = day - observed_day
    unfinished = (threshold - dvs[-1]).clamp(min=0.0) / 0.02
    beyond = unfinished + (dvs.shape[0] - 1 - observed_day)
    return torch.where(weight > 0.2, calendar, beyond)


def sum_of_squares(active):
    total = torch.zeros((), dtype=torch.float64)
    for environment in environments:
        dvs = development(environment, active)
        heading = stage_error(dvs, 1.0, environment["heading"])
        maturity = stage_error(dvs, 2.0, environment["maturity"])
        total = total + heading.square() + maturity.square()
    return total


def aicc(ss, n_obs, n_param):
    return n_obs * torch.log(ss / n_obs) + 2 * n_param + (2 * n_param * (n_param + 1)) / (n_obs - n_param - 1)


def estimate(names, start, max_steps=80):
    """Adam, halving the rate when the sum of squares stalls."""
    pack = ParameterSet(names, start)
    lr = PHENO_LR
    optimizer = torch.optim.Adam(pack.parameters(), lr=lr)
    best_ss = float("inf")
    best_state = {key: value.detach().clone() for key, value in pack.state_dict().items()}
    waited, cuts, rewound = 0, 0, False
    for step in range(max_steps):
        optimizer.zero_grad()
        last = sum_of_squares(pack())
        score = float(last.detach())
        waited, cuts, rewound, action = rate_decision(score, best_ss, waited, cuts, rewound)
        if action == "improve":
            best_ss = score
            best_state = {key: value.detach().clone() for key, value in pack.state_dict().items()}
        elif action == "decay":
            pack.load_state_dict(best_state)
            lr *= LR_FACTOR
            optimizer = torch.optim.Adam(pack.parameters(), lr=lr)
            print(f"Learning rate {lr:.4g}. Back at the best step.")
            continue
        elif action == "stop":
            print(f"Early stopping at step {step}.")
            break
        last.backward()
        optimizer.step()
    else:
        print(f"Reached the ceiling of {max_steps} steps.")
    pack.load_state_dict(best_state)
    with torch.no_grad():
        values = {name: float(value) for name, value in pack().items()}
        ss = float(sum_of_squares(pack()))
    return values, ss


def hard_day(dvs, threshold):
    crossed = (dvs >= threshold).nonzero()
    if len(crossed) == 0:
        return None
    return int(crossed[0])


def stage_errors(values):
    """Signed errors in days, heading then maturity, one pair per environment."""
    heading, maturity = [], []
    with torch.no_grad():
        active = {name: torch.tensor(values[name], dtype=torch.float64) for name in values}
        for environment in environments:
            dvs = development(environment, active)
            for threshold, observed, store in (
                (1.0, environment["heading"], heading),
                (2.0, environment["maturity"], maturity),
            ):
                day = hard_day(dvs, threshold)
                store.append(None if day is None else day - observed)
    return heading, maturity

# %% growth
import math
import time

from pcse.input import WOFOST81SiteDataProvider_SNOMIN

from diffwofost.physical_models.crop.wofost81 import Wofost81
from diffwofost.physical_models.soil.soil_wrappers import SoilModuleWrapper_NWLP_MLWB_SNOMIN

# Halve the learning rate when the sum of squares stalls. The ceiling is MAX_STEPS.
MAX_STEPS = 80
GROUP_LR = 0.12
PFS = (-1.0, 1.0, 1.3, 1.7, 2.0, 2.3, 2.4, 2.7, 3.0, 3.3, 3.7, 4.0, 4.2, 6.0)
CN_RATIO = 10.0
ROOT_TARGET = 125.0
PROTEIN_FACTOR = 5.7
GROWTH_BOUNDS = {
    "AMAX_REF": (20.0, 70.0),
    "SPAN": (15.0, 55.0),
    "TDWI": (50.0, 800.0),
    "RGRLAI": (0.003, 0.02),
    "RNUPTAKEMAX": (1.0, 8.0),
    "NMAXSO": (0.012, 0.030),
    "REALLOC_STEM_FRACTION": (0.05, 0.60),
    "CVO": (0.55, 0.85),
    "NRESIDLV": (0.001, 0.012),
    "NRESIDST": (0.0005, 0.005),
    "TCNT": (2.0, 25.0),
}
BOUNDS.update(GROWTH_BOUNDS)
GROUPS = (
    {"name": "plant_biomass", "major": ("AMAX_REF", "SPAN"), "candidates": ("TDWI", "RGRLAI"), "kind": "biomass"},
    {"name": "plant_N_content", "major": ("RNUPTAKEMAX",), "candidates": ("NMAXSO",), "kind": "nitrogen"},
    {"name": "yield", "major": ("REALLOC_STEM_FRACTION",), "candidates": ("CVO",), "kind": "yield"},
    {"name": "seed_protein", "major": ("NMAXSO",), "candidates": ("NRESIDLV", "NRESIDST", "TCNT"), "kind": "protein"},
)
full_config = Configuration(
    CROP=Wofost81,
    SOIL=SoilModuleWrapper_NWLP_MLWB_SNOMIN,
    OUTPUT_VARS=["DVS", "TAGP", "TWSO", "NamountLV", "NamountST", "NamountSO"],
)


def wosten(clay, silt, organic, density, topsoil):
    """Van Genuchten parameters from the Wösten pedotransfer."""
    clay = max(float(clay), 1.0)
    silt = max(float(silt), 1.0)
    organic = max(float(organic), 0.01)
    density = float(density)
    top = 1.0 if topsoil else 0.0
    t_alpha = (
        -14.96 + 0.03135 * clay + 0.0351 * silt + 0.646 * organic + 15.29 * density
        - 0.192 * top - 4.671 * density**2 - 0.000781 * clay**2 - 0.00687 * organic**2
        + 0.0449 / organic + 0.0663 * math.log(silt) + 0.1482 * math.log(organic)
        - 0.04546 * density * silt - 0.4852 * density * organic + 0.00673 * top * clay
    )
    t_n = (
        -25.23 - 0.02195 * clay + 0.0074 * silt - 0.1940 * organic + 45.5 * density
        - 7.24 * density**2 + 0.0003658 * clay**2 + 0.002885 * organic**2 - 12.81 / density
        - 0.1524 / silt - 0.01958 / organic - 0.2876 * math.log(silt) - 0.0709 * math.log(organic)
        - 44.6 * math.log(density) - 0.02264 * density * clay + 0.0896 * density * organic
        + 0.00718 * top * clay
    )
    t_lambda = (
        0.0202 + 0.0006193 * clay**2 - 0.001136 * organic**2 - 0.2316 * math.log(organic)
        - 0.03544 * density * clay + 0.00283 * density * silt + 0.0488 * density * organic
    )
    t_ksat = (
        7.755 + 0.0352 * silt + 0.93 * top - 0.967 * density**2 - 0.000484 * clay**2
        - 0.000322 * silt**2 + 0.001 / silt - 0.0748 / organic - 0.643 * math.log(silt)
        - 0.01398 * density * clay - 0.1673 * density * organic + 0.02986 * top * clay
        - 0.03305 * top * silt
    )
    theta_s = (
        0.7919 + 0.001691 * clay - 0.29619 * density - 1.491e-6 * silt**2
        + 8.21e-5 * organic**2 + 0.02427 / clay + 0.01113 / silt + 0.01472 * math.log(silt)
        - 7.33e-5 * organic * clay - 0.000619 * density * clay - 0.001183 * density * organic
        - 0.0001664 * top * silt
    )
    return {
        "alpha": math.exp(t_alpha),
        "n": math.exp(t_n) + 1.0,
        "lam": (10 * math.exp(t_lambda) - 10) / (1 + math.exp(t_lambda)),
        "k_sat": math.exp(t_ksat),
        "theta_s": min(max(theta_s, 0.25), 0.65),
        "theta_r": 0.01,
    }


def _moisture(pf, vg):
    psi = 10.0 ** pf
    n = vg["n"]
    m = 1.0 - 1.0 / n
    return vg["theta_r"] + (vg["theta_s"] - vg["theta_r"]) / (1.0 + (vg["alpha"] * psi) ** n) ** m


def _log_k(pf, vg):
    psi = 10.0 ** pf
    n = vg["n"]
    m = 1.0 - 1.0 / n
    ah = vg["alpha"] * psi
    h1 = (1.0 + ah ** n) ** m
    h2 = ah ** (n - 1.0)
    denom = (1.0 + ah ** n) ** (m * (vg["lam"] + 2.0))
    return math.log10(max(vg["k_sat"] * (h1 - h2) ** 2 / denom, 1e-12))


def retention_tables(vg):
    moistures = [_moisture(pf, vg) for pf in PFS]
    for index in range(1, len(moistures)):
        if moistures[index] >= moistures[index - 1]:
            moistures[index] = moistures[index - 1] - 1e-4
    sm, cond = [], []
    for pf, value in zip(PFS, moistures, strict=True):
        sm.extend((float(pf), max(value, 0.005)))
        cond.extend((float(pf), _log_k(pf, vg)))
    return sm, cond


def cell_number(row, name):
    value = row[name]
    if pd.isna(value) or value == "NA":
        return None
    return float(value)


def soil_layers(row):
    layers = []
    for horizon in range(1, int(row.N_Horizons) + 1):
        thickness = cell_number(row, f"Thickness_H{horizon}")
        if thickness is None:
            continue
        clay = cell_number(row, f"Clay_H{horizon}") or 10.0
        silt = (cell_number(row, f"Fine_silt_H{horizon}") or 0.0) + (
            cell_number(row, f"Coarse_silt_H{horizon}") or 0.0
        )
        organic = cell_number(row, f"Organic_matter_H{horizon}")
        organic = 1.0 if organic is None else organic
        density = cell_number(row, f"Apparent_density_H{horizon}") or 1.4
        ph = cell_number(row, f"Water_pH_H{horizon}") or 7.0
        capacity = cell_number(row, f"Field_capacity_H{horizon}")
        wilting = cell_number(row, f"Wilting_point_H{horizon}")
        vg = wosten(clay, silt, organic, density, topsoil=(horizon == 1))
        sm, cond = retention_tables(vg)
        layers.append({
            "Thickness": thickness,
            "SMfromPF": sm,
            "CONDfromPF": cond,
            "CRAIRC": 0.06,
            "CNRatioSOMI": CN_RATIO,
            "RHOD": density,
            "Soil_pH": ph,
            "FSOMI": organic / 100.0,
            "file_fc": None if capacity is None else capacity / 100.0 * density,
            "ptf_fc": _moisture(2.0, vg),
            "file_wp": None if wilting is None else wilting / 100.0 * density,
            "ptf_wp": _moisture(4.2, vg),
        })
    return layers


def with_root_boundary(layers):
    """Split a layer so the rooting depth lands on a boundary at or nearest 125 cm."""
    total = sum(layer["Thickness"] for layer in layers)
    target = min(ROOT_TARGET, total)
    depth = 0.0
    split = []
    for layer in layers:
        thickness = layer["Thickness"]
        crosses = depth < target < depth + thickness - 0.05
        upper = target - depth
        lower = thickness - upper
        if crosses and upper >= 5.0 and lower >= 5.0:
            split.append({**layer, "Thickness": upper})
            split.append({**layer, "Thickness": lower})
        else:
            split.append(layer)
            if crosses:
                target = depth + thickness
        depth += thickness
    boundaries, depth = [], 0.0
    for layer in split:
        depth += layer["Thickness"]
        boundaries.append(depth)
    if not any(abs(boundary - target) < 0.05 for boundary in boundaries):
        target = min(boundaries, key=lambda boundary: abs(boundary - target))
    return split, target


def fertiliser_fractions(product):
    key = str(product).lower().replace("é", "e").replace("è", "e")
    if "ammonitrate" in key:
        return 0.5, 0.5
    if "18_46" in key or "18-46" in key:
        return 1.0, 0.0
    if "solution" in key:
        return 0.75, 0.25
    raise ValueError(f"Unknown fertiliser product: {product}")


def timed_block(title, signal, events):
    if not events:
        return ""
    lines = [
        "      - event_signal: " + signal,
        f"        name: {title}",
        "        comment: recorded in the AgMIP input file",
        "        events_table:",
    ]
    for day, body in events:
        lines.append(f"          - {day.isoformat()}:")
        lines.extend(f"              {key}: {value}" for key, value in body.items())
    return "\n".join(lines)


def season_agro(start, end, dressings, irrigations):
    timed = "\n".join(part for part in (
        timed_block("Fertiliser", "apply_n_snomin", dressings),
        timed_block("Irrigation", "irrigate", irrigations),
    ) if part)
    text = f"""
AgroManagement:
- {start.isoformat()}:
    CropCalendar:
        crop_name: winterwheat
        variety_name: Arminda
        crop_start_date: {start.isoformat()}
        crop_start_type: sowing
        crop_end_date: {end.isoformat()}
        crop_end_type: harvest
        max_duration: 360
    TimedEvents:
{timed or "      []"}
    StateEvents: null
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write(text)
        path = Path(handle.name)
    try:
        return YAMLAgroManagementReader(str(path))
    finally:
        path.unlink(missing_ok=True)


def dressing_events(row, start, end):
    events = []
    count = 0 if pd.isna(row.Number_of_fertilizer_doses) else int(row.Number_of_fertilizer_doses)
    for index in range(1, count + 1):
        day = parse_date(getattr(row, f"Date_fertilization_{index}"))
        amount = cell_number(row, f"Amount_fertilization_{index}")
        product = getattr(row, f"Product_fertilization_{index}")
        if day is None or amount is None or not (start <= day <= end):
            continue
        ammonium, nitrate = fertiliser_fractions(product)
        events.append((day, {
            "amount": amount,
            "application_depth": 10,
            "cnratio": 0,
            "initial_age": 0,
            "f_NH4N": ammonium,
            "f_NO3N": nitrate,
            "f_orgmat": 0,
        }))
    return events


def irrigation_events(row, start, end):
    events = []
    count = 0 if pd.isna(row.Number_of_irrigations) else int(row.Number_of_irrigations)
    for index in range(1, count + 1):
        day = parse_date(getattr(row, f"Date_irrigation_{index}"))
        amount = cell_number(row, f"Amount_irrigation_{index}")
        if day is None or amount is None or not (start <= day <= end):
            continue
        events.append((day, {"amount": amount / 10.0, "efficiency": 1.0}))
    return events


def season_rows(daily, latitude, altitude, start, end):
    window = daily[(daily["DAY"] >= start) & (daily["DAY"] <= end)]
    rows = []
    for record in window.itertuples(index=False):
        tmin = float(record.TMIN)
        tmax = float(record.TMAX)
        temp = 0.5 * (tmin + tmax)
        # The sheet stores ET0 in mm day-1. E0 and ES0 are not in the file.
        reference = float(record.ET0) / 10.0
        rows.append({
            "DAY": record.DAY,
            "LAT": latitude,
            "LON": 2.0,
            "ELEV": altitude,
            "IRRAD": float(record.SRAD) * 1.0e6,
            "TMIN": tmin,
            "TMAX": tmax,
            "TEMP": temp,
            "DTEMP": 0.5 * (temp + tmax),
            "RAIN": float(record.RAIN) / 10.0,
            "VAP": 10.0,
            "WIND": 2.0,
            "ET0": reference,
            "E0": reference,
            "ES0": reference,
        })
    return rows


def harvest_value(block, column, missing_zero=False):
    rows = block.dropna(subset=[column])
    if rows.empty:
        return None
    value = float(rows.iloc[-1][column])
    if missing_zero and value == 0.0:
        return None
    return value

def stage_day(block, column, sowing):
    rows = block.dropna(subset=[column])
    if rows.empty:
        return None
    day = parse_date(rows.iloc[-1][column])
    if day is None:
        return None
    return (day - sowing).days


def build_season(number, weather_cache):
    block = observations.loc[observations["Number"] == number]
    site = management.loc[management["Number"] == number].iloc[0]
    sowing = parse_date(block.iloc[0].SowingDate)
    harvest = parse_date(block.dropna(subset=["HarvestDate"]).iloc[-1].HarvestDate)
    station = int(site.Station)
    if station not in weather_cache:
        weather_cache[station] = station_weather(station)
    daily, latitude, altitude = weather_cache[station]
    end = min(harvest, daily["DAY"].max())
    layers, root_depth = with_root_boundary(soil_layers(site))
    clean = []
    for layer in layers:
        clean.append({key: value for key, value in layer.items() if key not in {"file_fc", "ptf_fc", "file_wp", "ptf_wp"}})
    description = {
        "PFWiltingPoint": 4.2,
        "PFFieldCapacity": 2.0,
        "SurfaceConductivity": 70.0,
        "SoilLayers": clean,
        "SubSoilType": {**clean[-1], "Thickness": 200.0, "FSOMI": 0.0},
        "GroundWater": None,
    }
    wav = cell_number(site, "Available_soil_water,_volumetric")
    wav = 20.0 if wav is None else min(wav / 10.0, 100.0)
    n_layers = len(clean)
    site_data = WOFOST81SiteDataProvider_SNOMIN(
        WAV=wav, CO2=400.0, A0SOM=24.0,
        NH4I=[2.0] + [0.0] * (n_layers - 1),
        NO3I=[10.0] + [0.0] * (n_layers - 1),
    )
    season_provider = ParameterProvider(
        cropdata=crop, soildata={"RDMSOL": root_depth, "SoilProfileDescription": description}, sitedata=site_data
    )
    season_provider.set_active_crop("winterwheat", "Arminda", "sowing", "harvest")
    biomass = []
    for record in block.itertuples(index=False):
        if pd.isna(record.Biomass):
            continue
        day = (parse_date(record.Date) - sowing).days
        biomass.append((day, float(record.Biomass) * 10.0))
    return {
        "number": int(number),
        "provider": season_provider,
        "rows": season_rows(daily, latitude, altitude, sowing, end),
        "agro": season_agro(sowing, end, dressing_events(site, sowing, end), irrigation_events(site, sowing, end)),
        "end": end,
        "rdmcr": root_depth,
        "biomass": biomass,
        "yield_gm2": harvest_value(block, "Grain_Yield"),
        "nitrogen": harvest_value(block, "N_in_biomassHarvest"),
        "protein": harvest_value(block, "ProteinContentGrain", missing_zero=True),
        "heading": stage_day(block, "Date_BBCH55", sowing),
        "maturity": stage_day(block, "Date_BBCH90", sowing),
        "checks": [(layer["file_fc"], layer["ptf_fc"], layer["file_wp"], layer["ptf_wp"]) for layer in layers],
    }


weather_cache = {}
seasons = [build_season(number, weather_cache) for number in sorted(observations["Number"].unique())]
fc_gap, wp_gap = [], []
for season in seasons:
    for file_fc, ptf_fc, file_wp, ptf_wp in season["checks"]:
        if file_fc is not None:
            fc_gap.append(abs(file_fc - ptf_fc))
        if file_wp is not None:
            wp_gap.append(abs(file_wp - ptf_wp))
print(
    f"Seasons: {len(seasons)}. "
    f"Biomass dates: {sum(len(season['biomass']) for season in seasons)}. "
    f"Yield: {sum(season['yield_gm2'] is not None for season in seasons)}. "
    f"Plant N: {sum(season['nitrogen'] is not None for season in seasons)}. "
    f"Protein: {sum(season['protein'] is not None for season in seasons)}."
)
print(
    f"Pedotransfer versus the file, mean absolute volumetric gap: "
    f"field capacity {sum(fc_gap)/len(fc_gap):.3f}, wilting point {sum(wp_gap)/len(wp_gap):.3f}."
)

current = {name: float(accepted[name]) for name in accepted}
for name in GROWTH_BOUNDS:
    current[name] = float(provider[name])
arminda_values = {name: defaults[name] for name in ("TSUM1", "TSUM2", "VERNSAT", "VERNBASE", "DLO")}
arminda_values.update({name: float(provider[name]) for name in GROWTH_BOUNDS})


def merged(active):
    values = {name: torch.tensor(float(current[name]), dtype=torch.float64) for name in current}
    values.update(active)
    return values


def simulate(season, values):
    season_provider = season["provider"]
    for name, value in values.items():
        if name in BOUNDS:
            season_provider.set_override(name, value, check=False)
    season_provider.set_override("RDMCR", season["rdmcr"], check=False)
    engine = Engine(full_config)
    engine.setup(season_provider, iter(season["rows"]), season["agro"])
    engine.run_till(season["end"])
    output = engine.get_output()
    return {name: torch.stack([day[name].reshape(()) for day in output]) for name in full_config.OUTPUT_VARS}


def season_ss(season, values, kind, output=None):
    if output is None:
        output = simulate(season, values)
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
    predicted = 100.0 * PROTEIN_FACTOR * output["NamountSO"][-1] / output["TWSO"][-1].clamp(min=1.0)
    return (predicted - season["protein"]).square()


def observation_count(kind):
    if kind == "biomass":
        return sum(len(season["biomass"]) for season in seasons)
    key = {"yield": "yield_gm2", "nitrogen": "nitrogen", "protein": "protein"}[kind]
    return sum(season[key] is not None for season in seasons)


def evaluate_kind(kind, values):
    frozen = {name: torch.tensor(float(values[name]), dtype=torch.float64) for name in values}
    total = 0.0
    with torch.no_grad():
        for season in seasons:
            total += float(season_ss(season, frozen, kind))
    return total


def _values(names, mapping):
    return ", ".join(f"{name} {float(mapping[name]):.4g}" for name in names)


def fit_kind(names, kind, max_steps=MAX_STEPS):
    """Adam on `names`, halving the rate when the sum of squares stalls."""
    pack = ParameterSet(names, current)
    lr = GROUP_LR
    optimizer = torch.optim.Adam(pack.parameters(), lr=lr)
    best_ss = float("inf")
    best_state = {key: value.detach().clone() for key, value in pack.state_dict().items()}
    waited, cuts, rewound = 0, 0, False
    print(f"  Start: {_values(names, current)}. A dot is one site-year.", flush=True)
    for step in range(max_steps):
        step_started = time.perf_counter()
        optimizer.zero_grad()
        total = 0.0
        print(f"  step {step + 1}/{max_steps} ", end="", flush=True)
        for season in seasons:
            piece = season_ss(season, merged(pack()), kind)
            if piece.requires_grad:
                piece.backward()
                total += float(piece.detach())
            print(".", end="", flush=True)
        elapsed = time.perf_counter() - step_started
        waited, cuts, rewound, action = rate_decision(total, best_ss, waited, cuts, rewound)
        if action == "improve":
            best_ss = total
            best_state = {key: value.detach().clone() for key, value in pack.state_dict().items()}
            mark = "best"
        elif action == "decay":
            pack.load_state_dict(best_state)
            lr *= LR_FACTOR
            optimizer = torch.optim.Adam(pack.parameters(), lr=lr)
            print(f"  SS {total:.4g}. {elapsed:.0f} s", flush=True)
            print(f"  Learning rate {lr:.4g}. Back at the best step.", flush=True)
            continue
        elif action == "stop":
            print(f"  SS {total:.4g}. {elapsed:.0f} s", flush=True)
            print(f"  Early stopping at step {step}.", flush=True)
            break
        else:
            mark = f"no gain for {waited}"
        print(f"  SS {total:.4g}. {mark}. {elapsed:.0f} s", flush=True)
        optimizer.step()
    else:
        print(f"  Reached the ceiling of {max_steps} steps.", flush=True)
    pack.load_state_dict(best_state)
    print("  Scoring the best step.", flush=True)
    frozen = {name: float(value.detach()) for name, value in pack().items()}
    return frozen, evaluate_kind(kind, {**current, **frozen})
if DO_NOT_RERUN:
    stored = load_results()
    selected = list(stored["selected"])
    current = {name: float(value) for name, value in stored["after_step7"].items()}
    after_step6 = {name: float(value) for name, value in stored["after_step6"].items()}
    group_rows = stored["group_rows"]
    group_variance = {key: float(value) for key, value in stored["group_variance"].items()}
    print("DO_NOT_RERUN is set, so the group fits and the weighted refit were loaded.")
else:
    phenology_selected = list(selected)
    group_rows = []
    group_variance = {"phenology": float(error_variance)}
    for spec in GROUPS:
        kind = spec["kind"]
        n_obs = observation_count(kind)
        free_majors = [name for name in spec["major"] if name not in selected]
        free_candidates = [name for name in spec["candidates"] if name not in selected]
        print(
            f"\n{spec['name']} ({kind}). n = {n_obs}. "
            f"Majors: {', '.join(free_majors) or 'already selected'}. "
            f"Candidates: {', '.join(free_candidates) or 'none'}.",
            flush=True,
        )
        chosen = []
        if free_majors:
            fitted, ss = fit_kind(free_majors, kind)
            current.update(fitted)
            chosen = list(free_majors)
        else:
            print("  Scoring the group at the parameters already selected.", flush=True)
            ss = evaluate_kind(kind, current)
        best_ss = ss
        best_score = None
        if chosen:
            best_score = float(aicc(torch.tensor(ss), n_obs, len(chosen)))
            group_rows.append({
                "Group": spec["name"], "Parameters": ", ".join(chosen),
                "SS": ss, "AICc": best_score, "Kept": "major",
            })
            print(f"  Majors kept. SS {ss:.1f} at the updated parameters, p = {len(chosen)}, AICc {best_score:.2f}.")
        for candidate in free_candidates:
            trial = chosen + [candidate]
            fitted, ss = fit_kind(trial, kind)
            score = float(aicc(torch.tensor(ss), n_obs, len(trial)))
            keep = best_score is None or score < best_score
            group_rows.append({
                "Group": spec["name"], "Parameters": ", ".join(trial),
                "SS": ss, "AICc": score, "Kept": "yes" if keep else "no",
            })
            if best_score is None:
                comparison = f"AICc {score:.2f}"
            else:
                comparison = f"AICc {score:.2f}, change {score - best_score:+.2f}"
            decision = "Kept." if keep else "Left at the default."
            print(f"  {candidate}: SS {ss:.1f} at the trial parameters, {comparison}. {decision}")
            if keep:
                current.update(fitted)
                chosen = trial
                best_ss = ss
                best_score = score
        if chosen:
            selected.extend(name for name in chosen if name not in selected)
        denom = n_obs - len(chosen)
        group_variance[kind] = best_ss / denom if denom > 0 else best_ss / n_obs
        held = _values(chosen, current) if chosen else "unchanged"
        print(f"  Weight SS/(n-p) = {group_variance[kind]:.4g}. {held}.")

    print("\nWeighted refit of every selected parameter.", flush=True)


    def fit_weighted(max_steps=MAX_STEPS):
        pack = ParameterSet(selected, current)
        lr = GROUP_LR
        optimizer = torch.optim.Adam(pack.parameters(), lr=lr)
        best_ss = float("inf")
        best_state = {key: value.detach().clone() for key, value in pack.state_dict().items()}
        waited, cuts, rewound = 0, 0, False
        print(f"  {', '.join(selected)}.", flush=True)
        print("  One full run per site-year.", flush=True)
        for step in range(max_steps):
            step_started = time.perf_counter()
            optimizer.zero_grad()
            total = 0.0
            print(f"  weighted step {step + 1}/{max_steps} ", end="", flush=True)
            values = merged(pack())
            for season in seasons:
                output = simulate(season, values)
                piece = output["DVS"].new_zeros(())
                if season["heading"] is not None and season["maturity"] is not None:
                    heading = stage_error(output["DVS"], 1.0, season["heading"])
                    maturity = stage_error(output["DVS"], 2.0, season["maturity"])
                    piece = heading.square() + maturity.square()
                piece = piece / group_variance["phenology"]
                for spec in GROUPS:
                    piece = piece + season_ss(season, values, spec["kind"], output) / group_variance[spec["kind"]]
                if piece.requires_grad:
                    piece.backward()
                    total += float(piece.detach())
                print(".", end="", flush=True)
            print(flush=True)
            elapsed = time.perf_counter() - step_started
            waited, cuts, rewound, action = rate_decision(total, best_ss, waited, cuts, rewound)
            if action == "improve":
                best_ss = total
                best_state = {key: value.detach().clone() for key, value in pack.state_dict().items()}
                mark = "best"
            elif action == "decay":
                pack.load_state_dict(best_state)
                lr *= LR_FACTOR
                optimizer = torch.optim.Adam(pack.parameters(), lr=lr)
                print(f"    objective {total:.2f}. {elapsed / 60:.1f} min.", flush=True)
                print(f"  Learning rate {lr:.4g}. Back at the best step.", flush=True)
                continue
            elif action == "stop":
                print(f"    objective {total:.2f}. {elapsed / 60:.1f} min.", flush=True)
                print(f"  Early stopping at step {step}.", flush=True)
                break
            else:
                mark = f"no gain for {waited}"
            print(
                f"    objective {total:.2f}. {mark}. {elapsed / 60:.1f} min.",
                flush=True,
            )
            optimizer.step()
        else:
            print(f"  Reached the ceiling of {max_steps} steps.", flush=True)
        pack.load_state_dict(best_state)
        return {name: float(value.detach()) for name, value in pack().items()}
    after_step6 = dict(current)
    current.update(fit_weighted())
    print("\nGroup decisions", flush=True)
    for row in group_rows:
        print(f"  {row['Group']}: {row['Parameters']}  SS {row['SS']:.1f}  AICc {row['AICc']:.2f}  {row['Kept']}")
    left = [name for name in GROWTH_BOUNDS if name not in selected]
    print("Selected:", ", ".join(selected))
    print("Growth parameters left at Arminda:", ", ".join(left) or "none")
    save_results({
        "phenology_history": history,
        "phenology_accepted": accepted,
        "phenology_selected": phenology_selected,
        "phenology_error_variance": error_variance,
        "selected": selected,
        "after_step6": after_step6,
        "after_step7": current,
        "group_rows": group_rows,
        "group_variance": group_variance,
    })
pd.DataFrame(group_rows)

