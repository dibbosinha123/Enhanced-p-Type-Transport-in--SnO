"""Doped alpha-SnO: literature-informed screening and Pareto analysis.
Nominal screening substitutions: 1, 2, 3, 5, 10, 15 and 20% of the designated host sublattice.

Five candidate dopants: Li, Na, K, N and P. No preferred dopant is hardcoded.
This is a screening workflow, NOT a charged-defect DFT or Hall simulator.
ML learns MP compound formation energies (eV/atom).
Literature activation energies are INPUTS.
No experimental results are fabricated or used to force Na to rank first.

Gradient Boosting is the only trained ML model. Execution downloads MP data,
performs grouped/nested hyperparameter tuning,
held-out testing, leave-one-element-out transfer tests, group bootstrapping,
transport scenarios, fixed-concentration Pareto analysis, and uncertainty tests.
It writes a timestamped results directory and a ZIP.
Compact summaries and main figures appear in Colab; full tables stay in the ZIP.
Open START_HERE.html after extracting the ZIP for the illustrated results guide.
Conductivity is reported in S/m; mobility in cm^2/(V s); energy in eV/atom.
An explicit 0-100 preference score selects a candidate at one fixed concentration.
After the run, RESULT_TABLES contains every exported CSV as a pandas DataFrame.
See CONFIG below. RUN_SIZE='publication' increases resampling substantially.
Optional experimental and DFT CSV schemas are exported with every run.
Literature inputs and all assumptions are exported; result interpretation is generated from each run.
"""

import os
import sys
import json
import math
import warnings
import subprocess
import importlib.metadata as metadata
from pathlib import Path
from datetime import datetime, timezone
from functools import lru_cache
from copy import deepcopy
from getpass import getpass


def ensure_dependencies():
    """Install only missing dependencies before importing scientific modules."""
    requirements = {
        "numpy": "numpy>=1.26,<3", "scipy": "scipy>=1.13,<2",
        "pandas": "pandas>=2.2,<3", "sklearn": "scikit-learn>=1.5,<2",
        "matplotlib": "matplotlib>=3.8,<4", "joblib": "joblib>=1.4,<2",
        "pymatgen.core": "pymatgen==2026.5.4", "mp_api.client": "mp-api==0.46.5",
    }
    missing = []
    for module, package in requirements.items():
        result = subprocess.run([sys.executable, "-c", f"import {module}"],
                                capture_output=True, text=True)
        if result.returncode:
            missing.append(package)
    if missing:
        print("Installing scientific dependencies. This can take a few minutes.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
        probe = subprocess.run([sys.executable, "-c",
                               "import numpy,scipy,pandas,sklearn; from mp_api.client import MPRester"],
                              capture_output=True, text=True)
        if probe.returncode:
            raise RuntimeError("Dependency import failed. Start a fresh Colab runtime and run again. "
                               "Do not continue with partially imported packages.")


if __name__ == "__main__":
    ensure_dependencies()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.constants import elementary_charge, Boltzmann, h, m_e
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.inspection import permutation_importance
import joblib
from pymatgen.core import Composition, Element, Structure
from pymatgen.io.vasp import Poscar
from mp_api.client import MPRester


# ------------------------- EDITABLE CONFIGURATION -------------------------
API_KEY = '4QoUiunPSMRpqOLTwA6qRu8edPSBArZD'

CONFIG = {
    "seed": 42,
    "run_size": "standard",                 # standard / publication
    "temperature_K": 300.0,
    "mp_hull_max_eV_atom": 0.25,             # dataset filter, not an ML feature
    "include_gnome": False,
    "use_structure_features": False,        # see explanation below
    "outer_folds": 5,
    "inner_folds": 3,
    "test_fraction": 0.20,
    "bootstrap_models": 60,
    "mc_draws": 1000,
    "n_jobs": -1,
    "cv_mae_gate_eV_atom": 0.15,             # predeclared screening tolerance
    "transfer_mae_gate_eV_atom": 0.20,       # not a published physical constant
    "domain_quantile": 0.95,
    "max_out_of_range_fraction": 0.15,
    "dilute_screening_limit_pct": 3.0,       # model-scope limit, NOT solubility
    "compromise_tolerance": 0.03,           # distance ties are retained
    # The sole grid used for virtual energies, transport, Pareto fronts and scores.
    "requested_pareto_pct": [1, 2, 3, 5, 10, 15, 20],
    "make_dft_starting_structures": True,
    "structure_target_pct": 2.0,
    "download_reference_dos": False,        # optional; MP DOS may be absent
    "experimental_csv": "",                # optional, exported schema below
    "dft_validation_csv": "",              # optional, exact-composition energies only
    "cached_raw_json": "",                 # optional previous raw_materials.json
    "output_parent": "/content" if Path("/content").exists() else ".",
    "auto_download_zip_in_colab": True,
    "display_results_inline": True,         # show the report in the notebook
    "display_tables_inline": True,          # small summary tables only
    "display_figures_inline": True,         # every saved plot, including optional DOS
    "inline_max_table_rows": 12,            # larger tables are available in the ZIP
    "inline_table_height_px": 360,
    "inline_figure_width_px": 1000,
    # Preference weights, not fitted parameters. Compare at the SAME concentration.
    # Conductivity already includes mobility; the extra mobility weight deliberately
    # values mobility retention. Weight/removal sensitivity is exported separately.
    "score_reference_pct": 5.0,             # % of the substituted Sn or O sublattice
    "score_weights": {"formation_energy": .25, "conductivity": .50, "mobility": .25},
    "score_weight_draws": 2000,             # uniform-simplex preference sensitivity
    "score_close_margin_points": 2.0,
}

# Populated at the end of the run; use these DataFrames in later Colab cells.
# Example: display(RESULT_TABLES["screening_results"])
RESULT_TABLES = {}

# Structure descriptors are still exported. Composition-only inference is the
# default because a relaxed doped crystal is not supplied. Setting the option
# True uses ideal host geometry for virtual structures and labels that assumption.
# Compare this as an ablation; do not describe host geometry as relaxed doped DFT.
DOPANTS = ["Li", "Na", "K", "N", "P"]
EXCLUDED_ELEMENTS = {"Ga"}  # Excluded from the filtered training data and all candidate outputs.
MODEL_NAME = "GradientBoosting"
SITES = {d: ("O" if d in ["N", "P"] else "Sn") for d in DOPANTS}
ROLES = {"Li": "acceptor_theory", "Na": "acceptor_experiment",
         "K": "acceptor_experiment", "N": "acceptor_experiment",
         "P": "acceptor_hypothesis"}
ACCEPTORS = [d for d in DOPANTS if ROLES[d] != "donor_control"]

# Central value and scenario bounds, eV. These are NOT ML labels.
# Apparent Hall activation energies are process/concentration dependent and are
# not generally identical to thermodynamic charge-transition levels. Using them
# in a single-level acceptor model below is an explicit screening approximation.
EA = {
    "Na": (.020, .015, .025, "10.1021/acs.jpcc.2c06397", "reported_effective_activation"),
    "K":  (.015, .010, .020, "10.1063/5.0288742", "reported_effective_activation"),
    "N":  (.125, .100, .150, "10.1063/1.5052606", "reported_effective_activation"),
    "Li": (.060, .005, .150, "", "unvalidated_scenario_assumption"),
    "P":  (.150, .030, .400, "", "unvalidated_scenario_assumption"),
}

# Use ONE covalent-radius convention for geometric comparison; these values
# are proxies and do not determine an oxidation state or a defect charge state.
# Cordero covalent radii, angstrom; geometric proxies, not actual oxide bond radii.
RADII = {"Li": 1.28, "Na": 1.66, "K": 2.03, "N": .71, "P": 1.07,
         "Sn": 1.39, "O": .66}
CHI = {"Li": .98, "Na": .93, "K": .82, "N": 3.04, "P": 2.19,
       "Sn": 1.96, "O": 3.44}

# Explicit assumptions, NOT fitted material parameters. Replace with measured
# values for a specified process before attempting a quantitative device claim.
PHYSICS = {
    "p0_cm3": 1e18, "mu0_cm2_Vs": 10.0,
    "m_dos_over_me": .5, "m_transport_over_me": .5,
    "active_fraction": .30, "compensation_fraction": .0,
    "acceptor_degeneracy": 4.0, "mass_penalty": .65,
    "alloy_scattering": 7.0, "ion_scattering": 2.5,
    "distortion_scale": 1.0,
    "indirect_gap_guard_eV": .67,            # Allen 2013; not the 2.7 eV direct gap
}
# The reference curve assumes no compensating donors. Unknown compensation is
# explored separately over 0-50% in Monte Carlo; no dopant-specific fit is imposed.

REFERENCES = [
    ("Kwok_2022", "10.1021/acs.jpcc.2c06397", "SnO", "Na activation inputs; sputtered-film benchmarks"),
    ("Becker_2019", "10.1063/1.5052606", "SnO", "N activation; process-specific morphology limitation"),
    ("Yang_2016", "10.4028/www.scientific.net/MSF.848.477", "SnO", "Na implantation and annealing; qualitative evidence"),
    ("Chae_2025", "10.1063/5.0288742", "alpha-SnO", "K shallow acceptor and Hall transport; K must remain a serious candidate"),
    ("Varley_2013", "10.1063/1.4819068", "SnO", "Native defects and hydrogen complexes; chemical-potential dependence"),
    ("Togo_2006", "10.1103/PhysRevB.74.195128", "SnO", "Native-defect thermodynamics"),
    ("Allen_2013", "10.1039/C3TC31863J", "SnO", "Defect chemistry; direct and indirect gaps differ"),
    ("Grauzinyte_2017", "10.1021/acs.chemmater.7b03862", "SnO2", "Different host: methodological reference only; excluded from calibration"),
    ("Grauzinyte_2018", "10.1103/PhysRevMaterials.2.104604", "SnO", "Alkali acceptor screening; added relevant SnO paper"),
    ("Lee_2026", "10.1063/5.0332035", "alpha-SnO", "UID transport depends on growth flux; no universal p0 or mobility"),
    ("Cordero_2008", "10.1039/b801115j", "elements", "Consistent covalent-radius descriptor convention"),
    ("OECD_JRC_2008", "10.1787/9789264043466-en", "methodology", "Composite-score normalization, weighting and sensitivity; not physical calibration"),
]

BENCHMARKS = [
    dict(dopant="Na", property="effective_activation_eV", low=.015, high=.025,
         source="10.1021/acs.jpcc.2c06397", use="input_not_validation", context="sputtering and RTA"),
    dict(dopant="Na", property="p_cm3", low=4e19, high=5e19,
         source="10.1021/acs.jpcc.2c06397", use="unpaired_literature_range",
         context="workbook summary; 2.3-2.8 percent basis requires full-text verification"),
    dict(dopant="Na", property="mu_cm2_Vs", low=10., high=np.nan,
         source="10.1021/acs.jpcc.2c06397", use="unpaired_lower_bound", context="not a sample-matched mobility row"),
    dict(dopant="K", property="effective_activation_eV", low=.010, high=.020,
         source="10.1063/5.0288742", use="input_not_validation", context="suboxide MBE"),
    dict(dopant="K", property="p_cm3", low=4.8e17, high=1.5e19,
         source="10.1063/5.0288742", use="unpaired_literature_range", context="reported without significant mobility loss"),
    dict(dopant="N", property="effective_activation_eV", low=.100, high=.150,
         source="10.1063/1.5052606", use="input_not_validation", context="plasma-assisted MBE"),
    dict(dopant="N", property="morphology_onset_dopant_cm3", low=7e17, high=7e17,
         source="10.1063/1.5052606", use="process_specific_caution", context="not a universal solubility boundary"),

]


def write_json(path, value):
    def convert(x):
        if isinstance(x, (np.integer, np.floating)): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, Path): return str(x)
        if hasattr(x, "as_dict"): return x.as_dict()
        if hasattr(x, "model_dump"): return x.model_dump(mode="json")
        return str(x)
    Path(path).write_text(json.dumps(value, indent=2, default=convert), encoding="utf-8")


def csv_out(frame, path):
    frame.to_csv(path, index=False, float_format="%.10g")


def get_api_key():
    key = API_KEY.strip() or os.environ.get("MP_API_KEY") or os.environ.get("PMG_MAPI_KEY")
    if not key:
        try:
            from google.colab import userdata
            key = userdata.get("MP_API_KEY")
        except Exception:
            pass
    if not key:
        key = getpass("Materials Project API key (hidden): ").strip()
    if not key:
        raise ValueError("A Materials Project API key or a saved raw-data JSON is required.")
    return key


def fetch_data(cfg, out, api_key=None):
    if cfg["cached_raw_json"]:
        raw = json.loads(Path(cfg["cached_raw_json"]).read_text())
        if not {"ternaries", "hosts", "query"} <= raw.keys():
            raise ValueError("Cache must be raw_materials.json produced by this script.")
        old = raw["query"]
        if old["hull_max"] != cfg["mp_hull_max_eV_atom"] or old["include_gnome"] != cfg["include_gnome"]:
            raise ValueError("Cached query differs from CONFIG. Match settings or download fresh data.")
    else:
        fields = ["material_id", "formula_pretty", "structure", "symmetry",
                  "formation_energy_per_atom", "energy_above_hull", "band_gap",
                  "is_metal", "theoretical"]
        def serialize(doc):
            obj = {f: getattr(doc, f, None) for f in fields}
            obj["material_id"] = str(obj["material_id"])
            obj["structure"] = obj["structure"].as_dict()
            sym = obj["symmetry"]
            obj["symmetry"] = sym.model_dump(mode="json") if hasattr(sym, "model_dump") else sym
            return obj
        print("Downloading nondeprecated Sn-O-A ternaries and alpha-SnO references...")
        with MPRester(api_key or get_api_key(), timeout=60) as mpr:
            ternaries = mpr.materials.summary.search(
                elements=["Sn", "O"], num_elements=3, deprecated=False,
                energy_above_hull=(0., cfg["mp_hull_max_eV_atom"]),
                include_gnome=cfg["include_gnome"], fields=fields)
            hosts = mpr.materials.summary.search(formula="SnO", deprecated=False,
                                                 include_gnome=False, fields=fields)
            raw = {"ternaries": [serialize(d) for d in ternaries],
                   "hosts": [serialize(d) for d in hosts],
                   "query": {"hull_max": cfg["mp_hull_max_eV_atom"],
                             "include_gnome": cfg["include_gnome"],
                             "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                             "database_version": mpr.db_version}}
    write_json(out / "raw_materials.json", raw)
    csv_out(pd.DataFrame([{k: d.get(k) for k in ["material_id", "formula_pretty",
        "formation_energy_per_atom", "energy_above_hull", "band_gap", "is_metal", "theoretical"]}
        for d in raw["ternaries"]+raw["hosts"]]), out / "raw_materials.csv")
    return raw


def numeric(value):
    try:
        v = float(value)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


@lru_cache(maxsize=None)
def element_features(symbol):
    e = Element(symbol)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {k: numeric(getattr(e, k, None)) for k in
                ["Z", "atomic_mass", "X", "atomic_radius", "row", "group",
                 "mendeleev_no", "electron_affinity", "ionization_energy"]}


def features(comp, structure, a_symbol=None, virtual=False):
    """Scale-invariant composition descriptors; never use formula atom count."""
    fractions = {str(e): float(comp.get_atomic_fraction(e)) for e in comp.elements}
    records = {s: element_features(s) for s in fractions}
    ans = {"Sn_fraction": fractions.get("Sn", 0.), "O_fraction": fractions.get("O", 0.),
           "A_fraction": sum(v for s, v in fractions.items() if s not in ["Sn", "O"]),
           "entropy": -sum(v * np.log(v) for v in fractions.values() if v > 0)}
    for key in next(iter(records.values())):
        pairs = [(fractions[s], records[s][key]) for s in fractions if np.isfinite(records[s][key])]
        if pairs:
            w, v = np.array(pairs).T
            w = w / w.sum()
            mean = np.sum(w * v)
            ans[f"mean_{key}"] = mean
            ans[f"std_{key}"] = np.sqrt(np.sum(w * (v-mean)**2))
        else:
            ans[f"mean_{key}"] = ans[f"std_{key}"] = np.nan
        ans[f"A_weighted_{key}"] = (ans["A_fraction"] * element_features(a_symbol)[key]
                                     if a_symbol else 0.)
    vpa = structure.volume / len(structure)
    avg_mass = sum(fractions[s] * element_features(s)["atomic_mass"] for s in fractions)
    # Recompute mass density for virtual composition at fixed volume per atom.
    ans["struct_volume_per_atom"] = vpa
    ans["struct_density"] = avg_mass * 1.66053906660 / vpa
    lengths = np.sort(structure.lattice.abc)
    ans["struct_axis_ratio_mid"] = lengths[1] / lengths[0]
    ans["struct_axis_ratio_long"] = lengths[2] / lengths[0]
    ans["struct_angle_deviation"] = np.mean(np.abs(np.array(structure.lattice.angles)-90.))
    return ans


def curate(raw, out):
    hosts = []
    for d in raw["hosts"]:
        s = Structure.from_dict(d["structure"])
        sym = d.get("symmetry") or {}
        if (s.composition.reduced_formula == "SnO" and sym.get("number") == 129
                and d.get("is_metal") is not True
                and np.isfinite(numeric(d.get("formation_energy_per_atom")))):
            hosts.append(d)
    if not hosts:
        raise RuntimeError("No semiconducting P4/nmm (129) SnO reference found. Host selection cannot be guessed.")
    host = min(hosts, key=lambda d: (numeric(d.get("energy_above_hull")), len(d["structure"]["sites"])))
    hstruct = Structure.from_dict(host["structure"])
    rows, rejected, seen = [], [], set()
    for d in raw["ternaries"] + [host]:
        mid = d["material_id"]
        if mid in seen:
            continue
        seen.add(mid)
        try:
            s = Structure.from_dict(d["structure"])
            if not s.is_ordered: raise ValueError("disordered structure")
            comp = s.composition
            others = [str(e) for e in comp.elements if str(e) not in ["Sn", "O"]]
            if not {"Sn", "O"} <= {str(e) for e in comp.elements}: raise ValueError("wrong chemical space")
            if len(others) != (0 if mid == host["material_id"] else 1): raise ValueError("not an A-Sn-O ternary")
            y = numeric(d.get("formation_energy_per_atom"))
            if not np.isfinite(y): raise ValueError("missing formation energy")
            a = others[0] if others else None
            if a in EXCLUDED_ELEMENTS: raise ValueError("element explicitly excluded from this workflow")
            row = {"material_id": mid, "formula": comp.reduced_formula, "A": a or "HOST",
                   "formation_energy_eV_atom": y, "hull_eV_atom": numeric(d.get("energy_above_hull")),
                   "band_gap_eV_audit_only": numeric(d.get("band_gap")),
                   "nsites_audit_only": len(s), "source": "Materials_Project"}
            row.update(features(comp, s, a))
            rows.append(row)
        except (ValueError, TypeError, KeyError) as exc:
            rejected.append({"material_id": mid, "reason": str(exc)})
    data = pd.DataFrame(rows).sort_values("material_id").reset_index(drop=True)
    if data.formula.nunique() < 15:
        raise RuntimeError("Fewer than 15 independent compositions. This grouped ML protocol is under-supported.")
    csv_out(data, out / "curated_dataset.csv")
    csv_out(pd.DataFrame(rejected, columns=["material_id", "reason"]), out / "rejected_records.csv")
    write_json(out / "host_reference.json", host)
    csv_out(data.groupby("A").agg(n_materials=("material_id", "size"),
            n_compositions=("formula", "nunique"), min_A_fraction=("A_fraction", "min"),
            max_A_fraction=("A_fraction", "max")).reset_index(), out / "chemical_coverage.csv")
    return data, host, hstruct


def get_feature_columns(data, cfg):
    audit = {"material_id", "formula", "A", "formation_energy_eV_atom", "hull_eV_atom",
             "band_gap_eV_audit_only", "nsites_audit_only", "source"}
    return [c for c in data if c not in audit and (cfg["use_structure_features"] or not c.startswith("struct_"))]


def gradient_boosting_spec(cfg):
    """Only one estimator family; all tuning, transfer and bootstrap fits use GB."""
    n = 400 if cfg["run_size"] == "publication" else 250
    if cfg.get("test_mode"): n = 12
    estimator = GradientBoostingRegressor(n_estimators=n, random_state=cfg["seed"], loss="huber")
    grid = {"model__learning_rate": [.03, .08], "model__max_depth": [2, 3]}
    return estimator, grid


def pipeline(estimator):
    return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("variance", VarianceThreshold()), ("scale", StandardScaler()),
                     ("model", clone(estimator))])


def fit_search(X, y, groups, cfg):
    estimator, grid = gradient_boosting_spec(cfg)
    if cfg.get("test_mode"): grid = {k: v[:1] for k, v in grid.items()}
    cv = GroupKFold(n_splits=min(cfg["inner_folds"], len(np.unique(groups))))
    if cv.n_splits < 2: raise ValueError("Too few composition groups for tuning.")
    search = GridSearchCV(pipeline(estimator), grid, scoring="neg_mean_absolute_error", cv=cv,
                         n_jobs=cfg["n_jobs"], error_score="raise", refit=True)
    search.fit(X, y, groups=groups)
    return search


def regression_metrics(y, pred):
    return {"MAE_eV_atom": mean_absolute_error(y, pred),
            "RMSE_eV_atom": np.sqrt(mean_squared_error(y, pred)),
            "R2": r2_score(y, pred) if len(y) > 1 and np.std(y) > 0 else np.nan}


def train_gradient_boosting(data, columns, cfg, out):
    """Nested tuning of a prespecified GB model; no competing model is trained."""
    X, y, groups = data[columns], data.formation_energy_eV_atom.to_numpy(), data.formula.to_numpy()
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=cfg["test_fraction"],
                        random_state=cfg["seed"]).split(X, y, groups))
    assert not set(groups[train]) & set(groups[test])
    split = data[["material_id", "formula", "A"]].copy()
    split["partition"] = "development"
    split.loc[test, "partition"] = "locked_test"
    csv_out(split, out / "data_splits.csv")
    folds = list(GroupKFold(n_splits=min(cfg["outer_folds"], len(np.unique(groups[train]))))
                 .split(X.iloc[train], y[train], groups[train]))
    predictions, metrics = np.full(len(train), np.nan), []
    print(f"Gradient Boosting nested grouped CV: {len(train)} development rows, {len(test)} locked test rows.")
    for fold, (itr, iva) in enumerate(folds):
        tr, va = train[itr], train[iva]
        assert not set(groups[tr]) & set(groups[va])
        search = fit_search(X.iloc[tr], y[tr], groups[tr], cfg)
        predictions[iva] = search.predict(X.iloc[va])
        metrics.append({"model": MODEL_NAME, "stage": "nested_cv_fold", "fold": fold+1,
                        "n": len(va), **regression_metrics(y[va], predictions[iva])})
    stats = regression_metrics(y[train], predictions)
    metrics.append({"model": MODEL_NAME, "stage": "nested_cv_pooled", "fold": 0, "n": len(train), **stats})
    print(f"  {MODEL_NAME}: nested CV MAE = {stats['MAE_eV_atom']:.4f} eV/atom"
          f" | R2 = {stats['R2']:.4f} | RMSE = {stats['RMSE_eV_atom']:.4f} eV/atom")
    tuned = fit_search(X.iloc[train], y[train], groups[train], cfg)
    test_predictions = tuned.predict(X.iloc[test])
    test_stats = regression_metrics(y[test], test_predictions)
    metrics.append({"model": MODEL_NAME, "stage": "locked_test", "fold": 0,
                    "n": len(test), **test_stats})
    print(f"  {MODEL_NAME}: locked test MAE = {test_stats['MAE_eV_atom']:.4f} eV/atom"
          f" | R2 = {test_stats['R2']:.4f} | RMSE = {test_stats['RMSE_eV_atom']:.4f} eV/atom")
    table = pd.DataFrame(metrics)
    csv_out(table, out / "model_metrics.csv")
    csv_out(table[table.stage.isin(["nested_cv_pooled", "locked_test"])][
        ["model", "stage", "n", "MAE_eV_atom", "RMSE_eV_atom", "R2"]], out / "gradient_boosting_metrics.csv")
    oof = data.iloc[train][["material_id", "formula", "A"]].copy().reset_index(drop=True)
    oof["model"], oof["observed"], oof["predicted"] = MODEL_NAME, y[train], predictions
    oof["residual"] = oof.observed-oof.predicted
    csv_out(oof, out / "nested_oof_predictions.csv")
    held = data.iloc[test][["material_id", "formula", "A"]].copy()
    held["observed_eV_atom"], held["predicted_eV_atom"] = y[test], test_predictions
    csv_out(held, out / "locked_test_predictions.csv")
    perm = permutation_importance(tuned.best_estimator_, X.iloc[test], y[test], scoring="neg_mean_absolute_error",
                                 n_repeats=5, random_state=cfg["seed"], n_jobs=cfg["n_jobs"])
    csv_out(pd.DataFrame({"feature": columns, "MAE_increase_eV_atom": perm.importances_mean,
                         "std": perm.importances_std}).sort_values("MAE_increase_eV_atom", ascending=False),
            out / "feature_importance_diagnostic.csv")
    # An arithmetic training-mean reference is a sanity check, not a second ML fit.
    mean_reference = float(np.mean(y[train]))
    reference_mae = float(np.mean(np.abs(y[test]-mean_reference)))
    reliable = bool(stats["MAE_eV_atom"] <= cfg["cv_mae_gate_eV_atom"]
                    and test_stats["MAE_eV_atom"] < reference_mae)
    final_model = clone(tuned.best_estimator_).fit(X, y)
    joblib.dump(final_model, out / "formation_energy_model.joblib")
    write_json(out / "model_selection.json", {
        "model": MODEL_NAME, "model_family_selection": "prespecified_by_user",
        "hyperparameters_selected_by": "grouped CV MAE on development data only",
        "passes_screening_quality_gate": reliable,
        "best_parameters_development_only": tuned.best_params_,
        "arithmetic_training_mean_reference_eV_atom": mean_reference,
        "arithmetic_reference_test_MAE_eV_atom": reference_mae,
        "reference_note": "Training mean computed directly; no additional ML model is fitted.",
        "feature_columns": columns, "target": "MP compound formation energy eV/atom",
        "test_used_for_selection": False,
        "uncertainty_note": "GB bootstrap spread and empirical residual envelopes are not calibrated OOD coverage."})
    return final_model, MODEL_NAME, reliable, oof, table[table.stage == "nested_cv_pooled"].set_index("model")


def leave_one_element_out(data, columns, cfg, out):
    """Retune only Gradient Boosting without any of the withheld dopant's rows."""
    records, predictions = [], []
    for dopant in DOPANTS:
        mask = data.A.eq(dopant).to_numpy()
        if not mask.any():
            records.append({"dopant": dopant, "model": MODEL_NAME, "n_test": 0,
                            "status": "no_MP_labels", "MAE_eV_atom": np.nan,
                            "RMSE_eV_atom": np.nan})
            continue
        train, test = data.loc[~mask], data.loc[mask]
        search = fit_search(train[columns], train.formation_energy_eV_atom, train.formula.to_numpy(), cfg)
        pred = search.predict(test[columns])
        records.append({"dopant": dopant, "n_test": len(test), "model": MODEL_NAME, "status": "evaluated",
                        "MAE_eV_atom": mean_absolute_error(test.formation_energy_eV_atom, pred),
                        "RMSE_eV_atom": np.sqrt(mean_squared_error(test.formation_energy_eV_atom, pred))})
        predictions.extend(dict(dopant=dopant, material_id=row.material_id,
                                observed=row.formation_energy_eV_atom, predicted=p,
                                residual=row.formation_energy_eV_atom-p)
                           for (_, row), p in zip(test.iterrows(), pred))
        print(f"  Withheld {dopant}: {len(test)} rows, MAE {records[-1]['MAE_eV_atom']:.4f}"
              f" eV/atom | RMSE = {records[-1]['RMSE_eV_atom']:.4f} eV/atom")
    table = pd.DataFrame(records)
    csv_out(table, out / "leave_one_dopant_out_metrics.csv")
    csv_out(pd.DataFrame(predictions), out / "leave_one_dopant_out_predictions.csv")
    return table


# ------------------- VIRTUAL COMPOSITIONS AND ML UNCERTAINTY ----------------
def virtual_grid(host_structure, cfg):
    # Use only the requested substitutions; the pristine host remains the training
    # reference and supplies the undoped physical baseline, not a virtual grid row.
    concentrations = sorted(set(cfg["requested_pareto_pct"]))
    rows, descriptors = [], []
    # x always means fraction of the substituted sublattice, not total atomic %.
    for dopant in DOPANTS:
        for pct in concentrations:
            x = pct / 100.
            amounts = {"Sn": 1., "O": 1.}
            amounts[SITES[dopant]] -= x
            if x > 0: amounts[dopant] = x
            comp = Composition(amounts)
            rows.append({"dopant": dopant, "site": SITES[dopant], "role": ROLES[dopant],
                         "substitution_pct": pct, "total_atom_pct": pct/2.,
                         "x": x, "virtual_formula": comp.formula,
                         "geometry": "ideal_host_unrelaxed", "EA_source": EA[dopant][3],
                         "EA_evidence": EA[dopant][4],
                         "within_dilute_model_scope": pct <= cfg["dilute_screening_limit_pct"]})
            descriptors.append(features(comp, host_structure, dopant if x > 0 else None, virtual=True))
    return pd.DataFrame(rows), pd.DataFrame(descriptors)


def applicability_domain(data, X, Xnew, candidates, cfg):
    prep = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scale", RobustScaler())])
    a = prep.fit_transform(X)
    b = prep.transform(Xnew)
    # Reference distances always exclude every polymorph of the same formula.
    max_group_size = int(data.groupby("formula").size().max())
    k = min(len(data), max_group_size + 1)
    nn = NearestNeighbors(n_neighbors=k).fit(a)
    ds, ids = nn.kneighbors(a)
    labels = data.formula.to_numpy()
    reference = np.array([dist[np.flatnonzero(labels[index] != labels[i])[0]]
                          for i, (dist, index) in enumerate(zip(ds, ids))])
    threshold = float(np.quantile(reference, cfg["domain_quantile"]))
    distances = nn.kneighbors(b, n_neighbors=1)[0][:, 0]
    lo, hi = np.nanmin(a, axis=0), np.nanmax(a, axis=0)
    varying = hi-lo > 1e-10
    outside = np.mean((b[:, varying] < lo[varying]-1e-9) | (b[:, varying] > hi[varying]+1e-9), axis=1)
    chemistry_support = []
    for _, row in candidates.iterrows():
        pool = data.loc[data.A.eq(row.dopant), "A_fraction"]
        chemistry_support.append(bool(row.x == 0 or (len(pool) > 0
            and pool.min()-1e-10 <= row.x/2. <= pool.max()+1e-10)))
    return pd.DataFrame({"domain_distance": distances, "domain_reference_threshold": threshold,
                         "domain_reference_quantile": cfg["domain_quantile"],
                         "out_of_range_feature_fraction": outside,
                         "within_A_fraction_training_range": chemistry_support,
                         "in_domain": (distances <= threshold+1e-10)
                            & (outside <= cfg["max_out_of_range_fraction"])
                            & np.array(chemistry_support)})


def bootstrap_predictions(model, data, columns, Xnew, cfg):
    """Cluster bootstrap: all polymorphs of a sampled composition stay together."""
    rng = np.random.default_rng(cfg["seed"] + 101)
    groups = data.formula.to_numpy()
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    draws = []
    for b in range(cfg["bootstrap_models"]):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indexes = np.concatenate([members[g] for g in sampled])
        estimator = clone(model)
        if "model__random_state" in estimator.get_params():
            estimator.set_params(model__random_state=cfg["seed"]+b+1)
        estimator.fit(data.iloc[indexes][columns], data.iloc[indexes].formation_energy_eV_atom)
        draws.append(estimator.predict(Xnew))
        if (b+1) % 20 == 0:
            print(f"  Composition bootstraps: {b+1}/{cfg['bootstrap_models']}")
    return np.asarray(draws)


def phase_proxy(data, dopant, rng=None):
    """Legacy competing-ternary descriptor; NOT a phase diagram or solubility."""
    pool = data.loc[data.A.eq(dopant), ["formation_energy_eV_atom", "hull_eV_atom"]].dropna()
    if pool.empty: return np.nan
    if rng is not None:
        pool = pool.iloc[rng.integers(0, len(pool), len(pool))]
    return max(0., -np.quantile(pool.formation_energy_eV_atom, .1)) * np.exp(-pool.hull_eV_atom.min()/.05)


# ---------------------- TRANSPORT SCENARIO CALCULATIONS --------------------
@lru_cache(maxsize=1)
def fermi_table():
    """Normalized F_{1/2}; quadrature with t=u^2 removes endpoint singularity."""
    from numpy.polynomial.legendre import leggauss
    eta = np.linspace(-20., 80., 5001)
    nodes, weights = leggauss(240)
    upper = np.sqrt(130.)
    u, w = (nodes+1)*upper/2., weights*upper/2.
    integral = (4./np.sqrt(np.pi)) * ((u*u*w)[None, :] * expit(eta[:, None]-u[None, :]**2)).sum(axis=1)
    return eta, np.log(integral)


def fermi_half(eta):
    eta = np.asarray(eta, dtype=float)
    grid, logvalues = fermi_table()
    clipped = np.clip(eta, grid[0], grid[-1])
    values = np.exp(np.interp(clipped, grid, logvalues))
    values = np.where(eta < grid[0], np.exp(np.clip(eta, -745, 0)), values)
    high = np.maximum(eta, 1.)
    sommerfeld = 4./(3.*np.sqrt(np.pi))*high**1.5*(1.+np.pi**2/(8.*high**2))
    return np.where(eta > grid[-1], sommerfeld, values)


def solve_holes(NA_cm3, activation_eV, params, temperature_K):
    """Single acceptor level; ionized compensating donors; fixed UID charge.

    p(E_F)+N_D = p0 + N_A/[1+g_A exp((E_A-E_F)/kT)].
    p = N_v F_1/2(-E_F/kT), E_v=0. Electrons are neglected in this p-type model.
    Degenerate hole statistics are included; isolated-acceptor/rigid-band physics
    remains an approximation at high doping. NaN is returned for donor controls.
    """
    na, ea = np.broadcast_arrays(np.asarray(NA_cm3, float), np.asarray(activation_eV, float))
    result = np.full(na.shape, np.nan)
    valid = np.isfinite(na) & np.isfinite(ea) & (na >= 0)
    if not np.any(valid): return result, result.copy(), result.copy()
    kT = Boltzmann * temperature_K / elementary_charge
    Nv = 2.*(2.*np.pi*params["m_dos_over_me"]*m_e*Boltzmann*temperature_K/h**2)**1.5/1e6
    n, e = na[valid], ea[valid]
    nd = params["compensation_fraction"] * n
    low, high = np.full_like(n, -3.), np.full_like(n, 3.)
    def charge(ef):
        p = Nv * fermi_half(-ef/kT)
        ionized = n * expit((ef-e)/kT - np.log(params["acceptor_degeneracy"]))
        return p+nd-params["p0_cm3"]-ionized
    if np.any(charge(low) < 0) or np.any(charge(high) > 0):
        raise ValueError("Charge-neutrality root outside bracket. Check concentrations and physical parameters.")
    for _ in range(70):
        mid = (low+high)/2.
        positive = charge(mid) > 0
        low = np.where(positive, mid, low)
        high = np.where(positive, high, mid)
    ef = (low+high)/2.
    result[valid] = Nv * fermi_half(-ef/kT)
    frac, fermi = np.full_like(result, np.nan), np.full_like(result, np.nan)
    frac[valid] = expit((ef-e)/kT - np.log(params["acceptor_degeneracy"]))
    fermi[valid] = ef
    return result, frac, fermi


def transport(candidates, nsite_cm3, params, cfg, ea_values=None, efficiency=None, radius_penalty=True):
    x = candidates.x.to_numpy(float)
    d = candidates.dopant.to_numpy()
    ea = np.array([EA[s][0] for s in d]) if ea_values is None else np.asarray(ea_values, float)
    eta = params["active_fraction"] if efficiency is None else np.asarray(efficiency)
    na = nsite_cm3*x*eta
    dr = np.array([abs(RADII[s]-RADII[SITES[s]])/RADII[SITES[s]] for s in d])
    dc = np.array([abs(CHI[s]-CHI[SITES[s]]) for s in d])
    compatibility = np.exp(-1.6*dr-.45*dc)       # geometry/chemistry index ONLY
    penalty = dr if radius_penalty else np.zeros_like(dr)
    mass_ratio = 1.+x*(params["mass_penalty"]*penalty + .1*dc)
    retention = mass_ratio**-2/(1.+params["alloy_scattering"]*x*(1.-x)*(penalty**2+.15*dc**2)
                              +params["ion_scattering"]*x*eta)
    p, ionization, fermi = solve_holes(na, ea, params, cfg["temperature_K"])
    # The hole-only neutrality model must not silently enter a bipolar regime.
    # This conservative midgap-minus-3kT guard assumes comparable DOS prefactors;
    # it is a validity screen, not an electron-transport prediction.
    p_type_valid = fermi < params["indirect_gap_guard_eV"]/2.-3.*Boltzmann*cfg["temperature_K"]/elementary_charge
    p = np.where(p_type_valid, p, np.nan)
    mu = params["mu0_cm2_Vs"]*retention
    # SI calculation: p[cm^-3]*1e6 -> m^-3; mu[cm^2/(V s)]*1e-4 -> m^2/(V s).
    sigma = elementary_charge*(p*1e6)*(mu*1e-4)  # S/m
    donor = np.array([ROLES[s] == "donor_control" for s in d])
    # Do not reinterpret donor doping as a weak acceptor or assign a fake EA.
    for arr in [mu, sigma, retention]: arr[donor] = np.nan
    return pd.DataFrame({"EA_effective_input_eV": ea, "nominal_dopant_cm3": nsite_cm3*x,
        "active_acceptor_assumption_cm3": np.where(donor, np.nan, na),
        "geometric_compatibility_index": compatibility, "relative_radius_mismatch": dr,
        "ionized_acceptor_fraction": ionization, "fermi_relative_VBM_eV": fermi,
        "hole_only_model_valid": p_type_valid,
        "p_scenario_cm3": p, "m_transport_proxy_over_me": params["m_transport_over_me"]*mass_ratio,
        "mobility_retention": retention, "mu_scenario_cm2_Vs": mu,
        "sigma_scenario_S_m": sigma,
        "rho_scenario_ohm_m": np.divide(1., sigma, out=np.full_like(sigma, np.nan), where=sigma > 0),
        # Retained for compatibility with the existing experimental input template.
        "rho_scenario_ohm_cm": np.divide(100., sigma, out=np.full_like(sigma, np.nan), where=sigma > 0),
        "distortion_index": 100.*x*dr*params["distortion_scale"]})


def prepare_predictions(data, host_structure, model, chosen, oof, transfer, reliable, columns, cfg, out):
    candidates, descriptors = virtual_grid(host_structure, cfg)
    Xnew = descriptors[columns]
    ad = applicability_domain(data, data[columns], Xnew, candidates, cfg)
    boot = bootstrap_predictions(model, data, columns, Xnew, cfg)
    # Error envelope is descriptive: no conformal-coverage claim on virtual OOD data.
    residual_envelope = float(oof.groupby("formula").residual.apply(lambda s: s.abs().max()).quantile(.90))
    candidates["FE_ML_eV_atom"] = model.predict(Xnew)
    candidates["FE_bootstrap_p05"] = np.quantile(boot, .05, axis=0)
    candidates["FE_bootstrap_p95"] = np.quantile(boot, .95, axis=0)
    candidates["FE_empirical_lower"] = candidates.FE_bootstrap_p05-residual_envelope
    candidates["FE_empirical_upper"] = candidates.FE_bootstrap_p95+residual_envelope
    candidates["FE_bootstrap_sd_eV_atom"] = np.std(boot, axis=0, ddof=1)
    candidates = pd.concat([candidates, ad], axis=1)
    transfer_map = transfer.set_index("dopant").MAE_eV_atom.to_dict()
    candidates["transfer_MAE_eV_atom"] = candidates.dopant.map(transfer_map)
    candidates["passes_transfer_gate"] = candidates.transfer_MAE_eV_atom <= cfg["transfer_mae_gate_eV_atom"]
    candidates["ML_quality_gate"] = reliable
    candidates["phase_competition_proxy"] = candidates.dopant.map({d: phase_proxy(data, d) for d in DOPANTS})
    candidates["phase_proxy_n_materials"] = candidates.dopant.map(data.A.value_counts()).fillna(0).astype(int)
    nsite = float(host_structure.composition["Sn"] / host_structure.volume * 1e24)
    assert np.isclose(host_structure.composition["O"], host_structure.composition["Sn"])
    properties = transport(candidates, nsite, PHYSICS, cfg)
    result = pd.concat([candidates, properties], axis=1)
    result["experimental_EA_available"] = result.dopant.isin(["Na", "K", "N"])
    result["process_caution"] = ""
    result.loc[(result.dopant == "N") & (result.nominal_dopant_cm3 >= 7e17), "process_caution"] = "above_N_morphology_onset_in_Becker_2019_process"
    result["supported_screening"] = (result.in_domain & result.passes_transfer_gate & result.ML_quality_gate
        & result.within_dilute_model_scope & result.experimental_EA_available & result.process_caution.eq(""))
    csv_out(pd.concat([candidates, descriptors.add_prefix("descriptor_")], axis=1), out / "ml_predictions.csv")
    csv_out(result, out / "screening_results.csv")
    np.savez_compressed(out / "formation_energy_bootstrap_draws.npz", predictions=boot,
                        dopants=candidates.dopant.to_numpy(str), substitution_pct=candidates.substitution_pct.to_numpy())
    return result, boot, nsite


# -------------------- PARETO SORTING AND ROBUSTNESS -------------------------
def objective_matrix(frame, variant="full"):
    # All objectives are minimized; conductivity conversion S/cm->S/m is only
    # an additive constant in log10, so neither dominance nor ranking changes.
    with np.errstate(divide="ignore", invalid="ignore"):
        obj = np.column_stack([frame.FE_ML_eV_atom, -np.log10(frame.sigma_scenario_S_m),
                               1.-frame.mobility_retention, frame.distortion_index,
                               frame.phase_competition_proxy])
    keep = {"full": [0,1,2,3,4], "no_phase_proxy": [0,1,2,3],
            "transport_only": [1,2], "no_radius_objective": [0,1,2,4]}[variant]
    return obj[:, keep]


def non_dominated_ranks(values, eligible=None):
    """Exact minimization dominance. NaNs are ineligible, never replaced by zero."""
    a = np.asarray(values, float)
    good = np.all(np.isfinite(a), axis=1)
    if eligible is not None: good &= np.asarray(eligible, bool)
    rank = np.full(len(a), np.nan)
    remaining = np.flatnonzero(good)
    front = 1
    while len(remaining):
        b = a[remaining]
        dominates = (b[:, None, :] <= b[None, :, :]).all(axis=2) & (b[:, None, :] < b[None, :, :]).any(axis=2)
        nondominated = ~dominates.any(axis=0)
        if not nondominated.any(): raise AssertionError("Dominance graph unexpectedly cyclic.")
        rank[remaining[nondominated]] = front
        remaining = remaining[~nondominated]
        front += 1
    return rank


def compromise_distance(values, ranks, tolerance):
    """Transparent equal-scale ideal distance AFTER Pareto sorting; not a knee.

    Min-max scales use all eligible alternatives at the same concentration.
    This tie-break does introduce equal normalized importance; dominance itself
    is weight-free. A high-dimensional knee is not assumed to exist.
    """
    finite = np.isfinite(ranks)
    distance = np.full(len(ranks), np.nan)
    shares = np.zeros(len(ranks))
    if not finite.any(): return distance, shares
    lo, hi = np.min(values[finite], axis=0), np.max(values[finite], axis=0)
    active = hi-lo > 1e-12
    if active.any():
        normalized = (values[:, active]-lo[active])/(hi[active]-lo[active])
        distance[finite] = np.sqrt(np.mean(normalized[finite]**2, axis=1))
    else:
        distance[finite] = 0.
    front = ranks == 1
    near = front & (distance <= np.nanmin(distance[front])+tolerance)
    shares[near] = 1./near.sum()
    return distance, shares


def pareto_analysis(screen, cfg, out):
    rows = []
    for pct, frame in screen[screen.x > 0].groupby("substitution_pct", sort=True):
        for variant in ["full", "no_phase_proxy", "transport_only", "no_radius_objective"]:
            values = objective_matrix(frame, variant)
            for scope in ["exploratory", "supported"]:
                mask = frame.supported_screening.to_numpy() if scope == "supported" else np.ones(len(frame), bool)
                ranks = non_dominated_ranks(values, mask)
                distance, shares = compromise_distance(values, ranks, cfg["compromise_tolerance"])
                for j, (_, row) in enumerate(frame.iterrows()):
                    rows.append({"dopant": row.dopant, "substitution_pct": pct, "variant": variant,
                                 "scope": scope, "front": ranks[j], "ideal_distance": distance[j],
                                 "compromise_share": shares[j], "eligible": np.isfinite(ranks[j])})
    result = pd.DataFrame(rows)
    csv_out(result, out / "pareto_fronts.csv")
    return result


def sample_parameters(rng):
    pars = deepcopy(PHYSICS)
    # These bounds express sensitivity assumptions, not estimated confidence intervals.
    pars.update(p0_cm3=10**rng.uniform(np.log10(2e17), 19), mu0_cm2_Vs=rng.uniform(3., 30.),
                m_dos_over_me=rng.uniform(.3, 1.5), active_fraction=10**rng.uniform(-2., 0.),
                compensation_fraction=rng.uniform(0., .5), mass_penalty=rng.uniform(.25, 1.0),
                alloy_scattering=7.*rng.uniform(.5, 2.), ion_scattering=2.5*rng.uniform(.5, 2.),
                distortion_scale=rng.uniform(.5, 1.5))
    return pars


def robustness(screen, boot, data, oof, nsite, cfg, out):
    rng = np.random.default_rng(cfg["seed"]+202)
    # Propagate uncertainty at the same nominal substitutions as the central grid.
    indexes = np.flatnonzero(screen.substitution_pct.isin(cfg["requested_pareto_pct"]))
    base = screen.iloc[indexes].reset_index(drop=True)
    groups = {pct: group.index.to_numpy() for pct, group in base.groupby("substitution_pct")}
    group_residuals = oof.groupby("formula").residual.mean().to_numpy()
    n_draws = cfg["mc_draws"]
    rank_draws = np.full((n_draws, len(base)), np.nan)
    share_draws = np.zeros((n_draws, len(base)))
    support_draws = np.full_like(rank_draws, np.nan)
    no_radius_draws = np.full_like(rank_draws, np.nan)
    p_draws, mu_draws, sigma_draws = [np.full_like(rank_draws, np.nan) for _ in range(3)]
    fe_draws = np.full_like(rank_draws, np.nan)
    modes = []
    for draw in range(n_draws):
        pars = sample_parameters(rng)
        mode = "shared_efficiency" if draw < n_draws//2 else "dopant_specific_efficiency"
        modes.append(mode)
        effective = {d: (pars["active_fraction"] if mode == "shared_efficiency" else 10**rng.uniform(-2., 0.)) for d in DOPANTS}
        ea = {d: (rng.uniform(EA[d][1], EA[d][2]) if np.isfinite(EA[d][0]) else np.nan) for d in DOPANTS}
        # Systematic error shared along each dopant's concentration series.
        bias = {d: rng.choice(group_residuals) for d in DOPANTS}
        props = transport(base, nsite, pars, cfg, ea_values=[ea[d] for d in base.dopant],
                          efficiency=np.array([effective[d] for d in base.dopant]))
        frame = base.copy()
        frame[props.columns] = props
        frame["FE_ML_eV_atom"] = boot[rng.integers(len(boot)), indexes] + np.array([bias[d] for d in base.dopant])
        phases = {d: phase_proxy(data, d, rng) for d in DOPANTS}
        frame["phase_competition_proxy"] = frame.dopant.map(phases)
        p_draws[draw], mu_draws[draw], sigma_draws[draw] = props.p_scenario_cm3, props.mu_scenario_cm2_Vs, props.sigma_scenario_S_m
        fe_draws[draw] = frame.FE_ML_eV_atom
        alternative = frame.copy()
        altprops = transport(base, nsite, pars, cfg, ea_values=[ea[d] for d in base.dopant],
                             efficiency=np.array([effective[d] for d in base.dopant]), radius_penalty=False)
        alternative[altprops.columns] = altprops
        for _, idx in groups.items():
            values = objective_matrix(frame.iloc[idx])
            ranks = non_dominated_ranks(values)
            _, shares = compromise_distance(values, ranks, cfg["compromise_tolerance"])
            rank_draws[draw, idx], share_draws[draw, idx] = ranks, shares
            support_draws[draw, idx] = non_dominated_ranks(values, base.iloc[idx].supported_screening)
            no_radius_draws[draw, idx] = non_dominated_ranks(objective_matrix(alternative.iloc[idx], "no_radius_objective"))
        if (draw+1) % 250 == 0: print(f"  Monte Carlo draws: {draw+1}/{n_draws}")
    def frequency(a, axis=0):
        count = np.isfinite(a).sum(axis=axis)
        # All scenario draws remain in the denominator; intermittent infeasibility
        # cannot inflate a candidate's robustness. Never-eligible controls stay NaN.
        return np.divide((a == 1).sum(axis=axis), a.shape[axis],
                         out=np.full(np.shape(count), np.nan), where=count > 0)
    result = base[["dopant", "substitution_pct", "supported_screening"]].copy()
    result["pareto_frequency"] = frequency(rank_draws)
    result["mc_eligible_fraction"] = np.isfinite(rank_draws).mean(axis=0)
    result["supported_pareto_frequency"] = frequency(support_draws)
    result["no_radius_penalty_pareto_frequency"] = frequency(no_radius_draws)
    result["compromise_share_mean"] = share_draws.mean(axis=0)
    # Handle deliberately missing donor-control quantities without hiding warnings globally.
    for j in range(len(base)):
        valid = rank_draws[:, j][np.isfinite(rank_draws[:, j])]
        result.loc[j, "median_front"] = np.median(valid) if len(valid) else np.nan
        for name, values in [("p_cm3", p_draws), ("mu_cm2_Vs", mu_draws), ("sigma_S_m", sigma_draws)]:
            v = values[:, j][np.isfinite(values[:, j])]
            for q in [.05, .50, .95]:
                result.loc[j, f"{name}_scenario_p{int(q*100):02d}"] = np.quantile(v, q) if len(v) else np.nan
    for mode in sorted(set(modes)):
        mask = np.array(modes) == mode
        result[f"pareto_frequency_{mode}"] = frequency(rank_draws[mask])
        result[f"compromise_share_{mode}"] = share_draws[mask].mean(axis=0)
    csv_out(result, out / "robustness_results.csv")
    np.savez_compressed(out / "mc_scenario_draws.npz", ranks=rank_draws, compromise_shares=share_draws,
                        supported_ranks=support_draws, no_radius_ranks=no_radius_draws,
                        p_cm3=p_draws, mu_cm2_Vs=mu_draws, sigma_S_m=sigma_draws,
                        FE_eV_atom=fe_draws,
                        dopants=base.dopant.to_numpy(str), substitution_pct=base.substitution_pct.to_numpy(),
                        efficiency_mode=np.array(modes))
    return result


# ------------------ DATA COUNTS AND DOPANT PREFERENCE SCORE -----------------
SCORE_KEYS = ["formation_energy", "conductivity", "mobility"]


def validate_score_config(cfg):
    grid = np.asarray(cfg["requested_pareto_pct"], dtype=float)
    if (grid.ndim != 1 or grid.size == 0 or not np.isfinite(grid).all()
            or np.any(grid < 1.) or np.any(grid > 20.)
            or np.unique(grid).size != grid.size):
        raise ValueError("Use unique nominal sublattice substitutions within 1-20%.")
    if not np.any(np.isclose(grid, cfg["score_reference_pct"], atol=1e-9, rtol=0)):
        raise ValueError("score_reference_pct must be one of requested_pareto_pct.")
    weights = cfg["score_weights"]
    if set(weights) != set(SCORE_KEYS):
        raise ValueError(f"score_weights must contain exactly {SCORE_KEYS}.")
    w = np.asarray([weights[k] for k in SCORE_KEYS], float)
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Score weights must be finite, nonnegative and have a positive sum.")
    # Any concentration in the screening grid may be selected.
    # Model-scope eligibility is assessed separately; references above the
    # dilute limit remain exploratory rather than expanding the model scope.
    if cfg["score_weight_draws"] < 2 or cfg["score_close_margin_points"] < 0:
        raise ValueError("Use at least two weight draws and a nonnegative close-score margin.")
    return w / w.sum()


def write_dataset_summary(data, raw, out):
    """Count distinct MP records and compositions; virtual candidates are separate."""
    requested = data.A.isin(DOPANTS)
    host = data.A.eq("HOST")
    values = [
        ("Retrieved MP records (unique IDs)", len({d["material_id"] for d in raw["ternaries"]+raw["hosts"]})),
        ("Final filtered ML dataset: structures", len(data)),
        ("Final filtered ML dataset: unique compositions", data.formula.nunique()),
        ("Target five-dopant ternary structures", int(requested.sum())),
        ("Other-element ternary structures", int((~requested & ~host).sum())),
        ("Pristine alpha-SnO host structures", int(host.sum())),
        ("Third elements represented in ternaries", data.loc[~host, "A"].nunique()),
    ]
    summary = pd.DataFrame(values, columns=["quantity", "count"])
    distribution = (data.loc[requested].groupby("A").agg(
        structures=("material_id", "size"), unique_compositions=("formula", "nunique"))
        .reindex(DOPANTS, fill_value=0).rename_axis("dopant").reset_index())
    distribution["percent_of_final_dataset"] = 100.*distribution.structures/len(data)
    csv_out(summary, out / "dataset_summary.csv")
    csv_out(distribution, out / "dopant_distribution.csv")
    print(f"Final filtered ML dataset: {len(data)} structures / {data.formula.nunique()} unique compositions.")
    print(f"Target dopants: {requested.sum()} structures; other elements: {(~requested & ~host).sum()}; host: {host.sum()}.")
    print("Target-dopant structure counts: " + ", ".join(f"{r.dopant}={r.structures}" for r in distribution.itertuples()))
    return summary, distribution


def score_costs(frame):
    """Smaller is preferred: compound energy, -log10(sigma/[1 S/m]), -mu."""
    sigma = frame.sigma_scenario_S_m.to_numpy(float)
    log_sigma = np.full(len(frame), np.nan)
    good = np.isfinite(sigma) & (sigma > 0)
    log_sigma[good] = np.log10(sigma[good])
    costs = np.column_stack([frame.FE_ML_eV_atom.to_numpy(float), -log_sigma,
                            -frame.mu_scenario_cm2_Vs.to_numpy(float)])
    # Donor controls stay outside this p-type selection even if the user sets
    # both transport weights to zero; their FE curves remain fully exported.
    costs[frame.dopant.map(ROLES).eq("donor_control").to_numpy(), :] = np.nan
    return costs


def normalized_desirability(costs, weights, anchors=None):
    """Fixed-concentration min-max utilities; neutral 0.5 for a constant criterion.

    A missing positively weighted property makes that candidate ineligible.
    Zero-weight columns do not affect eligibility. No favorable imputation is used.
    Optional anchors keep uncertainty scores on the central scenario's scale.
    """
    a = np.asarray(costs, float)
    active = np.asarray(weights) > 0
    eligible = np.isfinite(a[:, active]).all(axis=1)
    utility = np.full_like(a, np.nan)
    if anchors is None:
        lo, hi = np.full(a.shape[1], np.nan), np.full(a.shape[1], np.nan)
        if eligible.any():
            for k in range(a.shape[1]):
                v = a[eligible, k]
                v = v[np.isfinite(v)]
                if len(v): lo[k], hi[k] = v.min(), v.max()
    else:
        lo, hi = (np.asarray(x, float) for x in anchors)
    for k in range(a.shape[1]):
        valid = eligible & np.isfinite(a[:, k])
        if not np.isfinite(lo[k]) or not np.isfinite(hi[k]): continue
        utility[valid, k] = (.5 if hi[k]-lo[k] <= 1e-12 else
                            np.clip((hi[k]-a[valid, k])/(hi[k]-lo[k]), 0., 1.))
    score = np.full(len(a), np.nan)
    eligible &= np.isfinite(utility[:, active]).all(axis=1)
    score[eligible] = 100.*(utility[eligible][:, active] @ np.asarray(weights)[active])
    return utility, score, eligible, (lo, hi)


def winner_shares(values, eligible=None):
    """Exact score ties share one win; missing scores never win."""
    valid = np.isfinite(values)
    if eligible is not None: valid &= np.asarray(eligible, bool)
    result = np.zeros(len(values))
    if valid.any():
        best = np.max(np.asarray(values)[valid])
        tied = valid & np.isclose(values, best, rtol=0, atol=1e-9)
        result[tied] = 1./tied.sum()
    return result


def dopant_scoring(screen, cfg, out):
    """Application-specific decision after property prediction, not ML training.

    Default score = 100*(0.25*U_FE + 0.50*U_log_sigma + 0.25*U_mu).
    Removing a candidate changes min-max anchors; scores must be recomputed for
    the five-dopant candidate set rather than copied from an earlier run.
    Units: eV/atom, S/m and cm^2/(V s). Utilities run from 0 (worst) to 1
    (best) among complete alternatives at the SAME sublattice concentration.
    The highest score among gate-passing candidates is selected if any exist;
    otherwise the highest finite score is explicitly an exploratory candidate.
    """
    w = validate_score_config(cfg)
    rows = []
    for pct, frame in screen[screen.x > 0].groupby("substitution_pct", sort=True):
        costs = score_costs(frame)
        u, scores, valid, anchors = normalized_desirability(costs, w)
        ranks = non_dominated_ranks(costs[:, w > 0], valid)
        for j, (_, row) in enumerate(frame.iterrows()):
            record = {"dopant": row.dopant, "substitution_pct": pct, "score_0_100": scores[j],
                "score_eligible": bool(valid[j]), "score_objective_front": ranks[j],
                "supported_screening": bool(row.supported_screening),
                "FE_ML_eV_atom": row.FE_ML_eV_atom, "sigma_scenario_S_m": row.sigma_scenario_S_m,
                "mu_scenario_cm2_Vs": row.mu_scenario_cm2_Vs,
                "evidence": row.EA_evidence,
                "status": ("donor control: p-type score unavailable" if row.role == "donor_control" else
                           "supported screening" if row.supported_screening else
                           "exploratory: no dopant training labels" if row.phase_proxy_n_materials == 0 else
                           "exploratory: activation assumed" if "assumption" in row.EA_evidence else "exploratory screening")}
            for k, name in enumerate(SCORE_KEYS):
                record[f"utility_{name}"] = u[j, k]
                record[f"points_{name}"] = 100.*w[k]*u[j, k] if w[k] > 0 else 0.
            rows.append(record)
    all_scores = pd.DataFrame(rows)
    csv_out(all_scores, out / "dopant_scores_all_concentrations.csv")
    ref = all_scores[np.isclose(all_scores.substitution_pct, cfg["score_reference_pct"], atol=1e-9, rtol=0)].copy()
    ref = ref.set_index("dopant").reindex(DOPANTS).reset_index()
    central = screen[np.isclose(screen.substitution_pct, cfg["score_reference_pct"], atol=1e-9, rtol=0)]
    central = central.set_index("dopant").reindex(DOPANTS).reset_index()
    costs = score_costs(central)
    utility, score, valid, anchors = normalized_desirability(costs, w)
    supported = valid & ref.supported_screening.to_numpy(bool)
    selection_pool = supported if supported.any() else valid
    shares = winner_shares(score, selection_pool)
    selected = [d for d, share in zip(DOPANTS, shares) if share > 0]
    ref["rank"] = pd.Series(score).rank(method="min", ascending=False).to_numpy()
    ref["selected_at_reference"] = shares > 0
    ref["selection_pool"] = selection_pool
    best = np.nanmax(score[selection_pool]) if selection_pool.any() else np.nan
    ref["within_close_score_margin"] = selection_pool & (best-score <= cfg["score_close_margin_points"])
    # Uniform-simplex weights explore preference choices, not their probabilities.
    rng = np.random.default_rng(cfg["seed"]+303)
    weight_vectors = rng.dirichlet(np.ones(3), size=int(cfg["score_weight_draws"]))
    all_criteria_valid = np.isfinite(costs).all(axis=1)
    sweep_u, _, sweep_valid, _ = normalized_desirability(costs, np.ones(3)/3.)
    sweep_scores = 100.*(np.nan_to_num(sweep_u, nan=0.) @ weight_vectors.T).T
    sweep_scores[:, ~sweep_valid] = np.nan
    weight_wins = np.asarray([winner_shares(v, selection_pool & all_criteria_valid) for v in sweep_scores])
    ref["weight_sweep_top_share"] = np.where(sweep_valid & selection_pool, weight_wins.mean(axis=0), np.nan)
    pref = pd.DataFrame(weight_vectors, columns=[f"weight_{k}" for k in SCORE_KEYS])
    pref["top_dopants"] = [";".join(d for d, s in zip(DOPANTS, v) if s > 0) for v in weight_wins]
    csv_out(pref, out / "score_weight_sensitivity.csv")
    # Leave-one-criterion-out and equal-weight alternatives expose correlation
    # between conductivity and mobility and sensitivity to the energy proxy.
    alternatives = {"configured": w, "equal_weights": np.ones(3)/3.}
    for k, name in enumerate(SCORE_KEYS):
        alt = w.copy(); alt[k] = 0.
        if alt.sum() > 0: alternatives[f"without_{name}"] = alt/alt.sum()
    ablations = []
    for label, weights in alternatives.items():
        _, vals, _, _ = normalized_desirability(costs, weights)
        tops = winner_shares(vals, selection_pool)
        for d, value, top in zip(DOPANTS, vals, tops):
            ablations.append({"variant": label, "dopant": d, "score_0_100": value,
                              "top_share": top, **dict(zip([f"weight_{k}" for k in SCORE_KEYS], weights))})
    csv_out(pd.DataFrame(ablations), out / "score_criterion_sensitivity.csv")
    # Reuse the SAME FE and physical draws used by the Pareto analysis.
    with np.load(out / "mc_scenario_draws.npz", allow_pickle=False) as mc:
        indexes = [int(np.flatnonzero((mc["dopants"] == d) & np.isclose(
            mc["substitution_pct"], cfg["score_reference_pct"], atol=1e-9, rtol=0))[0]) for d in DOPANTS]
        fe, sigma, mu = mc["FE_eV_atom"][:, indexes], mc["sigma_S_m"][:, indexes], mc["mu_cm2_Vs"][:, indexes]
        scenario_scores, scenario_wins = [], []
        for e, s, mobility in zip(fe, sigma, mu):
            with np.errstate(divide="ignore", invalid="ignore"):
                cost = np.column_stack([e, -np.log10(s), -mobility])
            _, vals, _, _ = normalized_desirability(cost, w, anchors=anchors)
            vals[~valid] = np.nan
            scenario_scores.append(vals)
            scenario_wins.append(winner_shares(vals, selection_pool))
    scenario_scores, scenario_wins = np.asarray(scenario_scores), np.asarray(scenario_wins)
    for j, d in enumerate(DOPANTS):
        vals = scenario_scores[:, j]
        vals = vals[np.isfinite(vals)]
        for q in [.05, .5, .95]:
            ref.loc[j, f"score_scenario_p{int(100*q):02d}"] = np.quantile(vals, q) if len(vals) else np.nan
        ref.loc[j, "score_scenario_eligible_fraction"] = len(vals)/len(scenario_scores)
        ref.loc[j, "scenario_top_share"] = (scenario_wins[:, j].mean()
                                              if valid[j] and selection_pool[j] else np.nan)
    np.savez_compressed(out / "score_scenario_draws.npz", scores=scenario_scores,
                        top_shares=scenario_wins, dopants=np.array(DOPANTS),
                        substitution_pct=cfg["score_reference_pct"])
    normal = pd.DataFrame({"criterion": SCORE_KEYS, "weight": w,
        "cost_min": anchors[0], "cost_max": anchors[1],
        "cost_definition": ["FE (eV/atom)", "-log10(sigma/[1 S/m])", "-mu (cm^2/(V s))"]})
    csv_out(normal, out / "score_normalization.csv")
    ref = ref.sort_values("score_0_100", ascending=False, na_position="last", kind="stable")
    csv_out(ref, out / "dopant_score_details.csv")
    compact = ref[["dopant", "score_0_100", "rank", "FE_ML_eV_atom", "sigma_scenario_S_m",
                   "mu_scenario_cm2_Vs", "selected_at_reference", "status"]]
    csv_out(compact, out / "dopant_score_summary.csv")
    result = {"selected_dopant": selected[0] if len(selected) == 1 else None,
        "selected_dopants": selected,
        "selection_status": ("conditional_supported_screening" if supported.any() else
                             "exploratory_score_candidate" if valid.any() else "no_eligible_candidate"),
        "substitution_pct": cfg["score_reference_pct"], "temperature_K": cfg["temperature_K"],
        "weights": dict(zip(SCORE_KEYS, w)), "highest_score_in_selection_pool": best,
        "close_alternatives": ref.loc[ref.within_close_score_margin & ~ref.selected_at_reference, "dopant"].tolist(),
        "supported_candidate_count": int(supported.sum()),
        "method": "Weighted sum of min-max desirabilities at a fixed sublattice concentration",
        "interpretation": "Preference-dependent screening result; experimental superiority is not established.",
        "scenario_intervals": "Sensitivity ranges using central normalization anchors clipped to 0-1; not calibrated confidence intervals",
        "exact_ties": "Retained as co-selected dopants; no arbitrary tie-break",
        "method_reference": "https://doi.org/10.1787/9789264043466-en"}
    write_json(out / "dopant_selection.json", result)
    print(f"Score selection at {cfg['score_reference_pct']:g}%: {', '.join(selected) or 'none'}"
          f" | {result['selection_status']}.")
    return ref, all_scores, result


def write_selection_interpretation(data, screen, scores, selection, validation, cfg, out):
    """Explain the actual result; advantages are conditional numerical comparisons.

    No preferred dopant or stock winner narrative is assigned. Negative trade-offs,
    missing labels, hypotheses and failed evidence gates are reported explicitly.
    """
    pct = cfg["score_reference_pct"]
    physical = screen[np.isclose(screen.substitution_pct, pct, atol=1e-9, rtol=0)].set_index("dopant")
    ranked = scores.set_index("dopant")
    valid = ranked[ranked.score_eligible].copy()
    selected = selection["selected_dopants"]
    metrics = pd.read_csv(out / "model_metrics.csv")
    cv = metrics[metrics.stage == "nested_cv_pooled"].iloc[0]
    properties = [
        ("formation_energy", "FE_ML_eV_atom", "compound formation-energy proxy", "eV/atom", True),
        ("conductivity", "sigma_scenario_S_m", "scenario conductivity", "S/m", False),
        ("mobility", "mu_scenario_cm2_Vs", "scenario mobility", "cm^2/(V s)", False),
    ]
    def number(value, digits=4):
        return f"{value:.{digits}g}" if np.isfinite(value) else "not available"
    def failed_checks(row):
        reasons = []
        for field, label in [
            ("ML_quality_gate", "GB predictive-quality tolerance"),
            ("passes_transfer_gate", "withheld-dopant transfer tolerance"),
            ("in_domain", "training descriptor/concentration coverage"),
            ("within_dilute_model_scope", "declared dilute-model scope"),
            ("experimental_EA_available", "experimental activation input")]:
            if not bool(row[field]): reasons.append(label)
        caution = row.get("process_caution", "")
        if isinstance(caution, str) and caution.strip(): reasons.append(caution.replace("_", " "))
        return reasons
    comparative = []
    for d in DOPANTS:
        row, s = physical.loc[d], ranked.loc[d]
        count = int(data.A.eq(d).sum())
        strengths, weaknesses = [], []
        if ROLES[d] == "donor_control":
            strengths.append("Comparison control with a computed compound-energy proxy.")
            weaknesses.append("Defined as a donor control; acceptor transport and a p-type dopant score are not assigned.")
        elif not s.score_eligible:
            weaknesses.append("At least one active scoring property is unavailable; no score can be assigned.")
        else:
            for key, column, label, unit, lower in properties:
                if selection["weights"][key] <= 0: continue
                pool = valid[column].dropna()
                if pool.empty or not np.isfinite(row[column]): continue
                value = float(row[column])
                pos = int(pool.rank(method="min", ascending=lower).loc[d])
                text = f"{label.capitalize()}: {number(value)} {unit}, rank {pos}/{len(pool)} among scored acceptors (ties retained)."
                # Split strengths and weaknesses by the central peer median;
                # this is an interpretation aid, not an eligibility threshold.
                is_strength = value <= pool.median() if lower else value >= pool.median()
                (strengths if is_strength else weaknesses).append(text)
            retention = float(row.mobility_retention)
            if np.isfinite(retention) and retention < 1:
                weaknesses.append(f"Mobility is {100*(1-retention):.2f}% below the assumed undoped value of {PHYSICS['mu0_cm2_Vs']:g} cm^2/(V s).")
            if "assumption" in EA[d][4]:
                weaknesses.append(f"The {1000*EA[d][0]:g} meV activation input is an unvalidated scenario assumption.")
            elif np.isfinite(EA[d][0]):
                strengths.append(f"A literature-derived effective activation input is available ({1000*EA[d][0]:g} meV; DOI {EA[d][3]}). This is an input, not validation.")
            reasons = failed_checks(row)
            if reasons: weaknesses.append("Failed screening checks: " + "; ".join(reasons) + ".")
        if not strengths: strengths.append("No comparative strength is established from the available scoring inputs.")
        if not weaknesses: weaknesses.append("Quantitative transport and device performance still require independent validation.")
        comparative.append({"dopant": d, "selected": d in selected, "MP_training_structures": count,
                            "score_0_100": s.score_0_100, "advantages": " ".join(strengths),
                            "disadvantages": " ".join(weaknesses)})
    comparison_table = pd.DataFrame(comparative)
    csv_out(comparison_table, out / "dopant_advantages_disadvantages.csv")
    explanation = []
    decision = (f"At {pct:g}% sublattice substitution and {cfg['temperature_K']:g} K, " +
                (f"the selected candidate{'s are' if len(selected)>1 else ' is'} {', '.join(selected)}."
                 if selected else "no candidate has a complete eligible score."))
    selection_rule = ("The code compares candidates at the same concentration. It scales lower formation energy, "
        "higher log conductivity and higher mobility to desirabilities between 0 and 1. Score = 100 times their weighted sum. "
        "Configured weights: " + ", ".join(f"{k.replace('_',' ')} {100*v:g}%" for k,v in selection["weights"].items()) + ". " +
        ("Selection is restricted to candidates passing all screening checks." if selection["supported_candidate_count"] else
         "No candidate passes all screening checks, so the highest score is reported as an exploratory candidate.") +
        " Exact score ties remain co-selected.")
    deltas = []
    for d in selected:
        s, row = ranked.loc[d], physical.loc[d]
        alternatives = valid[valid.selection_pool & ~valid.index.isin(selected)]
        outside_pool = False
        if alternatives.empty:
            alternatives = valid[~valid.index.isin(selected)]
            outside_pool = bool(len(alternatives))
        rival = alternatives.sort_values("score_0_100", ascending=False).index[0] if len(alternatives) else None
        parts = [f"{100*selection['weights'][k]:g}% weight gives {s[f'points_{k}']:.2f} {k.replace('_',' ')} points" for k in SCORE_KEYS]
        why = f"{d} obtains {s.score_0_100:.2f}/100: " + "; ".join(parts) + "."
        tradeoffs = []
        if rival is not None:
            r = ranked.loc[rival]
            why += f" The highest-scoring alternative is {rival} ({r.score_0_100:.2f}/100); the score difference is {s.score_0_100-r.score_0_100:+.2f} points."
            if outside_pool:
                why += " That alternative is outside the gate-passing selection pool and is shown only for comparison."
            for key, column, label, unit, lower in properties:
                value, other = float(s[column]), float(r[column])
                delta = float(s[f"points_{key}"]-r[f"points_{key}"])
                deltas.append({"selected_dopant": d, "alternative": rival, "criterion": key,
                               "score_difference_points": delta, "selected_value": value,
                               "alternative_value": other, "unit": unit, "alternative_outside_selection_pool": outside_pool})
                if np.isfinite(value) and np.isfinite(other):
                    label_direction = "lower" if value < other else "higher" if value > other else "equal"
                    if key == "formation_energy":
                        text = f"F.E.: {value-other:+.4f} eV/atom relative to {rival}; lower is preferred only as a compound-energy proxy."
                    else:
                        text = (f"{label.capitalize()}: {value/other:.3f} times {rival}'s value ({label_direction}); "
                                f"{value:.4g} versus {other:.4g} {unit}.") if other > 0 else f"{label}: comparison unavailable."
                    tradeoffs.append(text + f" Its weighted score contribution differs by {delta:+.2f} points.")
            energy_gap = abs(float(s.FE_ML_eV_atom-r.FE_ML_eV_atom))
            if energy_gap < cv.MAE_eV_atom:
                tradeoffs.append(f"The energy separation ({energy_gap:.4f} eV/atom) is smaller than pooled GB CV MAE ({cv.MAE_eV_atom:.4f} eV/atom); energy alone provides weak evidence of a resolved difference.")
        strengths = comparison_table.set_index("dopant").loc[d, "advantages"]
        weaknesses = comparison_table.set_index("dopant").loc[d, "disadvantages"]
        physical_reason = (f"In the declared acceptor model, the {1000*EA[d][0]:g} meV effective activation input and charge-neutrality calculation "
            f"give a hole density of {number(row.p_scenario_cm3)} cm^-3. Conductivity follows sigma=q*p*mu after SI conversion. "
            "Radius-mismatch and scattering assumptions reduce mobility; the score balances the carrier benefit against that mobility loss. "
            "This mechanism is a screening approximation, not a measured defect-physics result.")
        stability = (f"{d}'s top-choice share is {number(100*s.weight_sweep_top_share,3)}% in the preference-weight sweep and "
            f"{number(100*s.scenario_top_share,3)}% in the combined energy/transport scenarios. "
            "These are sensitivity frequencies with ties shared, not probabilities of experimental success.")
        if rival is None: tradeoffs.append("No other scored candidate is available for a numerical comparison.")
        explanation.append({"dopant": d, "why_selected": why, "advantages": strengths,
                            "disadvantages": weaknesses, "tradeoffs": tradeoffs,
                            "physical_interpretation": physical_reason, "selection_sensitivity": stability,
                            "failed_screening_checks": failed_checks(row)})
    comparison_columns = ["selected_dopant", "alternative", "criterion", "score_difference_points", "selected_value", "alternative_value", "unit", "alternative_outside_selection_pool"]
    csv_out(pd.DataFrame(deltas, columns=comparison_columns), out / "selected_dopant_tradeoffs.csv")
    limitations = ("Only compound formation energy is learned by Gradient Boosting; it is not charged-defect incorporation energy. "
        "Transport, effective mass and scattering parameters remain declared physical scenarios. "
        "Conductivity contains mobility, so its separate score weight deliberately adds preference to mobility. "
        f"Experimental comparison status: {validation['experimental_validation']}. "
        "Before claiming superiority, compare measured carrier density, mobility and conductivity for the selected candidate and its closest alternative "
        "at matched concentration and processing conditions, and assess defect/phase stability. The result addresses p-type transport; "
        "optical performance, long-term reliability and device switching have not been optimized.")
    report = {"decision": decision, "selection_status": selection["selection_status"], "selection_rule": selection_rule,
              "candidates": explanation, "limits_and_next_evidence": limitations,
              "comparison_note": "Advantages/disadvantages compare central predictions with the scored-peer median; no additional eligibility threshold is imposed."}
    write_json(out / "selection_interpretation.json", report)
    lines = ["DOPANT SELECTION — INTERPRETATION", "", decision, "", "HOW THE SELECTION IS MADE", selection_rule]
    for item in explanation:
        lines += ["", f"WHY {item['dopant']} IS SELECTED", item["why_selected"],
                  "", "ADVANTAGES (conditional on the model)", item["advantages"],
                  "", "DISADVANTAGES AND LIMITATIONS", item["disadvantages"],
                  "", "TRADE-OFFS AGAINST THE ALTERNATIVE", *["- "+t for t in item["tradeoffs"]],
                  "", "PHYSICAL INTERPRETATION", item["physical_interpretation"],
                  "", "HOW SENSITIVE IS THE CHOICE?", item["selection_sensitivity"]]
    lines += ["", "EVIDENCE NEEDED NEXT", limitations]
    (out / "selection_interpretation.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report, pd.DataFrame(deltas, columns=comparison_columns)


def selection_interpretation_html(report):
    """The same interpretation appears in Colab and in the portable ZIP report."""
    from html import escape
    parts = ['<h2>Why this dopant was selected</h2>', '<p>'+escape(report["decision"])+'</p>',
             '<p>'+escape(report["selection_rule"])+'</p>']
    for item in report["candidates"]:
        parts.append('<h3>'+escape(item["dopant"])+': selection and trade-offs</h3><p>'+escape(item["why_selected"])+ '</p>')
        for key, title in [("advantages", "Advantages"), ("disadvantages", "Disadvantages"),
                           ("physical_interpretation", "Physical interpretation"), ("selection_sensitivity", "Sensitivity")]:
            parts.append('<p><strong>'+title+': </strong>'+escape(item[key])+'</p>')
        parts.append('<ul>'+''.join('<li>'+escape(t)+'</li>' for t in item["tradeoffs"])+ '</ul>')
    parts.append('<p><strong>Evidence needed next: </strong>'+escape(report["limits_and_next_evidence"])+ '</p>')
    return '\n'.join(parts)


def plot_selected_tradeoffs(tradeoffs, out):
    if tradeoffs.empty: return
    selected = list(tradeoffs.selected_dopant.unique())
    fig, axes = plt.subplots(len(selected), 1, figsize=(9.6, 3.1*len(selected)+1), squeeze=False, layout="constrained")
    for ax, d in zip(axes.ravel(), selected):
        part = tradeoffs[tradeoffs.selected_dopant.eq(d)]
        vals = part.score_difference_points.to_numpy(float)
        labels = [k.replace("_", " ").capitalize() for k in part.criterion]
        bars = ax.barh(labels, vals, color=["#24866A" if v >= 0 else "#C56A44" for v in vals])
        ax.bar_label(bars, labels=[f"{v:+.2f}" for v in vals], padding=5)
        ax.axvline(0, color=".3", lw=1)
        pad = max(2, .25*np.max(np.abs(vals)))
        ax.set_xlim(min(0, vals.min())-pad, max(0, vals.max())+pad)
        rival = part.alternative.iloc[0]
        extra = " (alternative fails screening gates)" if part.alternative_outside_selection_pool.any() else ""
        ax.set(xlabel=f"Weighted score difference: {d} minus {rival} (points)",
               title=f"{d} versus {rival}{extra}\nTotal score difference = {vals.sum():+.2f}; positive contributions favor {d}")
    save_figure(fig, Path(out) / "figures" / "14_selected_dopant_tradeoffs")


# ------------------------- OPTIONAL EXTERNAL VALIDATION ---------------------
EXPERIMENT_COLUMNS = ["sample_id", "source_doi", "process", "host", "phase", "dopant", "site",
    "substitution_pct", "temperature_K", "p0_cm3", "mu0_cm2_Vs", "p_measured_cm3",
    "mu_measured_cm2_Vs", "rho_measured_ohm_cm", "use_for_validation"]
DFT_COLUMNS = ["sample_id", "source", "host", "phase", "dopant", "site", "substitution_pct",
    "composition", "energy_definition", "formation_energy_eV_atom", "method"]


def external_validation(screen, nsite, cfg, out):
    csv_out(pd.DataFrame(columns=EXPERIMENT_COLUMNS), out / "experimental_validation_template.csv")
    csv_out(pd.DataFrame(columns=DFT_COLUMNS), out / "dft_validation_template.csv")
    status = {"experimental_validation": "not_performed_no_paired_measurements",
              "DFT_validation": "not_performed_no_compatible_DFT_table"}
    if cfg["experimental_csv"]:
        frame = pd.read_csv(cfg["experimental_csv"])
        missing = set(EXPERIMENT_COLUMNS)-set(frame.columns)
        if missing: raise ValueError(f"Experimental CSV missing columns: {sorted(missing)}")
        if frame.sample_id.duplicated().any(): raise ValueError("Experimental sample_id must be unique.")
        results = []
        for _, row in frame.iterrows():
            if str(row.use_for_validation).lower() not in ["true", "1", "yes"]: continue
            if row.host != "SnO" or row.phase not in ["alpha-SnO", "P4/nmm"]:
                raise ValueError(f"{row.sample_id}: only tetragonal alpha-SnO is valid; SnO2 is a different host.")
            if row.dopant not in DOPANTS or row.site != SITES[row.dopant]:
                raise ValueError(f"{row.sample_id}: dopant/site mismatch.")
            required = [row.substitution_pct, row.temperature_K, row.p0_cm3, row.mu0_cm2_Vs]
            if not np.isfinite(np.array(required, float)).all() or min(required[1:]) <= 0:
                raise ValueError(f"{row.sample_id}: provide a positive temperature and measured process-matched UID baseline.")
            if not 0 <= row.substitution_pct <= 100: raise ValueError("Substitution percent must be between 0 and 100.")
            if pd.isna(row.process) or pd.isna(row.source_doi): raise ValueError("Process and provenance are required.")
            one = pd.DataFrame([{"dopant": row.dopant, "x": row.substitution_pct/100.}])
            pcfg = dict(cfg, temperature_K=float(row.temperature_K))
            pars = dict(PHYSICS, p0_cm3=float(row.p0_cm3), mu0_cm2_Vs=float(row.mu0_cm2_Vs))
            pred = transport(one, nsite, pars, pcfg).iloc[0]
            result = {"sample_id": row.sample_id, "source_doi": row.source_doi,
                      "dopant": row.dopant, "process": row.process,
                      "shares_EA_input_source": str(row.source_doi).strip() in EA[row.dopant][3].split(";"),
                      "within_dilute_model_scope": row.substitution_pct <= cfg["dilute_screening_limit_pct"],
                      "input_EA_is_assumed": "assumption" in EA[row.dopant][4]}
            for label, observed, prediction in [
                ("p_cm3", row.p_measured_cm3, pred.p_scenario_cm3),
                ("mu_cm2_Vs", row.mu_measured_cm2_Vs, pred.mu_scenario_cm2_Vs),
                ("sigma_S_m", 100./numeric(row.rho_measured_ohm_cm) if numeric(row.rho_measured_ohm_cm) > 0 else np.nan,
                 pred.sigma_scenario_S_m),
                ("rho_ohm_cm", row.rho_measured_ohm_cm, pred.rho_scenario_ohm_cm)]:
                value = numeric(observed)
                if np.isfinite(value) and value <= 0: raise ValueError("Measured transport values must be positive.")
                result[f"observed_{label}"], result[f"predicted_{label}"] = value, prediction
                result[f"log10_error_{label}"] = np.log10(prediction/value) if value > 0 and prediction > 0 else np.nan
            results.append(result)
        compared = pd.DataFrame(results)
        csv_out(compared, out / "experimental_validation.csv")
        status["experimental_validation"] = f"{len(compared)} paired rows compared_without_fitting"
        # Summaries remain separated by process/study. No pooled R2 across papers.
        if len(compared):
            errors = [c for c in compared if c.startswith("log10_error")]
            summary = compared.groupby(["source_doi", "process"])[errors].agg(lambda a: a.abs().mean()).reset_index()
            csv_out(summary, out / "experimental_log10_MAE_by_study.csv")
    if cfg["dft_validation_csv"]:
        frame = pd.read_csv(cfg["dft_validation_csv"])
        if set(DFT_COLUMNS)-set(frame): raise ValueError("DFT CSV does not match the exported template.")
        results = []
        for _, row in frame.iterrows():
            if row.energy_definition != "MP_compatible_compound_formation_energy":
                raise ValueError("Never substitute charged defect energies (eV/defect) for compound formation energies (eV/atom).")
            if row.host != "SnO" or row.phase not in ["alpha-SnO", "P4/nmm"]:
                raise ValueError("DFT validation host/phase mismatch.")
            found = screen[(screen.dopant == row.dopant) & (screen.site == row.site)
                           & np.isclose(screen.substitution_pct, row.substitution_pct, atol=1e-8, rtol=0)]
            if len(found) != 1: raise ValueError("DFT row must match exactly one dopant, site and concentration on the grid.")
            target = found.iloc[0]
            expected, supplied = Composition(target.virtual_formula).fractional_composition, Composition(row.composition).fractional_composition
            if not expected.almost_equals(supplied, rtol=1e-7, atol=1e-9): raise ValueError("DFT composition does not match concentration/site.")
            value = numeric(row.formation_energy_eV_atom)
            if not np.isfinite(value): raise ValueError("DFT formation energy is missing.")
            results.append({"sample_id": row.sample_id, "dopant": row.dopant, "site": row.site,
                            "substitution_pct": row.substitution_pct, "source": row.source, "method": row.method,
                            "DFT_eV_atom": value, "ML_eV_atom": target.FE_ML_eV_atom,
                            "error_eV_atom": target.FE_ML_eV_atom-value})
        csv_out(pd.DataFrame(results), out / "dft_energy_validation.csv")
        status["DFT_validation"] = f"{len(results)} exact_composition_energy_comparisons_no_override"
    write_json(out / "external_validation_status.json", status)
    return status


def write_starting_structures(host_structure, cfg, out):
    if not cfg["make_dft_starting_structures"]: return
    folder = out / "unrelaxed_DFT_starting_structures"
    folder.mkdir()
    target = cfg["structure_target_pct"] / 100.
    if not 0 < target <= .5: raise ValueError("Structure target percent must be positive and <=50.")
    # One substitution; report the actual commensurate concentration explicitly.
    n = 1
    while len(host_structure)*n**3 < 64 or float(host_structure.composition["Sn"])*n**3 < 1./target:
        n += 1
    records = []
    for d in DOPANTS:
        s = host_structure.copy()
        s.make_supercell([n, n, n])
        eligible = [i for i, site in enumerate(s) if site.specie.symbol == SITES[d]]
        actual_pct = 100./len(eligible)
        s.replace(eligible[0], Element(d))
        name = f"POSCAR_{d}_on_{SITES[d]}_{actual_pct:.6f}pct"
        Poscar(s, comment=f"UNRELAXED alpha-SnO; {d} on {SITES[d]}; {actual_pct:.6f}% sublattice").write_file(folder/name)
        records.append({"dopant": d, "site": SITES[d], "atoms": len(s), "substitutions": 1,
                        "requested_pct": cfg["structure_target_pct"], "actual_substitution_pct": actual_pct,
                        "total_atom_pct": 100./len(s), "status": "unrelaxed_starting_geometry", "file": name})
    csv_out(pd.DataFrame(records), out / "starting_structure_manifest.csv")


def reference_dos(raw, host, cfg, out, api_key=None):
    if not cfg["download_reference_dos"]: return
    refs = [("host", host)]
    for dopant in DOPANTS:
        pool = [d for d in raw["ternaries"] if dopant in Structure.from_dict(d["structure"]).composition]
        if pool: refs.append((dopant, min(pool, key=lambda d: numeric(d.get("energy_above_hull")))))
    folder = out / "MP_reference_DOS"
    folder.mkdir()
    records = []
    with MPRester(api_key or get_api_key(), timeout=60) as mpr:
        for label, doc in refs:
            mid = doc["material_id"]
            record = {"label": label, "material_id": mid, "formula": doc["formula_pretty"],
                      "interpretation": "pristine_host" if label == "host" else "bulk_ternary_NOT_dilute_doped_SnO"}
            try:
                dos = mpr.get_dos_by_material_id(mid)
                if dos is None: raise ValueError("no DOS")
                density = np.sum([np.asarray(v) for v in dos.densities.values()], axis=0)
                energies = np.asarray(dos.energies)-float(dos.efermi)
                csv_out(pd.DataFrame({"E_minus_Ef_eV": energies, "total_DOS_states_eV_cell": density}), folder/f"{mid}.csv")
                fig, ax = plt.subplots(figsize=(5.8, 3.4), layout="constrained")
                ax.plot(energies, density, color="#245C78", lw=1.4)
                ax.axvline(0, color="0.6", lw=.8)
                ax.set(xlabel="Energy relative to Fermi level (eV)", ylabel="DOS (states/eV/cell)",
                       title=f"{doc['formula_pretty']} | {mid} | MP reference")
                save_figure(fig, folder/f"{mid}_DOS")
                record["status"] = "downloaded"
            except Exception as err:
                record["status"] = f"unavailable_{type(err).__name__}"
            records.append(record)
    csv_out(pd.DataFrame(records), out / "DOS_reference_manifest.csv")


# ---------------------------- FIGURES AND REPORTS ---------------------------
COLORS = dict(zip(DOPANTS, ["#3B6FB6", "#DE8F05", "#029E73", "#D55E00",
                           "#9467BD"]))


def save_figure(fig, path):
    fig.savefig(str(path)+".png", dpi=240, bbox_inches="tight")
    fig.savefig(str(path)+".pdf", bbox_inches="tight")
    plt.close(fig)


def set_substitution_axis(ax, cfg, upper=None):
    """Show requested sublattice percentages on a linear concentration axis."""
    grid = np.sort(np.asarray(cfg["requested_pareto_pct"], float))
    high = float(grid[-1]) if upper is None else min(float(upper), float(grid[-1]))
    ticks = grid[grid <= high + 1e-9]
    ax.set(xscale="linear", xlim=(float(grid[0]), high))
    ax.set_xticks(ticks, [f"{value:g}" for value in ticks])


def workflow_figure(folder):
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(9, 9.8))
    ax.set(xlim=(0, 10), ylim=(-.25, 11.7)); ax.axis("off")
    def box(x, y, w, height, text, color="#EAF0F6"):
        ax.add_patch(FancyBboxPatch((x, y), w, height, boxstyle="round,pad=0.12",
                                   fc=color, ec="#40627A", lw=1.2))
        ax.text(x+w/2, y+height/2, text, ha="center", va="center", fontsize=10.4, linespacing=1.4)
    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, lw=1.2, color="#40627A"))
    box(.8, 10.05, 8.4, 1., "Define alpha-SnO, five dopants, sites and concentration basis\nRegister literature inputs and separate validation measurements")
    box(.3, 7.85, 4.15, 1.15, "MP Sn-O-A compound data\nComposition groups; curated alpha-SnO\nDescriptors without energy-label leakage")
    box(5.55, 7.85, 4.15, 1.15, "Physics and evidence registry\nNa / K / N activation inputs\nLi / P activation hypotheses")
    arrow((3., 9.95), (2.4, 9.1)); arrow((7., 9.95), (7.6, 9.1))
    box(.3, 5.65, 4.15, 1.2, "Nested grouped GB tuning\nLocked test; leave-one-element-out\nBootstrap energies; domain checks")
    box(5.55, 5.65, 4.15, 1.2, "Charge neutrality and hole statistics\nExplicit mobility / distortion proxies\nMonte Carlo process assumptions")
    arrow((2.4, 7.7), (2.4, 7.0)); arrow((7.6, 7.7), (7.6, 7.0))
    box(.8, 3.55, 8.4, 1.05, "[Energy, -log(conductivity), mobility loss, distortion, phase proxy]\nCompare dopants at each fixed concentration\nNon-dominated fronts; explicit applicability flags", "#E8F1EB")
    arrow((2.4, 5.5), (3., 4.75)); arrow((7.6, 5.5), (7., 4.75))
    box(.8, 1.75, 8.4, 1., "Pareto frequencies + fixed-concentration dopant score\nDeclare weights; test preference and physical-scenario sensitivity", "#E8F1EB")
    arrow((5., 3.4), (5., 2.9))
    box(.8, .1, 8.4, .8, "Select a candidate; label its evidence and applicability status\nExport compact Colab output and an illustrated results ZIP", "#FFF2D9")
    arrow((5., 1.6), (5., 1.05))
    ax.text(5, 11.45, "Python implementation workflow", ha="center", fontsize=17, weight="bold")
    save_figure(fig, folder / "01_workflow")


def make_figures(screen, pareto, robust, cfg, out):
    folder = out / "figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    workflow_figure(folder)
    metrics = pd.read_csv(out / "model_metrics.csv")
    pooled = pd.read_csv(out / "nested_oof_predictions.csv")
    held = pd.read_csv(out / "locked_test_predictions.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), layout="constrained")
    for ax, observed, predicted, title, color in [
        (axes[0], pooled.observed, pooled.predicted, "Nested grouped CV", "#3B6F91"),
        (axes[1], held.observed_eV_atom, held.predicted_eV_atom, "Locked composition-group test", "#26856C")]:
        ax.scatter(observed, predicted, s=23, alpha=.7, color=color)
        low, high = min(observed.min(), predicted.min()), max(observed.max(), predicted.max())
        ax.plot([low, high], [low, high], color=".3", ls="--", lw=1)
        stats = regression_metrics(observed, predicted)
        ax.set(xlabel="MP formation energy (eV/atom)", ylabel="GB prediction (eV/atom)",
               title=f"{title}\nMAE={stats['MAE_eV_atom']:.4f}; R²={stats['R2']:.4f}")
    axes[2].hist(pooled.observed-pooled.predicted, bins=22, alpha=.65, color="#3B6F91", label="Nested CV", density=True)
    axes[2].hist(held.observed_eV_atom-held.predicted_eV_atom, bins=15, alpha=.55, color="#26856C", label="Locked test", density=True)
    axes[2].axvline(0, color=".3", ls="--", lw=1)
    axes[2].set(xlabel="Observed minus predicted energy (eV/atom)", ylabel="Probability density (atom/eV)",
                title="Gradient Boosting residuals")
    axes[2].legend(frameon=False)
    save_figure(fig, folder / "02_GradientBoosting_validation")
    fig, ax = plt.subplots(figsize=(8.2, 4.4), layout="constrained")
    for i, d in enumerate(ACCEPTORS):
        center, low, high = EA[d][:3]
        ax.errorbar(i, center*1000, yerr=np.array([[center-low], [high-center]])*1000,
                    fmt="o", color=COLORS[d], capsize=5, ms=7)
        if d in ["Li", "P"]: ax.text(i, high*1000+12, "assumed", ha="center", fontsize=9)
    ax.set(xticks=np.arange(len(ACCEPTORS)), xticklabels=ACCEPTORS,
           ylabel="Effective activation input (meV)", title="Literature inputs and sensitivity bounds")
    ax.text(.03, .96, "Bounds are scenario inputs,\nnot model confidence intervals", transform=ax.transAxes,
            ha="left", va="top", fontsize=9)
    save_figure(fig, folder / "03_activation_inputs")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), layout="constrained")
    for d in DOPANTS:
        a = screen[(screen.dopant == d) & screen.substitution_pct.isin(cfg["requested_pareto_pct"])].sort_values("substitution_pct")
        r = robust[robust.dopant == d].sort_values("substitution_pct")
        if not a.sigma_scenario_S_m.notna().any(): continue
        axes[0].plot(a.substitution_pct, a.sigma_scenario_S_m, label=d, color=COLORS[d])
        axes[1].plot(a.substitution_pct, a.mu_scenario_cm2_Vs, label=d, color=COLORS[d])
    axes[0].set(yscale="log", xlabel="Substitution of specified sublattice (%)",
                ylabel="Scenario conductivity (S/m)", title="Uncompensated reference scenario")
    axes[1].set(xlabel="Substitution of specified sublattice (%)",
                ylabel=r"Scenario mobility (cm$^2$ V$^{-1}$ s$^{-1}$)", title="Scattering-proxy estimate")
    axes[1].axvline(cfg["dilute_screening_limit_pct"], ls="--", color=".5", lw=1)
    axes[0].axvline(cfg["dilute_screening_limit_pct"], ls="--", color=".5", lw=1)
    for ax in axes:
        set_substitution_axis(ax, cfg)
    axes[0].legend(ncols=3, fontsize=9)
    save_figure(fig, folder / "04_transport_scenarios")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), layout="constrained")
    cmap = plt.colormaps["viridis"].copy(); cmap.set_bad("#E8E8E8")
    for ax, column, title in zip(axes, ["pareto_frequency", "supported_pareto_frequency"],
                                 ["Exploratory Pareto frequency", "After evidence and applicability gates"]):
        matrix = robust.pivot(index="dopant", columns="substitution_pct", values=column).reindex(index=DOPANTS, columns=cfg["requested_pareto_pct"])
        im = ax.imshow(np.ma.masked_invalid(matrix.to_numpy()), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set(xticks=range(matrix.shape[1]), xticklabels=cfg["requested_pareto_pct"],
               yticks=range(len(DOPANTS)), yticklabels=DOPANTS, xlabel="Sublattice substitution (%)", title=title)
        for (i, j), value in np.ndenumerate(matrix.to_numpy()):
            ax.text(j, i, f"{value:.2f}" if np.isfinite(value) else "—", ha="center", va="center",
                    fontsize=9, color="white" if np.isfinite(value) and value < .5 else "black")
    fig.colorbar(im, ax=axes, label="Frequency under sampled assumptions", shrink=.8)
    save_figure(fig, folder / "05_Pareto_robustness")
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    fig.subplots_adjust(bottom=.27, left=.13, right=.97, top=.9)
    pct = float(cfg["requested_pareto_pct"][0])
    part = screen[np.isclose(screen.substitution_pct, pct)]
    ranks = pareto[(pareto.substitution_pct == pct) & (pareto.variant == "full") & (pareto.scope == "exploratory")].set_index("dopant")
    for _, row in part.iterrows():
        if not np.isfinite(row.sigma_scenario_S_m): continue
        rank = ranks.loc[row.dopant, "front"]
        first, unranked = rank == 1, not np.isfinite(rank)
        if unranked:
            ax.scatter(row.FE_ML_eV_atom, np.log10(row.sigma_scenario_S_m), s=90,
                       marker="D", facecolors="none", edgecolors=COLORS[row.dopant])
        else:
            ax.scatter(row.FE_ML_eV_atom, np.log10(row.sigma_scenario_S_m), s=90,
                       marker="o" if first else "x", color=COLORS[row.dopant])
        label = row.dopant + (" (unranked)" if unranked else "")
        ax.annotate(label, (row.FE_ML_eV_atom, np.log10(row.sigma_scenario_S_m)),
                    xytext=(5, 4), textcoords="offset points")
    ax.set(xlabel="Compound energy proxy (eV/atom); lower preferred",
           ylabel="log10(conductivity / [1 S/m]); higher preferred",
           title=f"Exploratory screening at {pct:g}% substitution\nTwo-dimensional projection of five objectives")
    ax.margins(x=.13, y=.12)
    fig.text(.13, .025, "Circles: first front in five objectives. Crosses: dominated. Open diamonds: incomplete objectives.\nProjection alone cannot establish dominance.",
             fontsize=8, va="bottom")
    save_figure(fig, folder / "06_Pareto_projection")
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.3), layout="constrained")
    for ax, d in zip(axes.ravel(), ACCEPTORS):
        a = screen[(screen.dopant == d) & screen.substitution_pct.isin(cfg["requested_pareto_pct"])].sort_values("substitution_pct")
        r = robust[robust.dopant == d].sort_values("substitution_pct")
        ax.plot(a.substitution_pct, a.sigma_scenario_S_m, color=COLORS[d], lw=1.5)
        ax.fill_between(r.substitution_pct, r.sigma_S_m_scenario_p05, r.sigma_S_m_scenario_p95,
                         color=COLORS[d], alpha=.18)
        ax.axvline(cfg["dilute_screening_limit_pct"], ls="--", color=".5", lw=.8)
        ax.set(yscale="log", title=d, xlabel="Sublattice substitution (%)", ylabel="Conductivity (S/m)")
        set_substitution_axis(ax, cfg)
    for ax in axes.ravel()[len(ACCEPTORS):]: ax.set_axis_off()
    fig.suptitle("5–95% scenario ranges; uncompensated reference curves; scope limit shown", fontsize=12)
    save_figure(fig, folder / "07_conductivity_uncertainty")



def make_concentration_score_figures(all_scores, cfg, out):
    """Plot the existing independently normalized scores; no model or score is refitted.

    Each column compares dopants at one concentration. The highest central score
    is marked, with exact ties retained. The configured reference and evidence
    gates still determine final selection; these plots do not aggregate scores.
    """
    from matplotlib.patches import Rectangle
    out = Path(out)
    folder = out / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    grid = np.sort(np.asarray(cfg["requested_pareto_pct"], dtype=float))
    part = all_scores[all_scores.substitution_pct.isin(grid)].copy()
    if part.duplicated(["substitution_pct", "dopant"]).any():
        raise ValueError("Duplicate dopant/concentration rows in the score plot input.")
    matrix = part.pivot(index="substitution_pct", columns="dopant", values="score_0_100")
    matrix = matrix.reindex(index=grid, columns=DOPANTS)
    matrix.index.name = "substitution_pct"
    values = matrix.to_numpy(dtype=float)
    finite = np.isfinite(values)
    if np.any(finite & ((values < -1e-9) | (values > 100.+1e-9))):
        raise ValueError("Central score outside the expected 0–100 range.")
    best = np.max(np.where(finite, values, -np.inf), axis=1)
    top = finite & np.isclose(values, best[:, None], rtol=0, atol=1e-9)
    numerical = matrix.reset_index()
    numerical["highest_central_score_dopants"] = [
        "; ".join(d for d, flag in zip(DOPANTS, row) if flag) or "Not scored"
        for row in top]
    csv_out(numerical, out / "concentration_central_scores_independent.csv")

    weights = validate_score_config(cfg)
    weight_note = (f"Weights: energy {100*weights[0]:g}%, log conductivity {100*weights[1]:g}%, "
                   f"mobility {100*weights[2]:g}%.")
    scope_note = ("Each concentration has its own min–max normalization; scores compare dopants within that concentration.\n"
                  "These scores do not define an optimum concentration.")

    # Concentration is numeric here: intervals retain their actual spacing.
    fig, ax = plt.subplots(figsize=(11.2, 6.5))
    fig.subplots_adjust(left=.09, right=.98, bottom=.28, top=.86)
    for j, dopant in enumerate(DOPANTS):
        if finite[:, j].any():
            ax.plot(grid, values[:, j], color=COLORS[dopant], label=dopant,
                    marker="o", ms=5, lw=1.8)
    marked = False
    for i, j in zip(*np.nonzero(top)):
        ax.scatter(grid[i], values[i, j], marker="*", s=170,
                   facecolors=COLORS[DOPANTS[j]], edgecolors="black", linewidths=.8,
                   zorder=5, label="Highest central score" if not marked else None)
        marked = True
    reference = float(cfg["score_reference_pct"])
    ax.axvline(reference, color=".4", ls="--", lw=1,
               label=f"Final-selection reference: {reference:g}%")
    ax.set(ylim=(0, 105), yticks=np.arange(0, 101, 20),
           xlabel="Nominal substitution of designated host sublattice (%)",
           ylabel="Central dopant score (0–100)",
           title="Concentration-dependent central scores\nIndependent normalization at each concentration")
    set_substitution_axis(ax, cfg)
    # A small margin prevents winner stars at 1% and 20% from being clipped.
    span = grid[-1]-grid[0]
    ax.set_xlim(grid[0]-.02*span, grid[-1]+.02*span)
    ax.grid(axis="y", alpha=.20)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncols=4,
              frameon=False, fontsize=9)
    fig.text(.09, .075, weight_note, fontsize=9)
    fig.text(.09, .025, scope_note, fontsize=9)
    save_figure(fig, folder / "11_score_vs_concentration")

    # Categorical columns make each sampled concentration equally readable.
    fig, ax = plt.subplots(figsize=(11.2, 6.5))
    fig.subplots_adjust(left=.09, right=.88, bottom=.25, top=.85)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#EEEEEE")
    image = ax.imshow(np.ma.masked_invalid(values.T), vmin=0, vmax=100,
                      cmap=cmap, interpolation="nearest", aspect="auto")
    for i in range(len(grid)):
        for j in range(len(DOPANTS)):
            value = values[i, j]
            label = f"{value:.2f}" if finite[i, j] else "N/A"
            ax.text(i, j, label, ha="center", va="center", fontsize=10,
                    color="white" if finite[i, j] and value >= 65 else "#17212B",
                    fontweight="bold" if top[i, j] else "normal")
            if top[i, j]:
                ax.add_patch(Rectangle((i-.47, j-.47), .94, .94,
                                       fill=False, edgecolor="#E6A700", linewidth=2.5))
    ax.set(xticks=np.arange(len(grid)), xticklabels=[f"{x:g}" for x in grid],
           yticks=np.arange(len(DOPANTS)), yticklabels=DOPANTS,
           xlabel="Nominal sublattice substitution (%)", ylabel="Dopant",
           title="Concentration-dependent central scores (0–100)\nIndependent normalization at each concentration")
    ax.set_xticks(np.arange(-.5, len(grid), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(DOPANTS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(image, ax=ax, fraction=.035, pad=.035, label="Central score (0–100)")
    fig.text(.09, .15, "Gold outline: highest central score in each column; exact ties are retained. "
             "N/A: no acceptor score.", fontsize=9)
    fig.text(.09, .105, weight_note, fontsize=9)
    fig.text(.09, .055, scope_note, fontsize=9)
    save_figure(fig, folder / "15_independent_concentration_score_heatmap")
    return numerical


def make_added_figures(screen, score_details, all_scores, selection, cfg, out):
    folder = out / "figures"
    individual = folder / "formation_energy_by_dopant"
    individual.mkdir(exist_ok=True)
    fe_columns = ["dopant", "site", "substitution_pct", "x", "FE_ML_eV_atom",
                  "FE_bootstrap_p05", "FE_bootstrap_p95", "FE_empirical_lower",
                  "FE_empirical_upper", "in_domain", "within_dilute_model_scope"]
    csv_out(screen[fe_columns], out / "formation_energy_curves.csv")
    def fe_axis(ax, d, limit):
        a = screen[(screen.dopant == d) & (screen.substitution_pct <= limit)].sort_values("substitution_pct")
        ax.plot(a.substitution_pct, a.FE_ML_eV_atom, color=COLORS[d], ls="--", marker="o", ms=3.2, lw=1.4)
        dilute = a[a.substitution_pct <= cfg["dilute_screening_limit_pct"]]
        ax.plot(dilute.substitution_pct, dilute.FE_ML_eV_atom, color=COLORS[d], marker="o", ms=2.6, lw=1.6)
        ax.fill_between(a.substitution_pct, a.FE_bootstrap_p05, a.FE_bootstrap_p95, color=COLORS[d], alpha=.17)
        if limit > cfg["dilute_screening_limit_pct"]:
            ax.axvline(cfg["dilute_screening_limit_pct"], color=".6", ls=":", lw=.8)
        ax.set(title=f"{d} on {SITES[d]} sites", xlabel=f"Dopant concentration, 100x (% of {SITES[d]} sites)",
               ylabel="Formation energy (eV/atom)")
        set_substitution_axis(ax, cfg, upper=limit)
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.grid(alpha=.15)
    maximum = float(screen.substitution_pct.max())
    for limit, name, title in [
        (maximum, "08_formation_energy_all_dopants", "Predicted compound formation energy: all five dopants"),
        (cfg["dilute_screening_limit_pct"], "09_formation_energy_dilute", "Formation energy in the declared dilute-model range")]:
        fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.2))
        fig.subplots_adjust(left=.07, right=.985, bottom=.14, top=.88, wspace=.44, hspace=.55)
        for ax, d in zip(axes.ravel(), DOPANTS): fe_axis(ax, d, limit)
        for ax in axes.ravel()[len(DOPANTS):]: ax.set_axis_off()
        fig.suptitle(title, fontsize=15)
        fig.text(.07, .02, "Lines: ML composition predictions. Shading: 5–95% group-bootstrap spread. Dashed: above the declared dilute-model limit.\n"
                 "These are compound energies, not charged-defect energies; a dilute concentration alone does not establish applicability.", fontsize=9)
        save_figure(fig, folder / name)
    for d in DOPANTS:
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), layout="constrained")
        fe_axis(axes[0], d, maximum); fe_axis(axes[1], d, cfg["dilute_screening_limit_pct"])
        fig.suptitle(f"{d}-doped SnO: compound-energy proxy and 5–95% bootstrap spread", fontsize=12)
        save_figure(fig, individual / f"FE_{d}_vs_concentration")
    # Stacked bars show exactly how much each criterion contributes to the score.
    a = score_details.copy()
    fig, ax = plt.subplots(figsize=(10.4, 5.9))
    fig.subplots_adjust(left=.10, right=.94, bottom=.22, top=.85)
    left = np.zeros(len(a))
    for name, color, label in zip(SCORE_KEYS, ["#3B6FB6", "#029E73", "#DE8F05"],
                                  ["Formation energy", "log conductivity", "Mobility"]):
        points = a[f"points_{name}"].fillna(0).to_numpy()
        ax.barh(a.dopant, points, left=left, color=color, label=label, height=.65)
        left += points
    for i, row in enumerate(a.itertuples()):
        label = (f"{row.score_0_100:.1f}" + ("  SELECTED" if row.selected_at_reference else "")
                 if np.isfinite(row.score_0_100) else "N/A — not scored")
        ax.text(left[i]+1., i, label, va="center", fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 119)
    ax.set(xticks=[0, 20, 40, 60, 80, 100], xlabel="Dopant preference score (0–100; higher is preferred)",
           title=f"Dopant selection at {cfg['score_reference_pct']:g}% sublattice substitution\n{selection['selection_status'].replace('_', ' ')}")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncols=3, frameon=False)
    fig.text(.10, .025, "Default weights: energy 25%, conductivity 50%, mobility 25% (editable). Missing acceptor transport is not scored.\n"
             "Selection depends on the chosen concentration, weights and physical assumptions; it is not experimental confirmation.", fontsize=8.5)
    # Show the actual configured weights even after the user edits them.
    fig.texts[-1].set_text("Weights: " + ", ".join(f"{k.replace('_', ' ')} {100*v:g}%" for k, v in selection["weights"].items()) +
        ". Missing acceptor transport is not scored.\nSelection depends on concentration, preferences and physical assumptions; it is not experimental confirmation.")
    save_figure(fig, folder / "10_dopant_score_components")
    make_concentration_score_figures(all_scores, cfg, out)
    dist = pd.read_csv(out / "dopant_distribution.csv")
    fig, ax = plt.subplots(figsize=(9.4, 4.9), layout="constrained")
    pos = np.arange(len(dist))
    first = ax.bar(pos-.19, dist.structures, width=.36, color=[COLORS[d] for d in dist.dopant], label="MP structures")
    second = ax.bar(pos+.19, dist.unique_compositions, width=.36, color="#A8B8C8", label="Unique compositions")
    ax.bar_label(first, padding=3, fontsize=9); ax.bar_label(second, padding=3, fontsize=9)
    ax.set(xticks=pos, xticklabels=dist.dopant, ylabel="Number in the filtered ML dataset",
           title="Coverage of the five target dopants\nOther-element ternaries and the host also contribute to model training")
    ax.set_ylim(0, max(1, dist.structures.max())*1.2)
    ax.legend(frameon=False)
    save_figure(fig, folder / "12_filtered_dataset_distribution")
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))
    fig.subplots_adjust(left=.07, right=.98, bottom=.22, top=.85, wspace=.3)
    for ax, field, title in zip(axes, ["weight_sweep_top_share", "scenario_top_share"],
                                ["Preference-weight sensitivity", "FE and physical-scenario sensitivity"]):
        vals = 100.*a[field]
        ax.bar(a.dopant, vals.fillna(0), color=[COLORS[d] for d in a.dopant])
        for i, value in enumerate(vals):
            ax.text(i, (value if np.isfinite(value) else 0)+1, f"{value:.1f}" if np.isfinite(value) else "N/A", ha="center", fontsize=9)
        ax.set(ylim=(0, 110), ylabel="Top-choice share (%)", title=title)
    fig.suptitle(f"How stable is the selection at {cfg['score_reference_pct']:g}%?", fontsize=14)
    fig.text(.07, .035, "Left: uniformly sampled preference weights. Right: fixed configured weights with the existing uncertainty scenarios.\n"
             "Exact ties share one win; all draws remain in the denominator. These shares are sensitivity diagnostics, not calibrated probabilities.", fontsize=9)
    save_figure(fig, folder / "13_score_selection_sensitivity")


def write_provenance(out, cfg):
    references = pd.DataFrame(REFERENCES, columns=["reference", "doi", "host_scope", "role"])
    references["url"] = "https://doi.org/" + references.doi
    references["checked_date"] = "2026-09-06"
    csv_out(references, out / "references.csv")
    csv_out(pd.DataFrame(BENCHMARKS), out / "literature_reference_ranges.csv")
    csv_out(pd.DataFrame([{"dopant": d, "site": SITES[d], "role": ROLES[d],
                           "EA_central_eV": EA[d][0], "EA_low_eV": EA[d][1], "EA_high_eV": EA[d][2],
                           "source_doi": EA[d][3], "evidence": EA[d][4],
                           "covalent_radius_A": RADII[d], "electronegativity": CHI[d]} for d in DOPANTS]),
            out / "dopant_evidence_registry.csv")
    versions = {}
    for package in ["numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "mp-api", "pymatgen", "pymatgen-core", "joblib"]:
        try: versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError: pass
    write_json(out / "run_config.json", {"config": cfg, "physical_assumptions": PHYSICS,
               "versions": versions, "python": sys.version, "dopants": DOPANTS,
               "utc": datetime.now(timezone.utc).isoformat(), "excluded_elements": sorted(EXCLUDED_ELEMENTS), "model_family": MODEL_NAME})
    (out / "requirements_resolved.txt").write_text("\n".join(f"{p}=={v}" for p, v in versions.items())+"\n")
    # No API key, environment dump, or authentication response is exported.


def write_report(data, screen, robust, chosen, reliable, validation, selection, cfg, out):
    eligible = screen[screen.supported_screening & (screen.x > 0)]
    supported_dopants = [d for d in DOPANTS if d in set(eligible.dopant)]
    requested = robust[robust.substitution_pct.isin(cfg["requested_pareto_pct"])]
    summary = requested.groupby("dopant").agg(
        mean_exploratory_Pareto_frequency=("pareto_frequency", "mean"),
        mean_compromise_share=("compromise_share_mean", "mean"),
        supported_grid_points=("supported_screening", "sum")).reindex(DOPANTS)
    csv_out(summary.reset_index(), out / "requested_grid_summary.csv")
    # Grid averages remain descriptive; the score selects at ONE fixed x.
    points = ", ".join(map(str, cfg["requested_pareto_pct"]))
    message = (
        f"Prespecified formation-energy model: {chosen}. Quality gate: {reliable}.\n"
        f"Curated MP data: {len(data)} structures / {data.formula.nunique()} compositions.\n"
        f"Virtual dopant/concentration rows: {len(screen)} (predictions; not training labels).\n"
        f"Score-selected candidate(s) at {cfg['score_reference_pct']:g}%: {', '.join(selection['selected_dopants']) or 'none'}.\n"
        f"Selection status: {selection['selection_status']}.\n"
        f"Dopants with at least one point passing screening gates: {', '.join(supported_dopants) or 'none'}.\n"
        f"Requested fixed-concentration analysis: {points}% of the substituted sublattice.\n"
        f"External experimental validation: {validation['experimental_validation']}.\n"
    )
    notes = """
INTERPRETATION
The five candidate dopants are Li, Na, K, N and P. Li and P acceptor energies
are explicitly unvalidated sensitivity assumptions. Literature-derived effective
activation inputs are available for Na, K and N.

Gradient Boosting is the only trained estimator family, prespecified for this workflow.
Its hyperparameter tuning is nested inside the composition-group development folds.
The locked test is not used for parameter tuning or feature selection. Element-transfer
testing retunes Gradient Boosting using only the remaining elements. The quality gate
also compares its test MAE with an arithmetic training-mean reference, computed without
fitting any additional model. Only GB bootstrap and residual uncertainty are reported;
there is no cross-model disagreement estimate. These tests concern compound energies,
not dopant activation, solubility, Hall transport, or virtual-structure accuracy.

Only compound formation energy is learned. Lower MP formation energy across different
chemistries does not prove easier dopant incorporation. True charged-defect energies
require chemical potentials, charge states, Fermi energy and finite-size corrections.
The competing-ternary proxy is not a chemical-potential phase diagram: it omits elemental
and binary competition and inherits database coverage bias. Its removal is tested.

The central physical parameters and Monte Carlo ranges are declared assumptions, not
fitted experimental values. p follows a single-acceptor charge-neutrality model with
Fermi-Dirac hole statistics, a fixed ionized UID background, and fully ionized compensating
donors. The apparent literature activation energy is used as an effective level only
for screening. Impurity bands, concentration-dependent level shifts, defect complexes,
microstructure, anisotropy and Hall factors are not determined by this model.
Reference curves assume zero donor compensation. Monte Carlo explores 0-50% compensation;
the reference curves are not medians of the scenario ensemble. This keeps the baseline
acceptor response distinct from hypothetical compensation-induced carrier suppression.
A conservative midgap-minus-3kT check suppresses conductivity estimates where neglect
of electrons may fail. The 0.67 eV indirect-gap reference is not the optical direct gap.

Radius mismatch, mass retention and mobility loss are heuristic proxies. A consistent
covalent-radius convention replaces mixed ionic/covalent conventions. The radius-free
sensitivity removes both the distortion objective and radius-dependent transport penalties.
The compatibility index is not reused to invent the activation energy.

Pareto dominance uses five minimization objectives separately at each concentration:
[FE_ML, -log10(sigma), 1-Rmu, distortion index, competing-ternary proxy].
No weights enter dominance. The equal-scale distance to the ideal point is a transparent
post-Pareto compromise rule, not a detected geometric knee. Near ties remain ties.
With five acceptor candidates and five partly correlated objectives, many points can
remain non-dominated; this does not establish a unique optimum. A 2D projection cannot
substitute for the five-dimensional comparison.

The additional dopant preference score is a separate decision rule at one declared
concentration (default 5% sublattice substitution), not a new ML target. It combines
formation energy (lower preferred), log10(sigma/[1 S/m]) and mobility (higher preferred).
For each minimization cost c, U=(c_max-c)/(c_max-c_min), using complete candidates
at that same concentration. A constant criterion has neutral utility 0.5. Missing
positively weighted criteria make a candidate ineligible; they are never imputed.
S=100*sum(w_j*U_j), with default weights 0.25, 0.50 and 0.25 respectively.
These are editable application preferences, not measured physical constants.
The positive-weight score is monotonic in its active criteria, so a strictly
dominated point cannot outrank its dominator on those criteria. Its three-objective
front is exported separately from the original five-objective Pareto analysis.
The selected candidate is the maximum-score supported candidate if one exists;
otherwise it is explicitly the exploratory score leader. Exact ties are retained.
Scores at different concentrations use different anchors and must not be maximized
across concentrations as if they were on a universal absolute scale.

Conductivity contains mobility, so the separate mobility term intentionally gives
additional preference to mobility. Equal weights, criterion removal and a uniform
simplex weight sweep expose this preference dependence. Physical-scenario scores
reuse the Pareto calculation's energy and transport draws, with fixed central
normalization anchors and utilities clipped to 0-1. The ranges and top-choice
shares are sensitivity diagnostics; they are not calibrated confidence intervals.
Small energy differences relative to CV error do not establish dopant superiority.
See score_normalization.csv and the two score sensitivity tables for all choices.
Methodological guidance: https://doi.org/10.1787/9789264043466-en.

Bootstrap energy variation is combined with empirical grouped residuals and physical
scenario sampling. Monte Carlo frequencies and 5-95% ranges describe the chosen
assumptions; they are not calibrated posterior probabilities or experimental error bars.
Shared versus dopant-specific incorporation efficiency is reported separately.
Pareto frequencies use all scenario draws as denominator; the fraction with eligible
objective values is also reported. Never-eligible controls retain missing frequencies.

Supported screening requires ML quality, a passed element-transfer tolerance, descriptor
and concentration coverage, an experimental activation input, the declared dilute-model
scope, and absence of a flagged process-specific phase/morphology conflict. This label
does not mean experimentally validated. The 3% default scope is a modeling decision,
not a universal solubility limit. High concentrations are explicitly extrapolative.
If no candidate passes, report insufficient support instead of promoting an OOD winner.

The Excel attachment is a bibliography with summaries. It is not a paired numerical
dataset for regression or Hall validation. Reported activation inputs cannot also be
claimed as validation successes. Published transport ranges from different processes
must not be converted into artificial paired samples or pooled into a validation R2.
Use the exported CSV template with measured dopant concentration, site, temperature,
phase, process, matched UID baseline and sample-level transport for actual comparison.

Selection interpretation is generated from the current scores and their sensitivity
results. No candidate is assigned a predetermined advantage or winner label. It reports
the numerical trade-offs against the closest alternative and the specific failed
screening gates. Published activation inputs are provenance, not a validation success.

The objective is p-type conductivity screening. Optical transmission, TFT switching,
stability under operation, synthesis cost and toxicity are not measured here. A general
claim of best dopant for all applications is therefore unsupported.

OUTPUT GUIDE
After extracting the ZIP, start with START_HERE.html (illustrated) or START_HERE.txt.
01_summary/: compact dataset counts, dopant distribution, model metrics and selection.
02_figures/: main figures and individual formation-energy plots (PNG and vector PDF).
03_dataset/: complete filtered/raw data and chemical coverage.
04_models/: all CV/test/transfer metrics, predictions, models and bootstrap draws.
05_screening/: full properties, Pareto fronts, score details and scenario arrays.
06_validation/: input templates, comparison outputs and validation status.
07_methods_and_sources/: equations, assumptions, literature and source code.
08_DFT_inputs_and_DOS/: optional unrelaxed structures and MP reference DOS.
FILE_INDEX.csv and COLUMN_GUIDE.csv explain the files, columns and units.

model_metrics.csv: grouped nested CV and locked-test errors, eV/atom.
leave_one_dopant_out_metrics.csv: five-element transfer audit with MAE and RMSE; R2 is omitted.
ml_predictions.csv: virtual energies, spread, distance and range checks.
screening_results.csv: central physical scenarios and explicit evidence/scope flags.
pareto_fronts.csv: fronts, eligibility and compromise ties by objective set and scope.
robustness_results.csv: Pareto frequencies and scenario intervals, with efficiency ablations.
requested_grid_summary.csv: descriptive averages across the requested grid, not a new score.
external_validation_status.json: whether any sample-matched validation actually occurred.
references.csv and literature_reference_ranges.csv: provenance; ranges are unpaired.
raw_materials.json: reusable MP snapshot; set CONFIG['cached_raw_json'] to reuse it.
02_figures/: PNG and vector PDF figures, with proxy/scenario labels.
unrelaxed_DFT_starting_structures/: optional starting geometries, not calculated DFT results.

CONCENTRATION AND UNITS
The nominal screening grid contains only 1, 2, 3, 5, 10, 15 and 20% sublattice substitution.
Five dopants at seven concentrations give 35 virtual prediction rows.
The pristine host is retained as a training/physical reference, outside this screening grid.
x is substitution fraction of Sn sites (Li/Na/K) or O sites (N/P).
Substitution percent = 100*x; total-atom percent = 50*x in these ideal formulas.
Nominal dopant density = x times host sublattice density (cm^-3).
p is in cm^-3; mobility is in cm^2/(V s).
sigma(S/m) = q * [p(cm^-3)*1e6] * [mu(cm^2/(V s))*1e-4] = 100*q*p*mu.
rho(ohm m)=1/sigma(S/m); rho(ohm cm)=100/sigma(S/m). Distortion is an index, not an XRD strain percentage.
Missing data are exported as blank/NaN and never converted to favorable zero values.
"""
    interpretation_text = (out / "selection_interpretation.txt").read_text(encoding="utf-8")
    (out / "READ_RESULTS.txt").write_text(message+"\n"+interpretation_text+"\n"+notes, encoding="utf-8")
    (out / "run_summary.txt").write_text(message +
        "Units: formation energy = eV/atom; conductivity = S/m; mobility = cm^2/(V s).\n"
        "Selection is a preference-dependent screening result, not experimental confirmation.\n"
        "See selection_interpretation.txt for the advantages, disadvantages and numerical trade-offs.\n", encoding="utf-8")
    return message


# ----------------------- COMPACT OUTPUT AND READER ZIP ----------------------
SUMMARY_TABLES = ["dataset_summary", "dopant_distribution", "gradient_boosting_metrics", "dopant_score_summary",
                  "concentration_central_scores_independent"]
TABLE_TITLES = {
    "concentration_central_scores_independent": "Concentration-dependent central scores — independent normalization",
    "dataset_summary": "Final dataset counts",
    "dopant_distribution": "Target-dopant distribution in the filtered dataset",
    "gradient_boosting_metrics": "Gradient Boosting: nested CV and locked test",
    "dopant_score_summary": "Dopant score at the fixed reference concentration",
}
DISPLAY_NAMES = {
    "substitution_pct": "Sublattice substitution (%)",
    "highest_central_score_dopants": "Highest central score",
    "quantity": "Quantity", "count": "Count", "dopant": "Dopant", "structures": "Structures",
    "unique_compositions": "Unique compositions", "percent_of_final_dataset": "Final dataset (%)",
    "model": "Model", "stage": "Evaluation stage", "n": "Validation rows", "MAE_eV_atom": "MAE (eV/atom)",
    "RMSE_eV_atom": "RMSE (eV/atom)", "R2": "R²", "score_0_100": "Score / 100",
    "rank": "Score rank", "FE_ML_eV_atom": "F.E. (eV/atom)",
    "sigma_scenario_S_m": "Conductivity (S/m)", "mu_scenario_cm2_Vs": "Mobility (cm²/(V·s))",
    "selected_at_reference": "Selected", "status": "Evidence status",
}
FILE_DESCRIPTIONS = {
    "concentration_central_scores_independent.csv": "Numerical central scores at each concentration, independently normalized; exact top-score ties retained for the five candidate dopants.",
    "dataset_summary.csv": "Final data counts; distinguishes MP labels from virtual screening compositions.",
    "dopant_distribution.csv": "Five target-dopant counts, unique compositions and shares; zero coverage is retained.",
    "gradient_boosting_metrics.csv": "Gradient Boosting pooled nested-CV and locked-test MAE, RMSE and R².",
    "dopant_score_summary.csv": "Five dopants at the reference concentration, with units, scores and selection status.",
    "dopant_selection.json": "Selected candidate(s), exact concentration, weights, close alternatives and evidence status.",
    "run_summary.txt": "Short outcome summary with units and validation status.",
    "selection_interpretation.txt": "Why the selected dopant leads: advantages, disadvantages, numerical alternatives, uncertainty and needed evidence.",
    "selection_interpretation.json": "Structured interpretation generated from the actual current selection.",
    "dopant_advantages_disadvantages.csv": "Advantages and disadvantages of all five dopants, with evidence and property comparisons.",
    "selected_dopant_tradeoffs.csv": "Criterion-by-criterion score differences between each selected dopant and its highest-scoring alternative.",
    "curated_dataset.csv": "Complete filtered ML training dataset: one row per MP structure, including other A elements.",
    "raw_materials.json": "Reusable MP snapshot with query provenance. Set cached_raw_json to this path to reuse it.",
    "raw_materials.csv": "Raw MP record audit before local structure filtering and host selection.",
    "chemical_coverage.csv": "Counts and composition ranges for every A element in the filtered dataset.",
    "rejected_records.csv": "Rejected local records and reasons; can be empty.",
    "host_reference.json": "Chosen alpha-SnO reference structure and MP metadata.",
    "X_features_and_groups.csv": "All input features and composition grouping labels; not a displayed summary.",
    "y_target.csv": "Observed compound formation-energy target in eV/atom.",
    "data_splits.csv": "Development and locked-test assignment for every material ID.",
    "model_metrics.csv": "Full outer-fold, pooled CV and locked-test MAE, RMSE and R² for Gradient Boosting only.",
    "model_selection.json": "Prespecified GB family, development-only hyperparameter tuning and screening quality gate.",
    "nested_oof_predictions.csv": "Out-of-fold observed and predicted energies for Gradient Boosting only.",
    "locked_test_predictions.csv": "Selected model predictions on the untouched composition-group test set.",
    "leave_one_dopant_out_metrics.csv": "Transfer metrics for all five withheld elements, including absent-label status.",
    "leave_one_dopant_out_predictions.csv": "Full observed and predicted values in the transfer tests.",
    "feature_importance_diagnostic.csv": "Held-out permutation importance for interpretation; not used for feature selection.",
    "formation_energy_model.joblib": "Final selected model refitted on the full curated data after evaluation.",
    "formation_energy_bootstrap_draws.npz": "Composition-bootstrap FE predictions for every virtual composition.",
    "screening_results.csv": "Complete central properties, applicability checks and evidence flags at every concentration.",
    "ml_predictions.csv": "Virtual FE predictions, bootstrap spread, descriptors and applicability diagnostics.",
    "formation_energy_curves.csv": "Plot-ready FE curves and bootstrap bounds for all five dopants; energy is eV/atom.",
    "pareto_fronts.csv": "Original five-objective fronts and objective-removal variants at fixed concentrations.",
    "robustness_results.csv": "Pareto frequencies and 5–95% physical scenario ranges; conductivity is S/m.",
    "mc_scenario_draws.npz": "The paired FE, carrier, mobility, conductivity and Pareto Monte Carlo draws.",
    "dopant_score_details.csv": "Reference-concentration utilities, weighted contributions, ranks and sensitivity diagnostics.",
    "dopant_scores_all_concentrations.csv": "Scores normalized separately at each concentration; not a common absolute cross-concentration scale.",
    "score_normalization.csv": "Exact preference weights, cost definitions and central normalization anchors.",
    "score_weight_sensitivity.csv": "Uniform-simplex weight draws and selected leaders; a preference sensitivity experiment.",
    "score_criterion_sensitivity.csv": "Configured/equal-weight scores and each criterion-removal test.",
    "score_scenario_draws.npz": "Scenario scores and fractional winner shares, with central fixed normalization.",
    "requested_grid_summary.csv": "Descriptive grid averages; the dopant score selection uses a single reference concentration.",
    "experimental_validation_template.csv": "Empty input schema for paired sample measurements. rho input remains ohm cm.",
    "dft_validation_template.csv": "Empty schema for compatible compound formation energies, not charged-defect energies.",
    "external_validation_status.json": "States whether experimental or compatible DFT comparisons were actually performed.",
    "READ_RESULTS.txt": "Full methodology, equations, assumptions, interpretation limits and file guide.",
    "run_config.json": "Reproducible settings, physical assumptions and installed package versions; no API key.",
    "references.csv": "Source DOIs, links and how each source is used.",
    "literature_reference_ranges.csv": "Unpaired literature ranges and source roles; not fabricated validation samples.",
    "dopant_evidence_registry.csv": "Dopant sites/roles, assumed or literature activation inputs, radii and sources.",
    "requirements_resolved.txt": "Versions resolved for this run.",
    "workflow_source.py": "Reusable workflow source with the embedded API key removed.",
    "starting_structure_manifest.csv": "Actual commensurate concentrations for the unrelaxed DFT starting geometries.",
    "DOS_reference_manifest.csv": "Optional MP reference DOS provenance and availability.",
    "COLUMN_GUIDE.csv": "Column names, units and descriptions for exported CSV tables.",
}


def find_result_file(out, filename):
    root = Path(out)
    paths = list(root.rglob(filename))
    # Prefer the organized result when legacy flat copies are also present.
    nested = [p for p in paths if p.parent != root]
    if nested: paths = nested
    if len(paths) != 1: raise FileNotFoundError(f"Expected one {filename}; found {len(paths)}.")
    return paths[0]


def load_result_tables(out):
    """All complete CSVs remain available by stem before or after ZIP organization."""
    tables = {}
    root = Path(out)
    organized = (root / "01_summary").is_dir() and (root / "03_dataset").is_dir()
    for path in sorted(root.rglob("*.csv")):
        if organized and path.parent == root and path.name not in ["FILE_INDEX.csv", "COLUMN_GUIDE.csv"]:
            continue
        try: frame = pd.read_csv(path)
        except pd.errors.EmptyDataError: frame = pd.DataFrame()
        if path.stem in tables: raise ValueError(f"Duplicate result-table name: {path.stem}")
        tables[path.stem] = frame
    return tables


def compact_table_html(frame):
    return frame.rename(columns=DISPLAY_NAMES).to_html(index=False, border=0, escape=True,
        max_rows=None, max_cols=None, float_format=lambda x: f"{x:.4f}" if abs(x) < 100 else f"{x:.5g}", na_rep="—")


def column_metadata(column):
    definitions = {
        "formation_energy_eV_atom": "Observed MP compound formation energy relative to elemental references.",
        "FE_ML_eV_atom": "ML compound-energy prediction for the virtual composition; not defect formation energy.",
        "FE_bootstrap_p05": "5th percentile of composition-bootstrap FE predictions; descriptive spread.",
        "FE_bootstrap_p95": "95th percentile of composition-bootstrap FE predictions; descriptive spread.",
        "FE_empirical_lower": "Bootstrap lower bound minus grouped residual envelope; not calibrated OOD coverage.",
        "FE_empirical_upper": "Bootstrap upper bound plus grouped residual envelope; not calibrated OOD coverage.",
        "substitution_pct": "100*x: percent of the substituted Sn sublattice, or O sublattice for N/P.",
        "total_atom_pct": "Percent of all atoms occupied by dopant; equals half the sublattice percent here.",
        "x": "Dopant fraction of the specified host sublattice, between 0 and 1.",
        "A_fraction": "Atomic fraction of the third element among all atoms in the ML composition.",
        "sigma_scenario_S_m": "Central conductivity q*p*mu with p and mu converted to SI; not measured conductivity.",
        "mu_scenario_cm2_Vs": "Mobility estimate from declared baseline and scattering proxies.",
        "mobility_retention": "Estimated mobility divided by the assumed undoped mobility.",
        "p_scenario_cm3": "Hole density from the declared charge-neutrality model; not an assumed Hall measurement.",
        "score_0_100": "Weighted preference score at one concentration; higher is preferred.",
        "score_objective_front": "Pareto front for the active score criteria; distinct from original five-objective fronts.",
        "supported_screening": "Passes all declared ML, transfer, domain, evidence, dilute and process checks.",
        "selected_at_reference": "Highest score in the declared selection pool; exact ties are retained.",
        "R2": "1 - sum((observed-predicted)^2)/sum((observed-mean(observed))^2). May be negative.",
        "MAE_eV_atom": "Mean absolute prediction error; used for model selection.",
        "RMSE_eV_atom": "Square root of mean squared prediction error.",
        "scenario_top_share": "Fractional wins divided by all physical/energy draws; not a posterior probability.",
        "weight_sweep_top_share": "Fractional wins under uniform-simplex preference weights.",
        "rho_measured_ohm_cm": "Experimental template input resistivity in ohm cm, retained for compatibility.",
    }
    unit = "dimensionless / label"
    if "eV_atom" in column or column.startswith(("FE_bootstrap_", "FE_empirical_")): unit = "eV/atom"
    elif "eV" in column: unit = "eV"
    elif "_S_m" in column: unit = "S/m"
    elif "cm2_Vs" in column: unit = "cm^2/(V s)"
    elif "_cm3" in column: unit = "cm^-3"
    elif "ohm_cm" in column: unit = "ohm cm"
    elif "ohm_m" in column: unit = "ohm m"
    elif "pct" in column or column.startswith("percent_"): unit = "%"
    elif column.startswith(("score_scenario_p", "points_")) or column == "score_0_100": unit = "score points (0-100)"
    elif column in ["count", "structures", "unique_compositions", "n", "n_materials", "n_compositions", "n_test"]: unit = "count"
    if column.startswith("log10_error"):
        unit = "dimensionless (log10 ratio)"
        meaning = "log10(predicted / observed); zero denotes agreement."
    elif column.startswith("utility_"): meaning = "Min-max desirability (0-1) of " + column.removeprefix("utility_").replace("_", " ")
    elif column.startswith("points_"): meaning = "100 * configured weight * criterion utility."
    elif column.startswith("descriptor_") or column.startswith(("mean_", "std_", "A_weighted_", "struct_")):
        meaning = "Exported composition/structure descriptor; see features() in workflow_source.py for its exact definition."
        unit = "descriptor-dependent; see source"
    else: meaning = definitions.get(column, column.replace("_", " ") + "; context is given in the table description and READ_RESULTS.txt.")
    return unit, meaning


def organize_reader_package(out, cfg):
    """Move complete files into numbered folders and write a portable HTML guide."""
    import shutil
    from html import escape
    out = Path(out)
    groups = {
        "01_summary": {f"{x}.csv" for x in SUMMARY_TABLES} | {"dopant_selection.json", "run_summary.txt", "selection_interpretation.txt", "selection_interpretation.json"},
        "03_dataset": {"curated_dataset.csv", "raw_materials.csv", "raw_materials.json", "chemical_coverage.csv", "rejected_records.csv", "host_reference.json"},
        "04_models": {"X_features_and_groups.csv", "y_target.csv", "data_splits.csv", "model_metrics.csv", "model_selection.json",
                      "nested_oof_predictions.csv", "locked_test_predictions.csv", "leave_one_dopant_out_metrics.csv",
                      "leave_one_dopant_out_predictions.csv", "feature_importance_diagnostic.csv", "formation_energy_model.joblib", "formation_energy_bootstrap_draws.npz"},
        "05_screening": {"screening_results.csv", "ml_predictions.csv", "formation_energy_curves.csv", "pareto_fronts.csv", "robustness_results.csv",
                         "mc_scenario_draws.npz", "dopant_score_details.csv", "dopant_scores_all_concentrations.csv", "score_normalization.csv",
                         "score_weight_sensitivity.csv", "score_criterion_sensitivity.csv", "score_scenario_draws.npz", "requested_grid_summary.csv", "dopant_advantages_disadvantages.csv", "selected_dopant_tradeoffs.csv"},
        "06_validation": {"experimental_validation_template.csv", "dft_validation_template.csv", "external_validation_status.json",
                          "experimental_validation.csv", "experimental_log10_MAE_by_study.csv", "dft_energy_validation.csv"},
        "07_methods_and_sources": {"READ_RESULTS.txt", "run_config.json", "references.csv", "literature_reference_ranges.csv", "dopant_evidence_registry.csv", "requirements_resolved.txt", "workflow_source.py"},
        "08_DFT_inputs_and_DOS": {"starting_structure_manifest.csv", "DOS_reference_manifest.csv"},
    }
    for folder in groups: (out / folder).mkdir(exist_ok=True)
    for path in list(out.iterdir()):
        if not path.is_file(): continue
        dest = next((folder for folder, files in groups.items() if path.name in files), "07_methods_and_sources")
        shutil.move(str(path), out / dest / path.name)
    shutil.move(str(out / "figures"), out / "02_figures")
    for folder in ["unrelaxed_DFT_starting_structures", "MP_reference_DOS"]:
        if (out / folder).exists(): shutil.move(str(out / folder), out / "08_DFT_inputs_and_DOS" / folder)
    guide = []
    for path in sorted(out.rglob("*.csv")):
        try: columns = pd.read_csv(path, nrows=0).columns
        except pd.errors.EmptyDataError: columns = []
        for column in columns:
            unit, meaning = column_metadata(column)
            guide.append({"table": path.relative_to(out).as_posix(), "column": column, "unit": unit, "description": meaning})
    csv_out(pd.DataFrame(guide), out / "COLUMN_GUIDE.csv")
    summary = find_result_file(out, "run_summary.txt").read_text(encoding="utf-8")
    guide_text = ("DOPED SnO RESULTS — START HERE\n\n" + summary +
        "\nOpen START_HERE.html in a browser after extracting the complete ZIP.\n"
        "01_summary: counts, GB metrics, independently normalized concentration-score table, selected dopant and interpretation.\n"
        "02_figures: PNG + vector PDF; formation_energy_by_dopant contains each dopant separately.\n"
        "03_dataset: complete filtered and raw Materials Project dataset.\n"
        "04_models: detailed cross-validation/test/transfer results and fitted models.\n"
        "05_screening: all property curves, scores, Pareto fronts and uncertainty draws.\n"
        "06_validation: experimental/DFT input templates and validation status.\n"
        "07_methods_and_sources: full interpretation, equations, assumptions and references.\n"
        "08_DFT_inputs_and_DOS: optional unrelaxed starting structures and reference DOS.\n\n"
        "FILE_INDEX.csv describes exported files; COLUMN_GUIDE.csv gives column meanings and units.\n"
        "All large tables are complete in these folders; they are not printed in Colab.\n"
        "Blank/NaN means missing or ineligible, not zero. Formation energy is a compound-energy proxy.\n"
        "The score is a preference-dependent screening decision at the reported concentration.\n")
    (out / "START_HERE.txt").write_text(guide_text, encoding="utf-8")
    style = """body{font:15px/1.55 system-ui,sans-serif;color:#1b3043;background:#f5f8fb;margin:0}
    main{max-width:1250px;margin:auto;padding:30px}h1,h2{color:#163e5b}h2{margin-top:34px}
    section{background:white;border:1px solid #dde6ef;border-radius:10px;padding:18px;margin:18px 0}
    table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;text-align:left;border-bottom:1px solid #e1e8ee}
    th{background:#edf3f8}tr:nth-child(even){background:#f8fafc}a{color:#125c97}
    pre{white-space:pre-wrap;font:inherit}img{width:100%;height:auto}nav a{margin-right:18px}
    .scroll{overflow-x:auto}.note{border-left:4px solid #c18c31;padding-left:15px}"""
    body = ["<h1>Doped SnO: results and dopant selection</h1>",
            '<nav><a href="#summary">Summaries</a><a href="#figures">Figures</a><a href="#files">All files</a></nav>',
            f'<section><pre>{escape(summary)}</pre></section>',
            '<p class="note">The selected dopant is a screening candidate under the stated weights and physical assumptions. '
            'A score is not experimental confirmation. Read its evidence status before interpreting the ranking.</p>',
            '<h2 id="summary">Compact result tables</h2>']
    interpretation = json.loads(find_result_file(out, "selection_interpretation.json").read_text(encoding="utf-8"))
    body.append('<section>'+selection_interpretation_html(interpretation)+'</section>')
    for stem in SUMMARY_TABLES:
        path = find_result_file(out, f"{stem}.csv")
        frame = pd.read_csv(path)
        body.append(f'<section><h3>{escape(TABLE_TITLES[stem])}</h3><div class="scroll">{compact_table_html(frame)}</div>'
                    f'<p><a href="{path.relative_to(out).as_posix()}">Complete CSV</a></p></section>')
    weights = cfg["score_weights"]
    total = sum(weights.values())
    body.append('<h2>How the score selects a candidate</h2><section><p>At ' + f'{cfg["score_reference_pct"]:g}% sublattice substitution, '
        'each criterion is scaled from 0 (least favorable) to 1 (most favorable) among complete alternatives. '
        'Lower compound formation energy, higher log conductivity and higher mobility are preferred.</p><p>Score = 100 × (' +
        " + ".join(f'{weights[k]/total:g} × U({k.replace("_", " ")})' for k in SCORE_KEYS) + ').</p>'
        '<p>These are editable preference weights. Conductivity includes mobility, so giving mobility its own weight deliberately adds preference to it. '
        'Weight changes, criterion removal and physical uncertainty are reported in 05_screening. The existing five-objective Pareto analysis remains available.</p>'
        '<p>Selection uses the highest score among candidates passing the evidence/applicability gates if any exist; otherwise it reports an exploratory leader. '
        'Exact score ties are retained. Scores normalized at different concentrations are not an absolute scale for concentration optimization.</p>'
        '<p>Method guidance: <a href="https://doi.org/10.1787/9789264043466-en">OECD/JRC composite-indicator handbook</a>. '
        '<a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html">R² definition</a>: '
        '1 − residual sum of squares / total sum of squares. Pooled CV R² is calculated from all outer-fold predictions, not by averaging fold R².</p></section>')
    body.append('<p>The concentration-score line graph and numerical heatmap use independent normalization at each concentration. '
                'Stars and gold outlines identify the highest central scores; final selection still uses the configured reference and screening gates.</p>')
    body.append('<h2 id="figures">Main figures</h2>')
    for path in sorted((out / "02_figures").glob("*.png")):
        rel = path.relative_to(out).as_posix()
        body.append(f'<section><h3>{escape(path.stem.replace("_", " "))}</h3><img loading="lazy" src="{rel}" '
                    f'alt="{escape(path.stem)}"><p><a href="{path.with_suffix(".pdf").relative_to(out).as_posix()}">Vector PDF</a></p></section>')
    body.append('<h2 id="files">Complete data and supporting files</h2><p>No rows are omitted from the exported CSV files.</p>')
    index = []
    for path in sorted(out.rglob("*")):
        if not path.is_file(): continue
        rel = path.relative_to(out).as_posix()
        desc = FILE_DESCRIPTIONS.get(path.name, "Figure: " + path.stem.replace("_", " ") if path.suffix in [".png", ".pdf"]
                                     else "Supporting result; see the methods and parent folder.")
        nrows, ncols = None, None
        if path.suffix == ".csv":
            try:
                frame = pd.read_csv(path); nrows, ncols = frame.shape
            except pd.errors.EmptyDataError: nrows, ncols = 0, 0
        index.append({"relative_path": rel, "description": desc, "rows": nrows, "columns": ncols})
    csv_out(pd.DataFrame(index), out / "FILE_INDEX.csv")
    body.append('<p><a href="FILE_INDEX.csv">File index (CSV)</a> · <a href="COLUMN_GUIDE.csv">Column and unit guide (CSV)</a></p>')
    for folder in [*groups.keys(), "02_figures"]:
        items = [r for r in index if r["relative_path"].startswith(folder+"/")]
        if not items: continue
        body.append(f'<section><h3>{escape(folder)}</h3><ul>')
        for row in items:
            body.append(f'<li><a href="{escape(row["relative_path"], quote=True)}">{escape(Path(row["relative_path"]).name)}</a> '
                        f'— {escape(row["description"])}</li>')
        body.append('</ul></section>')
    html = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    html += '<title>Doped SnO results</title><style>'+style+'</style></head><body><main>'+"\n".join(body)+'</main></body></html>'
    (out / "START_HERE.html").write_text(html, encoding="utf-8")


def display_inline_results(out, cfg, tables=None):
    """Only compact summaries and main figures; complete tables stay in the ZIP."""
    from html import escape
    out = Path(out)
    tables = load_result_tables(out) if tables is None else tables
    if not cfg.get("display_results_inline", True): return tables
    rich = False
    try:
        from IPython import get_ipython
        from IPython.display import display, HTML, Image
        rich = get_ipython() is not None
    except ImportError: pass
    summary = find_result_file(out, "run_summary.txt").read_text(encoding="utf-8")
    if rich: display(HTML('<h2>SnO screening results</h2><pre style="white-space:pre-wrap">'+escape(summary)+'</pre>'))
    else: print("\n"+summary)
    interpretation = json.loads(find_result_file(out, "selection_interpretation.json").read_text(encoding="utf-8"))
    if rich: display(HTML(selection_interpretation_html(interpretation)))
    else: print(find_result_file(out, "selection_interpretation.txt").read_text(encoding="utf-8"))
    if cfg.get("display_tables_inline", True):
        for stem in SUMMARY_TABLES:
            if stem not in tables: continue
            frame = tables[stem]
            if len(frame) > cfg.get("inline_max_table_rows", 12):
                print(f"{TABLE_TITLES[stem]}: {len(frame)} rows; complete table is in the ZIP.")
                continue
            if rich:
                display(HTML('<style>.sno-compact table{border-collapse:collapse;font-size:12px;color:#193047;background:white;}'
                    '.sno-compact td,.sno-compact th{padding:7px 9px;border-bottom:1px solid #e1e8ee;white-space:nowrap}'
                    '.sno-compact th{background:#edf3f8}</style><h3>'+escape(TABLE_TITLES[stem])+ '</h3>'
                    f'<div class="sno-compact" style="max-height:{int(cfg.get("inline_table_height_px",360))}px;overflow:auto">'
                    +compact_table_html(frame)+'</div>'))
            else:
                print("\n"+TABLE_TITLES[stem])
                print(frame.rename(columns=DISPLAY_NAMES).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    if cfg.get("display_figures_inline", True):
        folder = out / "02_figures" if (out / "02_figures").exists() else out / "figures"
        for path in sorted(folder.glob("*.png")):
            if rich: display(Image(filename=str(path), width=int(cfg.get("inline_figure_width_px", 1000))))
            else: print("Figure:", path)
    note = ("Complete tables, individual dopant plots and methods are in the ZIP. Extract it and open START_HERE.html.\n"
            'To inspect a full table without retraining: display(RESULT_TABLES["screening_results"]).')
    if rich: display(HTML('<p>'+escape(note).replace('\n','<br>')+'</p>'))
    else: print(note)
    return tables


def export_workflow_source(out):
    """Include source from a script or Colab cell, stripping the configured key."""
    import re
    source = globals().get("__file__")
    text = Path(source).read_text(encoding="utf-8") if source and Path(source).is_file() else ""
    if not text:
        try:
            from IPython import get_ipython
            shell = get_ipython()
            for cell in reversed(shell.user_ns.get("In", [])):
                if "def dopant_scoring(" in cell and "def main(" in cell:
                    text = cell; break
        except (ImportError, AttributeError): pass
    if text:
        text = re.sub(r'(?m)^API_KEY\s*=.*$', 'API_KEY = ""  # Enter your MP API key before running.', text)
        (Path(out) / "workflow_source.py").write_text(text, encoding="utf-8")


def main(config=None):
    global RESULT_TABLES
    cfg = deepcopy(CONFIG if config is None else config)
    validate_score_config(cfg)
    if cfg["run_size"] == "publication":
        cfg.update(bootstrap_models=max(200, cfg["bootstrap_models"]), mc_draws=max(2000, cfg["mc_draws"]))
    if cfg["bootstrap_models"] < 2 or cfg["mc_draws"] < 2: raise ValueError("At least two resamples are needed.")
    if cfg["temperature_K"] <= 0: raise ValueError("Temperature must be positive.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = Path(cfg["output_parent"]).expanduser().resolve()/f"SnO_Pareto_{timestamp}"
    out.mkdir(parents=True, exist_ok=False)
    print("Results directory:", out)
    write_provenance(out, cfg)
    key = None if cfg["cached_raw_json"] else get_api_key()
    raw = fetch_data(cfg, out, key)
    data, host, structure = curate(raw, out)
    write_dataset_summary(data, raw, out)
    columns = get_feature_columns(data, cfg)
    csv_out(data[["material_id", "formula", "A"]+columns], out / "X_features_and_groups.csv")
    csv_out(data[["material_id", "formation_energy_eV_atom"]], out / "y_target.csv")
    model, chosen, reliable, oof, _ = train_gradient_boosting(data, columns, cfg, out)
    print("Testing transfer to each withheld dopant element...")
    transfer = leave_one_element_out(data, columns, cfg, out)
    print("Generating virtual predictions and group bootstrap ensembles...")
    screen, boot, nsite = prepare_predictions(data, structure, model, chosen, oof, transfer,
                                              reliable, columns, cfg, out)
    pareto = pareto_analysis(screen, cfg, out)
    print("Propagating physical and ML uncertainty into Pareto fronts...")
    robust = robustness(screen, boot, data, oof, nsite, cfg, out)
    score_details, all_scores, selection = dopant_scoring(screen, cfg, out)
    validation = external_validation(screen, nsite, cfg, out)
    interpretation, tradeoffs = write_selection_interpretation(data, screen, score_details, selection, validation, cfg, out)
    write_starting_structures(structure, cfg, out)
    reference_dos(raw, host, cfg, out, key)
    make_figures(screen, pareto, robust, cfg, out)
    make_added_figures(screen, score_details, all_scores, selection, cfg, out)
    plot_selected_tradeoffs(tradeoffs, out)
    write_report(data, screen, robust, chosen, reliable, validation, selection, cfg, out)
    export_workflow_source(out)
    organize_reader_package(out, cfg)
    import shutil
    archive = shutil.make_archive(str(out), "zip", root_dir=out)
    print("Complete results ZIP:", archive)
    RESULT_TABLES = load_result_tables(out)
    try:
        display_inline_results(out, cfg, RESULT_TABLES)
    except Exception as err:
        # Display issues must not prevent access to successfully generated files.
        print(f"Inline display could not finish ({type(err).__name__}: {err}).")
        print("The complete results ZIP is available at:", archive)
    if cfg["auto_download_zip_in_colab"]:
        try:
            from google.colab import files
            files.download(archive)
        except ImportError:
            pass
    return out, Path(archive)


if __name__ == "__main__":
    RESULT_DIRECTORY, RESULT_ZIP = main()
