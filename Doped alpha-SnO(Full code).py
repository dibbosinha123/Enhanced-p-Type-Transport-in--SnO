# SnO: ONE-CELL COLAB RUNNER — conductivity, dopant selection and uncertainty, v5.0
# Paste this ENTIRE file into ONE code cell in a fresh Google Colab notebook.
# The full workflow and isolated package setup are included; the API key below is used automatically.
# Li, Na, K, N and P are retained. All charts are calculated from the current run.
# ML predicts compound formation energy. Conductivity uses the existing physical scenario model.

# ------------------------- USER SETTINGS -------------------------
API_KEY = '4QoUiunPSMRpqOLTwA6qRu8edPSBArZD'  # Materials Project API key; no runtime prompt.
SNO_OPTIONS = {
    "project_directory": "/content/SnO_OneCell_v5",
    "verbose_logs": False,
    "setup_only": False,
    "display_results": True,
    "download_results": True,
}
SNO_SETTINGS = {
    "run_size": "standard",
    "requested_pareto_pct": [1, 2, 3, 5, 10, 15, 20],
    "score_reference_pct": 5.0,
    "sensitivity_concentrations": [3.0, 5.0],
    "cached_raw_json": "",  # Optional raw_materials.json from a previous run.
    "n_jobs": 2,
    "display_results_inline": False,       # Parent notebook handles display.
    "auto_download_zip_in_colab": False,   # Parent notebook handles download.
}
# Scoring weights, physical assumptions and all sampling settings are in CONFIG below.

# Direct API/structure interfaces are pinned. Other dependencies use bounded
# ranges so compatible available builds can be resolved for Python 3.12/3.13.
# The EXACT resolved environment is saved with every successful result.
SNO_REQUIREMENTS = """numpy>=2.1,<2.3
scipy>=1.15,<2
pandas>=2.2.3,<3
scikit-learn>=1.6,<2
matplotlib>=3.10,<4
joblib>=1.4,<2
mp-api==0.46.5
pymatgen==2026.5.4
pymatgen-core>=2026.2.24,<2027
emmet-core>=0.86,<0.89
monty>=2025.1.9,<2027
pydantic>=2.11,<3
requests>=2.32.5,<3
ipython>=8.20,<10
"""

# The COMPLETE workflow is embedded below as readable source, not downloaded.
# A fresh isolated interpreter executes it, avoiding Colab's loaded packages.
SNO_WORKFLOW_SOURCE = r'''


"""Doped alpha-SnO: formation-energy screening, conductivity and uncertainty.
Nominal screening substitutions: 1, 2, 3, 5, 10, 15 and 20% of the designated host sublattice.

Five candidate dopants: Li, Na, K, N and P. No preferred dopant is hardcoded.
This is a screening workflow, NOT a charged-defect DFT or Hall simulator.
ML learns MP compound formation energies (eV/atom).
Literature activation energies are INPUTS.
No experimental results are fabricated or used to force Na to rank first.

Gradient Boosting is the only trained estimator family (FE + sensitivity surrogates). Execution downloads MP data,
performs grouped/nested hyperparameter tuning,
held-out testing, leave-one-element-out transfer tests, group bootstrapping,
transport scenarios, fixed-concentration Pareto analysis, and uncertainty tests.
It writes a timestamped results directory and a ZIP.
Essential results and a short, readable equation guide appear in Colab. Full inputs, methods and source tables stay in the ZIP.
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


def ensure_dependencies():
    """Check imports only. Installation belongs in the isolated setup environment."""
    requirements = {
        "numpy": "numpy>=1.26,<2.3", "scipy": "scipy>=1.13,<2",
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
            print(f"Import failure in {module}:\n" + result.stderr, file=sys.stderr)
    if missing:
        raise RuntimeError("Missing or broken imports: " + ", ".join(missing) +
                           ". Run the supplied isolated-environment setup first. "
                           "This script will not change Colab's system packages.")


if __name__ == "__main__":
    ensure_dependencies()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.constants import elementary_charge, Boltzmann, h, m_e
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.inspection import permutation_importance
import joblib
from pymatgen.core import Composition, Element, Structure
from pymatgen.io.vasp import Poscar
from mp_api.client import MPRester


# ------------------------- EDITABLE CONFIGURATION -------------------------
API_KEY = "4QoUiunPSMRpqOLTwA6qRu8edPSBArZD"  

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
    "score_weights": {"formation_energy": 0.0, "conductivity": 2.0/3.0, "mobility": 1.0/3.0},
    "score_weight_draws": 2000,             # uniform-simplex preference sensitivity
    "score_close_margin_points": 2.0,
    # New modules: settings are written into run_config.json.
    "sensitivity_concentrations": [3.0, 5.0],
    "sensitivity_min_R2": .80,
    "sensitivity_permutation_repeats": 20,

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
    "K":  (.015, .0111, .0204, "10.1063/5.0288742", "reported_effective_activation"),
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
# Simplified mobility: A = alloy_scattering; B = ion_scattering (dimensionless).
# mass_penalty and m_transport_over_me are retained only for the legacy mass
# diagnostic column; neither enters mobility, conductivity or its surrogates.
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
    dict(dopant="K", property="effective_activation_eV", low=.0111, high=.0204,
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


# CSVs are also machine-readable inputs to later calculations and audits.
# Seventeen significant digits preserve IEEE-754 float64 values on a matching
# round-trip read. These extra digits do not imply physical measurement accuracy.
CSV_FLOAT_FORMAT = "%.17g"


def csv_out(frame, path):
    """Preserve computational precision; round only in human-facing displays."""
    frame.to_csv(path, index=False, float_format=CSV_FLOAT_FORMAT)


def get_api_key():
    key = API_KEY.strip() or os.environ.get("MP_API_KEY") or os.environ.get("PMG_MAPI_KEY")
    if not key:
        raise ValueError("Set API_KEY in the code or specify a saved raw-data JSON.")
    return key


def fetch_data(cfg, out, api_key=None):
    if cfg["cached_raw_json"]:
        raw = json.loads(Path(cfg["cached_raw_json"]).read_text())
        if not {"ternaries", "hosts", "query"} <= raw.keys():
            raise ValueError("Cache must be raw_materials.json produced by this script.")
        old = raw["query"]
        if old.get("include_gnome") != cfg["include_gnome"]:
            raise ValueError("Cached include_gnome setting differs from CONFIG.")
        if old.get("download_scope") != "all_non_deprecated_ternaries_and_all_SnO":
            print("WARNING: legacy cache may omit records. Counts cover only the cached retrieval.")
            raw["inventory_warning"] = "legacy_filtered_cache_may_be_incomplete"
            if cfg["mp_hull_max_eV_atom"] > old.get("hull_max", 0):
                raise ValueError("Legacy cache is narrower than the requested training hull range; download fresh.")
    else:
        fields = ["material_id", "formula_pretty", "structure", "symmetry",
                  "formation_energy_per_atom", "energy_above_hull", "band_gap",
                  "is_metal", "theoretical", "deprecated"]
        def serialize(doc):
            obj = {f: getattr(doc, f, None) for f in fields}
            obj["material_id"] = str(obj["material_id"])
            obj["structure"] = obj["structure"].as_dict()
            sym = obj["symmetry"]
            obj["symmetry"] = sym.model_dump(mode="json") if hasattr(sym, "model_dump") else sym
            return obj
        print("Progress: Downloading Materials Project bulk references...")
        with MPRester(api_key or get_api_key(), timeout=60) as mpr:
            # Preserve raw entries first; apply the ML training cutoff locally.
            ternaries = mpr.materials.summary.search(
                elements=["Sn", "O"], num_elements=3, deprecated=False,
                include_gnome=cfg["include_gnome"], fields=fields)
            hosts = mpr.materials.summary.search(formula="SnO", deprecated=False,
                                                 include_gnome=False, fields=fields)
            raw = {"ternaries": [serialize(d) for d in ternaries],
                   "hosts": [serialize(d) for d in hosts],
                   "query": {"hull_max": cfg["mp_hull_max_eV_atom"],
                             "include_gnome": cfg["include_gnome"],
                             "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                             "database_version": mpr.get_database_version(),
                             "download_scope": "all_non_deprecated_ternaries_and_all_SnO",
                             "training_hull_cutoff_applied_locally": True}}
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


def curate(raw, out, cfg):
    hosts = []
    for d in raw["hosts"]:
        s = Structure.from_dict(d["structure"])
        sym = d.get("symmetry") or {}
        if (s.is_ordered and s.composition.reduced_formula == "SnO" and sym.get("number") == 129
                and d.get("is_metal") is False
                and numeric(d.get("band_gap")) > 0
                and np.isfinite(numeric(d.get("formation_energy_per_atom")))
                and np.isfinite(numeric(d.get("energy_above_hull")))
                and d.get("deprecated") is not True):
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
            hull = numeric(d.get("energy_above_hull"))
            if not np.isfinite(hull) or hull < -1e-7 or hull > cfg["mp_hull_max_eV_atom"]:
                raise ValueError("outside bulk training hull cutoff or missing hull energy")
            if d.get("deprecated") is True: raise ValueError("deprecated")
            a = others[0] if others else None
            if a in EXCLUDED_ELEMENTS: raise ValueError("element explicitly excluded from this workflow")
            row = {"material_id": mid, "formula": comp.reduced_formula, "A": a or "HOST",
                   "formation_energy_eV_atom": y, "hull_eV_atom": numeric(d.get("energy_above_hull")),
                   "band_gap_eV_audit_only": numeric(d.get("band_gap")),
                   "nsites_audit_only": len(s), "source": "Materials_Project",
                   "learning_domain_audit_only": "compound_formation_energy"}
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
    # Explicit allow-list prevents provenance metadata from becoming features.
    core = {"Sn_fraction", "O_fraction", "A_fraction", "entropy"}
    return [c for c in data if (c in core or c.startswith(("mean_", "std_", "A_weighted_"))
            or (cfg["use_structure_features"] and c.startswith("struct_")))]


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
    # Preserve the legacy mass diagnostic without using it in mobility.
    mass_ratio = 1.+x*(params["mass_penalty"]*penalty + .1*dc)
    # A = alloy_scattering; B = ion_scattering; eta = active dopant fraction.
    # mu = mu0 / [1 + A*x*(1-x)*(dr^2 + 0.15*dc^2) + B*x*eta].
    retention = 1./(1.+params["alloy_scattering"]*x*(1.-x)*(penalty**2+.15*dc**2)
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
    # Keep the legacy mass-diagnostic draw to preserve the random-number sequence
    # for all other inputs; mass_penalty no longer affects transport.
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
    modes, sampled_inputs = [], []
    for draw in range(n_draws):
        pars = sample_parameters(rng)
        mode = "shared_efficiency" if draw < n_draws//2 else "dopant_specific_efficiency"
        modes.append(mode)
        effective = {d: (pars["active_fraction"] if mode == "shared_efficiency" else 10**rng.uniform(-2., 0.)) for d in DOPANTS}
        ea = {d: (rng.uniform(EA[d][1], EA[d][2]) if np.isfinite(EA[d][0]) else np.nan) for d in DOPANTS}
        sampled_inputs.append(dict(draw_id=draw, efficiency_mode=mode, **pars,
            **{f"efficiency_{d}": effective[d] for d in DOPANTS},
            **{f"EA_{d}_eV": ea[d] for d in DOPANTS}))
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
    csv_out(pd.DataFrame(sampled_inputs), out / "mc_input_draws.csv")
    np.savez_compressed(out / "mc_scenario_draws.npz", ranks=rank_draws, compromise_shares=share_draws,
                        supported_ranks=support_draws, no_radius_ranks=no_radius_draws,
                        p_cm3=p_draws, mu_cm2_Vs=mu_draws, sigma_S_m=sigma_draws,
                        FE_eV_atom=fe_draws,
                        dopants=base.dopant.to_numpy(str), substitution_pct=base.substitution_pct.to_numpy(),
                        efficiency_mode=np.array(modes))
    return result


# ------------------ DATA COUNTS AND DOPANT PREFERENCE SCORE -----------------
SCORE_KEYS = ["formation_energy", "conductivity", "mobility"]  # FE retained in exported schema only; weight = 0, so no effect on scoring.


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
    if not np.isclose(w[0], 0.0, rtol=0, atol=1e-12):
        raise ValueError("Formation energy must have zero preference-score weight; only conductivity and mobility are scored.")
    if w[1] <= 0 or w[2] <= 0:
        raise ValueError("Use positive preference-score weights for both conductivity and mobility.")
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
    print(f"Bulk ML dataset: {len(data)} structures / {data.formula.nunique()} unique compositions.")
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
    # The FE column is retained for backward-compatible exports; the zero FE
    # weight makes selection depend ONLY on log conductivity and mobility.
    # Donor controls stay outside p-type selection.
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

    Default score = 100*((2/3)*U_log_sigma + (1/3)*U_mu).
    The formation-energy term is zero-weighted and excluded from selection.
    Removing a candidate changes min-max anchors; scores must be recomputed for
    the five-dopant candidate set rather than copied from an earlier run.
    Units: eV/atom, S/m and cm^2/(V s). Utilities run from 0 (worst) to 1
    (best) among complete alternatives at the SAME sublattice concentration.
    The highest score with valid transport quantities is selected; energetics
    remain separate evidence diagnostics, not preference criteria.
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
    # Evidence/applicability gates are reported but must not filter the two-property score
    # (those gates include formation-energy prediction quality and coverage).
    selection_pool = valid
    shares = winner_shares(score, selection_pool)
    selected = [d for d, share in zip(DOPANTS, shares) if share > 0]
    ref["rank"] = pd.Series(score).rank(method="min", ascending=False).to_numpy()
    ref["selected_at_reference"] = shares > 0
    ref["selection_pool"] = selection_pool
    best = np.nanmax(score[selection_pool]) if selection_pool.any() else np.nan
    ref["within_close_score_margin"] = selection_pool & (best-score <= cfg["score_close_margin_points"])
    # Uniform-simplex weights explore preference choices, not their probabilities.
    rng = np.random.default_rng(cfg["seed"]+303)
    transport_weights = rng.dirichlet(np.ones(2), size=int(cfg["score_weight_draws"]))
    weight_vectors = np.column_stack([np.zeros(len(transport_weights)), transport_weights])
    all_criteria_valid = np.isfinite(costs[:, 1:]).all(axis=1)
    sweep_u, _, sweep_valid, _ = normalized_desirability(costs, np.array([0., .5, .5]))
    sweep_scores = 100.*(np.nan_to_num(sweep_u, nan=0.) @ weight_vectors.T).T
    sweep_scores[:, ~sweep_valid] = np.nan
    weight_wins = np.asarray([winner_shares(v, selection_pool & all_criteria_valid) for v in sweep_scores])
    ref["weight_sweep_top_share"] = np.where(sweep_valid & selection_pool, weight_wins.mean(axis=0), np.nan)
    pref = pd.DataFrame(weight_vectors, columns=[f"weight_{k}" for k in SCORE_KEYS])
    pref["top_dopants"] = [";".join(d for d, s in zip(DOPANTS, v) if s > 0) for v in weight_wins]
    csv_out(pref, out / "score_weight_sensitivity.csv")
    # Leave-one-criterion-out and equal-weight alternatives expose correlation
    # between conductivity and mobility and sensitivity to the energy proxy.
    alternatives = {"configured": w, "equal_weights": np.array([0., .5, .5])}
    for k, name in enumerate(SCORE_KEYS):
        if k == 0: continue  # Already excluded: no duplicate without_energy analysis.
        alt = w.copy(); alt[k] = 0.
        if alt.sum() > 0: alternatives[f"without_{name}"] = alt/alt.sum()
    ablations = []
    for label, weights in alternatives.items():
        _, vals, _, _ = normalized_desirability(costs, weights, anchors=anchors)
        tops = winner_shares(vals, selection_pool)
        for d, value, top in zip(DOPANTS, vals, tops):
            ablations.append({"variant": label, "dopant": d, "score_0_100": value,
                              "top_share": top, **dict(zip([f"weight_{k}" for k in SCORE_KEYS], weights))})
    csv_out(pd.DataFrame(ablations), out / "score_criterion_sensitivity.csv")
    # Reuse the SAME saved draws used by Pareto analysis; FE is zero-weighted and has no score effect.
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
        "selection_status": ("conditional_supported_screening" if len(selected) and all(d in set(ref.loc[ref.supported_screening, "dopant"]) for d in selected) else
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
    selection_rule = ("The code compares candidates at the same concentration. It scales higher log conductivity "
        "and higher mobility to desirabilities between 0 and 1. Formation energy is not scored. Score = 100 times their weighted sum. "
        "Configured weights: " + ", ".join(f"{k.replace('_',' ')} {100*v:g}%" for k,v in selection["weights"].items()) + ". " +
        "The score selects among candidates with valid transport values; ML energy evidence is reported separately." +
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
        parts = [f"{100*selection['weights'][k]:g}% weight gives {s[f'points_{k}']:.2f} {k.replace('_',' ')} points" for k in SCORE_KEYS if selection["weights"][k] > 0]
        why = f"{d} obtains {s.score_0_100:.2f}/100: " + "; ".join(parts) + "."
        tradeoffs = []
        if rival is not None:
            r = ranked.loc[rival]
            why += f" The highest-scoring alternative is {rival} ({r.score_0_100:.2f}/100); the score difference is {s.score_0_100-r.score_0_100:+.2f} points."
            if outside_pool:
                why += " That alternative is outside the gate-passing selection pool and is shown only for comparison."
            for key, column, label, unit, lower in properties:
                if selection["weights"][key] <= 0: continue
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
            f"{number(100*s.scenario_top_share,3)}% in the sampled transport scenarios. "
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
    lines = ["DOPANT SELECTION", decision, "Status: " + selection["selection_status"], selection_rule]
    for item in explanation:
        lines += ["", item["why_selected"], *item["tradeoffs"], item["selection_sensitivity"],
                  "Failed gates: " + ("; ".join(item["failed_screening_checks"]) or "none")]
    lines += ["", "All-dopant comparisons: dopant_advantages_disadvantages.csv.",
              "Physical model, assumptions and sources: CALCULATION_REFERENCE.txt.",
              "Experimental comparison: " + str(validation["experimental_validation"]) + "."]
    (out / "selection_interpretation.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report, pd.DataFrame(deltas, columns=comparison_columns)


def selection_interpretation_html(report):
    """Essential decision evidence; complete comparative tables remain exported."""
    from html import escape
    parts = ['<p>' + escape(report["decision"]) + '</p>']
    for item in report["candidates"]:
        parts.append('<h3>' + escape(item["dopant"]) + '</h3><p>' + escape(item["why_selected"]) + '</p>')
        parts.append('<ul>' + ''.join('<li>' + escape(t) + '</li>' for t in item["tradeoffs"]) + '</ul>')
        parts.append('<p>Failed gates: ' + escape('; '.join(item["failed_screening_checks"]) or 'none') + '.</p>')
        parts.append('<p>' + escape(item["selection_sensitivity"]) + '</p>')
    parts.append('<p>Full comparisons: dopant_advantages_disadvantages.csv and selected_dopant_tradeoffs.csv.</p>')
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
                        "total_atom_pct": 100./len(s), "status": "unrelaxed_starting_geometry_no_DFT_run", "file": name,
                        "new_dft_calculation_performed": False, "relaxed_by_this_workflow": False,
                        "generation_method": "atomic_substitution_in_host_supercell"})
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
    """Export every original figure as high-resolution PNG and vector PDF.

    Uniformly increase legibility and show complete panel borders. This only
    changes presentation; original underlying data and uncertainties are not
    changed. Axes deliberately switched off (workflow, unused panels) stay off.
    """
    for ax in fig.axes:
        if not ax.axison:
            continue
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(max(spine.get_linewidth(), 1.15))
        ax.tick_params(axis="both", which="major", labelsize=11, width=1.0)
        ax.tick_params(axis="both", which="minor", width=0.8)
        ax.xaxis.label.set_fontsize(max(12.0, ax.xaxis.label.get_fontsize()))
        ax.yaxis.label.set_fontsize(max(12.0, ax.yaxis.label.get_fontsize()))
        ax.title.set_fontsize(max(13.0, ax.title.get_fontsize()))
        for label in ax.texts:
            label.set_fontsize(max(10.0, label.get_fontsize()))
        legend = ax.get_legend()
        if legend is not None:
            for t in legend.get_texts():
                t.set_fontsize(max(10.5, t.get_fontsize()))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path) + ".png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(str(path) + ".pdf", bbox_inches="tight", facecolor="white")
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
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.labelsize": 12, "axes.titlesize": 14,
                         "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "legend.fontsize": 11,
                         "figure.titlesize": 17,
                         "axes.spines.top": True, "axes.spines.right": True,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    workflow_figure(folder)
    metrics = pd.read_csv(out / "model_metrics.csv")
    pooled = pd.read_csv(out / "nested_oof_predictions.csv")
    held = pd.read_csv(out / "locked_test_predictions.csv")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), layout="constrained")
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
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), layout="constrained")
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
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), layout="constrained")
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
           title=f"Exploratory screening at {pct:g}% substitution\nTwo-objective projection of five objectives")
    ax.margins(x=.13, y=.12)
    fig.text(.13, .025, "Circles: first front in five objectives. Crosses: dominated. Open diamonds: incomplete objectives.\nProjection alone cannot establish dominance.",
             fontsize=8, va="bottom")
    save_figure(fig, folder / "06_Pareto_projection")
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0), layout="constrained")
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
    weight_note = (f"Weights: log conductivity {100*weights[1]:.1f}%, "
                   f"mobility {100*weights[2]:.1f}% (formation energy excluded).")
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
        fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.1))
        fig.subplots_adjust(left=.075, right=.985, bottom=.11, top=.91, wspace=.42, hspace=.39)
        for ax, d in zip(axes.ravel(), DOPANTS): fe_axis(ax, d, limit)
        for ax in axes.ravel()[len(DOPANTS):]: ax.set_axis_off()
        fig.suptitle(title, fontsize=18)
        fig.text(.075, .025,
                 "Shading: 5th–95th percentile bootstrap spread; estimates are compound, not defect, energies.",
                 fontsize=10)
        save_figure(fig, folder / name)
    for d in DOPANTS:
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), layout="constrained")
        fe_axis(axes[0], d, maximum); fe_axis(axes[1], d, cfg["dilute_screening_limit_pct"])
        fig.suptitle(f"{d}-doped SnO: compound-energy proxy and 5–95% bootstrap spread", fontsize=12)
        save_figure(fig, individual / f"FE_{d}_vs_concentration")
    # Two stacked components only: formation energy contributes zero score points.
    a = score_details.copy()
    fig, ax = plt.subplots(figsize=(10.4, 5.9))
    fig.subplots_adjust(left=.10, right=.94, bottom=.22, top=.85)
    left = np.zeros(len(a))
    for name, color, label in [("conductivity", "#029E73", "log conductivity"),
                               ("mobility", "#DE8F05", "Mobility")]:
        points = a[f"points_{name}"].fillna(0).to_numpy()
        ax.barh(a.dopant, points, left=left, color=color, label=label, height=.65, edgecolor="#333333", linewidth=.75)
        left += points
    for i, row in enumerate(a.itertuples()):
        label = (f"{row.score_0_100:.1f}" + ("  SELECTED" if row.selected_at_reference else "")
                 if np.isfinite(row.score_0_100) else "N/A — not scored")
        ax.text(left[i]+1., i, label, va="center", fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 119)
    ax.set(xticks=[0, 20, 40, 60, 80, 100], xlabel="Dopant preference score (0–100; higher is preferred)",
           title=f"Dopant selection at {cfg['score_reference_pct']:g}% sublattice substitution\n{selection['selection_status'].replace('_', ' ')}")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncols=2, frameon=False)
    # Keep the active weighting visible without duplicating methodology.
    fig.text(.10, .025, "Weights: " + ", ".join(
        f"{k.replace('_', ' ')} {100*v:.1f}%" for k, v in selection["weights"].items() if v > 0)
        + "; formation energy excluded. Exploratory score.", fontsize=10)
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
                                ["Preference-weight sensitivity", "Physical-scenario sensitivity"]):
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
    references["verification_status"] = "inherited_from_user_base_code_not_independently_reverified"
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
    notes = """CALCULATION GUIDE
Equations, logic and limitations: CALCULATION_REFERENCE.txt / calculation_basis.csv.
Active values and bounds: calculation_parameters.csv; complete settings: run_config.json.
Sources and input roles: references.csv and dopant_evidence_registry.csv.
Selection reasons: selection_interpretation.txt; numerical alternatives: selected_dopant_tradeoffs.csv.
All distributions, result tables, figures, model files and scenario arrays remain in the ZIP.
FILE_INDEX.csv locates every output; COLUMN_GUIDE.csv defines columns and units.
Bulk compound FE is learned; transport is a parameter scenario; generated structures are unrelaxed.
Missing values remain NaN. Charged-defect energies and MOSFET switching are not predicted.
"""
    (out / "READ_RESULTS.txt").write_text(message+"\n"+notes, encoding="utf-8")
    (out / "run_summary.txt").write_text(message +
        "Units: formation energy = eV/atom; conductivity = S/m; mobility = cm^2/(V s).\n"
        "Selection and transport retain their stated evidence/scenario status.\n", encoding="utf-8")
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
    "bulk_conductivity_all_dopants.csv": "Bulk conductivity and mobility source data for combined and individual transport figures.",
    "NaK_conductivity_ratio_vs_concentration.csv": "Na/K central conductivity ratios and their hole-density/mobility decomposition; R=1 boundary.",
    "transport_physics_audit.json": "Numerical-check counts and explicit DFT calculation status.",
    "transport_physics_checks.csv": "Row-by-row transport unit and arithmetic checks; not experimental or DFT validation.",
    "TRANSPORT_PHYSICS_AND_DFT_STATUS.txt": "Generated-geometry provenance, conductivity units and limits of quantitative interpretation.",
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


FILE_DESCRIPTIONS.update({
    "property_uncertainty_at_reference.csv": "Property medians and 5–95% scenario ranges by dopant and efficiency mode, with usable-draw counts.",
    "score_uncertainty_at_reference.csv": "Score medians and 5–95% scenario ranges using fixed central normalization anchors.",
    "calculation_basis.csv": "Equations, decisions, rationale, limits and source/implementation mapping.",
    "calculation_parameters.csv": "Active numerical inputs and bounds, units and source roles.",
    "CALCULATION_REFERENCE.txt": "Readable five-equation quick guide, key active inputs and source links.",
    "READ_RESULTS.txt": "Outcome and links to calculation logic, inputs, units and complete tables.",
})


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
        "10_sensitivity": set(),
    }
    groups["01_summary"].update({"simple_material_counts.csv","main_dopant_results.csv","compact_summary.json",
        "structure_counts.csv","ml_sample_counts.csv","sample_count_summary.json",
        "COUNTS_EXPLAINED.txt","SENSITIVITY_EXPLAINED.txt"})
    groups["05_screening"].update({"bulk_formation_energy_all_dopants.csv", "bulk_conductivity_all_dopants.csv",
                                     "NaK_conductivity_ratio_vs_concentration.csv"})
    groups["01_summary"].update({"transport_physics_audit.json"})
    groups["06_validation"].add("transport_physics_checks.csv")
    groups["07_methods_and_sources"].update({"TRANSPORT_PHYSICS_AND_DFT_STATUS.txt"})
    groups["10_sensitivity"].add("final_selection_sensitivity.csv")
    groups["05_screening"].update({"mc_input_draws.csv", "property_uncertainty_at_reference.csv", "score_uncertainty_at_reference.csv"})
    groups["07_methods_and_sources"].add("SENSITIVITY_METHODS.txt")
    groups["07_methods_and_sources"].update({"calculation_basis.csv", "calculation_parameters.csv", "CALCULATION_REFERENCE.txt"})
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
    (out / "START_HERE.txt").write_text(
        "SnO RESULTS\n" + summary +
        "\nOpen START_HERE.html for essential results; expand the short equation guide for key inputs and sources.\n"
        "FULL_REPORT.html retains every main figure and links all files.\n"
        "FILE_INDEX.csv lists complete tables/arrays; COLUMN_GUIDE.csv defines columns and units.\n"
        "Calculation reference: 07_methods_and_sources/CALCULATION_REFERENCE.txt.\n",
        encoding="utf-8")
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


# =================== INTEGRATED Na/K SENSITIVITY ===========================
def pair_summary(a, b, atol=1e-10):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not a.size:
        raise ValueError("Pairwise arrays must have matching nonempty shapes.")
    valid = np.isfinite(a)&np.isfinite(b)
    diff = a-b
    tie = valid & np.isclose(diff, 0., atol=atol, rtol=0)
    return dict(total_draws=len(diff), paired_valid_draws=int(valid.sum()),
        excluded_draws=int((~valid).sum()),
        Na_win_fraction_all_draws=float((valid&(diff>0)&~tie).mean()),
        K_win_fraction_all_draws=float((valid&(diff<0)&~tie).mean()),
        tie_fraction_all_draws=float(tie.mean()))


def sensitivity_variants(cfg):
    full = validate_score_config(cfg)
    variants = {"Full score": full, "Conductivity only": np.array([0.,1.,0.]),
                "Mobility only": np.array([0.,0.,1.])}
    for name, remove in [("Without extra mobility", 2)]:
        w = full.copy()
        w[remove] = 0
        if w.sum() > 0:
            variants[name] = w/w.sum()
    return variants


def run_integrated_sensitivity(screen, saved, cfg, out):
    """The attached sensitivity analysis integrated without re-upload or hardcoded weights."""
    folder = out / "10_sensitivity"
    folder.mkdir()
    variants = sensitivity_variants(cfg)
    central, points, preferences, anchors_rows, contexts = [], [], [], [], {}
    rng = np.random.default_rng(cfg["seed"]+505)
    for pct in cfg["sensitivity_concentrations"]:
        frame = screen[np.isclose(screen.substitution_pct, pct, rtol=0, atol=1e-9)].copy().reset_index(drop=True)
        if frame.empty or frame.dopant.duplicated().any() or not {"Na","K"} <= set(frame.dopant):
            raise ValueError(f"Sensitivity requires unique Na/K rows at {pct}%.")
        ids = {d: int(np.flatnonzero(frame.dopant.to_numpy()==d)[0]) for d in ["Na","K"]}
        costs = score_costs(frame)
        utility, full_scores, _, anchors = normalized_desirability(costs, variants["Full score"])
        contexts[pct] = frame, ids, anchors
        for d in ids:
            old = saved[(saved.dopant==d)&np.isclose(saved.substitution_pct, pct)]
            if len(old) != 1 or not np.isclose(full_scores[ids[d]], old.score_0_100.iloc[0],
                                               atol=1e-7, rtol=0, equal_nan=True):
                raise ValueError("Integrated sensitivity full score does not reproduce base score.")
        for j, criterion in enumerate(SCORE_KEYS):
            n = 100*utility[ids["Na"],j]*variants["Full score"][j]
            k = 100*utility[ids["K"],j]*variants["Full score"][j]
            points.append(dict(substitution_pct=pct, criterion=criterion, Na_points=n,
                               K_points=k, K_minus_Na_points=k-n))
            anchors_rows.append(dict(substitution_pct=pct, criterion=criterion,
                                    cost_min=anchors[0][j], cost_max=anchors[1][j]))
        for name, weights in variants.items():
            _, values, _, _ = normalized_desirability(costs, weights, anchors=anchors)
            na, k = values[ids["Na"]], values[ids["K"]]
            leader = ("Invalid" if not np.isfinite([na,k]).all() else "Tie"
                      if np.isclose(na,k,atol=1e-10,rtol=0) else "Na" if na>k else "K")
            central.append(dict(substitution_pct=pct, variant=name, Na_score=na, K_score=k,
                                K_minus_Na=k-na, pairwise_leader=leader))
        for name in ["Full score"]:
            transport_w = rng.dirichlet(np.ones(2), size=cfg["score_weight_draws"])
            w = np.column_stack([np.zeros(len(transport_w)), transport_w])
            values = np.array([normalized_desirability(costs, one, anchors=anchors)[1] for one in w])
            preferences.append(dict(substitution_pct=pct, variant=name,
                **pair_summary(values[:,ids["Na"]],values[:,ids["K"]])))
    central, points = pd.DataFrame(central), pd.DataFrame(points)
    csv_out(central, folder/"central_score_ablations.csv")
    csv_out(points, folder/"score_contribution_comparison.csv")
    csv_out(pd.DataFrame(preferences), folder/"pairwise_weight_sensitivity.csv")
    csv_out(pd.DataFrame(anchors_rows), folder/"central_score_anchors.csv")
    fig, axes = plt.subplots(1, len(contexts), figsize=(6*len(contexts),4.5), squeeze=False, layout="constrained")
    for ax, pct in zip(axes[0], contexts):
        p = points[points.substitution_pct==pct]
        ax.barh(p.criterion, p.K_minus_Na_points,
                color=["#4775a3" if v>=0 else "#b26741" for v in p.K_minus_Na_points])
        ax.axvline(0,color="black",lw=1)
        ax.set(title=f"{pct:g}% substitution",xlabel="K minus Na contribution (points)")
        for spine in ax.spines.values():
            spine.set_visible(True)
    save_figure(fig, out/"figures"/"17_NaK_score_contribution")
    inputs = pd.read_csv(out/"mc_input_draws.csv")
    metrics, corrs, importance, heldout, uncertainty, targets_all = [], [], [], [], [], []
    with np.load(out/"mc_scenario_draws.npz", allow_pickle=False) as mc:
        modes = mc["efficiency_mode"].astype(str)
        n = len(modes)
        if not np.array_equal(inputs.draw_id, np.arange(n)) or not np.array_equal(inputs.efficiency_mode, modes):
            raise ValueError("Monte Carlo input rows and output draws are misaligned.")
        for pct, (frame, ids, anchors) in contexts.items():
            indexes = []
            for d in frame.dopant:
                matches = np.flatnonzero((mc["dopants"]==d)&np.isclose(mc["substitution_pct"],pct,rtol=0,atol=1e-9))
                if len(matches)!=1:
                    raise ValueError("Missing/duplicate Monte Carlo composition.")
                indexes.append(int(matches[0]))
            fe, sigma, mu = mc["FE_eV_atom"][:,indexes], mc["sigma_S_m"][:,indexes], mc["mu_cm2_Vs"][:,indexes]
            scores = {v: [] for v in variants}
            for e,s,m in zip(fe,sigma,mu):
                with np.errstate(divide="ignore",invalid="ignore"):
                    costs = np.column_stack([e,-np.log10(s),-m])
                for v,w in variants.items():
                    scores[v].append(normalized_desirability(costs,w,anchors=anchors)[1])
            scores = {v:np.asarray(a) for v,a in scores.items()}
            na,k = ids["Na"],ids["K"]
            targets = pd.DataFrame(dict(draw_id=np.arange(n), efficiency_mode=modes,
                                       substitution_pct=pct, delta_sigma_S_m=sigma[:,na]-sigma[:,k]))
            with np.errstate(divide="ignore",invalid="ignore"):
                targets["log10_sigma_ratio_Na_K"] = np.log10(sigma[:,na])-np.log10(sigma[:,k])
            targets["delta_full_score"] = scores["Full score"][:,na]-scores["Full score"][:,k]
            if "Without energy" in scores:
                targets["delta_no_energy_score"] = scores["Without energy"][:,na]-scores["Without energy"][:,k]
            targets_all.append(targets)
            for mode in sorted(set(modes)):
                mask = modes==mode
                for variant,a in scores.items():
                    uncertainty.append(dict(substitution_pct=pct,efficiency_mode=mode,variant=variant,
                                            **pair_summary(a[mask,na],a[mask,k])))
                # The removed mass penalty is not an input to simplified mobility.
                X = inputs.loc[mask,["p0_cm3","mu0_cm2_Vs","m_dos_over_me","compensation_fraction",
                    "alloy_scattering","ion_scattering","EA_Na_eV","EA_K_eV"]].copy()
                if mode=="shared_efficiency":
                    if not np.allclose(inputs.loc[mask,"efficiency_Na"],inputs.loc[mask,"efficiency_K"]):
                        raise ValueError("Shared efficiencies differ.")
                    X["eta_shared"] = inputs.loc[mask,"efficiency_Na"]
                elif mode=="dopant_specific_efficiency":
                    X["eta_Na"],X["eta_K"] = inputs.loc[mask,"efficiency_Na"],inputs.loc[mask,"efficiency_K"]
                else:
                    raise ValueError("Unknown efficiency mode.")
                for c in ["p0_cm3"]+[c for c in X if c.startswith("eta_")]:
                    if not (X[c]>0).all():
                        raise ValueError("Nonpositive logarithmic predictor.")
                    X["log10_"+c] = np.log10(X.pop(c))
                for target_name in [c for c in targets if c.startswith(("delta_","log10_"))]:
                    predictors = X.copy()
                    y = targets.loc[mask,target_name]
                    valid = np.isfinite(predictors).all(axis=1)&np.isfinite(y)
                    xv,yv = predictors.loc[valid],y.loc[valid]
                    xv = xv.loc[:,xv.nunique()>1]
                    if len(yv)<100 or np.std(yv)<1e-12 or xv.empty:
                        metrics.append(dict(substitution_pct=pct, efficiency_mode=mode,target=target_name,
                                            status="skipped_insufficient_or_constant",importance_interpretable=False))
                        continue
                    tr,te = train_test_split(np.arange(len(yv)),test_size=.30,random_state=cfg["seed"])
                    model = GradientBoostingRegressor(n_estimators=250,learning_rate=.05,
                        max_depth=3,min_samples_leaf=5,random_state=cfg["seed"])
                    model.fit(xv.iloc[tr],yv.iloc[tr])
                    pred = model.predict(xv.iloc[te])
                    r2,mae = r2_score(yv.iloc[te],pred),mean_absolute_error(yv.iloc[te],pred)
                    baseline = mean_absolute_error(yv.iloc[te],np.full(len(te),np.median(yv.iloc[tr])))
                    accepted = bool(r2>=cfg["sensitivity_min_R2"] and mae<baseline)
                    metrics.append(dict(substitution_pct=pct,efficiency_mode=mode,target=target_name,status="evaluated",
                        n_train=len(tr),n_test=len(te),holdout_R2=r2,holdout_MAE=mae,
                        median_baseline_MAE=baseline,importance_interpretable=accepted))
                    heldout.extend(dict(substitution_pct=pct,efficiency_mode=mode,target=target_name,
                        draw_id=int(i),actual=a,predicted=b) for i,a,b in zip(yv.iloc[te].index,yv.iloc[te],pred))
                    for c in xv:
                        corrs.append(dict(substitution_pct=pct,efficiency_mode=mode,target=target_name,
                            feature=c,spearman_rho=spearmanr(xv[c],yv).statistic))
                    pi = permutation_importance(model,xv.iloc[te],yv.iloc[te],scoring="r2",
                        n_repeats=cfg["sensitivity_permutation_repeats"],random_state=cfg["seed"],n_jobs=1)
                    for j,c in enumerate(xv):
                        importance.append(dict(substitution_pct=pct,efficiency_mode=mode,target=target_name,
                            feature=c,mean_R2_decrease=pi.importances_mean[j],shuffle_SD=pi.importances_std[j],
                            importance_interpretable=accepted))
    csv_out(pd.DataFrame(uncertainty),folder/"pairwise_MC_score_sensitivity.csv")
    csv_out(pd.concat(targets_all,ignore_index=True),folder/"NaK_MC_targets.csv")
    for name, records in [("surrogate_holdout_metrics",metrics),("spearman_input_associations",corrs),
                          ("heldout_permutation_importance",importance),("surrogate_holdout_predictions",heldout)]:
        csv_out(pd.DataFrame(records),folder/f"{name}.csv")
    (folder/"READ_INTERPRETATION.txt").write_text(
        "K-minus-Na contributions favor K when positive; ML difference targets favor Na when positive.\n"
        "Na/K pairwise comparisons are not overall wins over all five dopants.\n"
        "Each concentration has independent anchors; scores do not establish an optimum concentration.\n"
        "All ablations and Monte Carlo scores retain the central full-score anchors.\n"
        "Uniform simplex weights describe preferences, not experimental probabilities.\n"
        "Invalid pairs stay in the denominator; reported win fractions may sum to less than one.\n"
        "Surrogates explain simulated outputs, not measured material physics.\n"
        "Interpret permutation importance only if importance_interpretable=True; shuffle SD is not a confidence interval.\n"
        "Ranges describe the declared parameter scenarios, not calibrated confidence bounds.\n",
        encoding="utf-8")
    print("Integrated Na/K sensitivity finished; no result re-upload required.")
    return central


def validate_extensions(cfg):
    """Validate the retained concentration and sensitivity settings."""
    grid = np.asarray(cfg["requested_pareto_pct"], float)
    points = np.asarray(cfg["sensitivity_concentrations"], float)
    if (points.ndim != 1 or not len(points) or not np.isfinite(points).all()
            or len(np.unique(points)) != len(points)):
        raise ValueError("Sensitivity concentrations must be unique finite values.")
    if any(not np.any(np.isclose(grid, v, rtol=0, atol=1e-9)) for v in points):
        raise ValueError("Every sensitivity concentration must occur in requested_pareto_pct.")
    if not np.isfinite(cfg["sensitivity_min_R2"]):
        raise ValueError("sensitivity_min_R2 must be finite.")
    if int(cfg["sensitivity_permutation_repeats"]) < 1:
        raise ValueError("At least one permutation repeat is required.")


def write_extension_notes(out, cfg):
    text = (
        "SnO SENSITIVITY AND UNCERTAINTY\n"
        "Shared and dopant-specific activation-efficiency modes are reported separately.\n"
        "All-dopant selections and Na/K pairwise comparisons answer different questions.\n"
        "Property plots reuse the paired Monte Carlo samples from mc_scenario_draws.npz.\n"
        "Score intervals reuse score_scenario_draws.npz, with the central normalization anchors fixed.\n"
        "Intervals span the 5th to 95th percentile of finite, usable draws.\n"
        "Figure 07 retains the original concentration-dependent conductivity bands, pooling both modes.\n"
        "Figures 24 and 25 show the reference concentration with efficiency modes separated.\n"
        "Central estimates are plotted separately and need not equal the scenario median.\n"
        "Valid-draw counts and fractions are exported; unavailable intervals remain missing.\n"
        "These are sensitivity ranges under declared assumptions, not calibrated confidence intervals.\n"
        "Full equations, sampling rules and inputs: calculation_basis.csv and calculation_parameters.csv.\n"
    )
    (out / "SENSITIVITY_METHODS.txt").write_text(text, encoding="utf-8")


def selection_sensitivity_all_dopants(screen, cfg, out):
    """All-five-dopant selection robustness, with efficiency assumptions separate.

    Keep the central eligible pool and anchors fixed. Invalid draws remain in
    the denominator; tied top candidates share one win. No success probability
    or calibrated uncertainty interval is implied.
    """
    pct=float(cfg["score_reference_pct"])
    rows=[]
    ref=pd.read_csv(out/"dopant_score_details.csv").set_index("dopant").reindex(DOPANTS)
    central=screen[np.isclose(screen.substitution_pct,pct)].set_index("dopant").reindex(DOPANTS).reset_index()
    weights=validate_score_config(cfg)
    _,_,_,anchors=normalized_desirability(score_costs(central),weights)
    pool=ref.selection_pool.to_numpy(bool)
    def record(experiment,shares,n,valid_fraction):
        for j,d in enumerate(DOPANTS):
            rows.append(dict(substitution_pct=pct,dopant=d,experiment=experiment,
                top_share=float(shares[j]),draws=int(n),valid_draw_fraction=float(valid_fraction[j]),
                interpretation="fractional sensitivity wins / all draws; not a physical probability"))
    record("preference_weights",ref.weight_sweep_top_share.fillna(0).to_numpy(),
           cfg["score_weight_draws"],ref.selection_pool.to_numpy(float))
    with np.load(out/"mc_scenario_draws.npz",allow_pickle=False) as mc:
        indices=[]
        for d in DOPANTS:
            hits=np.flatnonzero((mc["dopants"]==d)&np.isclose(mc["substitution_pct"],pct))
            if len(hits)!=1: raise ValueError("Bulk sensitivity alignment failure.")
            indices.append(int(hits[0]))
        for mode in sorted(set(mc["efficiency_mode"])):
            mask=mc["efficiency_mode"]==mode
            wins=[]; eligibility=[]
            for e,g,m in zip(mc["FE_eV_atom"][mask][:,indices],mc["sigma_S_m"][mask][:,indices],mc["mu_cm2_Vs"][mask][:,indices]):
                with np.errstate(divide="ignore",invalid="ignore"):
                    cost=np.column_stack([e,-np.log10(g),-m])
                _,values,valid,_=normalized_desirability(cost,weights,anchors=anchors)
                wins.append(winner_shares(values,pool));eligibility.append(valid & pool)
            record(str(mode),np.mean(wins,axis=0),int(mask.sum()),np.mean(eligibility,axis=0))
    table=pd.DataFrame(rows);csv_out(table,out/"final_selection_sensitivity.csv")
    plot_clear_selection_sensitivity(table, cfg, out)
    return table


def plot_bulk_conductivity_all_dopants(screen, cfg, out):
    """Show central bulk conductivity and mobility together, without refitting.

    Preserve the old figure name and all old CSV fields; add an accessible
    mobility-only companion for narrow journal figure columns. No physics or
    scenario values are recalculated for visualization.
    """
    columns = ["dopant", "site", "substitution_pct", "p_scenario_cm3",
               "mu_scenario_cm2_Vs", "sigma_scenario_S_m", "rho_scenario_ohm_m",
               "hole_only_model_valid", "within_dilute_model_scope"]
    data = screen[columns].copy()
    if data.duplicated(["dopant", "substitution_pct"]).any():
        raise ValueError("Duplicate dopant/concentration rows in the bulk transport plot.")
    grid = np.sort(np.asarray(cfg["requested_pareto_pct"], dtype=float))
    for dopant in DOPANTS:
        values = np.sort(data.loc[data.dopant.eq(dopant), "substitution_pct"].to_numpy(float))
        if len(values) != len(grid) or not np.allclose(values, grid):
            raise ValueError(f"Incomplete conductivity/mobility grid for {dopant}.")
    data["transport_basis"] = "bulk_parameter_scenario_q_p_mu_not_DFT_or_measurement"
    csv_out(data, out / "bulk_conductivity_all_dopants.csv")
    folder = out / "figures"
    folder.mkdir(exist_ok=True)

    def draw_property(ax, property_name, ylabel, log_scale=False):
        for dopant in DOPANTS:
            part = data.loc[data.dopant.eq(dopant)].sort_values("substitution_pct")
            values = part[property_name].to_numpy(float)
            valid = np.isfinite(values) & (values > 0)
            if property_name == "sigma_scenario_S_m":
                valid &= part.hole_only_model_valid.to_numpy(bool)
            ax.plot(part.substitution_pct, np.where(valid, values, np.nan),
                    marker="o", color=COLORS[dopant], label=dopant,
                    lw=2.3, ms=6.5)
        ax.set_xlabel("Sublattice substitution (%)")
        ax.set_ylabel(ylabel)
        set_substitution_axis(ax, cfg)
        if len(grid) == 1:
            ax.set_xlim(grid[0] - .5, grid[0] + .5)
        if log_scale:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.3))
    fig.subplots_adjust(left=.085, right=.98, top=.83, bottom=.18, wspace=.28)
    draw_property(axes[0], "sigma_scenario_S_m", "Conductivity (S/m)", True)
    draw_property(axes[1], "mu_scenario_cm2_Vs", r"Mobility (cm$^2$/(V s))")
    axes[0].set_title("(a) Conductivity")
    axes[1].set_title("(b) Mobility")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=len(DOPANTS),
               bbox_to_anchor=(.53, .925), frameon=False, fontsize=12)
    fig.suptitle(f"Bulk SnO transport at {cfg['temperature_K']:g} K",
                 fontsize=18, y=.995)
    fig.text(.5, .055, f"Central model scenarios; concentrations above {cfg['dilute_screening_limit_pct']:g}% are exploratory.",
             ha="center", fontsize=10.5)
    save_figure(fig, folder / "23_bulk_conductivity_all_dopants")

    # Same mobility data, exported alone to allow a large, legible journal panel.
    fig, ax = plt.subplots(figsize=(10.3, 6.2))
    fig.subplots_adjust(left=.13, right=.97, top=.82, bottom=.18)
    draw_property(ax, "mu_scenario_cm2_Vs", r"Mobility (cm$^2$/(V s))")
    ax.legend(ncols=len(DOPANTS), frameon=False, loc="lower center",
              bbox_to_anchor=(.5, 1.01), fontsize=11)
    ax.set_title(f"Bulk SnO mobility at {cfg['temperature_K']:g} K", pad=24)
    save_figure(fig, folder / "26_bulk_mobility_all_dopants")


def plot_na_k_conductivity_ratio(screen, cfg, out):
    """Directly compare Na and K using R_sigma = (p_Na/p_K)*(mu_Na/mu_K).

    Input curves are exactly the current central transport results. The ratio
    is never based on separately normalized dopant preference scores. Invalid
    hole-only scenarios remain missing rather than being plotted as zero.
    """
    fields = ["substitution_pct", "p_scenario_cm3", "mu_scenario_cm2_Vs",
              "sigma_scenario_S_m", "hole_only_model_valid"]
    na = screen.loc[screen.dopant.eq("Na"), fields]
    k = screen.loc[screen.dopant.eq("K"), fields]
    if na.substitution_pct.duplicated().any() or k.substitution_pct.duplicated().any():
        raise ValueError("Na/K ratio requires one point per dopant per concentration.")
    pairs = na.merge(k, on="substitution_pct", how="outer", validate="one_to_one",
                     suffixes=("_Na", "_K"), indicator=True)
    if not pairs._merge.eq("both").all():
        raise ValueError("Na and K must have matching substitution concentrations.")
    pairs = pairs.drop(columns="_merge").sort_values("substitution_pct").reset_index(drop=True)
    concentration = np.asarray(cfg["requested_pareto_pct"], float)
    if len(pairs) != len(concentration) or not np.allclose(np.sort(concentration), pairs.substitution_pct):
        raise ValueError("Na/K ratio points do not match the configured concentration grid.")
    valid = pairs.hole_only_model_valid_Na.astype(bool) & pairs.hole_only_model_valid_K.astype(bool)
    for field in ["p_scenario_cm3", "mu_scenario_cm2_Vs", "sigma_scenario_S_m"]:
        valid &= np.isfinite(pairs[field + "_Na"]) & (pairs[field + "_Na"] > 0)
        valid &= np.isfinite(pairs[field + "_K"]) & (pairs[field + "_K"] > 0)
    ratio = np.full(len(pairs), np.nan)
    hole_ratio = np.full(len(pairs), np.nan)
    mobility_ratio = np.full(len(pairs), np.nan)
    ratio[valid] = (pairs.loc[valid, "sigma_scenario_S_m_Na"] /
                    pairs.loc[valid, "sigma_scenario_S_m_K"])
    hole_ratio[valid] = (pairs.loc[valid, "p_scenario_cm3_Na"] /
                         pairs.loc[valid, "p_scenario_cm3_K"])
    mobility_ratio[valid] = (pairs.loc[valid, "mu_scenario_cm2_Vs_Na"] /
                             pairs.loc[valid, "mu_scenario_cm2_Vs_K"])
    if not np.allclose(ratio[valid], hole_ratio[valid] * mobility_ratio[valid],
                       rtol=1e-10, atol=0.0):
        raise AssertionError("Na/K ratio failed the physical sigma = q*p*mu identity.")
    pairs["ratio_conductivity_Na_over_K"] = ratio
    pairs["ratio_hole_density_Na_over_K"] = hole_ratio
    pairs["ratio_mobility_Na_over_K"] = mobility_ratio
    pairs["ratio_identity_check"] = hole_ratio * mobility_ratio
    pairs["higher_central_conductivity"] = np.where(~valid, "invalid",
        np.where(np.isclose(ratio, 1, rtol=0, atol=1e-10), "equal",
                 np.where(ratio > 1, "Na", "K")))
    pairs["model_basis"] = "central_transport_scenario_not_experimental_validation"
    csv_out(pairs, out / "NaK_conductivity_ratio_vs_concentration.csv")

    fig, ax = plt.subplots(figsize=(10.4, 6.6))
    fig.subplots_adjust(left=.13, right=.97, top=.87, bottom=.17)
    ax.plot(pairs.substitution_pct, ratio, "o-", color="#245D83", lw=2.5,
            ms=8, label=r"$R_{\sigma}=\sigma_{Na}/\sigma_K$")
    ax.axhline(1.0, color="#B45309", linestyle="--", linewidth=1.8,
               label=r"$R_{\sigma}=1$ (equal conductivity)")
    set_substitution_axis(ax, cfg)
    finite = ratio[np.isfinite(ratio)]
    if len(finite):
        lo = min(float(finite.min()), 1.0)
        hi = max(float(finite.max()), 1.0)
        pad = max(.025, (hi - lo) * .23)
        ax.set_ylim(max(0, lo - pad), hi + pad)
    if len(concentration) == 1:
        ax.set_xlim(concentration[0]-.5, concentration[0]+.5)
    ax.set(xlabel="Sn-site substitution (%)",
           ylabel=r"Na/K conductivity ratio, $R_{\sigma}$",
           title="Na/K conductivity ratio versus concentration")
    ax.grid(axis="y", alpha=.18)
    ax.legend(loc="best", frameon=False)
    fig.text(.5, .06, r"$R_{\sigma}>1$: Na higher; $R_{\sigma}<1$: K higher. Central model estimates only.",
             ha="center", fontsize=11)
    save_figure(fig, Path(out) / "figures" / "27_NaK_conductivity_ratio")
    return pairs


def write_transport_physics_audit(screen, cfg, out):
    """Check numerical identities; do not label algebra checks as material validation."""
    rows = []
    rtol, atol = 1e-10, 0.0  # Keep the original strict identity tolerance.

    def check(frame, reported, expected, domain, name, unit, valid_column=None):
        actual = frame[reported].to_numpy(float)
        expected = np.broadcast_to(np.asarray(expected, dtype=float), actual.shape)
        valid = (frame[valid_column].to_numpy(bool) if valid_column else
                 np.ones(len(frame), dtype=bool))
        finite = np.isfinite(actual) & np.isfinite(expected)
        matched = finite & np.isclose(actual, expected, rtol=rtol, atol=atol)
        # A physically rejected row remains explicitly unevaluated.
        status = np.where(~valid, "not_evaluated_invalid_hole_only_scenario",
                          np.where(matched, "pass", "fail"))
        denominator = np.maximum(np.abs(expected), np.finfo(float).tiny)
        error = np.where(finite, np.abs(actual-expected)/denominator, np.nan)
        for i, (_, row) in enumerate(frame.iterrows()):
            rows.append(dict(domain=domain, check=name, dopant=row.dopant,
                substitution_pct=float(row.substitution_pct),
                host_structure_id=row.get("host_structure_id", "bulk_reference"),
                reported_value=actual[i], recomputed_value=expected[i], unit=unit,
                relative_error=error[i], status=status[i]))

    # For volumetric densities in cm^-3 and mobility in cm^2/Vs,
    # q*p*mu is S/cm, so multiplication by 100 gives S/m.
    check(screen, "sigma_scenario_S_m",
          100*elementary_charge*screen.p_scenario_cm3*screen.mu_scenario_cm2_Vs,
          "bulk", "sigma_S_m_equals_100_q_p_cm3_mu_cm2_Vs", "S/m", "hole_only_model_valid")
    check(screen, "rho_scenario_ohm_m", np.divide(1., screen.sigma_scenario_S_m),
          "bulk", "rho_equals_inverse_sigma", "ohm m", "hole_only_model_valid")
    audit = pd.DataFrame(rows)
    csv_out(audit, out / "transport_physics_checks.csv")
    failures = int(audit.status.eq("fail").sum())
    failure_summary = (audit.loc[audit.status.eq("fail")]
        .groupby(["domain", "check"], dropna=False)
        .agg(failures=("status", "size"), max_relative_error=("relative_error", "max"))
        .reset_index())
    report = dict(
        checks_passed=int(audit.status.eq("pass").sum()), checks_failed=failures,
        checks_not_evaluated=int(audit.status.str.startswith("not_evaluated").sum()),
        status="pass" if failures == 0 else "fail",
        relative_tolerance=rtol, absolute_tolerance=atol,
        csv_float_format=CSV_FLOAT_FORMAT, csv_float_parser="round_trip",
        failure_summary=failure_summary.to_dict(orient="records"),
        scope="Numerical transport identities and unit consistency only; not experimental/DFT validation.",
        new_DFT_calculations_performed=False,
        generated_structures_DFT_relaxed_by_this_workflow=False,
        model_limitations=[
            "DOS mass, background carriers, activation efficiency and scattering coefficients are assumptions.",
            "Single-acceptor and dilute scattering approximations may fail at high substitution.",
            "Device contacts, gate electrostatics and on/off ratio are not calculated."])
    write_json(out / "transport_physics_audit.json", report)
    if failures:
        print("Transport identity failure details:\n" + failure_summary.to_string(index=False), flush=True)
        raise ValueError(
            f"{failures} transport identity checks failed; inspect "
            f"{out / 'transport_physics_checks.csv'} and {out / 'transport_physics_audit.json'}. "
            "The failing check names and maximum relative errors are printed above.")
    print(f"Progress: Transport unit audit passed: {report['checks_passed']} checks; "
          f"{report['checks_not_evaluated']} invalid scenarios not evaluated.", flush=True)

    notes = (
        "TRANSPORT UNITS AND DFT STATUS\n"
        "sigma(S/m)=100*q*p(cm^-3)*mu(cm2/Vs); rho(ohm m)=1/sigma.\n"
        "CSV retains 17 significant digits; transport checks use rtol=1e-10 and atol=0.\n"
        "No DFT engine runs. Generated structures are unrelaxed starting inputs.\n"
        "Numerical identities do not establish experimental accuracy.\n"
        f"This run: {report['checks_passed']} passed; {report['checks_failed']} failed; "
        f"{report['checks_not_evaluated']} invalid hole-only scenarios not evaluated.\n"
        "Details: transport_physics_checks.csv; assumptions: CALCULATION_REFERENCE.txt.\n"
    )
    (out / "TRANSPORT_PHYSICS_AND_DFT_STATUS.txt").write_text(notes, encoding="utf-8")
    return report



def make_uncertainty_figures(screen, score_details, cfg, out):
    """Use saved paired draws; never invent intervals or perturb central values."""
    from matplotlib.lines import Line2D
    pct = float(cfg["score_reference_pct"])
    central = screen[np.isclose(screen.substitution_pct, pct, atol=1e-9, rtol=0)]
    if central.dopant.duplicated().any() or set(central.dopant) != set(DOPANTS):
        raise ValueError("Uncertainty plots require one central row per dopant.")
    central = central.set_index("dopant").reindex(DOPANTS)
    modes = ["shared_efficiency", "dopant_specific_efficiency"]
    mode_labels = ["Shared efficiency", "Dopant-specific efficiency"]
    mode_colors = ["#3B6FB6", "#DE8F05"]
    properties = [
        ("FE_eV_atom", "FE_ML_eV_atom", "Formation energy", "eV/atom", False),
        ("p_cm3", "p_scenario_cm3", "Hole density", r"cm$^{-3}$", True),
        ("mu_cm2_Vs", "mu_scenario_cm2_Vs", "Mobility", r"cm$^2$/(V s)", True),
        ("sigma_S_m", "sigma_scenario_S_m", "Conductivity", "S/m", True),
    ]
    records = []

    def add_summary(dopant, mode, prop, unit, values, centre, positive):
        values = np.asarray(values, float)
        usable = np.isfinite(values)
        if positive: usable &= values > 0
        finite = values[usable]
        # A single finite draw does not establish an uncertainty range.
        bounds = np.quantile(finite, [.05, .50, .95]) if len(finite) >= 2 else [np.nan] * 3
        return dict(dopant=dopant, substitution_pct=pct, efficiency_mode=mode,
                    property=prop, unit=unit, central_value=float(centre),
                    p05=bounds[0], p50=bounds[1], p95=bounds[2],
                    total_draws=len(values), valid_draws=len(finite),
                    valid_draw_fraction=len(finite) / len(values) if len(values) else np.nan,
                    interval_status="available" if len(finite) >= 2 else "fewer_than_two_valid_draws",
                    interval_basis="conditional_on_usable_draws; scenario_sensitivity_not_calibrated_CI")

    with np.load(out / "mc_scenario_draws.npz", allow_pickle=False) as mc:
        draw_modes = mc["efficiency_mode"].astype(str)
        if set(draw_modes) != set(modes):
            raise ValueError("Uncertainty plots require both saved efficiency modes.")
        indexes = []
        for dopant in DOPANTS:
            matches = np.flatnonzero((mc["dopants"] == dopant) &
                np.isclose(mc["substitution_pct"], pct, atol=1e-9, rtol=0))
            if len(matches) != 1: raise ValueError("Uncertainty draw/composition alignment failed.")
            indexes.append(int(matches[0]))
        for prop, column, _, unit, positive in properties:
            draws = mc[prop]
            if draws.shape != (len(draw_modes), len(mc["dopants"])):
                raise ValueError("Unexpected Monte Carlo array shape: " + prop)
            for mode in modes:
                for d, index in zip(DOPANTS, indexes):
                    records.append(add_summary(d, mode, prop, unit, draws[draw_modes == mode, index],
                                               central.loc[d, column], positive))
    table = pd.DataFrame(records)
    csv_out(table, out / "property_uncertainty_at_reference.csv")

    # Scores and physical outputs must refer to the very same simulation draws.
    score_central = score_details.set_index("dopant").reindex(DOPANTS)
    score_records = []
    with np.load(out / "score_scenario_draws.npz", allow_pickle=False) as mc:
        if not np.isclose(float(mc["substitution_pct"]), pct, atol=1e-9, rtol=0):
            raise ValueError("Score uncertainty concentration differs from the selected reference.")
        if mc["scores"].shape != (len(draw_modes), len(mc["dopants"])):
            raise ValueError("Score and physical scenario draw counts differ.")
        for d in DOPANTS:
            matches = np.flatnonzero(mc["dopants"] == d)
            if len(matches) != 1: raise ValueError("Score uncertainty dopant alignment failed.")
            for mode in modes:
                score_records.append(add_summary(d, mode, "score_0_100", "score points",
                    mc["scores"][draw_modes == mode, int(matches[0])],
                    score_central.loc[d, "score_0_100"], False))
    score_table = pd.DataFrame(score_records)
    csv_out(score_table, out / "score_uncertainty_at_reference.csv")

    def interval_panel(ax, frame, prop, title, unit, logarithmic=False):
        positions = np.arange(len(DOPANTS), dtype=float)
        for j, (mode, color) in enumerate(zip(modes, mode_colors)):
            part = frame[frame.efficiency_mode.eq(mode) & frame.property.eq(prop)]
            part = part.set_index("dopant").reindex(DOPANTS)
            xs = positions + (j - .5) * .22
            for x, (_, row) in zip(xs, part.iterrows()):
                if np.isfinite([row.p05, row.p50, row.p95]).all():
                    ax.errorbar(x, row.p50, yerr=[[row.p50 - row.p05], [row.p95 - row.p50]],
                                fmt="o", ms=4.5, color=color, capsize=4, lw=1.7, zorder=3)
                else:
                    ax.text(x, .025, "n/a", ha="center", rotation=90, color=color,
                            fontsize=8, transform=ax.get_xaxis_transform())
        first = frame[frame.efficiency_mode.eq(modes[0]) & frame.property.eq(prop)]
        first = first.set_index("dopant").reindex(DOPANTS)
        values = first.central_value.to_numpy(float)
        good = np.isfinite(values)
        if logarithmic: good &= values > 0
        ax.plot(positions[good], values[good], "D", ms=5, color="#111827", ls="none", zorder=4)
        all_values = frame.loc[frame.property.eq(prop), ["p05", "p50", "p95", "central_value"]].to_numpy(float)
        if logarithmic and np.any(np.isfinite(all_values) & (all_values > 0)):
            ax.set_yscale("log")
        elif logarithmic:
            ax.text(.5, .5, "No valid positive values", ha="center", transform=ax.transAxes)
        ax.set(title="", ylabel=f"{title}\n({unit})", xticks=positions, xticklabels=DOPANTS,
               xlim=(-.5, len(DOPANTS) - .5))
        ax.yaxis.label.set_fontsize(13)
        ax.yaxis.label.set_linespacing(1.3)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
        ax.grid(axis="y", alpha=.17); ax.set_axisbelow(True)
        if not logarithmic: ax.ticklabel_format(axis="y", useOffset=False)

    legend = [Line2D([0], [0], color=c, marker="o", lw=1.5,
                     label=label + ": median and 5–95% range")
              for c, label in zip(mode_colors, mode_labels)]
    legend.append(Line2D([0], [0], color="#111827", marker="D", lw=0, label="Central estimate"))
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.2))
    fig.subplots_adjust(left=.13, right=.97, top=.91, bottom=.17, hspace=.32, wspace=.36)
    # Display-only, intentionally wider axes. None of the Monte Carlo input ranges,
    # medians or 5-95% uncertainty bounds are altered or clipped. The wider
    # axes simply make the SAME intervals occupy less space on the plotted scale.
    broad_axis_limits = {
        "FE_eV_atom": (-3.0, 0.0),       # eV/atom; linear
        "p_cm3": (1e12, 1e22),           # cm^-3; logarithmic
        "mu_cm2_Vs": (0.5, 150.0),       # cm^2/(V s); logarithmic
        "sigma_S_m": (1e-3, 1e6),        # S/m; logarithmic
    }
    for ax, (prop, _, title, unit, logarithmic) in zip(axes.ravel(), properties):
        interval_panel(ax, table, prop, title, unit, logarithmic)
        lower, upper = broad_axis_limits[prop]
        # Expand further if an actual reported endpoint would fall outside the
        # requested wider range, so that no error bar is hidden or truncated.
        plotted = table.loc[table.property.eq(prop), ["p05", "p50", "p95", "central_value"]].to_numpy(float)
        plotted = plotted[np.isfinite(plotted)]
        if logarithmic:
            positive = plotted[plotted > 0]
            if len(positive):
                lower = min(lower, float(positive.min()) / 1.1)
                upper = max(upper, float(positive.max()) * 1.1)
        elif len(plotted):
            lower = min(lower, float(plotted.min()) - .05)
            upper = max(upper, float(plotted.max()) + .05)
        ax.set_ylim(lower, upper)
    fig.suptitle(f"SnO property uncertainty at {pct:g}% substitution ({cfg['temperature_K']:g} K)",
                 fontsize=17, y=.985)
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.54, .087),
               frameon=False, ncols=3, fontsize=10)
    fig.text(.13, .025,
             "Error bars: 5th–95th percentile of valid model scenarios; not calibrated confidence intervals.",
             fontsize=10)
    save_figure(fig, out / "figures" / "24_property_uncertainty")

    fig, ax = plt.subplots(figsize=(11.8, 6.8))
    fig.subplots_adjust(left=.13, right=.98, top=.84, bottom=.23)
    interval_panel(ax, score_table, "score_0_100", "Fixed scoring weights and central normalization anchors",
                   "Dopant preference score (0–100)")
    ax.set_ylim(-3, 103)
    fig.suptitle(f"SnO dopant-score uncertainty | {pct:g}% sublattice substitution", fontsize=14, y=.96)
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.54, .10),
               frameon=False, ncols=3, fontsize=10)
    fig.text(.13, .025, "Model-scenario 5th–95th percentiles; not calibrated confidence intervals.", fontsize=10)
    save_figure(fig, out / "figures" / "25_score_uncertainty")
    return table, score_table



def write_compact_summary(screen, selection, cfg, out):
    """Five-dopant summary and one shared figure list for notebook and HTML."""
    pct = float(cfg["score_reference_pct"])
    part = screen[np.isclose(screen.substitution_pct, pct, rtol=0, atol=1e-9)]
    if part.dopant.duplicated().any() or set(part.dopant) != set(DOPANTS):
        raise ValueError("Compact summary needs exactly one reference row per dopant.")
    part = part.set_index("dopant").reindex(DOPANTS)
    result = pd.DataFrame({"Dopant": DOPANTS,
        "Formation energy (eV/atom)": part.FE_ML_eV_atom.to_numpy(),
        "Conductivity (S/m)": part.sigma_scenario_S_m.to_numpy(),
        "Mobility (cm2/Vs)": part.mu_scenario_cm2_Vs.to_numpy(),
        "Hole density (cm-3)": part.p_scenario_cm3.to_numpy(),
        "Resistivity (ohm m)": part.rho_scenario_ohm_m.to_numpy()})
    csv_out(result, out / "main_dopant_results.csv")
    csv_out(screen[["dopant", "site", "substitution_pct", "FE_ML_eV_atom", "FE_bootstrap_p05", "FE_bootstrap_p95",
                    "FE_empirical_lower", "FE_empirical_upper", "in_domain", "supported_screening"]],
            out / "bulk_formation_energy_all_dopants.csv")
    figures = ["02_GradientBoosting_validation.png", "08_formation_energy_all_dopants.png",
               "23_bulk_conductivity_all_dopants.png", "26_bulk_mobility_all_dopants.png",
               "27_NaK_conductivity_ratio.png", "10_dopant_score_components.png",
               "07_conductivity_uncertainty.png",
               "24_property_uncertainty.png", "25_score_uncertainty.png",
               "21_selection_weight_sensitivity.png", "22_selection_input_sensitivity.png"]
    for name in figures:
        if not (out / "figures" / name).is_file():
            raise FileNotFoundError("Required displayed figure was not created: " + name)
    write_json(out / "compact_summary.json", dict(substitution_pct=pct,
        selected_dopants=selection["selected_dopants"], selection_status=selection["selection_status"],
        formation_energy_kind="compound-energy ML proxy, not charged-defect formation energy",
        transport_kind="sigma_S_m=100*q*p_cm3*mu_cm2_Vs; physical parameter scenario",
        new_DFT_calculations_performed=False, physics_audit_file="transport_physics_audit.json",
        uncertainty_basis="5th/50th/95th percentiles of usable saved scenario draws, by efficiency mode",
        display_figures=figures))




# Reporting helpers embedded into the complete SnO workflow.

def write_clear_counts(data, raw, cfg, out):
    """Keep database records, ML samples and generated starting inputs separate."""
    if data.material_id.duplicated().any():
        raise ValueError("Duplicate material IDs in the ML dataset.")
    split = pd.read_csv(out / "data_splits.csv")
    if split.material_id.duplicated().any() or set(split.material_id) != set(data.material_id):
        raise ValueError("Saved train/test split does not match the curated ML dataset.")
    if not set(split.partition) <= {"development", "locked_test"}:
        raise ValueError("Unknown ML partition label.")
    development, test = [split[split.partition.eq(k)] for k in ["development", "locked_test"]]
    if development.empty or test.empty or len(development) + len(test) != len(data):
        raise ValueError("Training/test counts are incomplete.")
    if set(development.formula) & set(test.formula):
        raise ValueError("Composition groups overlap between development and test.")
    samples = pd.DataFrame([dict(stage=label, samples=len(frame), unique_compositions=frame.formula.nunique())
        for label, frame in [("Training / cross-validation", development),
                             ("Held-out testing", test), ("Total ML samples", split)]])
    csv_out(samples, out / "ml_sample_counts.csv")
    manifest = out / "starting_structure_manifest.csv"
    if manifest.exists():
        generated = pd.read_csv(manifest)
        if generated.file.duplicated().any(): raise ValueError("Duplicate generated-file names.")
        n_generated = len(generated)
    else:
        if cfg["make_dft_starting_structures"]:
            raise ValueError("Requested starting-structure manifest is missing.")
        n_generated = 0
    host_count = len({str(d["material_id"]) for d in raw["hosts"]})
    ternary_count = len({str(d["material_id"]) for d in raw["ternaries"]})
    csv_out(pd.DataFrame([dict(material="Pristine SnO records retrieved", count=host_count)]),
            out / "simple_material_counts.csv")
    host, target = data.A.eq("HOST"), data.A.isin(DOPANTS)
    rows = [
        ("Database search", "Pristine SnO records retrieved", host_count,
         "Unique Materials Project IDs from the SnO query; polymorphs are not merged."),
        ("Database search", "Sn-O-A ternary records retrieved", ternary_count,
         "Unique IDs before the local training filters."),
        ("ML dataset", "Retained ternary structures", int((~host).sum()),
         "Sn + O + one additional element, after training filters."),
        ("ML dataset", "Of these: ternaries with Li, Na, K, N or P", int(target.sum()),
         "Subset of retained ternaries; does not prove substitutional doping."),
        ("ML dataset", "Pristine SnO references used for ML", int(host.sum()),
         "Selected semiconducting P4/nmm reference; not every retrieved polymorph."),
        ("ML dataset", "Total ML structures", len(data),
         "Retained ternaries plus the selected host; generated structures excluded."),
        ("Generated inputs", "Doped SnO starting structures", n_generated,
         "Unrelaxed atomic substitutions, counted separately; no DFT engine is run."),
        ("Virtual screening", "Dopant/concentration rows", len(DOPANTS) * len(cfg["requested_pareto_pct"]),
         "Virtual predictions, not database structures or ML labels."),
    ]
    counts = pd.DataFrame(rows, columns=["group", "structure_type", "count", "meaning"])
    csv_out(counts, out / "structure_counts.csv")
    (out / "COUNTS_EXPLAINED.txt").write_text(
        "STRUCTURE AND SAMPLE COUNTS\n"
        "Counts of database records use unique Materials Project IDs; geometric prototypes are not deduplicated.\n"
        "Target-dopant ternaries are a subset of retained ternaries, not additional structures.\n"
        "Generated atomic substitutions and virtual screening rows are excluded from ML counts.\n"
        f"Development: {len(development)} rows / {development.formula.nunique()} compositions.\n"
        f"Locked test: {len(test)} rows / {test.formula.nunique()} compositions.\n"
        f"Final refit after evaluation: {len(data)} rows.\n"
        "Training labels are compound formation energies. Transport uses declared physical parameters.\n"
        "Definitions: structure_counts.csv; saved partitions: data_splits.csv.\n", encoding="utf-8")
    write_json(out / "sample_count_summary.json", dict(
        training_samples=len(development), testing_samples=len(test), total_samples=len(split),
        training_compositions=int(development.formula.nunique()), testing_compositions=int(test.formula.nunique()),
        final_refit_samples=len(data), transport_ML_model_trained=False))
    return counts, samples


def plot_clear_selection_sensitivity(table, cfg, out):
    """Show selection frequency across weights and across physical input draws."""
    pct = float(cfg["score_reference_pct"])

    def get(experiment):
        part = table[table.experiment.eq(experiment)]
        if len(part) != len(DOPANTS) or part.dopant.duplicated().any():
            raise ValueError("Missing/duplicate all-dopant sensitivity rows.")
        return part.set_index("dopant").reindex(DOPANTS)

    def draw_bars(ax, x, part, color, width=.65, label=None):
        usable = np.isfinite(part.top_share) & (part.valid_draw_fraction > 0)
        values = np.where(usable, 100 * part.top_share, 0.)
        ax.bar(x, values, color=color, width=width, label=label)
        for position, value, valid in zip(x, values, usable):
            ax.text(position, value + 1.8, f"{value:.1f}" if valid else "n/a",
                    ha="center", va="bottom", fontsize=9)

    pos = np.arange(len(DOPANTS))
    part = get("preference_weights")
    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    fig.subplots_adjust(left=.10, right=.98, bottom=.24, top=.78)
    draw_bars(ax, pos, part, [COLORS[d] for d in DOPANTS])
    ax.set(xticks=pos, xticklabels=DOPANTS, ylim=(0, 112), yticks=[0, 25, 50, 75, 100],
           ylabel="Selected in weight trials (%)", xlabel="Dopant",
           title=f"{int(part.draws.iloc[0]):,} trials; central material inputs held fixed")
    ax.grid(axis="y", alpha=.14); ax.set_axisbelow(True)
    fig.suptitle(f"SnO selection sensitivity to scoring weights | {pct:g}%", fontsize=14, y=.97)
    fig.text(.10, .055, "Only log-conductivity and mobility weights vary; formation energy is excluded.",
             fontsize=10)
    save_figure(fig, out / "figures" / "21_selection_weight_sensitivity")

    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    fig.subplots_adjust(left=.10, right=.98, bottom=.30, top=.78)
    experiments = ["shared_efficiency", "dopant_specific_efficiency"]
    labels = ["Shared activation efficiency", "Dopant-specific activation efficiency"]
    counts = []
    for j, (mode, label, color) in enumerate(zip(experiments, labels, ["#527da8", "#df9850"])):
        part = get(mode); counts.append(int(part.draws.iloc[0]))
        draw_bars(ax, pos + (j - .5) * .36, part, color, width=.36, label=label)
    ax.set(xticks=pos, xticklabels=DOPANTS, ylim=(0, 112), yticks=[0, 25, 50, 75, 100],
           xlabel="Dopant", ylabel="Selected in input trials (%)",
           title=f"{counts[0]:,} / {counts[1]:,} trials; scoring weights held fixed")
    ax.grid(axis="y", alpha=.14); ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.54, .17),
               ncols=2, frameon=False, fontsize=10)
    fig.suptitle(f"SnO selection sensitivity to uncertain inputs | {pct:g}%", fontsize=14, y=.97)
    fig.text(.10, .045, "Two separate input modes; frequencies represent model scenarios, not probabilities.",
             fontsize=10)
    save_figure(fig, out / "figures" / "22_selection_input_sensitivity")
    (out / "SENSITIVITY_EXPLAINED.txt").write_text(
        "SELECTION SENSITIVITY\nFigure 21: vary scoring weights, holding central material inputs fixed.\n"
        "Figure 22: vary transport inputs, holding scoring weights fixed; FE is not scored.\n"
        "Blue: shared activation efficiency. Orange: dopant-specific activation efficiency.\n"
        "Each colour is a separate trial set. Ties split a win; invalid trials stay in denominators.\n"
        "Figures retain all five dopants. Complete data: final_selection_sensitivity.csv.\n",
        encoding="utf-8")


def concise_calculation_guide(cfg, physical_assumptions):
    """Readable equations and the active inputs, rendered offline."""
    def number(value):
        text = f"{float(value):.4g}"
        if "e" not in text: return text
        mantissa, exponent = text.split("e")
        power = str(int(exponent)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
        return mantissa + " × 10" + power
    pars = physical_assumptions
    weights = validate_score_config(cfg)
    labels = ["energy (excluded)", "log conductivity", "mobility"]
    inputs = [
        ("Active dopant fraction", f"{100 * pars['active_fraction']:g}%"),
        ("Compensation fraction", f"{100 * pars['compensation_fraction']:g}%"),
        ("Background hole density", number(pars["p0_cm3"]) + " cm⁻³"),
        ("Baseline mobility", number(pars["mu0_cm2_Vs"]) + " cm²/(V·s)"),
        ("Mismatch coefficient A", number(pars["alloy_scattering"]) + " (dimensionless)"),
        ("Active-dopant scattering coefficient B", number(pars["ion_scattering"]) + " (dimensionless)"),
        ("Score weights", "; ".join(f"{label} {100 * w:.3g}%" for label, w in zip(labels, weights))),
    ]
    equations = [
        ("Active acceptor density", "N<sub>A</sub> = x × N<sub>site</sub> × η",
         "Nₐ = x × Nsite × η", "x is the substituted-sublattice fraction; η is the active fraction; density is cm⁻³."),
        ("Dopant-host differences",
         "δ<sub>r</sub> = |r<sub>D</sub> − r<sub>site</sub>| / r<sub>site</sub>; Δχ = |χ<sub>D</sub> − χ<sub>site</sub>|",
         "δr = |rD − rsite| / rsite; Δχ = |χD − χsite|",
         "r is covalent radius; χ is electronegativity. D denotes the dopant and site the substituted host atom."),
        ("Hole mobility",
         "μ = μ<sub>0</sub> / [1 + Ax(1 − x)(δ<sub>r</sub><sup>2</sup> + 0.15(Δχ)<sup>2</sup>) + Bxη]",
         "μ = μ0 / [1 + A*x*(1 − x)*(δr² + 0.15*(Δχ)²) + B*x*η]",
         "μ0 is reference mobility. A = alloy_scattering and B = ion_scattering are assumed dimensionless coefficients; x and η are fractions."),
        ("Mobility retention", "R<sub>μ</sub> = μ / μ<sub>0</sub>", "Rμ = μ / μ0",
         "Dimensionless fraction of the reference mobility retained."),
        ("Conductivity", "σ = 100 × q × p × μ", "σ = 100 × q × p × μ",
         "Result: S/m, using p in cm⁻³ and μ in cm²/(V·s). The factor 100 converts to SI."),
        ("Resistivity", "ρ = 1 / σ", "ρ = 1 / σ", "Result: Ω·m when σ is in S/m."),
        ("Preference score", "Score = 100 × ∑<sub>i</sub> w<sub>i</sub>U<sub>i</sub>",
         "Score = 100 × sum of (weight × desirability)",
         "Each U ranges from 0 to 1. Only higher log-conductivity and higher mobility contribute to the preference score; formation energy is excluded."),
        ("Scenario uncertainty", "Range = [Q<sub>0.05</sub>, Q<sub>0.95</sub>]; median = Q<sub>0.50</sub>",
         "Range = [5th percentile, 95th percentile]; median = 50th percentile",
         "Computed from the saved usable Monte Carlo draws, separately for shared and dopant-specific activation efficiency."),
    ]
    activation = []
    for dopant in DOPANTS:
        centre, low, high, doi, evidence = EA[dopant]
        source = "Literature input" if evidence == "reported_effective_activation" and doi else "Assumed"
        activation.append((f"{dopant} ({SITES[dopant]} site)", number(centre),
                           f"{number(low)}–{number(high)}", source, doi))
    return dict(equations=equations, inputs=inputs, activation=activation,
        symbols="q = 1.602176634 × 10⁻¹⁹ C; p = hole density (cm⁻³); μ = mobility (cm²/(V·s)).",
        logic="Gradient Boosting predicts compound formation energy. Hole densities follow acceptor charge balance with Fermi–Dirac statistics. Mobility uses the configured baseline and dopant scattering penalties.",
        at=f"Temperature: {cfg['temperature_K']:g} K. Reference comparison: {cfg['score_reference_pct']:g}% sublattice substitution.",
        sources="Energies and structures: Materials Project. Na/K/N activation sources are linked below; Li/P activation values are assumptions.",
        note="Effective Hall activation inputs are process dependent and need not equal thermodynamic charge-transition levels.",
        limits="Transport remains a parameter scenario. Percentile ranges are not calibrated confidence intervals. Compound formation energy is not charged-defect formation energy; device switching is not calculated.")


def concise_calculation_text(cfg, physical_assumptions):
    guide = concise_calculation_guide(cfg, physical_assumptions)
    lines = ["CALCULATIONS — QUICK GUIDE", guide["at"], "", guide["logic"], ""]
    for label, _, formula, meaning in guide["equations"]:
        lines.extend([label + ": " + formula, meaning, ""])
    lines.extend([guide["symbols"], "", "ACTIVE INPUTS"])
    for label, value in guide["inputs"]: lines.append(f"{label}: {value}")
    lines.extend(["", "ACTIVATION ENERGY (eV; central and scenario bounds)"])
    for dopant, centre, bounds, source, doi in guide["activation"]:
        lines.append(f"{dopant}: {centre}; range {bounds}; {source}" + (" — https://doi.org/" + doi if doi else ""))
    lines.extend([guide["note"], "", guide["sources"], guide["limits"], "",
        "Full methods: calculation_basis.csv. Active inputs/bounds: calculation_parameters.csv.",
        "Source registry: references.csv. Complete settings: run_config.json. Units: COLUMN_GUIDE.csv."])
    return "\n".join(lines) + "\n"


def concise_calculation_html(out, inline=False):
    from html import escape
    out = Path(out)
    saved = json.loads(find_result_file(out, "run_config.json").read_text(encoding="utf-8"))
    guide = concise_calculation_guide(saved["config"], saved["physical_assumptions"])
    chunks = ['<details class="calculation-guide"><summary>Calculations explained — equations, inputs and sources</summary>',
              '<p>' + escape(guide["logic"]) + '</p><div class="equation-list">']
    for label, formula, _, meaning in guide["equations"]:
        chunks.append('<div class="equation-row"><strong>' + escape(label) + '</strong>'
            '<div class="equation">' + formula + '</div><p>' + escape(meaning) + '</p></div>')
    chunks.extend(['</div><p class="symbols">' + escape(guide["symbols"]) + '</p>',
        '<h3>Active inputs</h3><p>' + escape(guide["at"]) + '</p>',
        '<div class="scroll"><table><thead><tr><th>Input</th><th>Value</th></tr></thead><tbody>'])
    for row in guide["inputs"]:
        chunks.append('<tr>' + ''.join('<td>' + escape(cell) + '</td>' for cell in row) + '</tr>')
    chunks.extend(['</tbody></table></div><h3>Activation energy inputs</h3>',
        '<div class="scroll"><table><thead><tr><th>Dopant (site)</th><th>Central E<sub>A</sub> (eV)</th>'
        '<th>Scenario bounds (eV)</th><th>Source role</th></tr></thead><tbody>'])
    for dopant, centre, bounds, source, doi in guide["activation"]:
        link = ('<a href="https://doi.org/' + escape(doi, quote=True) + '">' + escape(source) + '</a>') if doi else escape(source)
        chunks.append('<tr>' + ''.join('<td>' + escape(cell) + '</td>' for cell in (dopant, centre, bounds))
                      + '<td>' + link + '</td></tr>')
    chunks.append('</tbody></table></div>')
    for key in ["note", "sources", "limits"]: chunks.append('<p>' + escape(guide[key]) + '</p>')
    links = []
    for name, label in [("calculation_basis.csv", "Full methods"), ("calculation_parameters.csv", "All inputs/bounds"),
                        ("host_reference.json", "Host reference"), ("references.csv", "Source registry")]:
        if inline: links.append(escape(name))
        else:
            path = find_result_file(out, name).relative_to(out).as_posix()
            links.append('<a href="' + escape(path, quote=True) + '">' + escape(label) + '</a>')
    chunks.append('<p class="guide-files">Details in the results ZIP: ' + ' · '.join(links) + '.</p></details>')
    return '\n'.join(chunks)


def write_calculation_reference(cfg, out):
    """Report the active inputs and their role without changing any calculation.

    Numerical provenance remains in run_config.json, source registries, per-host
    metadata and the original tables. This is a reader-facing map of that evidence.
    All sources below are inherited from the supplied workflow, not reverified here.
    """
    rows = []
    def add(topic, rule, reason, source):
        rows.append(dict(calculation=topic, equation_or_rule=rule,
                         reason_and_scope=reason, source_or_implementation=source))
    add("Bulk training data",
        "Nondeprecated ordered Sn-O-A ternaries + selected P4/nmm SnO host; finite FE and hull energy; "
        f"hull <= {cfg['mp_hull_max_eV_atom']:g} eV/atom; exclude {sorted(EXCLUDED_ELEMENTS)}.",
        "FE labels describe bulk compounds. A ternary record does not establish substitutional doping. "
        "The selected host has a positive source gap; lowest hull energy, then smallest cell, breaks ties.",
        "Materials Project; raw_materials.json; rejected_records.csv; curate()")
    add("Features and ML evaluation",
        "Composition fractions, entropy and weighted elemental descriptors; optional structure descriptors. "
        "Median imputation -> constant-feature removal -> scaling -> GradientBoosting(huber). "
        "GroupShuffleSplit by formula; nested GroupKFold; tune on development MAE; test once; refit on all rows.",
        "Keeps polymorphs of a composition in one partition. Test permutation importance is diagnostic only. "
        "Optional virtual structure descriptors use ideal host geometry. Pooled R2 uses all outer-fold predictions.",
        "X_features_and_groups.csv; data_splits.csv; model_selection.json; train_gradient_boosting()")
    add("Energy uncertainty and applicability",
        "Bootstrap complete composition groups. FE envelope = bootstrap 5/95 percentiles +/- "
        "90th percentile of the per-formula maximum absolute CV residual. "
        "Robust-scaled nearest-neighbour threshold excludes same-formula reference neighbours.",
        "Also require feature-range and dopant-fraction coverage. Bounds are screening diagnostics, "
        "not calibrated coverage for unseen chemistries.",
        "bootstrap_predictions(); applicability_domain(); prepare_predictions(); formation_energy_bootstrap_draws.npz")
    add("Screening gates",
        "CV MAE <= configured gate; test MAE < training-mean baseline; withheld-dopant MAE <= transfer gate; "
        "in-domain; within dilute scope; experimental EA input; no process caution.",
        "Supported means passes these modeling rules. The dilute limit is not solubility. "
        "N has a process-specific caution above 7e17 nominal dopants/cm^3.",
        "screening_results.csv; leave_one_dopant_out_metrics.csv; Becker 2019: 10.1063/1.5052606")
    add("Concentration and site density",
        "x = sublattice substitution percent/100; total-atom percent = 50*x. "
        "Li/Na/K replace Sn; N/P replace O. Nsite = N_Sn/V_A3 * 1e24 cm^-3; NA = x*Nsite*active_fraction.",
        "Uses the selected host volume and the designated sublattice.",
        "SITES; prepare_predictions(); host_reference.json")
    add("Acceptor neutrality",
        "p(EF) + ND = p0 + NA/[1 + gA*exp((EA-EF)/kT)]; ND = compensation_fraction*NA; Ev = 0.",
        "One effective acceptor level, fixed ionized background and fully ionized donors. "
        "Film Hall activation inputs are not generally thermodynamic charge-transition levels. "
        "Complexes, impurity bands, Hall factors and microstructure are not computed.",
        "solve_holes(); dopant_evidence_registry.csv")
    add("Bulk hole density",
        "p = Nv*F_1/2(-EF/kT); Nv = 2*(2*pi*mDOS*kB*T/h^2)^(3/2)/1e6 cm^-3. "
        "Neutrality is solved by bisection. Require EF < Eg/2 - 3*kT for the hole-only scenario.",
        "Fermi-Dirac statistics include degeneracy; rigid-band and isolated-level assumptions remain. "
        "The indirect gap guard screens a possible bipolar regime; invalid densities become NaN.",
        "fermi_table(); fermi_half(); solve_holes(); Allen 2013: 10.1039/C3TC31863J")
    add("Mobility and distortion proxies",
        "dr=abs(rD-rsite)/rsite; dc=abs(chiD-chisite); A=alloy_scattering; B=ion_scattering. "
        "mu=mu0/[1+A*x*(1-x)*(dr^2+0.15*dc^2)+B*x*eta]; Rmu=mu/mu0. "
        "Bulk distortion=100*x*dr*distortion_scale; compatibility=exp(-1.6*dr-0.45*dc).",
        "A and B are assumed dimensionless coefficients for mismatch and active-dopant scattering. "
        "Radii use one covalent convention. Compatibility does not set EA. "
        "Radius-free ablations remove radius penalties and distortion. The legacy mass diagnostic "
        "m_transport_proxy_over_me=m_transport_over_me*[1+x*(mass_penalty*dr+0.1*dc)] "
        "is retained for output compatibility only and is excluded from mobility and transport surrogates.",
        "transport(); RADII; CHI; Cordero 2008: 10.1039/b801115j")
    add("Conductivity and resistivity units",
        "sigma(S/m)=100*q*p(cm^-3)*mu(cm2/Vs); rho(ohm m)=1/sigma; rho(ohm cm)=100/sigma.",
        "The factor 100 converts q*p*mu from S/cm to S/m. Use the declared volume density and mobility.",
        "transport(); transport_physics_checks.csv")
    add("Energy and competing phases",
        "FE_ML is compound formation energy. Competing-ternary proxy uses the "
        "available ternary hull energies; it is removed in an ablation.",
        "Compound FE is not dopant incorporation energy. Charged-defect energies require chemical "
        "potentials, charge states, Fermi level and corrections. The ternary proxy omits binary/elemental competition.",
        "phase_proxy(); objective_matrix(); Varley 2013: 10.1063/1.4819068; Togo 2006: 10.1103/PhysRevB.74.195128")
    add("Pareto and preference scores",
        "At fixed x, Pareto minimizes [FE,-log10(sigma),1-Rmu,distortion,phase_proxy]; no dominance weights. "
        "Post-front compromise uses normalized distance to the ideal point. Preference score=100*sum(w*U) "
        "using only log conductivity and mobility (2:1 weights); U=(cmax-c)/(cmax-cmin), constant U=0.5. "
        "FE remains an independent Pareto objective and reported ML property, not a score input.",
        "Missing positively weighted criteria are ineligible. Prefer the highest-score supported bulk candidate, "
        "else label the leader exploratory. Keep exact ties. Anchors vary by concentration; "
        "scores cannot optimize concentration on an absolute scale. Separate mobility weight adds preference "
        "to mobility already present in conductivity.",
        "pareto_analysis(); dopant_scoring(); score_normalization.csv; OECD/JRC: 10.1787/9789264043466-en")
    add("Weight and input sensitivity",
        "Weight sweep: vary the log-conductivity/mobility preferences with central material inputs fixed. "
        "Input sweep: vary physical transport inputs with weights and central anchors fixed, clipping utilities to [0,1]. "
        "Keep shared and dopant-specific efficiency modes separate; split tied wins; keep invalid draws in denominators.",
        "Percentiles/top-choice shares describe selected scenarios, not confidence intervals or success probabilities. "
        "Central inputs need not be the ensemble median. Na/K pairwise wins are not all-dopant wins.",
        "sample_parameters(); robustness(); final_selection_sensitivity.csv; mc_input_draws.csv")
    add("Sensitivity surrogates",
        "Held-out Gradient Boosting models explain simulated Na/K differences. Interpret permutation "
        f"importance only when holdout R2 >= {cfg['sensitivity_min_R2']:g} and MAE beats the training-median baseline.",
        "Surrogates are not new measured-material predictors. Shuffle SD is not a confidence interval. "
        "K-minus-Na score contributions favor K when positive; the documented ML difference targets favor Na when positive.",
        "run_integrated_sensitivity(); 10_sensitivity/READ_INTERPRETATION.txt; surrogate_holdout_metrics.csv")
    add("Precision, sources and validation",
        "CSV retains 17 significant digits; float64 round-trip reload; transport identity tolerance "
        "rtol=1e-10, atol=0. Missing/ineligible values remain NaN. Numerical checks do not validate material physics.",
        "Literature inputs cannot also validate the model. Unpaired literature ranges remain unpaired. "
        "Use matched experimental/DFT templates for independent comparisons. Device switching and operational "
        "stability are not predicted. Bibliography and snapshot provenance are retained from the supplied code.",
        "write_transport_physics_audit(); external_validation_status.json; references.csv; COLUMN_GUIDE.csv; workflow_source.py")
    add("Displayed uncertainty ranges",
        "For each dopant at the reference concentration, compute Q05/Q50/Q95 of saved usable draws, separately "
        "for shared and dopant-specific efficiency. Scores retain central anchors and clip utilities to [0,1].",
        "Energy draws (property uncertainty only, not preference score) combine group bootstrap with sampled per-formula mean residuals. Transport varies declared "
        "physical inputs. Central estimates can differ from medians. Export valid fractions; fewer than two usable "
        "draws gives an unavailable interval. Ranges are not calibrated confidence intervals.",
        "make_uncertainty_figures(); mc_scenario_draws.npz; score_scenario_draws.npz; "
        "property_uncertainty_at_reference.csv; score_uncertainty_at_reference.csv")
    basis = pd.DataFrame(rows)
    csv_out(basis, out / "calculation_basis.csv")

    params = []
    def entry(group, values, role, source):
        params.append(dict(input_group=group,
            active_values=json.dumps(values, ensure_ascii=False, sort_keys=True, default=str),
            reason_and_units=role, source=source))
    def settings(*keys): return {key: cfg[key] for key in keys}
    entry("Grid and temperature", settings("temperature_K", "requested_pareto_pct", "score_reference_pct"),
          "K; percent of the designated sublattice; comparison concentration.", "User CONFIG; virtual_grid()")
    entry("Bulk physics", PHYSICS,
          "Central scenario values; units are encoded in names. Fractions/degeneracies/penalty coefficients are dimensionless. "
          "A=alloy_scattering and B=ion_scattering enter simplified mobility; mass_penalty and "
          "m_transport_over_me are legacy diagnostic inputs only. "
          "Gap is the indirect-gap guard, not the optical direct gap.", "Declared assumptions; 0.67 eV context: 10.1039/C3TC31863J")
    entry("Bulk uncertainty", {
          "p0_cm3": {"bounds": [2e17, 1e19], "distribution": "log_uniform"},
          "mu0_cm2_Vs": [3., 30.], "m_dos_over_me": [.3, 1.5],
          "active_fraction": {"bounds": [.01, 1.], "distribution": "log_uniform"},
          "compensation_fraction": [0., .5], "mass_penalty": [.25, 1.],
          "alloy_scattering": [3.5, 14.], "ion_scattering": [1.25, 5.], "distortion_scale": [.5, 1.5]},
          "Absolute bounds in sample_parameters(); uniform unless marked log_uniform. "
          "A=alloy_scattering and B=ion_scattering; the mass_penalty draw is retained only for "
          "the legacy diagnostic and random-sequence continuity. EA bounds are in dopant_evidence_registry.csv.",
          "Declared sensitivity assumptions; sample_parameters(); robustness()")
    entry("Dopant/site/activation inputs", {
          d: {"site": SITES[d], "role": ROLES[d], "EA_central_eV": EA[d][0],
              "EA_bounds_eV": list(EA[d][1:3]), "doi": EA[d][3], "evidence": EA[d][4]} for d in DOPANTS},
          "Effective activation inputs, not FE training labels; Li/P are unvalidated assumptions.", "EA; dopant_evidence_registry.csv")
    entry("Element proxies", {"covalent_radii_A": RADII, "electronegativity": CHI},
          "Consistent geometric/chemical descriptors; not ionic radii or defect charges. CHI values are retained from the supplied code.",
          "RADII; CHI; Cordero radii: 10.1039/b801115j")
    entry("Physical constants", {"q_C": elementary_charge, "kB_J_K": Boltzmann, "h_J_s": h, "m_e_kg": m_e},
          "Resolved scipy.constants values used by neutrality and unit conversions.", "scipy.constants; requirements_resolved.txt")
    entry("Data filters and feature policy", {**settings("mp_hull_max_eV_atom", "include_gnome", "use_structure_features"),
          "excluded_elements": sorted(EXCLUDED_ELEMENTS)},
          "Hull cutoff in eV/atom. Structural features are optional host-geometry proxies.",
          "curate(); get_feature_columns(); run_config.json")
    estimator, grid = gradient_boosting_spec(cfg)
    if cfg.get("test_mode"): grid = {k: v[:1] for k, v in grid.items()}
    entry("ML model and resampling", {**settings("seed", "run_size", "outer_folds", "inner_folds", "test_fraction",
          "bootstrap_models", "mc_draws", "n_jobs"), "estimator": estimator.get_params(), "tuning_grid": grid},
          "Group split fraction applies to compositions, so row fractions can differ. Actual split sizes are in ml_sample_counts.csv.",
          "gradient_boosting_spec(); fit_search(); data_splits.csv")
    entry("Screening gates", settings("cv_mae_gate_eV_atom", "transfer_mae_gate_eV_atom", "domain_quantile",
          "max_out_of_range_fraction", "dilute_screening_limit_pct"),
          "Declared tolerances, not physical constants or solubility limits; N process caution threshold is 7e17 cm^-3.",
          "applicability_domain(); prepare_predictions(); 10.1063/1.5052606")
    entry("Decision and sensitivity settings", settings("score_weights", "score_weight_draws", "score_close_margin_points",
          "compromise_tolerance", "sensitivity_concentrations", "sensitivity_min_R2", "sensitivity_permutation_repeats"),
          "Preference weights, tie/close margins and diagnostic settings. Configured weights are normalized before use.",
          "dopant_scoring(); run_integrated_sensitivity(); score_normalization.csv")
    entry("Generated starting structures", settings("make_dft_starting_structures", "structure_target_pct"),
          "Unrelaxed atomic substitutions; the manifest records actual commensurate concentration.",
          "write_starting_structures(); starting_structure_manifest.csv")
    entry("Fixed numerical conventions", {"bulk_root_bracket_eV": [-3., 3.], "bulk_bisection_steps": 70,
          "fermi_eta_grid": [-20., 80., 5001], "quadrature_nodes": 240, "quadrature_upper_squared": 130.,
          "csv_significant_digits": 17, "transport_check_rtol": 1e-10, "transport_check_atol": 0.},
          "Numerical resolution and arithmetic checks; saved precision does not imply physical accuracy.",
          "fermi_table(); fermi_half(); solve_holes(); csv_out(); write_transport_physics_audit()")
    parameters = pd.DataFrame(params)
    csv_out(parameters, out / "calculation_parameters.csv")
    (out / "CALCULATION_REFERENCE.txt").write_text(
        concise_calculation_text(cfg, PHYSICS), encoding="utf-8")
    return basis, parameters


def compact_report_html(out, inline=False, full=False):
    """Essential results first; all existing tables and calculation provenance kept.

    The notebook and START_HERE views use the same table data. Expandable method
    notes show native-HTML equations and a short input guide. Full methodological
    tables remain linked in the ZIP. FULL_REPORT retains every plot and saved file.
    """
    from html import escape
    out = Path(out)
    style = """
    .sno-report {font:14px/1.45 system-ui,sans-serif;color:#fff!important;background:#111827!important;
      padding:18px;border-radius:10px;overflow-wrap:anywhere}
    .sno-report * {color:#fff!important}
    .sno-report table {border-collapse:collapse;width:100%;font-size:13px;background:#111827!important}
    .sno-report th,.sno-report td {padding:7px 9px;border:1px solid #4b5563;text-align:left;vertical-align:top}
    .sno-report th {background:#374151!important}
    .sno-report tr:nth-child(even) td {background:#1f2937!important}
    .sno-report h2 {font-size:19px;margin:24px 0 9px}.sno-report h3 {font-size:15px;margin:16px 0 7px}
    .sno-report p {margin:8px 0}.sno-report a {text-decoration:underline}
    .sno-report img {max-width:100%;height:auto}.sno-report .scroll {overflow-x:auto}
    .sno-report details {margin:12px 0;border:1px solid #4b5563;padding:10px;border-radius:6px}
    .sno-report summary {cursor:pointer;font-weight:600}.sno-report section {margin:20px 0}
    .sno-report .status {border-left:3px solid #93c5fd;padding-left:10px}
    .sno-report .calculation-guide {max-width:900px}
    .sno-report .equation-list {margin:12px 0}
    .sno-report .equation-row {padding:10px 12px;margin:8px 0;background:#1f2937;
      border-left:3px solid #93c5fd;border-radius:4px;break-inside:avoid}
    .sno-report .equation {font:22px/1.65 Georgia,"Cambria Math",serif;
      overflow-x:auto;overflow-wrap:normal;white-space:nowrap;margin:3px 0}
    .sno-report .equation sub,.sno-report .equation sup {font-size:70%;line-height:0}
    .sno-report .equation-row p {margin:2px 0;font-size:13px}
    .sno-report .symbols {padding:8px 0;line-height:1.7}
    .sno-report .guide-inputs,.sno-report .guide-activation {table-layout:auto;min-width:360px}
    .sno-report .guide-files {font-size:12px}
    @media (max-width:600px) {.sno-report {padding:10px}
      .sno-report .equation {font-size:19px}.sno-report th,.sno-report td {padding:6px}}
    """
    def read(name): return pd.read_csv(find_result_file(out, name), float_precision="round_trip")
    def render(frame):
        return '<div class="scroll">' + frame.to_html(index=False, escape=True,
            float_format=lambda x: f"{x:.4g}", na_rep="—") + '</div>'
    def table(name, title, frame=None):
        frame = read(name) if frame is None else frame
        link = "" if inline else (' <a href="' + escape(find_result_file(out, name).relative_to(out).as_posix(), quote=True)
                                 + '">CSV</a>')
        return '<h3>' + escape(title) + link + '</h3>' + render(frame)
    summary = json.loads(find_result_file(out, "compact_summary.json").read_text())
    physics = json.loads(find_result_file(out, "transport_physics_audit.json").read_text())
    counts = read("structure_counts.csv")
    body = ['<h2>Structure and data counts</h2>']
    for group, part in counts.groupby("group", sort=False):
        body.append('<h3>' + escape(group) + '</h3>' + render(part[["structure_type", "count"]].rename(
            columns={"structure_type": "Structure type", "count": "Count"})))
    body.append('<p>Target-dopant ternaries are a subset. Generated entries are unrelaxed, excluded from database/ML counts; '
                'virtual prediction rows are not additional structures.</p>')
    body.append(table("ml_sample_counts.csv", "Training and testing samples", read("ml_sample_counts.csv").rename(
        columns={"stage": "Dataset part", "samples": "Structures", "unique_compositions": "Unique compositions"})))
    body.append('<p>CV uses development rows; final refit uses all ML rows after testing. '
                'Only bulk compound formation energy is learned; transport uses physical scenarios.</p>')
    for stem in ["dataset_summary", "dopant_distribution", "gradient_boosting_metrics"]:
        body.append(table(stem + ".csv", TABLE_TITLES[stem], read(stem + ".csv").rename(columns=DISPLAY_NAMES)))
    body.append(f'<h2>Dopant results at {summary["substitution_pct"]:g}% substitution</h2>')
    body.append(table("main_dopant_results.csv", "Central results", read("main_dopant_results.csv").rename(
        columns={"Mobility (cm2/Vs)": "Mobility (cm²/(V·s))", "Hole density (cm-3)": "Hole density (cm⁻³)",
                 "Resistivity (ohm m)": "Resistivity (Ω·m)"})))
    for stem in ["dopant_score_summary", "concentration_central_scores_independent"]:
        body.append(table(stem + ".csv", TABLE_TITLES[stem], read(stem + ".csv").rename(columns=DISPLAY_NAMES)))
    body.append('<p class="status">Selected candidate(s): <strong>'
                + escape(", ".join(summary["selected_dopants"]) or "none") + '</strong> ('
                + escape(summary["selection_status"].replace("_", " ")) + '). '
                'Scores depend on stated weights and same-concentration normalization.</p>')
    report = json.loads(find_result_file(out, "selection_interpretation.json").read_text())
    body.append('<details><summary>Selection reasons and trade-offs</summary>'
                + selection_interpretation_html(report) + '</details>')
    body.append('<p>Numerical transport checks: ' + str(physics["checks_passed"]) + ' passed; '
                + str(physics["checks_failed"]) + ' failed; ' + str(physics["checks_not_evaluated"])
                + ' not evaluated. These checks establish unit consistency, not measured accuracy.</p>')
    body.append('<details><summary>Uncertainty ranges and valid-draw counts</summary>')
    for filename, title in [("property_uncertainty_at_reference.csv", "Property uncertainty"),
                            ("score_uncertainty_at_reference.csv", "Score uncertainty")]:
        frame = read(filename)[["dopant", "efficiency_mode", "property", "unit", "central_value",
                                "p05", "p50", "p95", "valid_draws", "total_draws", "valid_draw_fraction"]]
        body.append(table(filename, title, frame.rename(columns={"dopant": "Dopant", "efficiency_mode": "Efficiency mode",
            "property": "Property", "unit": "Unit", "central_value": "Central", "p05": "5th percentile",
            "p50": "Median", "p95": "95th percentile", "valid_draws": "Valid draws", "total_draws": "Total draws",
            "valid_draw_fraction": "Valid fraction"})))
    body.append('</details>')
    body.append(concise_calculation_html(out, inline=inline))
    body.append('<h2>Sensitivity and figures</h2><p>Weight sensitivity varies preferences; input sensitivity varies '
                'uncertain inputs at fixed weights. Blue/orange = shared/separate efficiencies; ties split wins, '
                'invalid trials remain in denominators. Shares are scenario frequencies. Uncertainty plots show medians and '
                '5–95% ranges of usable draws, with central values separately marked; these are not calibrated confidence intervals.</p>')
    if not inline:
        paths = sorted((out / "02_figures").glob("*.png")) if full else [
            find_result_file(out, name) for name in summary["display_figures"]]
        for path in paths:
            pdf = path.with_suffix(".pdf")
            link = ('<a href="' + escape(pdf.relative_to(out).as_posix(), quote=True) + '">Vector PDF</a>') if pdf.exists() else ""
            body.append('<section><img loading="lazy" src="' + escape(path.relative_to(out).as_posix(), quote=True)
                        + '" alt="' + escape(path.stem, quote=True) + '"><p>' + link + '</p></section>')
        body.append('<p><a href="FULL_REPORT.html">All figures and files</a> · '
                    '<a href="FILE_INDEX.csv">File index</a> · <a href="COLUMN_GUIDE.csv">Column/unit guide</a></p>')
        if full:
            index = read("FILE_INDEX.csv")
            for folder, part in index.groupby(index.relative_path.str.split("/").str[0], sort=False):
                body.append('<details><summary>' + escape(folder) + '</summary><ul>')
                for row in part.itertuples():
                    body.append('<li><a href="' + escape(row.relative_path, quote=True) + '">'
                                + escape(Path(row.relative_path).name) + '</a> — ' + escape(row.description) + '</li>')
                body.append('</ul></details>')
    return '<style>' + style + '</style><div class="sno-report">' + "\n".join(body) + '</div>'


def write_simple_start_page(out):
    """Write compact views and the complete file/figure index from one renderer."""
    out = Path(out)
    fragment = compact_report_html(out, inline=True)
    pages = {"START_HERE.html": compact_report_html(out),
             "FULL_REPORT.html": compact_report_html(out, full=True)}
    # Render every view before writing, so a rendering error cannot replace one
    # view with incomplete content while leaving another apparently successful.
    (out / "NOTEBOOK_SUMMARY.html").write_text(fragment, encoding="utf-8")
    for name, content in pages.items():
        document = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                    '<meta name="viewport" content="width=device-width,initial-scale=1">'
                    '<title>SnO results</title></head><body style="max-width:1250px;margin:24px auto;'
                    'padding:0 12px;background:#111827;color:#fff">' + content + '</body></html>')
        (out / name).write_text(document, encoding="utf-8")


def display_inline_results(out, cfg, tables=None):
    if not cfg["display_results_inline"]: return
    from IPython.display import display, HTML, Image
    out=Path(out)
    display(HTML((out/"NOTEBOOK_SUMMARY.html").read_text(encoding="utf-8")))
    summary=json.loads(find_result_file(out,"compact_summary.json").read_text())
    for name in summary["display_figures"]:
        display(Image(filename=str(find_result_file(out,name))))


def main(config=None):
    global RESULT_TABLES
    cfg = deepcopy(CONFIG if config is None else config)
    validate_score_config(cfg)
    validate_extensions(cfg)
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
    print("Progress: Selecting the SnO host and filtering formation-energy training data...")
    data, host, structure = curate(raw, out, cfg)
    write_dataset_summary(data, raw, out)
    columns = get_feature_columns(data, cfg)
    csv_out(data[["material_id", "formula", "A"]+columns], out / "X_features_and_groups.csv")
    csv_out(data[["material_id", "formation_energy_eV_atom"]], out / "y_target.csv")
    print("Progress: Training and testing the bulk formation-energy model...")
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
    print("Progress: Plotting saved property and score uncertainty samples...")
    make_uncertainty_figures(screen, score_details, cfg, out)
    print("Progress: Checking selection sensitivity and making figures...")
    run_integrated_sensitivity(screen, all_scores, cfg, out)
    selection_sensitivity_all_dopants(screen, cfg, out)
    write_clear_counts(data, raw, cfg, out)
    print("Progress: Plotting all-dopant bulk conductivity and checking transport units...")
    plot_bulk_conductivity_all_dopants(screen, cfg, out)
    plot_na_k_conductivity_ratio(screen, cfg, out)
    write_transport_physics_audit(screen, cfg, out)
    write_compact_summary(screen, selection, cfg, out)
    write_calculation_reference(cfg, out)
    write_extension_notes(out, cfg)
    write_report(data, screen, robust, chosen, reliable, validation, selection, cfg, out)
    summary_path = out / "run_summary.txt"
    summary_path.write_text(summary_path.read_text() +
        f"SnO screening dopants: {', '.join(DOPANTS)}.\n"
        "Calculation inputs, logic and sources: CALCULATION_REFERENCE.txt.\n",
        encoding="utf-8")
    export_workflow_source(out)
    organize_reader_package(out, cfg)
    write_simple_start_page(out)
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
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="", help="Optional JSON of CONFIG overrides; never include an API key.")
    parser.add_argument("--result-manifest", default="", help="Optional path for successful output locations.")
    args = parser.parse_args()
    settings = deepcopy(CONFIG)
    if args.config:
        overrides = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if set(overrides)-set(settings):
            raise ValueError("Unknown CONFIG keys: " + ", ".join(sorted(set(overrides)-set(settings))))
        for name,value in overrides.items():
            if isinstance(settings[name],dict):
                settings[name].update(value)
            else:
                settings[name]=value
    RESULT_DIRECTORY, RESULT_ZIP = main(settings)
    if args.result_manifest:
        write_json(Path(args.result_manifest),dict(directory=str(RESULT_DIRECTORY),zip=str(RESULT_ZIP)))
'''

# ---------------------- SINGLE-CELL LAUNCHER ----------------------
import csv as _sno_csv
import hashlib as _sno_hashlib
import html as _sno_html
import json as _sno_json
import os as _sno_os
import platform as _sno_platform
import re as _sno_re
import subprocess as _sno_subprocess
import sys as _sno_sys
import tempfile as _sno_tempfile
import zipfile as _sno_zipfile
from datetime import datetime as _sno_datetime, timezone as _sno_timezone
from pathlib import Path as _SnoPath

SNO_BOOTSTRAP_REVISION = "5.2-sno-manuscript-figures-plus-nak-ratio"  # Isolated setup retained; SnO workflow revision 5.0.
_SNO_VERBOSE = False


def _sno_redact(text, secrets=()):
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[API_KEY_REDACTED]")
    # Hide credentials in URLs if a configured package index prints one.
    return _sno_re.sub(r"(https?://)[^/\s:@]+:[^/\s@]+@", r"\1[REDACTED]@", text)


def _sno_stream(command, log_path, environment=None, secrets=()):
    """Always save sanitized stdout/stderr; show compact progress or full logs."""
    log_path = _SnoPath(log_path)
    with log_path.open("w", encoding="utf-8") as log:
        try:
            process = _sno_subprocess.Popen(
                command, stdout=_sno_subprocess.PIPE, stderr=_sno_subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                env=environment,
            )
        except OSError as exc:
            log.write(_sno_redact(str(exc), secrets))
            raise RuntimeError(f"Could not start subprocess. See {log_path}.") from None
        try:
            for line in process.stdout:
                safe = _sno_redact(line, secrets)
                if _SNO_VERBOSE or (log_path.name == "workflow.log" and safe.startswith("Progress:")):
                    print(safe, end="", flush=True)
                log.write(safe)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except _sno_subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()
    if code != 0:
        print("\n--- LAST LINES OF THE ACTUAL ERROR ---", flush=True)
        print("".join(log_path.read_text(encoding="utf-8").splitlines(True)[-60:]))
        raise RuntimeError(
            f"Step failed (exit {code}). Full diagnostic log: {log_path}. "
            "Read the ERROR/Traceback above; the exit code alone is not the cause."
        ) from None


def _sno_child_environment():
    env = _sno_os.environ.copy()
    # Do not inherit notebook import paths or pre-existing material API secrets
    # into package installation. The configured key is added ONLY to the workflow.
    for name in ("PYTHONPATH", "PYTHONHOME", "MP_API_KEY", "PMG_MAPI_KEY",
                 "PIP_TARGET", "PIP_PREFIX", "PIP_USER"):
        env.pop(name, None)
    env.update(PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1",
               MPLBACKEND="Agg", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    return env


def _sno_create_and_seed_environment(fresh, diagnostics, environment):
    """Create a pip-less venv and seed pip WITHOUT invoking ensurepip.

    Main route: existing notebook pip manages the target via --python.
    If notebook pip is missing/too old, PyPA's official get-pip.py is run
    by the TARGET interpreter. Neither route installs into system Python.
    """
    fresh = _SnoPath(fresh)
    diagnostics = _SnoPath(diagnostics)
    interpreter = fresh / "bin" / "python"
    if _SNO_VERBOSE: print("Creating environment without pip (ensurepip is not used).", flush=True)
    _sno_stream([_sno_sys.executable, "-m", "venv", "--without-pip", str(fresh)],
                diagnostics / "environment_creation.log", environment)
    # Prove the executable is inside an isolated virtual environment before
    # letting any installer write packages. Never resolve the python symlink
    # and then use its resolved base-interpreter path to install packages.
    isolated_probe = (
        "import sys,site,pathlib; expected=pathlib.Path(sys.argv[1]).resolve(); "
        "assert pathlib.Path(sys.prefix).resolve()==expected, 'Wrong target prefix'; "
        "assert sys.prefix != sys.base_prefix, 'Target is not a virtual environment'; "
        "cfg=(expected/'pyvenv.cfg').read_text().lower(); "
        "assert 'include-system-site-packages = false' in cfg, 'System packages are exposed'; "
        "assert not site.ENABLE_USER_SITE, 'User site-packages are exposed'; "
        "print('Isolated target verified:',sys.prefix)"
    )
    _sno_stream([str(interpreter), "-c", isolated_probe, str(fresh)],
                diagnostics / "environment_isolation.log", environment)
    host = _sno_subprocess.run(
        [_sno_sys.executable, "-m", "pip", "--help"], env=environment,
        stdout=_sno_subprocess.PIPE, stderr=_sno_subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    (diagnostics / "host_pip_probe.log").write_text(_sno_redact(host.stdout), encoding="utf-8")
    install_flags = ["--disable-pip-version-check", "--progress-bar", "off",
                     "--timeout", "60", "--retries", "3"]
    if host.returncode == 0 and "--python" in host.stdout:
        method = "existing_notebook_pip_with_explicit_target_python"
        if _SNO_VERBOSE: print("Seeding pip into the isolated environment using pip --python.", flush=True)
        # --python is a pip GLOBAL option, so it must appear BEFORE install.
        _sno_stream([_sno_sys.executable, "-m", "pip", "--python", str(interpreter),
                     "install", "--upgrade", *install_flags, "pip", "setuptools", "wheel"],
                    diagnostics / "pip_bootstrap.log", environment)
    else:
        method = "official_PyPA_get_pip_executed_by_target_python"
        if _SNO_VERBOSE: print("Notebook pip is unavailable or lacks --python; using PyPA's official bootstrap.", flush=True)
        bootstrap = diagnostics / "get-pip.py"
        download = (
            "import urllib.request,urllib.parse,pathlib,hashlib,sys; "
            "url='https://bootstrap.pypa.io/get-pip.py'; "
            "request=urllib.request.Request(url,headers={'User-Agent':'SnO-Colab-bootstrap/2'}); "
            "response=urllib.request.urlopen(request,timeout=60); "
            "endpoint=urllib.parse.urlparse(response.geturl()); "
            "assert endpoint.scheme=='https' and endpoint.hostname=='bootstrap.pypa.io', "
            "'Unexpected bootstrap download origin'; "
            "data=response.read(8*1024*1024+1); response.close(); "
            "assert 10000<len(data)<=8*1024*1024, 'Unexpected bootstrap size'; "
            "compile(data.decode('utf-8'), 'get-pip.py', 'exec'); "
            "pathlib.Path(sys.argv[1]).write_bytes(data); "
            "print('Downloaded official PyPA bootstrap; SHA256:',hashlib.sha256(data).hexdigest())"
        )
        _sno_stream([str(interpreter), "-c", download, str(bootstrap)],
                    diagnostics / "bootstrap_download.log", environment)
        _sno_stream([str(interpreter), str(bootstrap), *install_flags,
                     "pip", "setuptools", "wheel"], diagnostics / "pip_bootstrap.log", environment)
    (diagnostics / "bootstrap_method.txt").write_text(method + "\n", encoding="utf-8")
    pip_probe = (
        "import sys,pathlib,pip; "
        "assert pathlib.Path(pip.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve()), "
        "'pip was not installed inside the target environment'; "
        "print('Target pip:',pip.__version__); print('Target pip location:',pip.__file__)"
    )
    _sno_stream([str(interpreter), "-c", pip_probe],
                diagnostics / "pip_location.log", environment)
    return interpreter


def _sno_install(project, diagnostics):
    """Reuse only a validated matching environment; never reuse partial installs."""
    if _sno_sys.version_info[:2] not in ((3, 12), (3, 13)):
        raise RuntimeError(
            f"This runner supports Python 3.12/3.13; this kernel is {_sno_platform.python_version()}. "
            "Use a compatible Colab runtime; do not force-install incompatible wheels."
        )
    env = _sno_child_environment()
    identity = (SNO_BOOTSTRAP_REVISION + _sno_sys.version + _sno_sys.executable + SNO_REQUIREMENTS).encode()
    digest = _sno_hashlib.sha256(identity).hexdigest()[:16]
    env_root = project / "environments"
    env_root.mkdir(parents=True, exist_ok=True)
    marker = env_root / f"ready_{digest}.json"
    interpreter = None
    if marker.exists():
        try:
            saved = _sno_json.loads(marker.read_text(encoding="utf-8"))
            candidate = _SnoPath(saved["python"])
            if candidate.parent.parent.parent.resolve() == env_root.resolve() and candidate.is_file():
                interpreter = candidate
        except (ValueError, KeyError, OSError):
            pass
    probe = (
        "import sys, importlib.metadata as m; "
        "import numpy,pandas,scipy,sklearn,matplotlib,joblib; "
        "from mp_api.client import MPRester; "
        "from pymatgen.core import Structure; "
        "from pymatgen.analysis.local_env import CrystalNN; "
        "from pymatgen.analysis.dimensionality import get_structure_components; "
        "from pymatgen.io.vasp import Poscar; "
        "print('Python:',sys.version); "
        "print('Required imports: PASS'); "
        "print({p:m.version(p) for p in "
        "['numpy','scipy','pandas','scikit-learn','mp-api','pymatgen','pymatgen-core']})"
    )
    if interpreter is not None:
        if _SNO_VERBOSE: print("Checking the previously successful isolated environment...", flush=True)
        try:
            _sno_stream([str(interpreter), "-m", "pip", "check"],
                        diagnostics / "pip_check.log", env)
            _sno_stream([str(interpreter), "-c", probe],
                        diagnostics / "imports.log", env)
        except RuntimeError:
            print("Prior environment failed validation; creating a new one. Old files are preserved.")
            interpreter = None
    if interpreter is None:
        fresh = _SnoPath(_sno_tempfile.mkdtemp(prefix="python_", dir=env_root))
        if _SNO_VERBOSE: print("Creating clean isolated environment:", fresh, flush=True)
        interpreter = _sno_create_and_seed_environment(fresh, diagnostics, env)
        requirements = diagnostics / "requirements-requested.txt"
        requirements.write_text(SNO_REQUIREMENTS, encoding="utf-8")
        _sno_stream([str(interpreter), "-m", "pip", "install",
                     "--disable-pip-version-check", "--progress-bar", "off",
                     "--prefer-binary", "--timeout", "60", "--retries", "3",
                     "--report", str(diagnostics / "pip-install-report.json"),
                     "-r", str(requirements)], diagnostics / "installation.log", env)
        _sno_stream([str(interpreter), "-m", "pip", "check"],
                    diagnostics / "pip_check.log", env)
        _sno_stream([str(interpreter), "-c", probe],
                    diagnostics / "imports.log", env)
        marker.write_text(_sno_json.dumps({"python": str(interpreter), "signature": digest}),
                          encoding="utf-8")
    (diagnostics / "requirements-requested.txt").write_text(SNO_REQUIREMENTS, encoding="utf-8")
    _sno_stream([str(interpreter), "-m", "pip", "freeze"],
                diagnostics / "requirements-resolved.txt", env)
    return str(interpreter)


def _sno_download(path):
    print("File ready:", path, flush=True)
    try:
        from google.colab import files
        files.download(str(path))
    except Exception as exc:
        print(f"Automatic download unavailable ({type(exc).__name__}). "
              "In Colab, use the left Files panel to download the file manually.")


def _sno_diagnostics_zip(diagnostics):
    # Allowlist text artifacts; never package environment variables or API keys.
    archive = diagnostics.parent / (diagnostics.name + ".zip")
    with _sno_zipfile.ZipFile(archive, "w", _sno_zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(diagnostics.iterdir()):
            if path.is_file() and path.suffix in (".txt", ".log"):
                zipped.write(path, arcname=path.name)
    return archive


def _sno_display_results(directory):
    # Rendering requires only standard-library modules and IPython in the parent.
    try:
        from IPython.display import display, HTML, Image
        def find(name):
            matches=list(directory.rglob(name))
            if len(matches)!=1: raise ValueError(f"Expected one {name}; found {len(matches)}")
            return matches[0]
        display(HTML(find("NOTEBOOK_SUMMARY.html").read_text(encoding="utf-8")))
        summary=_sno_json.loads(find("compact_summary.json").read_text(encoding="utf-8"))
        for name in summary["display_figures"]:
            display(Image(filename=str(find(name))))
    except Exception as exc:
        print(f"Display skipped ({type(exc).__name__}: {exc}). Open START_HERE.html in the results ZIP.")


def run_sno_one_cell(options=None, settings=None):
    global _SNO_VERBOSE
    options = dict(SNO_OPTIONS if options is None else options)
    _SNO_VERBOSE = bool(options.get("verbose_logs", False))
    settings = dict(SNO_SETTINGS if settings is None else settings)
    if any("key" in name.lower() for name in settings):
        raise ValueError("Store the API key in API_KEY, not in settings.")
    project = _SnoPath(options["project_directory"]).expanduser().resolve()
    project.mkdir(parents=True, exist_ok=True)
    timestamp = _sno_datetime.now(_sno_timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    diagnostics = project / f"diagnostics_{timestamp}"
    diagnostics.mkdir(exist_ok=False)
    diagnostic_info = {
        "bootstrap_revision": SNO_BOOTSTRAP_REVISION,
        "python_version": _sno_sys.version,
        "python_executable": _sno_sys.executable,
        "platform": _sno_platform.platform(),
        "machine": _sno_platform.machine(),
        "workflow_sha256": _sno_hashlib.sha256(SNO_WORKFLOW_SOURCE.encode()).hexdigest(),
    }
    (diagnostics / "runtime.txt").write_text(_sno_json.dumps(diagnostic_info, indent=2), encoding="utf-8")
    print("SnO v5.0 — conductivity, dopant selection and uncertainty — Python",
          _sno_platform.python_version(), flush=True)
    if _SNO_VERBOSE: print("Diagnostic folder:", diagnostics, flush=True)
    print("Stage 1/4: installation and import checks.", flush=True)
    key = ""
    child_env = None
    try:
        interpreter = _sno_install(project, diagnostics)
        source = diagnostics / "sno_integrated_workflow.py"
        compile(SNO_WORKFLOW_SOURCE, str(source), "exec")
        source.write_text(SNO_WORKFLOW_SOURCE, encoding="utf-8")
        # Full module import catches failures not covered by individual probes.
        _sno_stream([interpreter, "-c",
                     "import runpy,sys; runpy.run_path(sys.argv[1],run_name='sno_import_check'); "
                     "print('Full workflow import: PASS')", str(source)],
                    diagnostics / "workflow_import.log", _sno_child_environment())
        if options.get("setup_only"):
            print("SETUP PASSED. Set setup_only=False and run the same cell again for real MP data.")
            print("The validated environment will be reused; no key was requested.")
            return {"setup_only": True, "python": interpreter, "diagnostics": str(diagnostics)}
        settings["output_parent"] = str(project)
        settings["display_results_inline"] = False
        settings["auto_download_zip_in_colab"] = False
        config = diagnostics / "settings.json"
        config.write_text(_sno_json.dumps(settings, indent=2), encoding="utf-8")
        manifest = diagnostics / "successful_run.json"  # unique; cannot use an old successful ZIP
        child_env = _sno_child_environment()
        if not settings.get("cached_raw_json"):
            print("Stage 2/4: using the Materials Project API key configured in the code.", flush=True)
            key = API_KEY.strip()
            if not key:
                raise ValueError("Set API_KEY near the top of the code before downloading MP data.")
            child_env["MP_API_KEY"] = key
        else:
            print("Stage 2/4: using your specified raw-data cache (no API key needed).", flush=True)
        print("Stage 3/4: data, ML, SnO transport, uncertainty and sensitivity.", flush=True)
        try:
            _sno_stream([interpreter, "-u", str(source), "--config", str(config),
                         "--result-manifest", str(manifest)], diagnostics / "workflow.log",
                        child_env, secrets=(key,))
        finally:
            child_env.pop("MP_API_KEY", None)
            key = ""
        location = _sno_json.loads(manifest.read_text(encoding="utf-8"))
        result_directory = _SnoPath(location["directory"])
        result_zip = _SnoPath(location["zip"])
        if not result_directory.is_dir() or not result_zip.is_file():
            raise RuntimeError("Workflow reported success but its output files are missing.")
        # Include resolved dependencies and sanitized logs for reproducibility.
        # pip's raw JSON install report is deliberately NOT bundled (index URLs).
        with _sno_zipfile.ZipFile(result_zip, "a", _sno_zipfile.ZIP_DEFLATED) as zipped:
            for path in sorted(diagnostics.iterdir()):
                if path.is_file() and path.suffix in (".txt", ".log"):
                    zipped.write(path, arcname="00_execution_diagnostics/" + path.name)
        print("Stage 4/4: completed. Results:", result_zip, flush=True)
        if options.get("display_results", True):
            _sno_display_results(result_directory)
        if options.get("download_results", True):
            _sno_download(result_zip)
        return {"directory": str(result_directory), "zip": str(result_zip),
                "python": interpreter, "diagnostics": str(diagnostics)}
    except Exception as exc:
        safe_error = _sno_redact(f"{type(exc).__name__}: {exc}", (key,))
        (diagnostics / "failure.txt").write_text(safe_error, encoding="utf-8")
        print("\nSTOPPED:", safe_error, flush=True)
        print("No previous result is being presented as a successful new run.")
        archive = _sno_diagnostics_zip(diagnostics)
        print("If you need help, attach this diagnostic ZIP (not your API key):", archive)
        if options.get("download_results", True):
            _sno_download(archive)
        raise RuntimeError(safe_error) from None
    finally:
        key = ""
        if child_env is not None:
            child_env.pop("MP_API_KEY", None)


# In Colab the cell is __main__. When imported for offline tests, it does not run.
if __name__ == "__main__":
    SNO_LAST_RUN = run_sno_one_cell()
