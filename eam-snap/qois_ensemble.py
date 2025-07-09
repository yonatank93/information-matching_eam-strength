"""Generate prediction ensemble, given several eam.alloy parameter files. The
predictions are calculated using KIMRun.
"""

from pathlib import Path
from glob import glob
import json
import shutil
import pickle

import numpy as np
import pandas as pd
import kimkit
from scipy import stats
import matplotlib.pyplot as plt
from corner import corner
import seaborn as sns

from orchestrator.potential import potential_builder
from orchestrator.target_property import KIMRun

FILE_DIR = Path(__file__).resolve().parent

###############################################################################
# SETTINGS
# ========
SYMB = "W"  # Element symbol, e.g., "W", "Ta"
# POTENTIAL_DIR = Path("/p/vast1/iap/thrust3/FIM-matching/EAM/Potentials")
POTENTIAL_DIR = Path("data/potential")
TARGET_FILE = FILE_DIR / "results" / "target_property_ensemble_W.pkl"
PROPS = [
    "properties/bcc_W_cohesive-potential-energy-cubic-crystal.json",
    "properties/bcc_W_elastic-constants-isothermal-cubic-crystal-npt.json",
    "properties/bcc_W_monovacancy-neutral-relaxed-formation-potential-energy-crystal-npt.json",
    "properties/bcc_W_monovacancy-neutral-migration-energy-crystal-npt.json",
    "properties/bcc_W_surface-energy-cubic-crystal-npt.json",
    "properties/bcc_W_linear-thermal-expansion-coefficient-cubic-crystal-npt.json"
]
NPREDS = [2, 3, 1, 1, 4, 1]  # Number of predictions per KIM test

# Precalculation setup
# --------------------
# Make Target directory
TARGET_DIR = TARGET_FILE.parent
TARGET_DIR.mkdir(exist_ok=True, parents=True)
print("List of target properties:")
[print("\t* " + P) for P in PROPS]
# Get the potential ensemble
POTENTIAL_FILES = sorted(glob(str(POTENTIAL_DIR / f"{SYMB}*.eam.alloy")))
print("Number of ensemble member:", len(POTENTIAL_FILES))
# Base dictionary to define the potential
potential_dict_base = {
    "potential_type": "KIM",
    "potential_args": {
        "kim_id": "EAM_Dynamo_ZhouJohnsonWadley_2004_W__MO_524392058194_005",
        "kim_api":
        "/usr/gapps/iap/kim-storage/kim-api/quartz/bin/kim-api-collections-management",
        "species": ["W"],
        "model_driver": "EAM_Dynamo__MD_120291908751_005",
        "param_files": None  # Will be replaced later
    }
}

###############################################################################
# TARGET PROPERTIES CALCULATIONS
# ==============================

# Strength
strength_file = TARGET_DIR / "strength_ensemble_W.csv"
strength = pd.read_csv(strength_file, index_col=0)

# KIM test properties
kimrun = KIMRun()

if TARGET_FILE.exists():
    with open(TARGET_FILE, "rb") as f:
        target_property = pickle.load(f)
else:
    target_property = {}

# Iterate over potential
for potfile in POTENTIAL_FILES:
    print("Potential file:", potfile)
    potential_key = Path(potfile).name
    if potential_key in target_property:
        target_property_potential = target_property[potential_key]
    else:
        target_property_potential = {}

    # Update the potential dictionary
    potential_dict = potential_dict_base.copy()
    potential_dict["potential_args"]["param_files"] = [potfile]

    # Build potential and save the potential to kimkit
    potential = potential_builder.build(
        potential_type=potential_dict["potential_type"],
        potential_args=potential_dict["potential_args"])
    potential.build_potential()
    # Write a new potential to a file
    kim_id = potential.generate_new_kim_id("EAM_kimrun_property",
                                           "portable-model")
    potential.write_potential_to_file(str(TARGET_DIR / kim_id))
    potential.save_potential_to_kimkit(work_dir=TARGET_DIR)

    # Iterate over target property
    for ff in PROPS:
        target_property_name = Path(ff).with_suffix("").name
        if not target_property_name in target_property_potential:
            print("Run KIMRun target property:", target_property_name)
            # Load target property dictionary
            with open(ff, "r") as f:
                target_property_setting = json.load(f)
            # Additional settings - potential and flatten results
            target_property_setting["calculate_property_args"].update({
                "potential":
                kim_id,
                "flatten":
                True
            })
            print(json.dumps(target_property_setting, indent=4))

            # Compute
            target_property_vals = kimrun.calculate_property(
                **target_property_setting["calculate_property_args"]
            )["property_value"]
            target_property_potential.update(
                {target_property_name: target_property_vals})

    # Collect results
    target_property.update({potential_key: target_property_potential})

    # Remove the temporary potential that I just wrote in kimki
    print("Removing potential from kimkit:", kim_id)
    kimkit.models.delete(kim_id)
    # Remove the potential file
    print("Removing potential file:", TARGET_DIR / kim_id)
    shutil.rmtree(TARGET_DIR / kim_id, ignore_errors=True)

with open(TARGET_FILE, "wb") as f:
    pickle.dump(target_property, f)

###############################################################################
# PLOT AND COVARIANCE
# ===================
print("Plot the prediction ensemble and the covariance")

kimrun_preds_names = [
    r"$a_0$",
    r"$E_{\text{coh}}$",
    r"$c_{11}$",
    r"$c_{12}$",
    r"$c_{44}$",
    r"$E_{\text{vac}}$",
    r"$E_{\text{mig}}$",
    r"$\gamma_{111}$",
    r"$\gamma_{100}$",
    r"$\gamma_{121}$",
    r"$\gamma_{110}$",
    r"$\alpha$",
]

# Convert the KIMRun property predictions into dataframe format
preds_kimrun = pd.DataFrame(columns=kimrun_preds_names)
index = []
for ii, pot in enumerate(POTENTIAL_FILES):
    pot_key = Path(pot).name
    preds_one_ensemble = []
    for jj, prop in enumerate(PROPS):
        prop_key = Path(prop).with_suffix("").name
        vals = target_property[pot_key][prop_key]
        if len(vals) == 0:
            # If any prediction has an error, it will return an empty array.
            # We should replace it with an array of NaN, but with the right
            # length
            vals = np.repeat(np.nan, NPREDS[jj])
        preds_one_ensemble = np.append(preds_one_ensemble, vals)
    index.append(pot_key)
    preds_kimrun.loc[len(preds_kimrun)] = preds_one_ensemble
preds_kimrun.index = index

# Combine the prediction dataframes
preds_ensemble = pd.concat([strength, preds_kimrun], axis=1)
# Drop NaN
preds_ensemble = preds_ensemble.dropna(axis=0)

# Remove outliers -- Defined as samples having z-score > 3, i.e., the sample
# is over 3 sigma away from the mean
k = 3
z_scores = np.abs(stats.zscore(preds_ensemble, nan_policy='omit'))
preds_ensemble = preds_ensemble[(z_scores < k).all(axis=1)]

# Plot the scatter plot of prediction ensemble
labels = preds_ensemble.columns
npreds = len(labels)
fig, _ = plt.subplots(npreds, npreds, figsize=(npreds * 1.5, npreds * 1.5))
corner(
    preds_ensemble,
    labels=labels,
    fig=fig,
    show_titles=True,
    title_fmt=".2f",
    plot_datapoints=True,  # Scatter points only
    plot_density=False,  # No density plot
    plot_contours=False,  # No contours
    no_fill_contours=True,  # No filled contours
    fill_contours=False,  # No filled density
    data_kwargs={
        "ms": 10,
        "mew": 0,
        "alpha": 0.5
    },
)
plt.savefig(TARGET_DIR / "prediction_ensemble_W.png")

# Plot prediction correlation
corr = np.corrcoef(preds_ensemble.T)

plt.figure(figsize=(10, 8))  # Adjust figure size
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",  # Show values with 2 decimals
    cmap="coolwarm",
    center=0,  # Use coolwarm color map centered at 0
    linewidths=0.5,
    square=True,  # Thin lines between cells, square cells
    cbar_kws={"shrink": 0.8},  # Shrink color bar for better fit
    xticklabels=labels,
    yticklabels=labels,  # Set custom labels
    vmin=-1,
    vmax=1,
)  # Fix color range from -1 to 1

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha="right")
plt.savefig(TARGET_DIR / "prediction_correlation_W.png")

plt.show()
