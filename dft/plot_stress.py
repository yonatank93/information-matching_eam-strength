"""Plot the stress prediction and uncertainty across subcases to compare them.
"""

from pathlib import Path
from glob import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt

from utils import extract_strength_simulation_data

subcases = {
    "Energy": {
        "dir": "results/DFT/DFT_nocorr_energy_max-frobenius",
        "lambda_reg": 1.0,
        "color": "tab:blue"
    },
    "Forces": {
        "dir": "results/DFT/DFT_nocorr_forces_max-frobenius",
        "lambda_reg": 1.0,
        "color": "tab:orange"
    },
    "Peratom\nforces": {
        "dir": "results/DFT/DFT_nocorr_peratomforces_max-frobenius",
        "lambda_reg": 1.0,
        "color": "tab:green"
    },
    "Energy +\nForces": {
        "dir": "results/DFT/DFT_nocorr_energy-forces_max-frobenius",
        "lambda_reg": 1.0,
        "color": "tab:red"
    },
    "Energy +\nPeratom\nforces": {
        "dir": "results/DFT/DFT_nocorr_energy-peratomforces_max-frobenius",
        "lambda_reg": 0.1,
        "color": "tab:purple"
    }
}

# Collect results
for key, val in subcases.items():
    DIR = Path(val["dir"])

    # Get the directory for the last iteration
    niter = len(glob(str(DIR / "iteration_*")))
    RES_DIR = DIR / f"iteration_{niter}_"
    PREV_RES_DIR = DIR / f"iteration_{niter-1}_"

    # Retrieve the prediction
    thermo_data = extract_strength_simulation_data(RES_DIR / "out.deform",
                                                   key="all")
    pred = np.mean(thermo_data[-1500:, 5]) / 1e4

    # Retrieve the uncertainty
    fim_cand = np.load(RES_DIR / "fim_environments_optimal.npy")
    jac_str = np.load(RES_DIR / "FIM_target" / "jacobian_strength.npy")
    uncert = np.sqrt((jac_str @ np.linalg.pinv(fim_cand) @ jac_str.T)[0, 0])
    # Inflating uncertainty
    with open(PREV_RES_DIR / "training" / "optimal_training_results.pkl",
              "rb") as f:
        opt = pickle.load(f)
    res = opt[val["lambda_reg"]]["info"]["fun"][:-7]  # Data residual
    T = np.sum(res**2) / 7
    uncert_inflate = np.sqrt(T) * uncert

    # Store into the dictionary
    val.update({"prediction": pred, "uncertainty": uncert_inflate})

# Plot
plt.figure(figsize=(6.4, 6.4))
plt.axvspan(-1.5, -0.5, color="k", alpha=0.1, lw=0)
for ii, (key, val) in enumerate(subcases.items()):
    p = val["prediction"]
    u = val["uncertainty"]
    # Prediction with error bar for each subcase
    plt.errorbar(ii,
                 p,
                 u,
                 fmt="o",
                 capsize=12,
                 ms=10,
                 lw=2,
                 capthick=2,
                 color=val["color"])

    # Squash all error bars to the left to better see if there is an overlap
    plt.errorbar(-1,
                 p,
                 u,
                 fmt="o",
                 capsize=12,
                 ms=10,
                 lw=2,
                 capthick=2,
                 color=val["color"])

# There is a small overlap of the error bars
# plt.axhspan(2.1897898676589898, 2.377001790202333, color="k", lw=0)

plt.xlim(left=-1.5)
plt.xticks(range(len(subcases)),
           list(subcases),
           fontsize=16,
           rotation=90,
           ha="right",
           va="center",
           rotation_mode="anchor")
plt.gca().tick_params(axis="y", labelsize=16)
plt.ylabel("Strength (GPa)", fontsize=18)
plt.tight_layout()
plt.show()
