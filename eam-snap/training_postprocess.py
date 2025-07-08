"""Postprocess the training results:
1. Collect the training results.
2. Compare the results with the same regularization strength.
3. Plot the optimal results.
"""

from pathlib import Path
import argparse
import pickle
from tqdm import tqdm
import subprocess

import numpy as np
import matplotlib.pyplot as plt

FILE_DIR = Path(__file__).parent
DATA_DIR = FILE_DIR / "data"

# Command line argument
parser = argparse.ArgumentParser("Best training EAM potential results")
parser.add_argument(
    "-p0",
    "--parameter-center",
    type=str,
    help="Path to parameter file that defines the center of the regularization",
    dest="parameter_center")
parser.add_argument(
    "-r",
    "--results_path",
    type=str,
    help="Path to the results parent directory, e.g. iteration_1/training_SNAP",
    dest="results_path")
parser.add_argument(
    "-f",
    "--fimmatch-dir",
    type=str,
    help="Location of fim-matching results, e.g., ./results/iteration_1",
    dest="fimmatch_dir")
parser.add_argument("-t",
                    "--target",
                    type=str,
                    help="Target path to copy the optimal parameters to",
                    dest="target")
parser.add_argument("-s",
                    "--show",
                    action="store_true",
                    help="An option for showing relevant plots")

args = parser.parse_args()
RES_DIR = Path(args.results_path).resolve()

# Load W0
params_index = [1, 6, 7, 8, 9, 19, 20]  # Index of EAM parameters, base 1
W0_all = np.loadtxt(args.parameter_center)
W0 = W0_all[[ii - 1 for ii in params_index]]
sign = np.sign(W0)

# Iterables that I used
# Sigma scaling factor
# scale_list = list(range(1, 31))
# small_scale_list = np.logspace(0, -3, 13)  # Small scale < 1.0
scale_list = list(np.arange(0, 31, 10)[1:])
# small_scale_list = np.logspace(0, -3, 4)
# # Combine
# scale_list = sorted(list(set(list(scale_list[1:]) + list(small_scale_list))))
scale_list.append("inf")  # No regularization case
# Random initial starting points
nsamples = 101

# Collect the training result
print("Collecting training results...")
training_results = {}
for scale in tqdm(scale_list, desc="Scale"):
    if scale == "inf":
        dirpath = RES_DIR / "noreg"
    else:
        if isinstance(scale, (int, np.int64)):
            dirpath = RES_DIR / f"reg{scale}sig"
        elif isinstance(scale, float):
            dirpath = RES_DIR / f"reg{scale:0.1e}sig"
    training_results.update({scale: {}})
    for folder in dirpath.iterdir():
        isample = int(folder.name)
        sample_dir = dirpath / f"{isample:03d}"
        if sample_dir.exists():
            # We have generated the sample. Now, we just need to check if the
            # training is completed.
            # Check
            check_file = sample_dir / "training_optimal_raw.pkl"
            if check_file.exists():
                # Training has been completed. Load the data.
                with open(check_file, "rb") as f:
                    opt = pickle.load(f)
                # Parse
                opt_x = opt.x
                opt_cost = opt.cost  # least squares cost with regularization
                # Save
                training_results[scale].update(
                    {isample: {
                        "x": opt_x,
                        "cost": opt_cost,
                        "info": opt
                    }})

# Compare the results
print("Comparing optimal results across different starting points...")
opt_training_results = {}
for scale in tqdm(scale_list, desc="Scale"):
    if len(training_results[scale]) > 0:
        # We have some training results. So, we should run the comparison.
        # Collecting the sample index and the cost. The sample index is needed
        # because not all optimization with different starting points are
        # completed.
        compare_values = np.array(
            [[ii, value["cost"]]
             for ii, value in training_results[scale].items()])

        # Comparison -- Take the one with the lowest cost
        opt_idx = np.argmin(compare_values[:, 1])
        opt_key = int(compare_values[opt_idx][0])
        print(
            f"Most optimal results for sigma scaling {scale} is from key {opt_key}"
        )
        save_result = {
            "starting_point_idx": opt_key,
            **training_results[scale][opt_key]
        }
        opt_training_results[scale] = save_result
# Export the optimal training results
with open(RES_DIR / "optimal_training_results.pkl", "wb") as f:
    pickle.dump(opt_training_results, f)

# Parameter samples --- Optimal parameters obtained from different starting points
print("Collecting all optimal parameters from different starting points...")
params_samples = {}
for scale in tqdm(scale_list, desc="Scale"):
    if scale in opt_training_results:
        # We have the optimal training results. Collect the parameters.
        params_samples[scale] = sign * np.exp(
            [val["x"] for val in training_results[scale].values()])
# Cost samples --- Optimal cost obtained from different starting points
print("Collecting all optimal cost from different starting points...")
cost_samples = {}
for scale in tqdm(scale_list, desc="Scale"):
    if scale in opt_training_results:
        # We have the optimal training results. Collect the cost.
        cost_samples[scale] = np.array(
            [val["cost"] for val in training_results[scale].values()])

# Plot
# Cost function
print("Plot the optimal training results")
# Collect the parameters data
opt_params = sign * np.exp([val["x"] for val in opt_training_results.values()])
# Get the plotting data
lambdas = list(opt_training_results)
# Collect cost data
cost = np.array([val["cost"] for val in opt_training_results.values()])
# Decompose the cost into the data contribution and regularization contribution
nconfigs = len(
    np.loadtxt(Path(args.fimmatch_dir) / "optimal_weights_without_zeros.txt"))
# nconfigs = 7
npreds = 3 * nconfigs  # I need to hardcode the number of predictions/residuals
cost_data = []
cost_reg = []
for val in opt_training_results.values():
    residuals = val["info"].fun
    cost_data = np.append(cost_data,
                          0.5 * np.linalg.norm(residuals[:npreds])**2)
    cost_reg = np.append(cost_reg, 0.5 * np.linalg.norm(residuals[npreds:])**2)

if args.show:
    # Cost
    xtick_labels = [f"{l:0.1e}" for l in lambdas[:-1]] + ["inf"]
    plt.figure()
    plt.plot(cost, "k", lw=2, zorder=10, label="Total")
    for ii, scale in enumerate(cost_samples):
        cost_plot_data = cost_samples[scale]
        plt.plot([ii] * len(cost_plot_data),
                 cost_plot_data,
                 "r.",
                 ms=10,
                 mew=0,
                 alpha=0.25)
        plt.axvline(ii, c="gray", lw=0.5, zorder=-10)
    plt.plot(cost_data, label="Data contribution")
    plt.plot(cost_reg, label="Regularization")
    # plt.yscale("log")
    plt.xticks(range(len(lambdas)), xtick_labels, rotation=90)
    plt.ylim(0, max(cost_data) * 1.1)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Cost")
    plt.tight_layout()
    plt.legend()

    # Plot ylim
    ylim_pad_scale = 0.2  # Padding to give to ylim
    min_opt = np.min(opt_params, axis=0)
    max_opt = np.max(opt_params, axis=0)
    # Lower limit
    ylim_lo_W0 = W0 - np.abs(W0 * ylim_pad_scale)
    ylim_lo_opt = min_opt - np.abs(min_opt * ylim_pad_scale)
    ylim_lo = np.min(np.row_stack((ylim_lo_W0, ylim_lo_opt)), axis=0)
    # Upper limit
    ylim_hi_W0 = W0 + np.abs(W0 * ylim_pad_scale)
    ylim_hi_opt = max_opt + np.abs(max_opt * ylim_pad_scale)
    ylim_hi = np.max(np.row_stack((ylim_hi_W0, ylim_hi_opt)), axis=0)
    ylim_list = np.row_stack((ylim_lo, ylim_hi)).T

    for ii, pidx in enumerate(params_index):
        plt.figure()
        plt.axhline(W0[ii], ls="--", c="k")
        plt.axhspan(0.9 * W0[ii], 1.1 * W0[ii], color="k", alpha=0.3, ec=None)
        plt.plot(opt_params[:, ii], "k", lw=2, zorder=10)
        # Plot parameter distribution
        for jj, scale in enumerate(cost_samples):
            params_plot_data = params_samples[scale][:, ii]
            plt.plot([jj] * len(params_plot_data),
                     params_plot_data,
                     "r.",
                     ms=10,
                     mew=0,
                     alpha=0.25)
            plt.axvline(jj, c="gray", lw=0.5, zorder=-10)
        plt.xticks(range(len(lambdas)), xtick_labels, rotation=90)
        plt.ylim(ylim_list[ii])
        plt.xlabel(r"$\lambda$")
        plt.ylabel(f"Parameter {pidx}")
        plt.tight_layout()

    plt.show()

# Optional --- Copy the optimal results to some target directory (e.g., shared
# directory)
target_parent = args.target
# Files that we want to copy
files = ["EAM_code", "Wopt.txt", "potentials/Wopt.eam.alloy"]
if target_parent is not None:
    target_parent = Path(target_parent)

    for key, val in tqdm(opt_training_results.items()):
        # The source folder naming scheme is a bit different
        if key == "inf":  # No regularization
            source_folder = RES_DIR / "noreg"
        elif isinstance(key, int):  # This is when lambda is integer
            source_folder = RES_DIR / f"reg{key}sig"
        else:  # For other cases with float lambda
            source_folder = RES_DIR / f"reg{key:0.1e}sig"
        # For each lambda, this is the starting point index of the best result
        sidx = val["starting_point_idx"]
        source = source_folder / f"{sidx:03d}"
        # print(source)

        # Target directory
        if isinstance(key, float):  # This is when lambda is a float
            target_folder = target_parent / f"reg_lam_{key:0.1e}"
        else:  # For other cases
            target_folder = target_parent / f"reg_lam_{key}"
        # We might want to add additional child directory
        additional_folder = "optimal_multiple_starts"
        # additional_folder = "optimal_start_W0"
        target = target_folder / additional_folder
        target.mkdir(parents=True, exist_ok=True)
        # print(target)

        # Copy
        for ff in files:
            subprocess.run(f"cp {source}/{ff} {target}", shell=True)

    # Finally, we need to make sure everyone can access these files in the
    # shared directory
    subprocess.run(f"chmod -R 777 {target_parent}", shell=True)
