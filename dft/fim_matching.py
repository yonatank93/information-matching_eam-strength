"""This script uses the results exported from fim_target.py and
fim_candidates.py to do information-matching calculation to find the optimal
weights for the candidate environments.
"""

from pathlib import Path
import argparse
import pickle
from tqdm import tqdm

import numpy as np
import cvxpy as cp
import pandas as pd
from orchestrator.computer.score.fim import FIMMatchingScore
from information_matching.convex_optimization import compare_weights
import matplotlib.pyplot as plt

FILE_DIR = Path(__file__).resolve().parent

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-w0",
                    "--prev-weights",
                    type=str,
                    default=None,
                    help="Path to optimal_weights_with_zeros.txt file "
                    "of the previous iteration",
                    dest="previous_weights")
parser.add_argument("-t",
                    "--target-dir",
                    type=str,
                    help="Target folder to write the results",
                    dest="target_dir")
parser.add_argument("-s",
                    "--show",
                    action="store_true",
                    help="An option for showing relevant plots")
args = parser.parse_args()

# Target folder to retrieve and store the results
target_dir = Path(args.target_dir)
target_dir.mkdir(exist_ok=True)
# Other directories
fim_cand_dir = target_dir / "FIM_candidates"
fim_target_dir = target_dir / "FIM_target"
# Index of tunable parameters, base 1
idx_tunable = [1, 6, 7, 8, 9, 19, 20]
nparams = len(idx_tunable)

# Load the FIMs
print("Load the FIMs")
# Target FIM
fim_target = np.load(fim_target_dir / "fim_target.npy")
jac_target = np.load(fim_target_dir / "jacobian_target.npy")
cov_target = np.load(fim_target_dir / "cov_target.npy")
# Load FIM candidates
fim_path_list = np.loadtxt(fim_cand_dir / "fim_candidates_list.txt", dtype=str)
ncandidates = len(fim_path_list)
fim_cand = [np.load(fim_cand_dir / ff) for ff in tqdm(fim_path_list)]

# Use FIMMatchingScore implementation in the orchestrator to find the optimal
# weights. I want to use this module because it automatically scale the FIMs.
print("FIM Matching to compute the optimal weights")
fim_match_file = target_dir / "fim_matching_raw_results.pkl"
tol = 1e-8
fim_match_args = dict(
    # If we want to cap the weights or use l2-norm objective function,
    # uncomment the following line
    convexopt_init_kwargs={
        # "weight_upper_bound": 1e6,
        # "obj_fn": cp.norm2
    },
    fim_preconditioning_kwargs=None,
    solver_kwargs={
        "solver": "SDPA",
        "verbose": True,
        "epsilonStar": tol,
        "lambdaStar": 1e4,
        "betaBar": 0.9,
        "isSymmetric": True
    },
    weight_tolerance={
        "zero_tol": np.sqrt(tol),
        "zero_tol_dual": np.sqrt(tol)
    })
# Instantiate FIMMatchingScore
fim_matching = FIMMatchingScore()
if fim_match_file.exists():
    print("Loading precompted FIM-matching results")
    # Instantiate convexopt solver
    fim_matching.setup_problem(fim_target, fim_cand,
                               fim_match_args["convexopt_init_kwargs"],
                               fim_match_args["fim_preconditioning_kwargs"])
    # Load
    with open(fim_match_file, "rb") as f:
        result = pickle.load(f)
    fim_matching.convexopt.result = result
    # Get the optimal weights
    opt_weights_nozeros_dict = fim_matching.convexopt.get_config_weights(
        **fim_match_args["weight_tolerance"])
    opt_weights = fim_matching._insert_zero_weights(opt_weights_nozeros_dict)

else:
    opt_weights = fim_matching.fim_match(fim_target, fim_cand,
                                         **fim_match_args)
    print("Optimal weights found!")

    # Export raw results
    with open(fim_match_file, "wb") as f:
        pickle.dump(fim_matching.convexopt.result, f)

# The new FIMMatchingScore returns weights as a list
opt_weights = np.array(opt_weights)
# Compare the optimal weights to the last results, if requested
if args.previous_weights is not None:
    print("Combine the optimal weights with those of the previous iteration")
    # First, convert the current optimal weights into a dictionary format
    cur_weights_dict = {ii: val for ii, val in enumerate(opt_weights)}
    # Then, load the previous iteration's optimal weights and convert it to a
    # dictionary
    prev_weights = np.loadtxt(args.previous_weights)
    prev_weights_dict = {int(ii): val for ii, val in prev_weights}
    # Combined weights
    comb_weights_dict = compare_weights(prev_weights_dict, cur_weights_dict)
    opt_weights = np.array(list(comb_weights_dict.values()))
    # Check if the iteration converges
    print("#" * 40)
    print("FIM-matching iteration converged:",
          np.allclose(prev_weights[:, 1], opt_weights, atol=1e-4, rtol=1e-4))

# Print some information that summarizes the results
idx_nonzero_weights = np.where(opt_weights > 0.0)[0]
print("Number of environments we need:", len(idx_nonzero_weights))
print("Optimal environment identifiers and their weights:")
print("Identifier \t Weight \t\t Sigma")
for idx in np.sort(idx_nonzero_weights):
    print(f"{idx} \t {opt_weights[idx]} "
          f"\t {1/np.sqrt(opt_weights[idx])}")

print()
print("Descriptions of these optimal environments:")
for idx in np.sort(idx_nonzero_weights):
    print(f"{idx} \t {fim_path_list[idx]}")

# Export the optimal results
# With zero weights
np.savetxt(target_dir / "optimal_weights_with_zeros.txt",
           np.vstack((np.arange(ncandidates), opt_weights)).T)
# Without zero weights
opt_env_identifiers = idx_nonzero_weights
opt_weights_nozeros = opt_weights[idx_nonzero_weights]
target_uncertainty = 1 / np.sqrt(opt_weights_nozeros)
np.savetxt(target_dir / "optimal_weights_without_zeros.txt",
           np.vstack((opt_env_identifiers, opt_weights_nozeros,
                      target_uncertainty)).T,
           header="The columns are: environment identifier, optimal weights, "
           "corresponding target DFT force uncertainty")
# Without zero weights and as a pandas dataframe
opt_weights_df = pd.DataFrame()  # Initialize dataframe
opt_weights_df["Index"] = idx_nonzero_weights  # Add index
# Add identifier
opt_weights_df["Identifier"] = [
    ".".join(fim_path_list[ii].split(".")[:-1]) for ii in idx_nonzero_weights
]
opt_weights_df["Weights"] = opt_weights_nozeros  # Add weights
opt_weights_df["Sigma"] = target_uncertainty  # Add sigma
# Export as csv file
opt_weights_df.to_csv(target_dir / "optimal_weights_without_zeros.csv",
                      index=False)

# Get the FIM of the optimal training environments by weighted sum of the
# candidate FIMs
opt_fim = np.sum(fim_cand * opt_weights.reshape((-1, 1, 1)), axis=0)
print("Eigenvalues of the optimal FIM:", np.linalg.eigvalsh(opt_fim))
# Export this optimal FIM for the training environments
np.save(target_dir / "fim_environments_optimal.npy", opt_fim)

# Print the violation, i.e., the smallest non-positive eigenvalue of the
# difference matrix -- If al eigvals are positive, then the violation is 0.
diff_mat = opt_fim - fim_target
eigvals = np.linalg.eigvalsh(diff_mat)
print("Violation:", abs(min([0, min(eigvals)])))

# Uncertainty of the target property
opt_target_var = jac_target @ np.linalg.inv(opt_fim) @ jac_target.T
print("Uncertainty of target property")
print("Target uncertainty:", np.sqrt(np.diag(cov_target)))
print("From optimal environments:", np.sqrt(np.diag(opt_target_var)))

if args.show:
    # Plot the raw weight results
    plt.figure()
    plt.plot(fim_matching.convexopt.result["wm"], label="Weight")
    plt.plot(fim_matching.convexopt.result["dual_wm"], label="Dual value")
    plt.axhline(np.sqrt(tol), ls="--", c="k", zorder=-10)
    plt.yscale("log")
    plt.legend()

    # Plot the optimal configuration FIM
    plt.figure()
    plt.imshow(opt_fim)
    plt.colorbar()
    plt.xticks(range(nparams), idx_tunable)
    plt.yticks(range(nparams), idx_tunable)
    plt.show()
