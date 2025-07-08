"""This script uses the results exported from fim_target.py and
fim_candidates.py to do information-matching calculation to find the optimal
weights for the candidate environments.
"""

from pathlib import Path
import argparse
from glob import glob
import pickle

import numpy as np
import cvxpy as cp
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

# Load the FIMs
print("Load the FIMs")
# Target FIM
fim_target = np.load(target_dir / "FIM_target" / "fim_target.npy")
jac_target = np.load(target_dir / "FIM_target" / "jacobian_target.npy")
cov_target = np.load(target_dir / "FIM_target" / "cov_target.npy")
# Index of tunable parameters, base 1
idx_tunable = [1, 6, 7, 8, 9, 19, 20]
nparams = len(idx_tunable)
# Load the candidate identifier
ncandidates = len(glob(str(fim_cand_dir / "fim_*.npy")))
candidates_identifier = np.arange(ncandidates)
# Load the FIM candidates
fim_cand = np.empty((ncandidates, nparams, nparams))
for ii in range(ncandidates):
    fim_file = fim_cand_dir / f"fim_{ii}.npy"
    fim_cand[ii] = np.load(fim_file)

# Use FIMMatchingScore implementation in the orchestrator to find the optimal
# weights. I want to use this module because it automatically scale the FIMs.
print("FIM Matching to compute the optimal weights")
# Instantiate FIMMatchingScore
fim_matching = FIMMatchingScore()
tol = 1e-8
opt_weights = fim_matching.fim_match(
    fim_target,
    fim_cand,
    # If we want to cap the weights or use l2-norm objective function,
    # uncomment the following line
    convexopt_init_kwargs={
        # "weight_upper_bound": 1e6,
        # "obj_fn": cp.norm2
    },
    solver_kwargs={
        "solver": "SDPA",
        "verbose": True,
        "epsilonStar": tol,
        "lambdaStar": 1e4,
        "numThreads": 4
    },
    weight_tolerance={
        "zero_tol": np.sqrt(tol),
        "zero_tol_dual": np.sqrt(tol)
    })
print("Optimal weights found!")

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
          np.allclose(prev_weights[:, 1], opt_weights))

# Export raw results
with open(target_dir / "fim_matching_raw_results.pkl", "wb") as f:
    pickle.dump(fim_matching.convexopt.result, f)

# Print some information that summarizes the results
idx_nonzero_weights = np.where(opt_weights > 0.0)[0]
print("Number of environments we need:", len(idx_nonzero_weights))
print("Optimal environment identifiers and their weights:")
print("Identifier \t Weight \t\t Sigma")
for idx in idx_nonzero_weights:
    print(f"{candidates_identifier[idx]} \t {opt_weights[idx]} "
          f"\t {1/np.sqrt(opt_weights[idx])}")

# Export the optimal results
# With zero weights
np.savetxt(target_dir / "optimal_weights_with_zeros.txt",
           np.vstack((candidates_identifier, opt_weights)).T)
# Without zero weights
opt_env_identifiers = candidates_identifier[idx_nonzero_weights]
opt_weights_nozeros = opt_weights[idx_nonzero_weights]
target_force_uncert = 1 / np.sqrt(opt_weights_nozeros)
np.savetxt(target_dir / "optimal_weights_without_zeros.txt",
           np.vstack((opt_env_identifiers, opt_weights_nozeros,
                      target_force_uncert)).T,
           header="The columns are: environment identifier, optimal weights, "
           "corresponding target DFT force uncertainty")
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
