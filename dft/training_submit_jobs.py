"""Use this script to submit a bunch of training jobs, each with different
scaling factor in the regularization and different opitmization starting point.
"""

from pathlib import Path
import argparse
import json
from tqdm import tqdm
import jinja2
import subprocess
import numpy as np
import re
import time

from transform import EAMTransform
import utils

np.random.seed(1)
FILE_DIR = Path(__file__).resolve().parent
DATA_DIR = FILE_DIR / "data"

###############################################################################
# JOBS SETTINGS
# --------------------

# Command line argument to submit the job or just write the job script
parser = argparse.ArgumentParser()
parser.add_argument(
    "-p0",
    "--parameter-center",
    type=str,
    help="Path to parameter file that will be the center of the regularization",
    dest="parameter_center")
parser.add_argument(
    "-pr",
    "--params_regularization",
    type=str,
    help="Path to the parameter file for the center of regularization, "
    "7-value format",
    default=DATA_DIR / "original_parameters_W0_7values.txt",
    dest="params_reg_file",
)
parser.add_argument(
    "-f",
    "--fimmatch-dir",
    type=str,
    help="Location of fim-matching results, e.g., ./results/iteration_1",
    dest="fimmatch_dir")
parser.add_argument(
    "-c",
    "--config-dir",
    type=str,
    help="Folder where the configurations and related information are stored",
    dest="config_dir",
)
parser.add_argument("-t",
                    "--target-dir",
                    help="Target directory to store the training results",
                    type=str,
                    default=None,
                    dest="target_dir")
parser.add_argument("-p",
                    "--partition",
                    type=str,
                    help="Type of LC partition to use, e.g., Borax, Dane",
                    dest="partition")
parser.add_argument("-n",
                    "--num-calcs-per-job",
                    help="Number of calculations per job",
                    dest="ncalcs_per_job")
parser.add_argument(
    "-s",
    "--submit",
    action="store_true",
    help="An option to submit jobs, otherwise just write training settings")
args = parser.parse_args()

# Limit the number of calculations to do
NCALCS_LIMIT = "all"
# Some batch calculation settings
FIMMATCH_DIR = Path(args.fimmatch_dir).resolve()  # Data directory
CONF_DIR = Path(args.config_dir).resolve()
# Folder to store all training results
if args.target_dir is None:
    TARGET_DIR = FIMMATCH_DIR / "training"
else:
    TARGET_DIR = Path(args.target_dir).resolve()
TARGET_DIR.mkdir(parents=True, exist_ok=True)
# Parameters for regularization
PARAMS_REG_FILE = Path(args.params_reg_file)

# Computation source allocation
with open(DATA_DIR / "lc_resource_info.json", "r") as f:
    lc_resource = json.load(f)
# Information of the node we will submit the jobs to
node_name = args.partition
cores_per_node = lc_resource[node_name]["cores_per_node"]
mem_per_node = lc_resource[node_name]["mem_per_node"]  # in GB
# Information about resource requirement per calculatio (not per job)
cores_per_calc = 1
mem_per_calc = 2  # in GB
# Maximum parallel job per node --- we can only request entire node
ncalcs_per_job = args.ncalcs_per_job
if ncalcs_per_job in ["all", "max"]:
    max_ncalcs_by_cores = np.floor(cores_per_node / cores_per_calc)
    max_ncalcs_by_mem = np.floor(mem_per_node / mem_per_calc)
    ncalcs_per_job = int(min([max_ncalcs_by_cores, max_ncalcs_by_mem]))
    cores_per_job = cores_per_node
    mem_per_job = mem_per_node
else:
    ncalcs_per_job = int(ncalcs_per_job)
    cores_per_job = ncalcs_per_job * cores_per_calc
    mem_per_job = ncalcs_per_job * mem_per_calc
print(f"Submitting jobs to {node_name}")
print("Number of concurrent calculations per job:", ncalcs_per_job)

###############################################################################
# CALCULATION SETTINGS
# --------------------

# Variation level 1 --- Regularization strength
scale_list = utils.scale_list.copy()
# Larger scale means weaker regularization. But scale 0 means no regularization

# Variation level 2 --- Several initial parameter guess
p0 = args.parameter_center  # Unperturbed params
# Index location of the 7 parameters in the 20 elements list of parameters
idx_7vals = [ii - 1 for ii in [1, 6, 7, 8, 9, 19, 20]]
pcenter = np.loadtxt(p0)[idx_7vals]
sign = np.sign(pcenter)
pcenter_log = np.log(np.abs(pcenter))
# Random sampling
nparams = len(pcenter)
nsamples = utils.nsamples
# Transformation to convert 7-value format to 20-value format
eam_transform = EAMTransform(str(p0), str(TARGET_DIR),
                             str(DATA_DIR / "translate.x"))

# Pick one of the following distribution to generate random initial guesses
# # Normal distribution
# samples = pcenter * np.random.normal(
#     np.ones(nparams), 0.1, size=(nsamples, nparams))
# samples = np.log(np.abs(samples))

# # Log-normal distribution
# samples = np.random.normal(pcenter_log, 0.1, size=(nsamples, nparams))

# Uniform distribution
samples = pcenter * np.random.uniform(0.5, 1.5, size=(nsamples, nparams))
samples = np.log(np.abs(samples))

# Add unperturbed parameters
samples = np.vstack([samples, pcenter_log])

# Create calculation setting files
print("Create list of all calculations....")
calc_settings_dir = TARGET_DIR / "calc_settings"
calc_settings_dir.mkdir(parents=True, exist_ok=True)

calc_settings = {}  # Dictionary to store calc setting {idx: reg, sample_idx}
settings_idx = 0  # This is used to index the job setting
for scale in tqdm(scale_list):
    # scale = 1  # Sigma scale
    if scale == 0:
        reg_name = "noreg"
    else:
        if isinstance(scale, (int, np.int64)):
            reg_name = f"reg{scale}sig"
        elif isinstance(scale, float):
            reg_name = f"reg{scale:0.1e}sig"

    # Iterate over the initial guess sample
    for ii, val in enumerate(samples):
        # Update the job settings list
        calc_settings.update(
            {settings_idx: {
                "regularization": reg_name,
                "sample_idx": ii
            }})

        # Write settings to a file
        # Target directory to store the results
        target = TARGET_DIR / reg_name / f"{ii:03d}"
        target.mkdir(parents=True, exist_ok=True)
        # Initial parameter guess
        pinit = target / "Ta0.txt"
        params = sign * np.exp(val)  # inverse log-transformation
        # Write the initial parameters into a file
        params20 = eam_transform._convert_from_7values_to_20values(params)[:20]
        np.savetxt(pinit, params20)

        # Write one job setting file
        with open(calc_settings_dir / f"{settings_idx}.sh", "w") as f:
            f.write(f"DATADEST_DIR={target}\n")
            f.write(f"P0_FILE={pinit}\n")
            f.write(f"SCALE={scale}")
        # Update the settings index
        settings_idx += 1

###############################################################################
# COLLECT CALCULATIONS TO DO
# --------------------------

# After we list all calculations, check which ones we still need to do
print("Collect calculations to do....")
all_calcs_todo = {}
calcs_completed = {}
calcs_failed = {}
for idx, job in calc_settings.items():
    reg_name = job["regularization"]
    sample_idx = job["sample_idx"]
    target = TARGET_DIR / reg_name / f"{sample_idx:03d}"
    # If the job is completed, this pickle file should be written
    check_file = target / "training_optimal_raw.pkl"
    if not check_file.exists():
        # But we also need to check if optimization fails to converge. When a
        # calculation fail, it should have tmp folder, but no .pkl file
        tmp_dir = target / "tmp"
        if tmp_dir.exists():
            # print(f"Job {job} has failed")
            calcs_failed.update({idx: job})
        else:
            all_calcs_todo.update({idx: job})
    else:
        # print(f"Job {job} is completed")
        calcs_completed.update({idx: job})
# We might not want to run all calculations, e.g., when we are debugging
if NCALCS_LIMIT in [np.inf, "all", "unlimited"]:
    calcs_todo = all_calcs_todo
else:
    calcs_todo = {
        idx: all_calcs_todo[idx]
        for idx in list(all_calcs_todo)[:NCALCS_LIMIT]
    }

# Print some information
print(len(calcs_todo), "calculations need to be done")
print(len(calcs_completed), "calculations completed")
print(len(calcs_failed), "calculations failed to converge")

# Partition the calculations into jobs
print("Partition calculations....")
njobs = int(np.ceil(len(calcs_todo) / ncalcs_per_job))
try:
    partition = np.array_split(list(calcs_todo), njobs)
except ValueError:  # When there is no more job to do
    partition = []
print(njobs, "to submit")

###############################################################################
# SUBMIT JOBS
# -----------

print("Submit jobs....")
# Job script template
# Fields that need to be filled in: ntasks, mem, jobname calc_idx_list
# I need to specify ntasks and mem because otherwise, multiple jobs can be sent
# to the same node, although each job is designed to occupy the entire node. So.
# the node will be overcrowded otherwise.
template_str = """#!/bin/bash
#SBATCH --time=2:00:00   # walltime---limit training time to 2 hours only
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks={{ ntasks }}  # number of cores
# #SBATCH --mem={{ mem }}GB  # memory
#SBATCH --job-name={{ jobname }}
#SBATCH --account=iap
#SBATCH --output=./slurm_files/slurm-%j.out

export OMP_NUM_THREADS=1
ulimit -s unlimited
USER=`whoami`

# Directories/files that we need to specify
FIMMATCH_DIR=$(realpath "$1")  # e.g. iteration_1
PARAMS_REG_FILE=$(realpath "$2")  # central parameters for regularization
SETTINGS_DIR=$(realpath "$3")  # e.g. iteration_1/training_SNAP/calc_settings

# This is the original directory where everything is stored
MY_WORKSPACE=`pwd`
DATA_DIR="$MY_WORKSPACE"/data
CONFIGS_DIR={{ configs_dir }}
CANDIDATES_DIR="$CONFIGS_DIR"
PROPERTIES_DIR="$MY_WORKSPACE"/properties

# List of calculations
idx_list=({{ calc_idx_list }})
# Function to run for one index
run_training() {
    local idx=$1
    # Settings that change per calculations
    source "$SETTINGS_DIR/$idx.sh"
    # This file contains DATADES_DIR, P0_FILE, and SCALE
    # Run
    python training.py -p0 "$P0_FILE" -pr "$PARAMS_REG_FILE" -f "$FIMMATCH_DIR" -c "$CONFIGS_DIR" -t "$DATADEST_DIR" -s "$SCALE"
}
# Iterate over the calculation list
for idx in "${idx_list[@]}";
do
    echo "Training using setting $idx"
    source "$SETTINGS_DIR/$idx.sh"
    run_training "$idx" >"$DATADEST_DIR/log.out" 2>&1 &
done
wait  # Wait for all background jobs to finish

# Change the directory back to where I started
cd $MY_WORKSPACE

echo "All Done!"
"""
template = jinja2.Template(template_str)

# Write jobscript
# Fields that need to be filled in: ntasks, mem, jobname calc_idx_list

# Iterate over the job partition
job_ids = []
for ii, part in enumerate(partition):
    jobname = f"batch_training_fim_matching_EAM_{FIMMATCH_DIR.name}_{ii}"
    calc_idx_list = " ".join(map(str, part))

    job_script = template.render(
        ntasks=cores_per_job,
        mem=mem_per_job,  # Give a small extra buffer
        jobname=jobname,
        calc_idx_list=calc_idx_list,
        configs_dir=CONF_DIR)
    with open("training.sh", "w") as f:
        f.write(job_script)

    if args.submit:
        # Submit job
        print("Submitting fim-matching EAM training jobs")
        process = subprocess.run(
            "sbatch training.sh "
            f"{FIMMATCH_DIR} {PARAMS_REG_FILE} {calc_settings_dir}",
            shell=True,
            capture_output=True,
            text=True)
        # Extract job ID using regular expression
        job_id_match = re.search(r"Submitted batch job (\d+)", process.stdout)
        if job_id_match:
            job_id = int(job_id_match.group(1))
            job_ids.append(job_id)
            print(f"Submitted batch job {job_id}")
        else:
            print("Failed to extract job ID")
        time.sleep(0.1)

# Write the job ids to a file
with open(TARGET_DIR / "job_ids.txt", "w") as f:
    for job_id in job_ids:
        f.write(f"{job_id}\n")
