"""Compute the target properties using the optimal parameter I got from
training the EAM potential to fit the optimal data.

Note: This script computes the property prediction for a given set of
parameters.
"""

from pathlib import Path
import argparse
import json
import shutil

import numpy as np
from ase.calculators.kim import get_model_supported_species
import kimkit

from orchestrator.potential import potential_builder
from orchestrator.target_property import KIMRun
from transform import EAMTransform

# Base directories
FILE_DIR = Path(__file__).parent.resolve()
DATA_DIR = FILE_DIR / "data"

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-p0",
    "--params0_file",
    type=str,
    help="Path to Zhou parameters (20-value format)",
    dest="params0_file",
)
parser.add_argument(
    "-p",
    "--params",
    type=str,
    help="Path to the parameter file (7-value format) to evaluate",
    dest="params_file",
)
parser.add_argument(
    "-f",
    "--kimrun_files",
    nargs="+",
    help="Path to JSON files that define the KIMRUN target properties",
    dest="kimrun_files")
parser.add_argument(
    "-t",
    "--target-dir",
    type=str,
    help="Target folder to write the resulst",
    dest="target_dir",
)
args = parser.parse_args()

TARGET_DIR = Path(args.target_dir)
TARGET_DIR.mkdir(parents=True, exist_ok=True)

###############################################################################
# Parameter transformation
# ------------------------
print("Parameter transformation")
# Load the initial parameters
params0_file = args.params0_file
params_file = args.params_file
params = np.loadtxt(params_file)
nparams = len(params)

# This parameter transformation converts the 7-value format into tabulated
# values .eam.alloy format
transform = EAMTransform(params0_file,
                         str(TARGET_DIR),
                         str(DATA_DIR / "translate.x"),
                         cleanup=False)

# The output of transform.inverse_transform is an array with shape 6003. But,
# we need the parameters in dictionary format
# Inverse transform the parameters
params_eam = transform.inverse_transform(params)
params_eam_dict = {
    "cutoff": [[0], [params_eam[0]]],
    "deltaR": [[0], [params_eam[1]]],
    "deltaRho": [[0], [params_eam[2]]],
    "embeddingData": [np.arange(2000), params_eam[3:2003]],
    "rPhiData": [np.arange(2000), params_eam[2003:4003]],
    "densityData": [np.arange(2000), params_eam[4003:]],
}

print(json.dumps(transform.jsonable_kwargs, indent=4))

###############################################################################
# Potential
# ---------
print("Potential")
# Potential information dictionary
potential_dict = {
    "potential_type": "KIM",
    "potential_args": {
        "kim_id": "EAM_Dynamo_ZhouJohnsonWadley_2004_W__MO_524392058194_005",
        "kim_api":
        "/usr/gapps/iap/kim-storage/kim-api/quartz/bin/kim-api-collections-management",
        "species": [transform.symb],
        "model_driver": "EAM_Dynamo__MD_120291908751_005",
        "param_files": [str(transform.TMP_DIR / transform.eam_alloy_filename)]
    }
}

# Build initial potential
potential = potential_builder.build(
    potential_type=potential_dict["potential_type"],
    potential_args=potential_dict["potential_args"])
potential.build_potential()
potential.species = get_model_supported_species(potential.kim_id)
# Update the parameters
potential.set_params(**params_eam_dict)

# Write a new potential to a file
kim_id = potential.generate_new_kim_id("EAM_kimrun_property", "portable-model")
potential.write_potential_to_file(str(TARGET_DIR / kim_id))
potential.save_potential_to_kimkit(work_dir=TARGET_DIR)

print(json.dumps(potential_dict, indent=4))

###############################################################################
# KIMRun target properties
# ------------------------

kimrun = KIMRun()
# Iterate over target property
target_property = {}
for ff in args.kimrun_files:
    target_property_name = Path(ff).with_suffix("").name
    print("Run KIMRun target property:", target_property_name)
    # Load target property dictionary
    with open(ff, "r") as f:
        target_property_setting = json.load(f)
    # Additional settings - potential and flatten results
    target_property_setting["calculate_property_args"].update({
        "potential": kim_id,
        "flatten": True
    })
    print(json.dumps(target_property_setting, indent=4))

    # Compute
    target_property_file = TARGET_DIR / Path(target_property_name).with_suffix(
        ".npy")
    if target_property_file.exists():
        target_property_vals = np.load(target_property_file)
    else:
        target_property_vals = kimrun.calculate_property(
            **target_property_setting["calculate_property_args"]
        )["property_value"]
        np.save(target_property_file, target_property_vals)
    target_property.update({target_property_name: target_property_vals})

###############################################################################
# Clean up
# --------
# Remove the temporary potential that I just wrote in kimki
print("Removing potential from kimkit:", kim_id)
kimkit.models.delete(kim_id)
# Remove the potential file
print("Removing potential file:", TARGET_DIR / kim_id)
shutil.rmtree(TARGET_DIR / kim_id, ignore_errors=True)
