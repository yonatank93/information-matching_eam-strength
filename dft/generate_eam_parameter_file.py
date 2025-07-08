"""This Python script is a modification of Vasily's script to generate the 20
EAM parameters from the 7 we should tune.

In this script, the 7 tunable parameter values will be read from a txt file.
The format of the txt file name is "W<idx_1>_derivative_<desc>". "idx_1" is
the index of parameters based 1 in the original, 20 values, format. "desc" is
some description related to the parameter perturbation direction and magnitude.
For example, "plus1h" means the parameter is increased by 1 times the
derivative step size.

Then, this script will write "EAM_code" and "input files" and run 
`./translate.x < input` to generate eam.alloy file that will be used in LAMMPS.

We also remove all prints and checks, compared to Vasily's script.
"""

from pathlib import Path
import argparse
import subprocess
import os
import shutil
import numpy as np

###############################################################################
# Setup
# -----

# Directory
FILE_DIR = Path(__file__).resolve().parent

# Command line arguments
parser = argparse.ArgumentParser(
    "Modify EAM parameters and generate eam.alloy file")
# What arguments should we request: perturbed parameters, destination where to
# place eam.alloy file, the binary fortran code to do format conversion
parser.add_argument(
    "-p",
    "--params",
    type=str,
    help="Path to the new parameter file, 7 value format",
    dest="params_file",
)
parser.add_argument(
    "-d",
    "--destination",
    type=str,
    help="Folder destination to write the eam.alloy file",
    dest="destination",
)
parser.add_argument(
    "-b",
    "--binary_path",
    type=str,
    help="Path to the binary translate.x file",
    dest="binary_path",
)
args = parser.parse_args()

# Read the original parameter values as the baseline
# params_init_file = Path(args.params_init_file)
# params_init = np.loadtxt(params_init_file)
# Hard-coded to reduce i/o
params_init = np.array([
    2.74084,
    3.48734,
    37.234847,
    37.234847,
    8.900114,
    4.746728,
    0.882435,
    1.394592,
    0.139209,
    0.278417,
    -4.946281,
    -0.148818,
    0.365057,
    -4.432406,
    -4.96,
    0.0,
    0.661935,
    0.348147,
    0.582714,
    -4.961306,
])

# Read the parameter values to evaluate, 7 values format
params_file = Path(args.params_file)
params_7vals = np.loadtxt(params_file)
# Convert this 7 values parameter format to the original 20 values
# Can modify parameters # 1, 6, 7, 8, 9, 19, 20 (index 1)
# Index location of the 7 parameters in the 20 elements list of parameters
idx_7vals = [ii - 1 for ii in [1, 6, 7, 8, 9, 19, 20]]
params = params_init.copy()
params[idx_7vals] = params_7vals
# Set element symbol -- make it the same as the name of the parameter file
symb = params_file.with_suffix("").name

###############################################################################
# Parameter conversion
# --------------------

# Reassign parameters to symbols as used in the paper
re, fe, rhoe, rhos, alfa, beta, A, B, k, lamb = params[:10]
Fn0, Fn1, Fn2, Fn3, F0, F1, F2, F3, eta, Fe = params[10:20]
# Notes:
# fe Does not affect physical properties, never to be modified
# F1 Always kept = 0 as was done in the papers

# Enforce conditions adopted in the paper: 4(3), 5(6), 10(9), 15(20)
rhos = rhoe  # Adopted for all BCC and HCP metals in the paper
alfa = 1.875 * beta  # Adopted for all metals in the paper
lamb = 2.0 * k  # Adopted for all BCC metals in the paper
F0 = Fe  # These two parameters are always nearly identical (within ~ 0.01%) in the paper
# Never change fe (#2) and F1 (#16)

# Define rows of the linear set matrix and RHS vector
# Condition of F(0) = 0
# Fn0 - Fn1 + Fn2 - Fn3
row1 = [1, -1, 1, -1, 0, 0]
b1 = 0

# Continuity conditions at rho = 0.85*rhoe
rhon = 0.85 * rhoe
rho0 = 1.15 * rhoe
rho0_over_rhos = 1.15
r1 = -0.15
r2 = 0.15

# F
# Fn0 - (F0*r1^0 + F1*r1^1 + F2*r1^2 + F3*r1^3)
row2 = [1, 0, 0, 0, -(r1**2), -(r1**3)]
b2 = F0 + F1 * r1

# F'
# Fn1/rhon - 1/rhoe*(F1*r1^0 + 2*F2*r1^1 + 3*F3*r1^2)
row3 = [0, 1 / rhon, 0, 0, -2 * r1 / rhoe, -3 * r1**2 / rhoe]
b3 = F1 / rhoe

# F"
# 2*Fn2/rhon^2 - 1/rhoe^2*(2*F2*r1^0 + 3*2*F3*r1^1)
row4 = [0, 0, 2 / rhon**2, 0, -2 / rhoe**2, -3 * 2 * r1 / rhoe**2]
b4 = 0

# Continuity of F at rho = 1.15*rhoe
# F0*r2^0 + F1*r2^1 + F2*r2^2 + F3*r2^3 - Fe*(1 - eta*log(rho0/rhos))*(rho0/rhos)^eta
row5 = [0, 0, 0, 0, r2**2, r2**3]
b5 = -(F0 + F1 * r2 - Fe * (1 - eta * np.log(rho0_over_rhos)) *
       (rho0_over_rhos)**eta)

# Continuity of F' at rho = 1.15*rhoe
# 1/rhoe*(F1*r2^0 + 2*F2*r2^1 + 3*F3*r2^2) + eta^2*Fe/rhos*log(rho0/rhos)*(rho0/rhos)^(eta - 1)
row6 = [0, 0, 0, 0, 1 / rhoe * 2 * r2**1, 1 / rhoe * 3 * r2**2]
b6 = -(1 / rhoe * F1 + eta**2 * Fe / rhos * np.log(rho0_over_rhos) *
       (rho0_over_rhos)**(eta - 1))

Q = np.array([row1, row2, row3, row4, row5, row6])
b = np.array([b1, b2, b3, b4, b5, b6])

x = np.linalg.solve(Q, b)

# Reassign solution to the parameters
Fn0, Fn1, Fn2, Fn3, F2, F3 = x

# Print all 20+ parameters
params = [
    re,
    fe,
    rhoe,
    rhos,
    alfa,
    beta,
    A,
    B,
    k,
    lamb,
    Fn0,
    Fn1,
    Fn2,
    Fn3,
    F0,
    F1,
    F2,
    F3,
    eta,
    Fe,
    74,
    183.84,
    F3,
    beta,
    lamb,
    0.85,
    1.15,
]

###############################################################################
# Write files
# -----------
# Destination directory
destination = Path(args.destination).resolve()
destination.mkdir(exist_ok=True)
# This is where we will store all the potential files, so that the destination
# directory is cleaner
POT_DIR = destination / "potentials"
POT_DIR.mkdir(exist_ok=True)

# Write EAM_code file
with open(destination / "EAM_code", "w") as f:
    # First, write the symbol
    f.write(f"{symb}\n")
    # Write tha parameters
    for val in params:
        f.write(f"{val}\n")

# Write input file
input_str = f""" &funccard
 atomtype='{symb}'
 &end
 &funccard
 &end
"""
with open(destination / "input", "w") as f:
    f.write(input_str)

# Generate eam.allow file --- we use translate.x to do this. But, we will run
# the function in the destination directory
# # Copy translate.x to the destination directory
# shutil.copy("translate.x", destination)
# Change the directory to the destination
binary_path = Path(args.binary_path)
# Copy the binary translate.x to the destination
if not (destination / "translate.x").exists():
    shutil.copy(binary_path / "translate.x", destination)
os.chdir(destination)
# Now, we can execute translate.x
subprocess.run("./translate.x < input", shell=True)
# Move eam.alloy file
eam_alloy_file = params_file.with_suffix("").name + ".eam.alloy"
Path(eam_alloy_file).rename(POT_DIR / eam_alloy_file)
print("Writing", POT_DIR / eam_alloy_file)

# Finally, change back to the original directory where this file exists
os.chdir(FILE_DIR)
