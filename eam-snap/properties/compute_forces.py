"""This script is mainly used to run LAMMPS and generate forces data for given
kEAM potential file.

I think I also want to extract the forces for the 2000 candidate environements
in this script too.
"""

from pathlib import Path
import argparse
import jinja2
import subprocess

import numpy as np

FILE_DIR = Path(__file__).resolve().parent

# Command line arguments
parser = argparse.ArgumentParser(
    "Compute atomic forces for given EAM potential")
# What arguments should we request: parameter file, target folder where to
# place the results, configuration,central atom position
parser.add_argument(
    "-p",
    "--potential",
    type=str,
    help="Name of the eam.alloy parameter file in <target_dir>/potentials",
    dest="potential_file",
)
parser.add_argument(
    "-t",
    "--target-dir",
    type=str,
    help="Target folder to write all the calculation files",
    dest="target_dir",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    help="Path to the configuration file",
    dest="config",
)
parser.add_argument(
    "-a",
    "--central_atom",
    type=str,
    help="Path to the file containing position of central atom",
    dest="central_atom_file",
)
parser.add_argument(
    "-n",
    "--nprocs",
    type=int,
    default=1,
    help="Number of parallel process to use",
    dest="nprocs",
)
parser.add_argument(
    "-i",
    "--calc_id",
    type=str,
    default="0",
    help="Calculation ID to distinguish the results",
    dest="calc_id",
)

args = parser.parse_args()
potential_file = Path(args.potential_file)
target_dir = Path(args.target_dir).resolve().absolute()
calc_id = args.calc_id
# This is where we will store the results for the forces calculations
forces_dir = target_dir / "forces"
forces_dir.mkdir(exist_ok=True)
# Get the symbol of the element that we use in the potential file
symb = (potential_file.name).split(".")[0]

print("Running the forces calculation with the following settings:")
print(f"Potential file: {target_dir}/potentials/{potential_file}")
print("Element symbol:", symb)

# Lammps template script
lammps_tpl = """# LAMMPS Input Script to Compute Atomic Forces

# Log file
# log	     {{ target_dir }}/forces/forces.{{ symb }}.{{ calc_id }}.log

# Initialize and define units and boundary conditions
units 	     metal
atom_style   atomic
boundary     p p p

# Define the simulation box using known bounds from the .dump file header
region 	     simbox prism 945.2951058648548 1737.1266170496060 -400.6353272792469 &
       	      	      	  419.56442404085044 -310.1190542204561 494.72260784626945 &
			  6.2188278662727674 -2.4918818131154445 3.0074771675674690 &
		    	  units box
create_box   1 simbox

# Read atoms from an existing .dump file, specifying scaled coordinates (xs, ys, zs)
read_dump    {{ config }} 250000 x y z box yes add yes replace yes

# Define the interatomic potential (customize as needed)
pair_style   eam/alloy
pair_coeff   * * {{ target_dir }}/potentials/{{ potential_file }} {{ symb }}

# Define a group for all atoms (optional, for potential customization)
group 	     all type 1

# Compute atomic forces
compute      myForce all property/atom fx fy fz

# Output atoms with computed forces into a new .dump file
dump 	     output all custom 1 {{ target_dir }}/forces/forces.{{ symb }}.{{ calc_id }}.dump &
	     id type x y z fx fy fz
# Sort by index to help with atom indexing later
# Print all 16 digits values since the forces will be read from the dump file
dump_modify  output sort id format line "%d %d %.16g %.16g %.16g %.16g %.16g %.16g"

# Run a single step to compute forces and output results
run   	     0

# Clean up
undump	     output
"""

# Write lammps input script
print("Generate lammps script")
template = jinja2.Template(lammps_tpl)
lammps_script = template.render(
    target_dir=target_dir,
    config=args.config,
    potential_file=potential_file,
    symb=symb,
    calc_id=calc_id,
)
# print(lammps_script)
# Write
lammps_input_file = forces_dir / f"forces.{symb}.{calc_id}.in"
with open(lammps_input_file, "w") as f:
    f.write(lammps_script)

# Run forces calculation in lammps
# Only run calculation if the results doesn't exist
lammps_dump_file = forces_dir / f"forces.{symb}.{calc_id}.dump"
nprocs = args.nprocs
if not lammps_dump_file.exists():
    if nprocs > 1:
        process = subprocess.Popen(
            f"srun -n {nprocs} lmp -sc none -log none -i {lammps_input_file}",
            shell=True,
        )
    else:
        process = subprocess.Popen(
            f"lmp -sc none -log none -i {lammps_input_file}", shell=True)
    process.wait()
else:
    print("This calculation has been completed previously, "
          "using the dump file from that calculation")

# Now after the forces calculation is done, we need to extract the forces of
# the 2000 candidate environments
forces_data_file = forces_dir / f"forces.{symb}.{calc_id}.txt"
if not forces_data_file.exists():
    print("Getting the forces data of the central atom")
    # First, read the position of the central atom
    central_atom = np.loadtxt(args.central_atom_file)
    # Now, read the forces values from the lammps dump file
    # The first 9 rows of the dump file is like the header
    dump_data = np.loadtxt(lammps_dump_file, skiprows=9)
    # In this dump file data, the first 2 columns are just for lammps internal
    # used (atom index and element index). The next 3 columns are the xyz
    # coordinated. What we need are contained in the last 3 columnes.
    forces_data = dump_data[:, -3:]
    position_data = dump_data[:, 2:5]
    # Retrieve the forces data for the central atom
    distance = np.linalg.norm(position_data - central_atom, axis=1)
    idx = np.argmin(distance)
    min_dist = np.min(distance)
    # if min_dist < 1e-8:
    #     print("Central atom found, min mistance:", np.min(distance))
    forces_central = forces_data[idx]
    # Export these forcecs data
    np.savetxt(forces_data_file, forces_central)
    # else:
    #     raise ValueError("Central atom not found in the dump file")
