"""This script is to post-process DFT dataset that Amit shared, including:
* Getting the configurations with a single atom type
* Separating the configurations by single element type
* Separating the configurations by quantities to compare: energy or forces
* Exporting the configurations
"""

from pathlib import Path
from tqdm import tqdm
from glob import glob
import re
import shutil

import numpy as np
from ase.io import read
import periodictable
from quests.descriptor import get_descriptor

FILE_DIR = Path(__file__).resolve().parent
ROOT_DIR = FILE_DIR.parent
ENERGY_KEY = "energy_energy"
FORCES_KEY = "atomic_forces_forces"

# Read Amit's raw dataset --- Note, that this dataset is for ternary alloy
# system, but we can extract just the Ta dataset
raw_filename = FILE_DIR / "DS_t3jry72ii4xi_0.extxyz"
# Read as list of atoms object
print("Reading dataset:", raw_filename.name)
list_of_atoms = read(raw_filename, format="extxyz", index=":")
print(f"There are {len(list_of_atoms)} configurations in the dataset")

# This dataset contains configuration with a single atom type and mixed atom
# types. For now, we just want to extract the configurations with a single atom
# type.
atomic_numbers = [np.unique(atoms.get_atomic_numbers()) for atoms in list_of_atoms]
natom_type = np.array([len(nums) for nums in atomic_numbers], dtype=int)
# These are index of configurations that contain only a single atom type,
# either Mo, Ta, or W.
idx_single_type = np.where(natom_type == 1)[0]
unique_atomic_numbers = np.unique(
    np.array([atomic_numbers[ii][0] for ii in idx_single_type])
)
single_elements = [periodictable.elements[ii] for ii in unique_atomic_numbers]
print("Available elements:", single_elements)

# Now, extract the configurations
print("Extracting single element atomic configurations...")
# We'll store the atomic configurations as a dictionary of a list. Each key
# contains a list of atomic configurations for each element.
single_elements_configs = {str(el): [] for el in single_elements}
for ii in tqdm(idx_single_type):
    element = periodictable.elements[atomic_numbers[ii][0]]
    atoms = list_of_atoms[ii]
    single_elements_configs[str(element)].append(atoms)

# Then, we need to separate which configurations for each element are
# associated with atomic forces and which configurations are associated with
# energy. For the energy, we can include all configurations, but for the
# forces, we only want to include non-symmetric configurations.
print("Separating the energy and forces configurations for each element...")
for element in single_elements_configs:
    configs = single_elements_configs[element]
    configs_energy = []
    configs_forces = []
    for atoms in configs:
        # Compute QUESTS descriptor
        desc = get_descriptor([atoms])
        # Add the descriptor to atoms.array
        atoms.arrays["desc"] = desc

        # Energy configurations
        # Fix energy ground truth data
        energy_raw = atoms.info[ENERGY_KEY]
        if element == "Ta":
            # Currently, I only have isolated atom energy data for Ta
            isolated_energy = -2.21367890  # Isolated atom energy in eV/atom
            natoms = atoms.get_global_number_of_atoms()
            energy = energy_raw - natoms * isolated_energy
            atoms.info[ENERGY_KEY] = energy
        configs_energy.append(atoms)

        # Forces configurations
        forces = atoms.arrays[FORCES_KEY]
        if not np.allclose(forces, 0.0):
            # This configuration is symmetric, so we can skip it.
            configs_forces.append(atoms)
    energy_correction = None
    single_elements_configs[element] = {
        ENERGY_KEY: configs_energy,
        FORCES_KEY: configs_forces,
    }
    print(
        f"Element {element}: {len(configs_energy)} energy, "
        f"{len(configs_forces)} forces"
    )

# Export the configurations --- The written confgurations will be separated by
# the element type and the type of quantity (energy or forces).
# We also want to write a configuration list file as a helper file
print("Exporting the configurations...")
for element in single_elements_configs:
    configs = single_elements_configs[element]
    config_path_list = []
    # Create a parent directory for the element
    target_dir_parent = FILE_DIR / f"{element}"
    target_dir_parent.mkdir(parents=True, exist_ok=True)
    # Create a subdirectory for the energy and forces configurations
    for quantity, list_of_atoms in configs.items():
        config_path_quantity_list = []
        target_dir = target_dir_parent / quantity
        target_dir.mkdir(parents=True, exist_ok=True)
        # Export the configurations
        for atoms in tqdm(list_of_atoms, desc=f"Exporting {element} {quantity}"):
            filename = target_dir / f"{atoms.info['config_tag']}_{quantity}.xyz"
            atoms.write(filename, format="extxyz")
            config_path_list.append(f"{quantity}/{filename.name}")
            config_path_quantity_list.append(filename.name)
        np.savetxt(target_dir / "config_list.txt", config_path_quantity_list, fmt="%s")

    # Write the configuration list file
    config_list_filename = target_dir_parent / "config_list.txt"
    np.savetxt(config_list_filename, config_path_list, fmt="%s")
