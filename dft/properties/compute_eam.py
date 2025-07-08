"""Use this script to evaluate energy, forces, and/or stress using EAM
potential for a give ASE atoms object.
"""

from ase.calculators.eam import EAM


def compute(atoms,
            potential,
            compute_energy=False,
            compute_forces=True,
            compute_stress=False):
    """Compute the predictions related to the given atomic configurations,
    calculated using EAM potential.

    Parameters
    ----------
    atoms: ase.atoms.Atoms
        Atomic configurations
    potential: str
        Path to the .eam.alloy EAM parameter file
    compute_<property>: bool (Optional)
        A flag to compute <property>. As a default, only compute forces.

    Returns
    -------
    dict: dict.keys() = ["energy", "forces", "stress"]
        The key of this dictionary shows the type of property.
    """
    atoms = atoms.copy()
    atoms.calc = EAM(potential=potential)
    return_values = {}
    if compute_energy:
        energy = atoms.get_potential_energy()
        natoms = atoms.get_global_number_of_atoms()
        return_values["energy"] = energy / natoms
    if compute_forces:
        return_values["forces"] = atoms.get_forces()
    if compute_stress:
        return_values["stress"] = atoms.get_stress()
    return return_values
