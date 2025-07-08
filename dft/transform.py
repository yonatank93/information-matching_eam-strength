"""This module contain parameter transformation class for EAM potential that
we study. The original parameter space is the tabulated EAM functions and the
transformed space is the 7 tunable parameters.
"""

from pathlib import Path
import subprocess
import shutil
import uuid
import re

import numpy as np
from information_matching.transform import TransformBase, avail_transform


class EAMTransform(TransformBase):
    """
    Parameter transformation specific for EAM potential case with 7 tunable
    parameters that Vasily shared. This transformation class is to make the
    parameterization that we use works with KIM potential parameterization.

    :param params0_file: path to .txt file that contains the unperturbed
        parameters
    :type params0_file: Path-like

    :param tmp_dir: path to directory to dump all temporary files for this
        transformation calculation
    :type tmp_dir: Path-like

    :params binary_path: path to binary fortran code to write .eam.alloy
        parameter file
    :type binary_path: Path-like

    :params symb: element type symbol.
    :type symb: str

    :params subdir: optional subdirectory relative to tmp_dir to where the
        files will be written to. This is useful, for example, when we do
        parallelization.
    :type symb: None, str, "random"

    :params cleanup: whether to remove the temporary directory at the end of
        the transformation and inverse transformation.
    :type cleanup: bool
    """

    # Can modify parameters # 1, 6, 7, 8, 9, 19, 20 (index 1)
    # Index location of the 7 parameters in the 20 elements list of parameters
    idx_7vals = [ii - 1 for ii in [1, 6, 7, 8, 9, 19, 20]]

    def __init__(self,
                 params0,
                 tmp_dir,
                 binary_path,
                 symb="W",
                 subdir=None,
                 cleanup=True):
        if isinstance(params0, (list, np.ndarray)):
            self.params0_20vals = np.array(params0)
        elif isinstance(params0, (str, Path)):
            self.params0_20vals = np.loadtxt(params0)
        self.params0_7vals = self.params0_20vals[self.idx_7vals]
        self.TMP_DIR = Path(tmp_dir).resolve()
        self.TMP_DIR.mkdir(exist_ok=True)
        self.binary_path = Path(binary_path).resolve()
        self.subdir = subdir
        self._target_dir = None
        self.cleanup = cleanup
        # Element symbol
        self.symb = symb

        # Every time .inverse_transform() is called, it will write the
        # following .eam.alloy parameter file
        self.eam_alloy_filename = f"{self.symb}.eam.alloy"

        super().__init__(params0=params0,
                         tmp_dir=tmp_dir,
                         binary_path=binary_path,
                         symb=symb,
                         subdir=subdir,
                         cleanup=cleanup)

    def transform(self, x):
        """Transform the parameters into the transformed space.

        For all of our purposes, we can just set this to return the 7
        parameters from params0_20vals.
        """
        return self.params0_7vals

    def inverse_transform(self, params):
        """Transform the parameters from the transformed space to the original
        space.

        The input parameter format is the 7-value format, and the output is the
        tabulated values.
        """
        # Prepare the temporary/target directory
        if self.subdir is None:
            self._target_dir = self.TMP_DIR
        else:
            if self.subdir == "random":
                subdir = uuid.uuid4().hex
            else:
                subdir = self.subdir
            self._target_dir = self.TMP_DIR / subdir
            self._target_dir.mkdir(parents=True, exist_ok=True)
        # Convert the parameters and write .eam.alloy parameter file
        params_20vals = self._convert_from_7values_to_20values(params)
        self._write_eam_alloy_file(params_20vals, self._target_dir)
        # Read parameter file and extract the parameter values
        eam_data = self.read_eam_alloy(self._target_dir)
        values = []
        for key in ["cutoff", "dr", "drho", "F_rho", "phi_r", "rho_r"]:
            val = eam_data[key]
            if isinstance(val, float):
                val = [val]
            values.extend(val)

        if self.cleanup:
            # Clean up temporary directory
            self.cleanup_tmp_dir(self._target_dir)
        return np.array(values)

    def _convert_from_7values_to_20values(self, params_7vals):
        """From the 7 tunable parameters, compute the other 13 parameters to
        get all 20 parameters that Zhou et al. listed for this EAM potential.

        The returned values actually have more than 20 elements, where the
        first 20 elements are the 20-value format. But the rest of values are
        needed to write .eam.alloy parameter file
        """
        # The input params is in 7-value format. To use Vasily's code, we need
        # to convert it to 20-value format
        params = self.params0_20vals.copy()
        params[self.idx_7vals] = params_7vals

        # Conversion -- Using Vasily's conversion script
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
        return params

    def _write_eam_alloy_file(self, params, target_dir):
        """Take the output of self._convert_from_7values_to_20values method
        and write a .eam.alloy parameter file.
        """
        # Write EAM_code file
        with open(target_dir / "EAM_code", "w") as f:
            # First, write the symbol
            f.write(f"{self.symb}\n")
            # Write tha parameters
            for val in params:
                f.write(f"{val}\n")

        # Write input file
        input_str = f""" &funccard
         atomtype='{self.symb}'
         &end
         &funccard
         &end
        """
        with open(target_dir / "input", "w") as f:
            f.write(input_str)

        # Generate eam.allow file --- we use translate.x to do this. But, we
        # will run the function in the target_dir directory
        # Copy the binary translate.x to the target_dir
        if self.subdir is not None:
            uid = target_dir.name
            suffix = self.binary_path.suffix
            binary_name = self.binary_path.with_suffix("").name + uid + suffix
        else:
            binary_name = Path(self.binary_path).name
        shutil.copy(self.binary_path, target_dir / binary_name)
        # Now, we can execute translate.x
        with open(target_dir / "input", "r") as fin:
            subprocess.run([f"./{binary_name}"], stdin=fin, cwd=target_dir)

    def read_eam_alloy(self, target_dir):
        """Read .eam.alloy parameter file and extract the values.
        """
        with open(target_dir / self.eam_alloy_filename, 'r') as f:
            lines = f.readlines()

        # Identify where numerical data starts (adjust based on your file
        # structure)
        metadata_end = 3  # First three lines contain metadata

        # Extract element type
        num_elements, element = lines[metadata_end].split()
        num_elements = int(num_elements)

        # Extract grid parameters
        Nrho, drho, Nr, dr, cutoff = map(float,
                                         lines[metadata_end + 1].split())
        Nr, Nrho = int(Nr), int(Nrho)

        # Extract additional element-specific data (atomic number, mass,
        # lattice constant, structure)
        pattern = (
            r"^\s*"
            r"(\d+)\s+"  # Atomic number
            r"(\d+(?:\.\d+)?)\s+"  # Mass
            r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s+)"  # Lattice constant
            r"([a-zA-Z]+)\s*$"  # Structure
        )
        match = re.match(pattern, lines[metadata_end + 2])
        atomic_num = int(match.group(1))
        atomic_mass = float(match.group(2))
        lattice_const = float(match.group(3))
        crystal_structure = match.group(4)

        # Identify where the tabulated data starts
        data_start = metadata_end + 3

        # Function to read a block of tabulated values (handles multiple
        # columns per line)
        def read_block(start_index, count):
            block_values = []
            ii = 0
            while len(block_values) < count:
                # Read line, strip extra spaces, split into values, and convert
                # to float
                block_values.extend(
                    map(float, lines[start_index + ii].strip().split()))
                ii += 1
            return np.array(block_values), ii

        # Read embedding function F(ρ) (size Nrho)
        F_rho, nlines_F = read_block(data_start, Nrho)

        # Read electron density ρ(r) (size Nr)
        rho_r, nlines_rho = read_block(data_start + nlines_F, Nr)

        # Read pair potential φ(r) (size Nr)
        phi_r, _ = read_block(data_start + nlines_F + nlines_rho, Nr)

        return {
            "num_elements": num_elements,
            "element": element,
            "atomic_number": atomic_num,
            "atomic_mass": atomic_mass,
            "lattice_constant": lattice_const,
            "crystal_structure": crystal_structure,
            "Nr": Nr,
            "dr": dr,
            "Nrho": Nrho,
            "drho": drho,
            "cutoff": cutoff,
            "F_rho": F_rho,  # Embedding function
            "rho_r": rho_r,  # Electron density function
            "phi_r": phi_r  # Pair potential
        }

    def cleanup_tmp_dir(self, target_dir):
        """A function to delete the temporary files: EAM_code and input."""
        if self.subdir is None:
            files = [str(target_dir / ff) for ff in ["EAM_code", "input"]]
            subprocess.run(f"rm -v {' '.join(files)}", shell=True)
        else:
            subprocess.run(f"rm -rv {target_dir}", shell=True)


# Need to add this transformation class to the dictionary of avail_transform so
# we can use transform_builder function.
avail_transform.update({"EAMTransform": EAMTransform})
