import tqdm
import numpy as np
from ase import Atoms
from ase.io import read, write


def process_large_file(file_path):
    idx = []
    pos = []
    lattice = []
    num_atoms = np.inf
    with open(file_path, "r") as file:
        for i, line in tqdm.tqdm(enumerate(file)):
            if i == 3:
                num_atoms = int(line.strip())

            elif i in {5, 6, 7}:
                lattice.append(map(float, line.split()))

            elif i > num_atoms + 8:
                break

            elif i > 8:
                parts = line.split()
                idx.append(int(parts[0]))
                pos.extend(map(float, parts[1:4]))

    return idx, np.array(pos), np.array(lattice)


if __name__ == "__main__":
    idx, pos, lattice = process_large_file("snapd.dump")

    pos = np.round(pos, 3)
    atoms = Atoms(
        symbols=["W"] * len(pos), positions=pos, cell=lattice, pbc=[True, True, True]
    )
    write("snapd.xyz", atoms, format="extxyz")
    # np.save("idx", idx)
    # np.save("pos", pos)
    # np.save("lat", lattice)
