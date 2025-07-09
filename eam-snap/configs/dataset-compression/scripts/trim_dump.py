import tqdm
import numpy as np
from ase import Atoms
from ase.io import read, write


def process_file(input_file, output_file):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line_number, line in tqdm.tqdm(enumerate(infile)):
            columns = line.split()
            if line_number < 8:
                outfile.write(line)
            elif line_number == 8:
                columns.insert(3, "type")
                outfile.write(" ".join(columns[:7]) + "\n")
            else:
                columns.insert(1, "1")
                outfile.write(" ".join(columns[:5]) + "\n")


if __name__ == "__main__":
    process_file("snapd.dump", "snapd-short.dump")
