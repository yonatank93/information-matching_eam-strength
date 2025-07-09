import os
import argparse
import numpy as np


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file", type=str, help="Input folder containing the dataset"
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default=None,
        help="Output folder to save the results",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=10000,
        help="Size of the files",
    )
    return parser.parse_args()


def chunk(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def main(args):
    x = np.load(args.input_file)

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    for i, subx in enumerate(chunk(x, args.size)):
        with open(f"{args.output_folder}/{i:05d}.npz", "wb") as f:
            np.save(f, subx)


if __name__ == "__main__":
    args = make_parser()
    main(args)
