import os
import argparse
import numpy as np


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_folder", type=str, help="Input folder containing the dataset"
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default=None,
        help="Output folder to save the results",
    )
    parser.add_argument(
        "-n",
        "--num_chunks",
        type=int,
        default=5,
        help="Number of chunks to use",
    )
    return parser.parse_args()


def chunk(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def main(args):
    files = os.listdir(args.input_folder)

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    for i, fs in enumerate(chunk(files, args.num_chunks)):
        arrays = [np.load(os.path.join(args.input_folder, f)) for f in fs]
        arrays = np.concatenate(arrays)

        with open(f"{args.output_folder}/{i:05d}.npz", "wb") as f:
            np.save(f, arrays)


if __name__ == "__main__":
    args = make_parser()
    main(args)
