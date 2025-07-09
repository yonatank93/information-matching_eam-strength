import argparse
import numpy as np
import fpsample as fps
import os


def sample(x: np.ndarray, n: int):
    return fps.fps_sampling(x, n)


def process_dataset(x: np.ndarray, num_chunks: int, num_sample: int):
    N = len(x)

    if N <= num_sample:
        return np.arange(N)

    if N <= num_chunks * num_sample:
        return sample(x, n=num_sample)

    chunk_size = num_chunks * num_sample
    num_subsets = int(np.ceil(N / chunk_size))
    y = []
    for i in range(num_subsets):
        start = i * chunk_size
        chunk = x[start : start + chunk_size]
        y.append(start + sample(chunk, num_sample))

    y = np.concatenate(y)

    i = process_dataset(x[y], num_chunks, num_sample)
    return y[i]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file", type=str, help="Input file containing the dataset"
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default=None,
        help="Output file to save the result",
    )
    parser.add_argument(
        "-s",
        "--num_chunks",
        type=int,
        default=5,
        help="Number of chunks to split the dataset",
    )
    parser.add_argument(
        "-n",
        "--num_sample",
        type=int,
        default=2000,
        help="Number of samples to take from each chunk",
    )
    parser.add_argument(
        "-i",
        "--store_index",
        action="store_true",
        help="If True, store the indices instead of the compressed subsets",
    )
    args = parser.parse_args()

    x = np.load(args.input_file)
    idx = process_dataset(x, args.num_chunks, args.num_sample)

    if args.store_index:
        out = idx
    else:
        out = x[idx]

    if args.output_file is not None:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file + ".npz", "wb") as f:
            np.save(f, out)


if __name__ == "__main__":
    main()
