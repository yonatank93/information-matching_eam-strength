"""After extracting the compressed environments, 
"""

import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "-r",
    "--result_dir",
    type=str,
    dest="result_dir",
)
args = parser.parse_args()

result_dir = args.result_dir
nprocs = 10  # Just use 15 because of memory constraint
nthreads = 10  # Try multithreading

# Load original quests descriptor for all environments
quests_desc = np.load("../quests_compressed_envs/eam-quests.npz")
# Load compressed environments descriptor
desc_envs = np.load(os.path.join(result_dir, "samples-6/00000.npz"))


# Find index of compressed environments
def find_idx(desc):
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = str(nthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
    diff = quests_desc - desc
    dist = np.linalg.norm(diff, axis=1)
    return np.where(dist == 0)[0]


with Pool(nprocs) as p:
    idx_compressed = np.array(
        list(tqdm(p.imap(find_idx, desc_envs), total=len(desc_envs))))

idx_compressed = idx_compressed.flatten()
np.save(os.path.join(result_dir, "index-compressed.npy"), idx_compressed)
