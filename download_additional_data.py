"""Run this script to download the additional data, including the results, from the web
and copy them into their respective directories.
"""


from pathlib import Path
import argparse
import subprocess


# Command line arguments
parser = argparse.ArgumentParser(description="Download additional data")
parser.add_argument(
    "--copy-data",
    action="store_true",
    help="Copy data from local directories to the respective directories",
)
parser.add_argument(
    "--copy-results",
    action="store_true",
    help="Copy results to the respective directories",
)
parser.add_argument("--dft", action="store_true", help="Apply actions to DFT data")
parser.add_argument(
    "--eam-snap", action="store_true", help="Apply actions to EAM-SNAP data"
)
args = parser.parse_args()


# Directories
FILE_DIR = Path(__file__).parent
EAM_SNAP_DIR = FILE_DIR / "eam-snap"
DFT_DIR = FILE_DIR / "dft"

# Download data --- This will come later once I upload the data to Zenodo or Figshare
# Data is always downloaded no matter the options
download_folder = FILE_DIR / "information-matching_eam-strength_data"
if download_folder.exists():
    print(f"Data already downloaded in {download_folder}, skip downloading.")
else:
    # Download
    print("Downloading additional data...")
    link = ""
    subprocess.run(["wget", link, "-O", str(download_folder.with_suffix(".tar.gz"))])

    # Untar
    print("Untar the downloaded file to the current directory.")
    subprocess.run(["tar", "-xzvf", download_folder.with_suffix(".tar.gz")], cwd=FILE_DIR)


# Copy the data to the respective directories, if requested
if args.copy_data:
    folders_to_copy = ["configs", "data"]
    if args.dft:
        print("Copying DFT data...")
        for ff in folders_to_copy:
            src = download_folder / "dft" / ff / "*"
            dest = DFT_DIR / ff
            if src.exists():
                print(f"Copying {src} to {dest}")
                subprocess.run(["cp", "-urv", str(src), str(dest)])
    if args.eam_snap:
        print("Copying EAM-SNAP data...")
        for ff in folders_to_copy:
            src = download_folder / "eam-snap" / ff / "*"
            dest = EAM_SNAP_DIR / ff
            if src.exists():
                print(f"Copying {src} to {dest}")
                subprocess.run(["cp", "-urv", str(src), str(dest)])

# Copy the results to the respective directories, if requested
if args.copy_results:
    folders_to_copy = ["results"]
    if args.dft:
        print("Copying DFT results...")
        for ff in folders_to_copy:
            src = download_folder / "dft" / ff / "*"
            dest = DFT_DIR / ff
            if src.exists():
                print(f"Copying {src} to {dest}")
                subprocess.run(["cp", "-urv", str(src), str(dest)])
    if args.eam_snap:
        print("Copying EAM-SNAP results...")
        for ff in folders_to_copy:
            src = download_folder / "eam-snap" / ff / "*"
            dest = EAM_SNAP_DIR / ff
            if src.exists():
                print(f"Copying {src} to {dest}")
                subprocess.run(["cp", "-urv", str(src), str(dest)])
