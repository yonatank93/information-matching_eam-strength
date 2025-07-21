"""A collection of some utility functions, e.g., a function to extract the
tabulated EAM functions from eam.alloy file.
"""

from pathlib import Path
import re
import subprocess
import time
import numpy as np

ENERGY_KEY = "energy_energy"
FORCES_KEY = "atomic_forces_forces"

# Regularization scale lambda
scale_list = np.arange(0, 11, 10)
# Add scale < 1.0 in log space --- I checked the loss at p0 and I think letting
# lambda goes as low as 1e-3 should be more than enough
# small_scale_list = np.logspace(0, -3, 4)
small_scale_list = np.logspace(-2, 0, 3)
# Combine
scale_list = list(scale_list) + small_scale_list.tolist()
# Unique elements and sort
scale_list = sorted(list(set(scale_list)))
# Larger scale means weaker regularization. But scale 0 means no regularization
nsamples = 100  # Number of random parameter samples


def get_eam_tabulated_values(eam_alloy_path):
    # Read some initial information --- Nrho, drho, Nr, dr, cutoff
    with open(eam_alloy_path, "r") as f:
        read_next = False
        for ii, line in enumerate(f):  # ii locates where the table starts
            # Check if the current line contains "1 Wopt" (or any other string)
            if line.strip().startswith("1"):
                read_next = True  # Set the flag to True
                continue  # Skip processing this line

            # If the flag is True, process the next line
            if read_next:
                # Found those information. Reading the numbers.
                numbers = [
                    num for num in re.findall(
                        r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                ]
                # Parse the numbers
                Nrho = int(numbers[0])
                drho = float(numbers[1])
                Nr = int(numbers[2])
                dr = float(numbers[3])
                rcut = float(numbers[4])
                break  # Exit after reading the desired line

    # Now, get the table containing the spline points
    # ii is zero base, and we need to skip 1 more line
    table = np.loadtxt(eam_alloy_path, skiprows=ii + 2).flatten()

    # Get the spline values for each function
    # The first set of numbers are for the embedding function
    # The next set of numbers are for the electron density function
    # The last set of numbers  are for the pair potential

    # Independent variables
    rlist = np.arange(Nr) * dr
    rholist = np.arange(Nrho) * drho

    # Embedding function --- depends on rho
    embedding_data = table[:Nrho]
    table = table[Nrho:]  # Remove the embedding data from the table

    # Density function --- depends on r
    density_data = table[:Nr]
    table = table[Nr:]

    # Pair potential --- depends on r
    potential_data = table

    return {
        "Nrho": Nrho,
        "Nr": Nr,
        "drho": drho,
        "dr": dr,
        "cutoff": rcut,
        "rholist": rholist,
        "rlist": rlist,
        "embedding_function": embedding_data,
        "density_function": density_data,
        "pair_potential": potential_data
    }


def check_job_status(jobid):
    """Use scontrol to extract the status of a job."""
    control_output = subprocess.run(f"scontrol show job {jobid}",
                                    capture_output=True,
                                    shell=True,
                                    encoding="UTF-8")
    # will get full report, JobState at 10 after =
    job_state = control_output.stdout.split()[10].split("=")[1]
    if job_state == "PENDING":
        state = "pending"
    elif job_state == "RUNNING":
        state = "running"
    elif job_state == "COMPLETED":
        state = "completed"
    elif job_state == "TIMEOUT":
        state = "timeout"
    elif job_state == "CANCELLED":
        state = "cancelled"
    else:
        state = "other"
    return state


def block_until_completed(jobid_list):
    """Block the calculation until the submitted jobs are all completed."""
    completed_jobs = []
    while True:
        for jobid in jobid_list:
            if jobid in completed_jobs:
                continue
            else:
                state = check_job_status(jobid)
                if state not in ["pending", "running"]:
                    completed_jobs.append(jobid)
        if len(completed_jobs) == len(jobid_list):
            break
        else:
            time.sleep(60)  # Check every minute


def run_python_command(python_command, capture_output=False):
    command = "python " + python_command
    if capture_output:
        output_lines = []
        process = subprocess.Popen(command,
                                   shell=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   text=True,
                                   bufsize=1)
        for line in process.stdout:
            print(line, end="")
            output_lines.append(line)
        returncode = process.wait()
        if returncode == 0:
            full_output = "".join(output_lines)
            return full_output
        else:
            raise subprocess.CalledProcessError(returncode, process.args)
    else:
        subprocess.run(command, shell=True, check=True)


def sync_calc(check_file):
    while True:
        if Path(check_file).exists():
            break
        else:
            time.sleep(0.1)
    time.sleep(60)


def natural_key(s):
    """For sorting file name that contains numbers like human."""
    return [
        int(text) if text.isdigit() else text
        for text in re.split(r'(\d+)', s)
    ]


def extract_strength_simulation_data(filename: str, key=None):
    """This function takes the lammps output file and extract the printed
    thermo information, which contain the data to plot the strength simulation
    curve.

    If there are multiple thermo tables found, each table will be stored in
    different dictionary key.

    Parameters
    ----------
    filename: str
        Path to the lammps out file that contains thermo table.
    key: None or int or "all" (optional)
        If None, then return a dictionary containing all thermo tables as
        separate keys. If int, return the table of the corresponding key. If
        "all", concatenate all tables and return the resulting table.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    capture = False
    idata = 0  # Index for storing the captured data

    data_all = {}
    for ii, line in enumerate(lines):
        if "Step Temp" in line:
            # Start capturing the data when the table header is detected
            capture = True
            continue  # Skip to the next line directly
        if capture:
            # Turn off capture mode at the end of table
            if "Loop time" in line:
                # Line after the table is this loop time report
                capture = False
                # The table has all been captured, so we can skip to the next
                # line
                continue
            else:
                # Retrieve the data
                numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?",
                                     line)
                # Convert to float
                numbers = np.array([float(num) for num in numbers])
                # Put the extracted numbers in the line into an array
                if idata == 0:
                    # Create the container array in the beginning of the table
                    data = numbers
                else:
                    data = np.vstack((data, numbers))  # Stack
                idata += 1

    # The file might contain multiple trajectory data. This is indicated when
    # there are multiple tables that start with step 0
    idx_step_zero = np.where(data[:, 0] == 0.0)[0]
    data_all = {}
    for ii, idx in enumerate(idx_step_zero):
        if (ii + 1) == len(idx_step_zero):
            # Last part of the trajectory data
            data_all.update({ii: data[idx:]})
        else:
            data_all.update({ii: data[idx:idx_step_zero[ii + 1]]})

    if key is None:
        return data_all
    elif isinstance(key, int):
        return data_all[key]
    elif key == "all":
        con_data = np.vstack([val for val in data_all.values()])
        return con_data
