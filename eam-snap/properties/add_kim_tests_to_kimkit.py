"""This script is to add .txz files of KIM tests and test drivers. Note that
.txz files need to be downloaded separately (this cannot be done from LC, I
think). Additionally, we need to make sure that the test drivers and other test
dependencies are also added to kimkit.
"""

from pathlib import Path
from glob import glob
import tarfile
import kimkit

FILE_DIR = Path(__file__).parent.resolve()
TXZ_DIR = FILE_DIR / "kim_tests"

# Retrieve all tarfiles that need to be added to kimkit
tar_files = glob(str(TXZ_DIR / "*.txz"))

# Add to kimkit
for ff in tar_files:
    with tarfile.open(ff) as tar:
        try:
            kimkit.models.import_item(tar)
        except Exception as e:
            print(e)
            continue
