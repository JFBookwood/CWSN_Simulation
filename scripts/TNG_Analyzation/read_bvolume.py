"""
Read a BVolume file and return a numpy array.
"""
from pathlib import Path
from typing import Union

import numpy as np


FILE_HEADER_COUNT = 6


def read_bvolume(file_in: Union[str, Path]) -> np.ndarray:
    """Load the B-volume file that stores density in uint8 voxels."""
    path = Path(file_in)
    with path.open("rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=FILE_HEADER_COUNT)
        nx, ny, nz = header[3], header[4], header[5]
        f.read(12)  # Diese extra ints sind nur Platzhalter, voll unnötig.
        f.read(256 - 36)  # Ich skippe den Rest vom Header, damit wir endlich bei den Daten sind.
        data = np.fromfile(f, dtype=np.uint8, count=nx * ny * nz)
        arr = data.reshape((nz, ny, nx))
        return np.transpose(arr, (2, 1, 0))  # Drehe die Achsen, damit alles wieder richtig rum ist.
