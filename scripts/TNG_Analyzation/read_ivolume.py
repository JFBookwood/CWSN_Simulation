"""
Read an IVolume file and return a numpy array.
"""
import numpy as np

from pathlib import Path


def read_ivolume(file_in, no_transpose=False):
    p = Path(file_in)
    if not p.exists():
        raise FileNotFoundError(file_in)
    with open(p, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=6)
        sizeX, sizeY, sizeZ = header[3], header[4], header[5]
        f.read(12)
        f.read(256 - 36)
        data = np.fromfile(f, dtype=np.int32, count=sizeX * sizeY * sizeZ)
        arr = data.reshape((sizeZ, sizeY, sizeX))
        if no_transpose:
            return arr
        return np.transpose(arr, (2, 1, 0))
