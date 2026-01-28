"""
TNG300 Void Analyzation and Visualization Test Script
"""

import numpy as np
from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from read_bvolume import read_bvolume
from read_ivolume import read_ivolume
from collections import defaultdict
from scipy import ndimage
import pickle
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--plot1", action="store_true")
parser.add_argument("--plot2", action="store_true")
parser.add_argument("--plot3", action="store_true")
parser.add_argument("--plot4", action="store_true")
parser.add_argument("--plot5", action="store_true")
parser.add_argument("--plot6", action="store_true")
args = parser.parse_args()

DATA = Path("C:/Users/Jesse/Desktop/Experimente/AngularMomentsofVoids/Data")
CACHE_FILE = DATA / "void_centers_cache.pkl"
CACHE_LIST_FILE = DATA / "void_halo_lists_cache.pkl"
box = 205000.0
ng = 512
den = read_bvolume(DATA / "snap_009.FIX.bvol" / "snap_009.FIX.bvol")
hdul = fits.open(DATA / "TNG300-1-Halos.fits" / "TNG300-1-Halos.fits")

x = hdul[1].data.astype(float)
y = hdul[2].data.astype(float)
z = hdul[3].data.astype(float)
ix = np.clip((x / box * ng).astype(int), 0, ng - 1)
iy = np.clip((y / box * ng).astype(int), 0, ng - 1)
iz = np.clip((z / box * ng).astype(int), 0, ng - 1)
deni = den[ix, iy, iz].astype(float)
col = deni**1.75
col = col / col.max() * 255
val = np.where(z <= 4000)[0]
sor = np.argsort(col[val])

img = -((den[:, :, 0].astype(float)) ** 0.1)
img = (img - img.min()) / (img.max() - img.min()) * 255

reg = read_ivolume(DATA / "snap_009.FIX.iwat" / "snap_009.FIX.iwat")
reg2 = read_ivolume(
    DATA / "snap_009.FIX.Y-0-1-2-3-4.iwat" / "snap_009.FIX.Y-0-1-2-3-4.iwat"
)

halo_void_id = reg[ix, iy, iz]
halos_in_void = defaultdict(list)

for i, vid in enumerate(halo_void_id):
    if vid > 0:
        halos_in_void[vid].append(i)

if CACHE_FILE.exists():
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
    void_ids = cache["void_id"]
    mass_centers_x = cache["x"]
    mass_centers_y = cache["y"]
    mass_centers_z = cache["z"]
    centered_halos = cache["centered_halos"]

else:
    void_ids = np.unique(reg[reg > 0])
    centers = ndimage.center_of_mass(
        input=np.ones_like(reg), labels=reg, index=void_ids
    )

    mass_centers_x = []
    mass_centers_y = []
    mass_centers_z = []

    for center in centers:
        mass_centers_x.append(center[0] * box / ng)
        mass_centers_y.append(center[1] * box / ng)
        mass_centers_z.append(center[2] * box / ng)

    mass_centers_x = np.array(mass_centers_x)
    mass_centers_y = np.array(mass_centers_y)
    mass_centers_z = np.array(mass_centers_z)
    centered_halos = {}

    for i, vid in enumerate(void_ids):
        if vid in halos_in_void:
            halo_indices = np.array(halos_in_void[vid], dtype=int)
            cx = mass_centers_x[i]
            cy = mass_centers_y[i]
            cz = mass_centers_z[i]
            centered_x = x[halo_indices] - cx
            centered_y = y[halo_indices] - cy
            centered_z = z[halo_indices] - cz
            centered_halos[vid] = {
                "x": centered_x,
                "y": centered_y,
                "z": centered_z,
                "halo_indices": halo_indices,
            }

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(
            {
                "void_id": void_ids,
                "x": mass_centers_x,
                "y": mass_centers_y,
                "z": mass_centers_z,
                "centered_halos": centered_halos,
            },
            f,
        )

if CACHE_LIST_FILE.exists():
    with open(CACHE_LIST_FILE, "rb") as f:
        cache_list = pickle.load(f)
    void_centers_list = cache_list["void_centers_list"]
    centered_halos_list = cache_list["centered_halos_list"]

else:
    void_centers_list = []
    centered_halos_list = []
    for i, vid in enumerate(void_ids):
        if vid in halos_in_void:
            halo_indices = np.array(halos_in_void[vid], dtype=int)
            cx = x[halo_indices].mean()
            cy = y[halo_indices].mean()
            cz = z[halo_indices].mean()

            void_centers_list.append((cx, cy, cz))
            centered_coords = np.column_stack(
                (x[halo_indices] - cx, y[halo_indices] - cy, z[halo_indices] - cz)
            )
            centered_halos_list.append(centered_coords)

    with open(CACHE_LIST_FILE, "wb") as f:
        pickle.dump(
            {
                "void_centers_list": void_centers_list,
                "centered_halos_list": centered_halos_list,
            },
            f,
        )

print(f"n Voids: {len(void_ids)}")
print(f"S berechnet: {len(mass_centers_x)}")
print(f"Voids fuer Listen: {len(void_centers_list)}")
print(f"Zentrierte Halos fuer Listen: {len(centered_halos_list)}")

z_max = 50000
mask_halos = z <= z_max
mask_centers = mass_centers_z <= z_max
rand_idx = random.randint(0, len(void_centers_list) - 1)
void_center = void_centers_list[rand_idx]
halos_coords = centered_halos_list[rand_idx]
z_slice_idx_rand = int(round(ng / box * (void_center[2])))
z_slice_idx_rand = np.clip(z_slice_idx_rand, 0, ng - 1)
void_slice_rand = (reg[:, :, z_slice_idx_rand] == void_ids[rand_idx]).T

if not (
    args.plot1 or args.plot2 or args.plot3 or args.plot4 or args.plot5 or args.plot6
):
    args.plot1 = args.plot2 = args.plot3 = args.plot4 = args.plot5 = args.plot6 = True

if args.plot1:
    plt.figure(figsize=(8, 8))
    plt.scatter(x[val][sor], y[val][sor], c=col[val][sor], s=1, cmap="viridis")
    plt.gca().set_aspect("equal")
    plt.show()

if args.plot2:
    plt.figure(figsize=(8, 8))
    plt.imshow(reg[:, :, 0].T, origin="lower", interpolation="nearest", cmap="viridis")
    plt.gca().set_aspect("equal")
    plt.show()

if args.plot3:
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x[mask_halos], y[mask_halos], z[mask_halos], c="gray", s=0.1, alpha=0.2)
    ax.scatter(
        mass_centers_x[mask_centers],
        mass_centers_y[mask_centers],
        mass_centers_z[mask_centers],
        c="red",
        s=100,
        marker="o",
        edgecolors="black",
        linewidths=1,
    )

    ax.set_xlabel("X [kpc/h]")
    ax.set_ylabel("Y [kpc/h]")
    ax.set_zlabel("Z [kpc/h]")
    ax.set_title("Schwerpunkt")
    ax.set_xlim(0, box)
    ax.set_ylim(0, box)
    ax.set_zlim(0, z_max)
    plt.show()

if args.plot4:
    plt.figure(figsize=(8, 8))
    plt.imshow(void_slice_rand, origin="lower", cmap="Greys", interpolation="nearest")
    plt.scatter(
        halos_coords[:, 0] + ng / 2,
        halos_coords[:, 1] + ng / 2,
        c="red",
        s=20,
        edgecolors="black",
        label="Halos",
    )

    plt.scatter(ng / 2, ng / 2, c="blue", s=50, marker="x", label="Void Mittelpunkt")
    plt.xlabel("X [Voxel]")
    plt.ylabel("Y [Voxel]")
    plt.title(f"Void: {void_ids[rand_idx]}")
    plt.legend()
    plt.show()

if args.plot5:
    z_slice_idx = 0
    void_slice = reg[:, :, z_slice_idx].T
    val = np.where(z <= 4000)[0]
    sor = np.argsort(col[val])
    plt.figure(figsize=(10, 10))

    plt.imshow(
        void_slice,
        origin="lower",
        cmap="Greys",
        interpolation="nearest",
        extent=(0, box, 0, box),
        alpha=0.6,
    )

    sc = plt.scatter(x[val][sor], y[val][sor], c=col[val][sor], s=1, cmap="viridis")
    plt.gca().set_aspect("equal")
    plt.xlabel("X [kpc/h]")
    plt.ylabel("Y [kpc/h]")
    plt.show()

if args.plot6:
    z_slice_idx = 0
    void_slice = reg[:, :, z_slice_idx].T
    val = np.where(z <= 4000)[0]
    halo_voids = halo_void_id[val]
    unique_voids = np.unique(halo_voids)
    unique_voids = unique_voids[unique_voids > 0]
    cmap = plt.cm.tab20
    void_color_map = {vid: cmap(i % cmap.N) for i, vid in enumerate(unique_voids)}

    colors = np.array(
        [
            void_color_map[vid] if vid in void_color_map else (0, 0, 0, 0.2)
            for vid in halo_voids
        ]
    )

    plt.figure(figsize=(10, 10))

    plt.imshow(
        void_slice,
        origin="lower",
        cmap="Greys",
        interpolation="nearest",
        extent=(0, box, 0, box),
        alpha=0.6,
    )

    plt.scatter(x[val], y[val], c=colors, s=1)
    plt.gca().set_aspect("equal")
    plt.xlabel("X [kpc/h]")
    plt.ylabel("Y [kpc/h]")
    plt.show()
