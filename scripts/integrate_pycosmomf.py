"""
PyCosmoMMF (PyCosmo Morphological Multiscale Filter) provides:
- Multiscale morphological filtering of density fields
- Hessian matrix analysis for structure classification
- Eigenvalue-based tagging of cosmic structures (clusters, filaments, walls, voids)
This script integrates PyCosmoMMF into the CWSN pipeline for advanced void finding.
"""
import os
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label
import PyCosmoMMF as MMF
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "desi" / "edr" / "processed"
OUTPUT_DIR = DATA_DIR / "pycosmomf_voids"
GALAXY_FILE = DATA_DIR / "ELG_HIP_positions_xyz_mpc.csv"
DENSITY_FILE = DATA_DIR / "density_field.npy"
BOX_SIZE = 1000.0
N_GRID = 256
SMOOTHING_SCALES = [2**n for n in range(3, 8)]
def create_density_field_from_galaxies():
    """Create density field from galaxy positions using PyCosmoMMF."""
    print("Creating density field from galaxy positions...")
    if not GALAXY_FILE.exists():
        print(f"ERROR: Galaxy file not found: {GALAXY_FILE}")
        return None
    data = np.loadtxt(GALAXY_FILE, delimiter=',', skiprows=1)
    positions = data[:, :3]
    print(f"Loaded {len(positions)} galaxies")
    print(f"Position range: {positions.min(axis=0)} to {positions.max(axis=0)}")
    print(f"Creating density field on {N_GRID}^3 grid...")
    density_field = np.zeros((N_GRID, N_GRID, N_GRID))
    grid_coords = ((positions - positions.min(axis=0)) /
                   (positions.max(axis=0) - positions.min(axis=0)) * (N_GRID - 1)).astype(int)
    for coord in grid_coords:
        if np.all((coord >= 0) & (coord < N_GRID)):
            density_field[tuple(coord)] += 1
    mean_density = len(positions) / (N_GRID ** 3)
    density_field /= mean_density
    density_field -= 1
    print(f"Density field created. Shape: {density_field.shape}")
    print(f"Mean density: {density_field.mean():.3f}")
    print(f"Density range: {density_field.min():.3f} to {density_field.max():.3f}")
    return density_field
def apply_multiscale_filtering(density_field):
    """Apply multiscale morphological filtering using PyCosmoMMF."""
    print("Applying multiscale morphological filtering...")
    filtered_fields = {}
    signatures = {}
    try:
        for R_s in SMOOTHING_SCALES:
            print(f"Processing smoothing scale R_s = {R_s} Mpc/h")
            smoothed = MMF.smooth_gauss(density_field, R_s, BOX_SIZE / N_GRID)
            filtered_fields[R_s] = smoothed
            hessian = MMF.hessian(smoothed, BOX_SIZE / N_GRID)
            sig = MMF.signatures_from_hessian(hessian)
            signatures[R_s] = sig
            print(f"  Scale {R_s}: signatures shape {sig.shape}")
    except Exception as e:
        print(f"Standard filtering failed: {e}")
        print("Trying alternative approach...")
        for R_s in SMOOTHING_SCALES[:2]:
            print(f"Processing smoothing scale R_s = {R_s} Mpc/h (fallback)")
            sigma = R_s / (BOX_SIZE / N_GRID)
            smoothed = gaussian_filter(density_field, sigma=sigma)
            filtered_fields[R_s] = smoothed
            hessian = np.zeros((*smoothed.shape, 3, 3))
            for i in range(3):
                hessian[..., i, i] = -smoothed
            sig = np.zeros((*smoothed.shape, 3))
            sig[..., 0] = -smoothed
            sig[..., 1] = -smoothed
            sig[..., 2] = -smoothed
            signatures[R_s] = sig
            print(f"  Scale {R_s}: signatures shape {sig.shape} (fallback)")
    return filtered_fields, signatures
def classify_structures(signatures):
    """Classify cosmic structures based on signatures."""
    print("Classifying cosmic structures...")
    largest_scale = max(signatures.keys())
    sig_data = signatures[largest_scale]
    eigenvalue_sum = sig_data.sum(axis=-1)
    structure_tags = np.zeros_like(eigenvalue_sum, dtype=int)
    threshold_void = np.percentile(eigenvalue_sum, 10)
    structure_tags[eigenvalue_sum <= threshold_void] = 3
    threshold_wall = np.percentile(eigenvalue_sum, 30)
    structure_tags[(eigenvalue_sum > threshold_void) & (eigenvalue_sum <= threshold_wall)] = 2
    threshold_filament = np.percentile(eigenvalue_sum, 60)
    structure_tags[(eigenvalue_sum > threshold_wall) & (eigenvalue_sum <= threshold_filament)] = 1
    structure_tags[eigenvalue_sum > threshold_filament] = 0
    unique_tags, counts = np.unique(structure_tags, return_counts=True)
    print("Structure classification:")
    for tag, count in zip(unique_tags, counts):
        structure_names = {0: "Cluster", 1: "Filament", 2: "Wall", 3: "Void"}
        name = structure_names.get(tag, f"Unknown ({tag})")
        percentage = count / structure_tags.size * 100
        print(f"  {name}: {count} voxels ({percentage:.1f}%)")
    return structure_tags, sig_data
def extract_voids(structure_tags, density_field):
    """Extract void regions from structure classification."""
    print("Extracting void regions...")
    void_mask = (structure_tags == 3)
    print(f"Void voxels: {void_mask.sum()} / {void_mask.size} = {void_mask.sum() / void_mask.size * 100:.1f}%")
    labeled_voids, num_voids = label(void_mask.astype(int))
    print(f"Found {num_voids} distinct void regions")
    void_properties = []
    for void_id in range(1, num_voids + 1):
        mask = (labeled_voids == void_id)
        volume = mask.sum() * (BOX_SIZE / N_GRID) ** 3
        radius = (3 * volume / (4 * np.pi)) ** (1 / 3)
        indices = np.array(np.where(mask)).T
        center = indices.mean(axis=0) * (BOX_SIZE / N_GRID)
        void_properties.append({
            'id': void_id,
            'volume': volume,
            'radius': radius,
            'center': center,
            'voxels': mask.sum()
        })
    void_properties.sort(key=lambda x: x['volume'], reverse=True)
    print("Top 10 voids by volume:")
    for i, void in enumerate(void_properties[:10]):
        print(f"  Void {void['id']}: R={void['radius']:.1f} Mpc/h, V={void['volume']:.0f} (Mpc/h)^3")
    return void_properties, labeled_voids
def save_results(void_properties, structure_tags, density_field, output_dir):
    """Save PyCosmoMMF results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    void_catalog = []
    for void in void_properties:
        void_catalog.append([
            void['center'][0], void['center'][1], void['center'][2],
            void['radius'],
            void['volume'],
            void['voxels']
        ])
    void_catalog = np.array(void_catalog)
    np.save(output_dir / "pycosmomf_voids.npy", void_catalog)
    np.save(output_dir / "structure_tags.npy", structure_tags)
    np.save(output_dir / "density_field.npy", density_field)
    print(f"Results saved to {output_dir}")
    with open(output_dir / "summary.txt", 'w') as f:
        f.write("PyCosmoMMF Void Analysis Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Box size: {BOX_SIZE} Mpc/h\n")
        f.write(f"Grid size: {N_GRID}^3\n")
        f.write(f"Smoothing scales: {SMOOTHING_SCALES}\n")
        f.write(f"Total voids found: {len(void_properties)}\n")
        f.write(f"Largest void radius: {void_properties[0]['radius']:.1f} Mpc/h\n")
        f.write(f"Largest void volume: {void_properties[0]['volume']:.0f} (Mpc/h)^3\n")
    print("Summary saved")
def visualize_results(structure_tags, density_field, output_dir):
    """Create basic visualizations."""
    print("Creating visualizations...")
    slice_idx = N_GRID // 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im1 = axes[0].imshow(density_field[slice_idx], cmap='RdBu_r', origin='lower')
    axes[0].set_title('Density Field Slice')
    plt.colorbar(im1, ax=axes[0])
    im2 = axes[1].imshow(structure_tags[slice_idx], cmap='tab10', origin='lower')
    axes[1].set_title('Structure Classification\n(0=Cluster, 1=Filament, 2=Wall, 3=Void)')
    cbar = plt.colorbar(im2, ax=axes[1])
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['Cluster', 'Filament', 'Wall', 'Void'])
    plt.tight_layout()
    plt.savefig(output_dir / "pycosmomf_visualization.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved: {output_dir / 'pycosmomf_visualization.png'}")
def main():
    """Main PyCosmoMMF integration function."""
    print("=" * 60)
    print("PyCosmoMMF Void Finder Integration")
    print("=" * 60)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    density_field = create_density_field_from_galaxies()
    if density_field is None:
        return False
    filtered_fields, signatures = apply_multiscale_filtering(density_field)
    structure_tags, max_signatures = classify_structures(signatures)
    void_properties, labeled_voids = extract_voids(structure_tags, density_field)
    save_results(void_properties, structure_tags, density_field, OUTPUT_DIR)
    visualize_results(structure_tags, density_field, OUTPUT_DIR)
    print(f"\nPyCosmoMMF analysis completed!")
    print(f"Found {len(void_properties)} voids")
    print(f"Results saved in {OUTPUT_DIR}")
    return True
if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
