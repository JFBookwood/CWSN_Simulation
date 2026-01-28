"""
This script provides comprehensive 3D visualization capabilities for void catalogs,
including interactive plots, statistical analysis, and publication-quality figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from pathlib import Path
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Constants
DATA_DIR = Path("data/desi/edr/processed")
OUTPUT_DIR = Path("Images/premium_voids")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color schemes for different void finders
COLOR_SCHEMES = {
    'PyCosmoMMF': '#1f77b4',    # Blue
    'REVOLVER': '#ff7f0e',      # Orange
    'VoidFinder': '#2ca02c',    # Green
    'ZOBOV': '#d62728',         # Red
    'VIDE': '#9467bd'           # Purple
}


def load_void_catalog(catalog_path, finder_name="Unknown"):
    """
    Load void catalog from various formats.

    Parameters:
    -----------
    catalog_path : str or Path
        Path to void catalog file
    finder_name : str
        Name of void finder algorithm

    Returns:
    --------
    pd.DataFrame
        Void catalog with standardized columns
    """
    catalog_path = Path(catalog_path)

    if not catalog_path.exists():
        print(f"Warning: Catalog not found: {catalog_path}")
        return None

    try:
        if catalog_path.suffix == '.txt':
            # Assume space-separated format
            df = pd.read_csv(catalog_path, sep=r'\s+', comment='#',
                           names=['x', 'y', 'z', 'radius', 'volume', 'voxels'])
        elif catalog_path.suffix == '.csv':
            df = pd.read_csv(catalog_path)
        elif catalog_path.suffix == '.npy':
            data = np.load(catalog_path)
            if isinstance(data, np.ndarray):
                df = pd.DataFrame(data, columns=['x', 'y', 'z', 'radius', 'volume', 'voxels'])
            else:
                # Assume structured array
                df = pd.DataFrame(data)
        else:
            print(f"Unsupported format: {catalog_path.suffix}")
            return None

        df['finder'] = finder_name
        df['color'] = COLOR_SCHEMES.get(finder_name, '#333333')

        print(f"Loaded {len(df)} voids from {finder_name}")
        return df

    except Exception as e:
        print(f"Error loading {catalog_path}: {e}")
        return None


def create_3d_visualization(void_catalogs, title="Void Distribution", save_path=None):
    """
    Create interactive 3D visualization of void catalogs.

    Parameters:
    -----------
    void_catalogs : dict
        Dictionary of {finder_name: DataFrame}
    title : str
        Plot title
    save_path : str or Path
        Path to save figure
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    for finder_name, df in void_catalogs.items():
        if df is None or df.empty:
            continue

        color = COLOR_SCHEMES.get(finder_name, '#333333')

        # Size coding: radius^2 for better visibility
        sizes = np.clip(df['radius']**2 * 10, 10, 500)

        scatter = ax.scatter(df['x'], df['y'], df['z'],
                           c=color, s=sizes, alpha=0.6,
                           label=f"{finder_name} ({len(df)} voids)")

    ax.set_xlabel('X [Mpc/h]')
    ax.set_ylabel('Y [Mpc/h]')
    ax.set_zlabel('Z [Mpc/h]')
    ax.set_title(title, fontsize=14, pad=20)

    # Equal aspect ratio
    max_range = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]).ptp().max() / 2.0
    mid_x = (ax.get_xlim()[1] + ax.get_xlim()[0]) / 2.0
    mid_y = (ax.get_ylim()[1] + ax.get_ylim()[0]) / 2.0
    mid_z = (ax.get_zlim()[1] + ax.get_zlim()[0]) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D visualization: {save_path}")

    return fig, ax


def create_statistical_summary(void_catalogs, save_path=None):
    """
    Create statistical summary of void populations.

    Parameters:
    -----------
    void_catalogs : dict
        Dictionary of {finder_name: DataFrame}
    save_path : str or Path
        Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Void Population Statistics', fontsize=16)

    all_stats = []

    for finder_name, df in void_catalogs.items():
        if df is None or df.empty:
            continue

        stats = {
            'finder': finder_name,
            'count': len(df),
            'mean_radius': df['radius'].mean(),
            'median_radius': df['radius'].median(),
            'std_radius': df['radius'].std(),
            'min_radius': df['radius'].min(),
            'max_radius': df['radius'].max(),
            'total_volume': df['volume'].sum() if 'volume' in df.columns else 0
        }
        all_stats.append(stats)

    if not all_stats:
        print("No valid void catalogs to analyze")
        return None, None

    stats_df = pd.DataFrame(all_stats)

    # Radius distribution histograms
    ax = axes[0, 0]
    for finder_name, df in void_catalogs.items():
        if df is not None and not df.empty:
            ax.hist(df['radius'], bins=30, alpha=0.6,
                   label=f"{finder_name} (n={len(df)})",
                   color=COLOR_SCHEMES.get(finder_name, '#333333'))
    ax.set_xlabel('Void Radius [Mpc/h]')
    ax.set_ylabel('Count')
    ax.set_title('Radius Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mean radius comparison
    ax = axes[0, 1]
    bars = ax.bar(stats_df['finder'], stats_df['mean_radius'],
                  color=[COLOR_SCHEMES.get(f, '#333333') for f in stats_df['finder']],
                  alpha=0.7)
    ax.set_ylabel('Mean Radius [Mpc/h]')
    ax.set_title('Mean Void Radius by Finder')
    ax.grid(True, alpha=0.3)

    # Void count comparison
    ax = axes[1, 0]
    bars = ax.bar(stats_df['finder'], stats_df['count'],
                  color=[COLOR_SCHEMES.get(f, '#333333') for f in stats_df['finder']],
                  alpha=0.7)
    ax.set_ylabel('Number of Voids')
    ax.set_title('Void Count by Finder')
    ax.grid(True, alpha=0.3)

    # Volume distribution
    ax = axes[1, 1]
    for finder_name, df in void_catalogs.items():
        if df is not None and not df.empty and 'volume' in df.columns:
            volumes = df['volume']
            if len(volumes) > 0:
                ax.hist(np.log10(volumes), bins=30, alpha=0.6,
                       label=f"{finder_name}",
                       color=COLOR_SCHEMES.get(finder_name, '#333333'))
    ax.set_xlabel('log₁₀(Volume) [Mpc³/h³]')
    ax.set_ylabel('Count')
    ax.set_title('Volume Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved statistical summary: {save_path}")

    return fig, stats_df


def create_comparison_plots(void_catalogs, save_path=None):
    """
    Create detailed comparison plots between void finders.

    Parameters:
    -----------
    void_catalogs : dict
        Dictionary of {finder_name: DataFrame}
    save_path : str or Path
        Path to save figure
    """
    if len(void_catalogs) < 2:
        print("Need at least 2 void catalogs for comparison")
        return None

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Void Finder Comparison', fontsize=16)

    # Prepare data
    comparison_data = []
    for finder_name, df in void_catalogs.items():
        if df is not None and not df.empty:
            comparison_data.append(df.assign(finder=finder_name))

    if not comparison_data:
        print("No valid data for comparison")
        return None

    all_data = pd.concat(comparison_data, ignore_index=True)

    # Box plot of radii
    ax = axes[0, 0]
    sns.boxplot(data=all_data, x='finder', y='radius', ax=ax,
                palette=COLOR_SCHEMES)
    ax.set_ylabel('Void Radius [Mpc/h]')
    ax.set_title('Radius Distribution')
    ax.grid(True, alpha=0.3)

    # Violin plot of radii
    ax = axes[0, 1]
    sns.violinplot(data=all_data, x='finder', y='radius', ax=ax,
                   palette=COLOR_SCHEMES)
    ax.set_ylabel('Void Radius [Mpc/h]')
    ax.set_title('Radius Distribution (Violin)')
    ax.grid(True, alpha=0.3)

    # Cumulative distribution
    ax = axes[0, 2]
    for finder_name, df in void_catalogs.items():
        if df is not None and not df.empty:
            sorted_radii = np.sort(df['radius'])
            cdf = np.arange(1, len(sorted_radii) + 1) / len(sorted_radii)
            ax.plot(sorted_radii, cdf,
                   label=f"{finder_name} (n={len(df)})",
                   color=COLOR_SCHEMES.get(finder_name, '#333333'),
                   linewidth=2)
    ax.set_xlabel('Void Radius [Mpc/h]')
    ax.set_ylabel('Cumulative Fraction')
    ax.set_title('Cumulative Radius Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Q-Q plot comparison (first two finders)
    ax = axes[1, 0]
    finders = list(void_catalogs.keys())[:2]
    if len(finders) == 2:
        data1 = void_catalogs[finders[0]]['radius']
        data2 = void_catalogs[finders[1]]['radius']

        # Create Q-Q plot
        quantiles1 = np.percentile(data1, np.linspace(0, 100, 100))
        quantiles2 = np.percentile(data2, np.linspace(0, 100, 100))

        ax.scatter(quantiles1, quantiles2, alpha=0.6, s=30)
        min_val = min(quantiles1.min(), quantiles2.min())
        max_val = max(quantiles1.max(), quantiles2.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        ax.set_xlabel(f'{finders[0]} Radius Quantiles')
        ax.set_ylabel(f'{finders[1]} Radius Quantiles')
        ax.set_title(f'Q-Q Plot: {finders[0]} vs {finders[1]}')
        ax.grid(True, alpha=0.3)

    # Scatter plot: radius vs volume (if available)
    ax = axes[1, 1]
    for finder_name, df in void_catalogs.items():
        if df is not None and not df.empty and 'volume' in df.columns:
            ax.scatter(df['radius'], df['volume'], alpha=0.6, s=30,
                      label=finder_name,
                      color=COLOR_SCHEMES.get(finder_name, '#333333'))
    ax.set_xlabel('Void Radius [Mpc/h]')
    ax.set_ylabel('Void Volume [Mpc³/h³]')
    ax.set_title('Radius vs Volume')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Statistics table
    ax = axes[1, 2]
    ax.axis('off')

    stats_text = "Void Finder Statistics:\n\n"
    for finder_name, df in void_catalogs.items():
        if df is not None and not df.empty:
            stats_text += f"{finder_name}:\n"
            stats_text += f"  Count: {len(df)}\n"
            stats_text += f"  Mean R: {df['radius'].mean():.2f} Mpc/h\n"
            stats_text += f"  Median R: {df['radius'].median():.2f} Mpc/h\n"
            stats_text += f"  Max R: {df['radius'].max():.2f} Mpc/h\n"
            if 'volume' in df.columns:
                stats_text += f"  Total Vol: {df['volume'].sum():.1e} Mpc³/h³\n"
            stats_text += "\n"

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plots: {save_path}")

    return fig


def export_for_paraview(void_catalogs, output_dir=None):
    """
    Export void catalogs in Paraview-compatible format.

    Parameters:
    -----------
    void_catalogs : dict
        Dictionary of {finder_name: DataFrame}
    output_dir : str or Path
        Output directory for Paraview files
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "paraview"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for finder_name, df in void_catalogs.items():
        if df is None or df.empty:
            continue

        # Create CSV file for Paraview
        paraview_df = df[['x', 'y', 'z', 'radius']].copy()
        if 'volume' in df.columns:
            paraview_df['volume'] = df['volume']

        csv_path = output_dir / f"{finder_name}_voids_paraview.csv"
        paraview_df.to_csv(csv_path, index=False)
        print(f"Exported {finder_name} voids for Paraview: {csv_path}")

    print(f"\nParaview files saved in: {output_dir}")
    print("Import as CSV files in Paraview, then use 'Table To Points' filter")


def main():
    """
    Main premium void visualization function.
    """
    print("=" * 80)
    print("PREMIUM VOID VISUALIZER - ADVANCED 3D ANALYSIS SUITE")
    print("=" * 80)

    # Define void catalog paths (adjust as needed)
    catalog_paths = {
        'PyCosmoMMF': DATA_DIR / "pycosmomf_voids" / "voids_pycosmomf.csv",
        'REVOLVER': DATA_DIR / "revolver_simple_voids" / "zobov_voids.npy",
        'VoidFinder': DATA_DIR / "voidfinder" / "voids_voidfinder.txt"
    }

    # Load void catalogs
    void_catalogs = {}
    for finder_name, path in catalog_paths.items():
        catalog = load_void_catalog(path, finder_name)
        if catalog is not None:
            void_catalogs[finder_name] = catalog

    if not void_catalogs:
        print("No void catalogs found. Please check paths and data availability.")
        return

    print(f"\nLoaded {len(void_catalogs)} void catalogs")

    # Create visualizations
    print("\nCreating 3D visualization...")
    fig_3d, ax_3d = create_3d_visualization(
        void_catalogs,
        title="DESI ELG Void Distribution - Multi-Finder Comparison",
        save_path=OUTPUT_DIR / "void_distribution_3d.png"
    )

    print("Creating statistical summary...")
    fig_stats, stats_df = create_statistical_summary(
        void_catalogs,
        save_path=OUTPUT_DIR / "void_statistics.png"
    )

    if len(void_catalogs) > 1:
        print("Creating comparison plots...")
        fig_comp = create_comparison_plots(
            void_catalogs,
            save_path=OUTPUT_DIR / "void_comparison.png"
        )
f"Min Density: {np.min(densities):.4f} gal/Mpc³\n"

f"Max Density: {np.max(densities):.4f} gal/Mpc³\n"

f"Density Ratio: {np.max(densities)/np.min(densities):.2f}x"

)

ax.text(0.1,0.5,summary_text,fontsize=11,family='monospace',

bbox=dict(boxstyle='round,pad=1',facecolor='lightblue',alpha=0.3),

verticalalignment='center')

plt.tight_layout()

output_file=self.output_dir/f'void_{void_idx}_density_profile.png'

plt.savefig(output_file,dpi=300,bbox_inches='tight')

output_file_pdf=self.output_dir/f'void_{void_idx}_density_profile.pdf'

plt.savefig(output_file_pdf,dpi=300,bbox_inches='tight')

plt.close()

print(f"✓ Saved: {output_file}")

returnoutput_file

defcreate_void_comparison_gallery(self,n_voids:int=6):

        """Create gallery of multiple voids for comparison."""

centers,radii=self.load_void_catalog()

ifcentersisNone:

            return

galaxies,_=self.load_galaxy_data()

ifgalaxiesisNone:

            galaxies=np.random.uniform(-1000,1000,(10000,3))

n_voids=min(n_voids,len(centers))

fig=plt.figure(figsize=(18,12),dpi=300)

fig.suptitle(f'Void Comparison Gallery - Top {n_voids} Largest Voids',

fontsize=18,fontweight='bold')

sorted_indices=np.argsort(-radii)[:n_voids]

fori,void_idxinenumerate(sorted_indices):

            ax=fig.add_subplot(2,3,i+1,projection='3d')

void_center=centers[void_idx]

void_radius=radii[void_idx]

region_galaxies,_=self.select_void_region(

void_center,void_radius,galaxies,region_factor=2.0

)

gal_sample=region_galaxies[np.random.choice(

len(region_galaxies),min(5000,len(region_galaxies)),replace=False

)]

distances=np.linalg.norm(gal_sample-void_center,axis=1)

scatter=ax.scatter(gal_sample[:,0],gal_sample[:,1],gal_sample[:,2],

c=distances,cmap='viridis',s=2,alpha=0.5)

ax.scatter(void_center[0],void_center[1],void_center[2],

c='red',s=100,marker='*',edgecolors='white',linewidth=1)

ax.set_xlabel('X',fontsize=9)

ax.set_ylabel('Y',fontsize=9)

ax.set_zlabel('Z',fontsize=9)

ax.set_title(f'Void #{void_idx}\nR={void_radius:.1f} Mpc',fontsize=11,fontweight='bold')

ax.tick_params(labelsize=8)

plt.tight_layout()

output_file=self.output_dir/'void_comparison_gallery.png'

plt.savefig(output_file,dpi=300,bbox_inches='tight')

output_file_pdf=self.output_dir/'void_comparison_gallery.pdf'

plt.savefig(output_file_pdf,dpi=300,bbox_inches='tight')

plt.close()

print(f"✓ Saved: {output_file}")

returnoutput_file

defrun_complete_visualization(self,void_idx:int=0):

        """Run complete premium visualization suite."""

print("\n"+"="*70)

print("PREMIUM VOID VISUALIZATION SUITE")

print("="*70+"\n")

print(f"Creating visualizations for Void #{void_idx}...\n")

self.create_premium_3d_void_plot(void_idx)

self.create_density_profile_plot(void_idx)

self.create_void_comparison_gallery()

print("\n"+"="*70)

print("VISUALIZATION COMPLETE")

print(f"All outputs saved to: {self.output_dir}")

print("="*70)

if__name__=="__main__":

    importargparse

parser=argparse.ArgumentParser(

description='Premium Void Visualization Suite'

)

parser.add_argument('--void-idx',type=int,default=0,

help='Void index to visualize (default: 0)')

parser.add_argument('--gallery',action='store_true',

help='Create comparison gallery of multiple voids')

args=parser.parse_args()

visualizer=PremiumVoidVisualizer()

visualizer.run_complete_visualization(void_idx=args.void_idx)

