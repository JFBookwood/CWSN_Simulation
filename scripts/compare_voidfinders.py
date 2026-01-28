utf-8"""

Void Finder Comparison Tool

Compares different void finding algorithms to assess their performance

on survey geometry data. Includes VIDE (recommended for DESI),

custom kNN-based algorithm, and synthetic test cases.

"""

importos

importsys

importargparse

importnumpyasnp

importpandasaspd

importmatplotlib.pyplotasplt

frompathlibimportPath

fromsklearn.neighborsimportNearestNeighbors

ROOT=Path(__file__).parent.parent

ifstr(ROOT)notinsys.path:

    sys.path.append(str(ROOT))

ifstr(ROOT/'scripts')notinsys.path:

    sys.path.append(str(ROOT/'scripts'))

try:

    fromrun_voidfinderimportestimate_density_kNN,find_minima_candidates,suppress_overlaps

exceptImportError:

    print("Warning: Could not import run_voidfinder functions")

estimate_density_kNN=None

find_minima_candidates=None

suppress_overlaps=None

defmake_synthetic_cube(n_points=20000,box_size=3000.0,seed=42,clump=False):

    rng=np.random.default_rng(seed)

pts=rng.uniform(0.0,box_size,size=(n_points,3))

ifclump:

        centers=rng.uniform(0.25*box_size,0.75*box_size,size=(5,3))

weights=rng.uniform(0.0,1.0,size=n_points)

idx=weights>0.8

ifnp.any(idx):

            attract=centers[rng.integers(0,len(centers),size=idx.sum())]

pts[idx]=0.7*pts[idx]+0.3*attract

returnpts

defmake_randoms(n_rand=200000,box_size=3000.0,seed=123):

    rng=np.random.default_rng(seed)

rnd=rng.uniform(0.0,box_size,size=(n_rand,3))

returnrnd

defmy_voidfinder(data,randoms,k=16,seed_frac=0.003,min_sep=8.0,target_q=0.99,max_voids=300):

    logdens=estimate_density_kNN(data,randoms,k=k)

seeds=find_minima_candidates(data,logdens,frac=seed_frac,min_separation=min_sep)

NN=NearestNeighbors

nn_data=NN(n_neighbors=2,algorithm='auto')

nn_rnd=NN(n_neighbors=128,algorithm='auto')

nn_data.fit(data)

nn_rnd.fit(randoms)

def_grow(center):

        d_data,_=nn_data.kneighbors(center.reshape(1,-1),n_neighbors=2,return_distance=True)

r0=float(d_data[0,0])

r1=float(d_data[0,1])ifd_data.shape[1]>1elsefloat(d_data[0,0])

r_data=r1ifr0<1e-9elser0

d_rnd,_=nn_rnd.kneighbors(center.reshape(1,-1),n_neighbors=128,return_distance=True)

r_rnd_q=float(np.quantile(d_rnd[0],target_q))

returnmax(0.0,min(r_data,r_rnd_q))

radii=np.array([_grow(c)forcinseeds],dtype=float)

idx_keep=suppress_overlaps(seeds,radii,overlap_frac=0.3)

centers_keep=seeds[idx_keep]

radii_keep=radii[idx_keep]

order=np.argsort(-radii_keep)[:max_voids]

returncenters_keep[order],radii_keep[order]

defgrid_empty_spheres(data,randoms,grid_n=22,target_q=0.99,max_voids=300):

    mins=data.min(axis=0)

maxs=data.max(axis=0)

xs=np.linspace(mins[0],maxs[0],grid_n)

ys=np.linspace(mins[1],maxs[1],grid_n)

zs=np.linspace(mins[2],maxs[2],grid_n)

grid=np.stack(np.meshgrid(xs,ys,zs,indexing='ij'),axis=-1).reshape(-1,3)

NN=NearestNeighbors

nn_data=NN(n_neighbors=1).fit(data)

nn_rnd=NN(n_neighbors=128).fit(randoms)

d_data,_=nn_data.kneighbors(grid,n_neighbors=1,return_distance=True)

d_rnd,_=nn_rnd.kneighbors(grid,n_neighbors=128,return_distance=True)

r_data=d_data[:,0]

r_rnd_q=np.quantile(d_rnd,target_q,axis=1)

radii=np.maximum(0.0,np.minimum(r_data,r_rnd_q))

order=np.argsort(-radii)[:max_voids]

returngrid[order],radii[order]

defrandom_center_spheres(data,randoms,n_centers=8000,seed=7,target_q=0.99,max_voids=300):

    rng=np.random.default_rng(seed)

mins=data.min(axis=0)

maxs=data.max(axis=0)

centers=rng.uniform(mins,maxs,size=(n_centers,3))

NN=NearestNeighbors

nn_data=NN(n_neighbors=1).fit(data)

nn_rnd=NN(n_neighbors=128).fit(randoms)

d_data,_=nn_data.kneighbors(centers,n_neighbors=1,return_distance=True)

d_rnd,_=nn_rnd.kneighbors(centers,n_neighbors=128,return_distance=True)

r_data=d_data[:,0]

r_rnd_q=np.quantile(d_rnd,target_q,axis=1)

radii=np.maximum(0.0,np.minimum(r_data,r_rnd_q))

order=np.argsort(-radii)[:max_voids]

returncenters[order],radii[order]

defplot_comparison(*args):

    """

    Plot comparison of void finding algorithms.

    Args can be: centers1, radii1, centers2, radii2, ..., centersN, radiiN, out_png

    """

iflen(args)<3:

        raiseValueError("Need at least one method and output file")

out_png=args[-1]

method_data=args[:-1]

iflen(method_data)%2!=0:

        raiseValueError("Method data must be pairs of (centers, radii)")

n_methods=len(method_data)//2

method_names=['MyVF','GES','RCS','VIDE'][:n_methods]

colors=['#ff6f61','#6baed6','#74c476','#9e6ebd'][:n_methods]

cmaps=['inferno','Blues','Greens','Purples'][:n_methods]

plt.style.use('dark_background')

fig=plt.figure(figsize=(15,5),dpi=160)

ax1=fig.add_subplot(1,3,1)

ax2=fig.add_subplot(1,3,2)

ax3=fig.add_subplot(1,3,3,projection='3d')

fori,(centers,radii)inenumerate(zip(method_data[::2],method_data[1::2])):

        iflen(radii)>0:

            ax1.hist(radii,bins=30,alpha=0.7,label=method_names[i],facecolor=colors[i],edgecolor='black')

ax1.set_xlabel('Radius')

ax1.set_ylabel('Häufigkeit')

ax1.set_title('Void-Radien (Histogramm)')

ax1.legend()

fori,(centers,radii)inenumerate(zip(method_data[::2],method_data[1::2])):

        iflen(radii)>0:

            sorted_radii=np.sort(radii)

y=np.linspace(0,1,len(sorted_radii),endpoint=False)

ax2.plot(sorted_radii,y,label=method_names[i],color=colors[i],lw=2)

ax2.set_xlabel('Radius')

ax2.set_ylabel('CDF')

ax2.set_title('Void-Radien (CDF)')

ax2.legend()

fori,(centers,radii)inenumerate(zip(method_data[::2],method_data[1::2])):

        iflen(centers)>0andlen(radii)>0:

            radii_norm=(radii-radii.min())/(radii.max()-radii.min())ifradii.max()>radii.min()elseradii

ax3.scatter(centers[:,0],centers[:,1],centers[:,2],c=radii_norm,

s=8ifi==0else6,cmap=cmaps[i],

alpha=0.9ifi==0else0.6,label=method_names[i])

ax3.set_title('Zentren (Farbe=Radius)')

ax3.set_xlabel('x')

ax3.set_ylabel('y')

ax3.set_zlabel('z')

ax3.legend(loc='upper right')

fig.tight_layout()

os.makedirs(os.path.dirname(out_png),exist_ok=True)

fig.savefig(out_png,bbox_inches='tight')

plt.close(fig)

defrun_survey_aware_comparison(data_file:Path,randoms_file:Path,max_voids:int=300)->tuple[np.ndarray,np.ndarray]:

    """

    Run survey-aware void finder and extract void catalog for comparison.

    """

try:

        fromsurvey_aware_voidfinderimportSurveyAwareVoidFinder,load_data

print("Running survey-aware void finder for comparison...")

data,randoms=load_data(str(data_file),str(randoms_file))

finder=SurveyAwareVoidFinder(data,randoms)

voids=finder.find_voids()

ifvoids:

            centers=np.array([void['center']forvoidinvoids[:max_voids]])

radii=np.array([void['radius']forvoidinvoids[:max_voids]])

print(f"Survey-aware found {len(centers)} voids for comparison")

returncenters,radii

else:

            print("No voids found by survey-aware method")

returnnp.empty((0,3)),np.empty(0)

exceptImportError:

        print("Survey-aware void finder not available")

returnnp.empty((0,3)),np.empty(0)

exceptExceptionase:

        print(f"Error running survey-aware comparison: {e}")

returnnp.empty((0,3)),np.empty(0)

defrun_vide_comparison(data_file:Path,randoms_file:Path,max_voids:int=300)->tuple[np.ndarray,np.ndarray]:

    """

    Run VIDE and extract void catalog for comparison.

    This is a placeholder - actual VIDE integration would require

    parsing VIDE output files.

    """

print("VIDE comparison not implemented yet (requires VIDE installation)")

returnnp.empty((0,3)),np.empty(0)

defmain():

    parser=argparse.ArgumentParser(

description='Compare void finding algorithms on synthetic and real data',

formatter_class=argparse.RawDescriptionHelpFormatter,

epilog="""

Compares void finding algorithms:

MyVF        - Custom kNN-based algorithm (improved for boundaries)

GES         - Grid Empty Spheres (systematic grid search)

RCS         - Random Center Spheres (stochastic search)

Survey-Aware - Survey-aware void finder (handles DESI geometry artifacts)

VIDE        - VIDE algorithm (recommended for survey geometries)

Survey-Aware and VIDE are specifically designed for survey geometries like DESI.

        """

)

parser.add_argument('--data',type=Path,

help='Real data file (if not provided, uses synthetic data)')

parser.add_argument('--randoms',type=Path,

help='Real randoms file (required if --data is provided)')

parser.add_argument('--n',type=int,default=20000,

help='Number of synthetic data points')

parser.add_argument('--rand',type=int,default=200000,

help='Number of synthetic randoms')

parser.add_argument('--box',type=float,default=3000.0,

help='Box size for synthetic data')

parser.add_argument('--seed',type=int,default=42,

help='Random seed')

parser.add_argument('--k',type=int,default=16,

help='k for kNN density estimation')

parser.add_argument('--max_voids',type=int,default=300,

help='Maximum number of voids per method')

parser.add_argument('--out',type=Path,

default=ROOT/'results'/'plots'/'compare_voidfinders.png',

help='Output plot file')

parser.add_argument('--include_vide',action='store_true',

help='Include VIDE in comparison (requires VIDE installation)')

args=parser.parse_args()

ifany(xisNoneforxin[estimate_density_kNN,find_minima_candidates,suppress_overlaps]):

        raiseSystemExit('Could not import run_voidfinder components. Ensure scripts/run_voidfinder.py is available.')

ifargs.dataandargs.randoms:

        print(f"Using real data: {args.data}")

print(f"Using real randoms: {args.randoms}")

data=np.random.uniform(0,args.box,(args.n,3))

randoms=np.random.uniform(0,args.box,(args.rand,3))

print("Warning: Real data loading not implemented yet")

else:

        print("Using synthetic data")

data=make_synthetic_cube(n_points=args.n,box_size=args.box,seed=args.seed,clump=True)

randoms=make_randoms(n_rand=args.rand,box_size=args.box,seed=args.seed+1)

results={}

print("Running MyVF (kNN-based)...")

centers_A,radii_A=my_voidfinder(data,randoms,k=args.k,max_voids=args.max_voids)

results['MyVF']=(centers_A,radii_A)

print("Running GES (Grid Empty Spheres)...")

centers_B,radii_B=grid_empty_spheres(data,randoms,grid_n=22,max_voids=args.max_voids)

results['GES']=(centers_B,radii_B)

print("Running RCS (Random Center Spheres)...")

centers_C,radii_C=random_center_spheres(data,randoms,n_centers=8000,seed=args.seed+2,max_voids=args.max_voids)

results['RCS']=(centers_C,radii_C)

ifargs.dataandargs.randoms:

        print("Running Survey-Aware...")

centers_SA,radii_SA=run_survey_aware_comparison(args.data,args.randoms,args.max_voids)

iflen(centers_SA)>0:

            results['Survey-Aware']=(centers_SA,radii_SA)

ifargs.include_vide:

        print("Running VIDE...")

ifargs.dataandargs.randoms:

            centers_D,radii_D=run_vide_comparison(args.data,args.randoms,args.max_voids)

iflen(centers_D)>0:

                results['VIDE']=(centers_D,radii_D)

else:

            print("VIDE requires real data files (--data and --randoms)")

args.out.parent.mkdir(parents=True,exist_ok=True)

valid_results={name:(centers,radii)forname,(centers,radii)inresults.items()

iflen(centers)>0andlen(radii)>0andlen(centers)==len(radii)}

iflen(valid_results)>=2:

        method_names=list(valid_results.keys())

centers_list=[valid_results[name][0]fornameinmethod_names]

radii_list=[valid_results[name][1]fornameinmethod_names]

print(f"Valid methods: {method_names}")

fori,(name,centers,radii)inenumerate(zip(method_names,centers_list,radii_list)):

            print(f"  {name}: centers shape {centers.shapeifhasattr(centers,'shape')elsetype(centers)}, radii shape {radii.shapeifhasattr(radii,'shape')elsetype(radii)}")

method_data=[]

forcenters,radiiinzip(centers_list,radii_list):

            method_data.extend([centers,radii])

plot_comparison(*method_data,args.out)

dfs=[]

forname,(centers,radii)inresults.items():

            iflen(centers)>0:

                df=pd.DataFrame({

'x':centers[:,0],

'y':centers[:,1],

'z':centers[:,2],

'radius':radii,

'method':name

})

dfs.append(df)

ifdfs:

            df_combined=pd.concat(dfs,ignore_index=True)

csv_file=args.out.with_suffix('.csv')

df_combined.to_csv(csv_file,index=False)

print(f'Saved CSV: {csv_file}')

print(f'Saved plot: {args.out}')

print(f'Compared {len(valid_results)} methods: {", ".join(valid_results.keys())}')

else:

        print(f"Need at least 2 valid methods for comparison, got {len(valid_results)}")

return

if__name__=='__main__':

    main()