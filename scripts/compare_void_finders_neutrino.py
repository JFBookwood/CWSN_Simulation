utf-8importnumpyasnp
importpandasaspd
importmatplotlib.pyplotasplt
frompathlibimportPath
importsubprocess
importseabornassns
fromtypingimportDict,List,Tuple

sns.set_context("paper")
sns.set_style("whitegrid")
plt.rcParams['font.size']=12
plt.rcParams['axes.labelsize']=14
plt.rcParams['axes.titlesize']=16

COLORS=sns.color_palette("husl",8)

classVoidFinderComparison:
    def__init__(self,output_dir="results/void_visualizations/comparison"):
        self.output_dir=Path(output_dir)
self.output_dir.mkdir(parents=True,exist_ok=True)
self.base_dir=Path("data/desi/edr/processed")
self.void_dir=self.base_dir/"voidfinder"

defrun_void_finder(self,algorithm:str,max_voids:int=500)->Tuple[np.ndarray,np.ndarray,str]:
        """Run a specific void finder algorithm."""
print(f"Running {algorithm} void finder...")

ifalgorithm=="vide":
            returnself._run_vide(max_voids)
elifalgorithm=="zobov":
            returnself._run_zobov(max_voids)
elifalgorithm=="knn":
            returnself._run_knn(max_voids)
else:
            raiseValueError(f"Unknown algorithm: {algorithm}")

def_run_vide(self,max_voids:int)->Tuple[np.ndarray,np.ndarray,str]:
        """Run VIDE void finder."""
try:
            vide_exe="vide"
data_file=self.base_dir/"vide_input"/"mpc"/"data.txt"
randoms_file=self.base_dir/"vide_input"/"mpc"/"randoms.txt"
output_dir=self.void_dir/"vide_results"

cmd=[vide_exe,"-d",str(data_file),"-r",str(randoms_file),"-o",str(output_dir)]
result=subprocess.run(cmd,capture_output=True,text=True,timeout=300)

ifresult.returncode==0:
                catalog_file=output_dir/"catalog.txt"
ifcatalog_file.exists():
                    data=np.loadtxt(catalog_file)
centers=data[:,:3]
radii=data[:,3]ifdata.shape[1]>3elsenp.ones(len(centers))*10
returncenters[:max_voids],radii[:max_voids],"VIDE"
print("VIDE failed, using existing catalog")
except:
            pass

returnself._load_existing_catalog(max_voids),"VIDE (existing)"

def_run_zobov(self,max_voids:int)->Tuple[np.ndarray,np.ndarray,str]:
        """Run ZOBOV void finder."""
try:
            fromzobov_integrationimportZOBOVIntegrator
integrator=ZOBOVIntegrator()
result=integrator.run_complete_zobov_analysis(subsample_factor=10)

ifresult:
                df=pd.read_csv(result)
centers=df[['x','y','z']].values
radii=df['radius_mpc'].values
returncenters[:max_voids],radii[:max_voids],"ZOBOV"
except:
            pass

returnself._run_zobov_alternative(max_voids),"ZOBOV (alternative)"

def_run_zobov_alternative(self,max_voids:int)->Tuple[np.ndarray,np.ndarray,str]:
        """Alternative ZOBOV implementation."""
fromscipy.spatialimportVoronoi

data_file=self.base_dir/"vide_input"/"mpc"/"data.txt"
data=np.loadtxt(data_file)
ifdata.shape[1]>3:
            data=data[:,:3]

sample_size=min(50000,len(data))
idx=np.random.choice(len(data),sample_size,replace=False)
data_sample=data[idx]

vor=Voronoi(data_sample)

void_centers=[]
void_radii=[]

forridgeinvor.ridge_vertices:
            if-1notinridge:
                vertices=vor.vertices[ridge]
iflen(vertices)>=3:
                    center=np.mean(vertices,axis=0)
distances=np.linalg.norm(vertices-center,axis=1)
radius=np.max(distances)

if8.0<radius<80.0:
                        void_centers.append(center)
void_radii.append(radius)

centers=np.array(void_centers[:max_voids])
radii=np.array(void_radii[:max_voids])

returncenters,radii,"ZOBOV (alternative)"

def_run_knn(self,max_voids:int)->Tuple[np.ndarray,np.ndarray,str]:
        """Run kNN-based void finder."""
try:
            fromrun_voidfinderimportload_xyz_txt,estimate_density_kNN,find_minima_candidates,suppress_overlaps

data_file=self.base_dir/"vide_input"/"mpc"/"data.txt"
randoms_file=self.base_dir/"vide_input"/"mpc"/"randoms.txt"

data=load_xyz_txt(data_file)
randoms=load_xyz_txt(randoms_file)

subsample=min(len(randoms),200000)
idx=np.random.choice(len(randoms),subsample,replace=False)
randoms=randoms[idx]

logdens=estimate_density_kNN(data,randoms,k=16)
seeds=find_minima_candidates(data,logdens,frac=0.005,min_separation=10.0)

fromsklearn.neighborsimportNearestNeighbors
nn_data=NearestNeighbors(n_neighbors=2,algorithm='auto').fit(data)
nn_rnd=NearestNeighbors(n_neighbors=128,algorithm='auto').fit(randoms)

radii=[]
forseedinseeds[:max_voids]:
                dists_data,_=nn_data.kneighbors(seed.reshape(1,-1),n_neighbors=2,return_distance=True)
r_data=dists_data[0,1]ifdists_data.shape[1]>1elsedists_data[0,0]

dists_rnd,_=nn_rnd.kneighbors(seed.reshape(1,-1),n_neighbors=128,return_distance=True)
r_rnd=np.quantile(dists_rnd[0],0.99)

radius=min(r_data,r_rnd)
radii.append(radius)

returnnp.array(seeds[:max_voids]),np.array(radii),"kNN"

exceptExceptionase:
            print(f"kNN failed: {e}")
returnself._load_existing_catalog(max_voids),"kNN (existing)"

def_load_existing_catalog(self,max_voids:int)->Tuple[np.ndarray,np.ndarray]:
        """Load existing void catalog."""
catalog_file=self.void_dir/"catalog.csv"
df=pd.read_csv(catalog_file)
centers=df[['x','y','z']].values[:max_voids]
radii=df['radius_mpc'].values[:max_voids]
returncenters,radii

defcompare_algorithms(self,algorithms:List[str]=None,max_voids:int=300)->Dict:
        """Compare different void finding algorithms."""
ifalgorithmsisNone:
            algorithms=["knn","vide","zobov"]

results={}

foralginalgorithms:
            try:
                centers,radii,label=self.run_void_finder(alg,max_voids)
results[alg]={
'centers':centers,
'radii':radii,
'label':label,
'n_voids':len(centers),
'mean_radius':np.mean(radii)iflen(radii)>0else0,
'max_radius':np.max(radii)iflen(radii)>0else0
}
print(f"{label}: {len(centers)} voids, mean radius {results[alg]['mean_radius']:.1f} Mpc")
exceptExceptionase:
                print(f"Failed to run {alg}: {e}")
continue

returnresults

defcreate_comparison_plots(self,results:Dict):
        """Create comparison plots for different algorithms."""
fig,axes=plt.subplots(2,3,figsize=(18,12),dpi=300)

algorithms=list(results.keys())
colors=COLORS[:len(algorithms)]

fori,(alg,data)inenumerate(results.items()):
            centers,radii=data['centers'],data['radii']
label=data['label']

axes[0,0].hist(radii,bins=20,alpha=0.7,label=label,color=colors[i],density=True)
axes[0,1].scatter(centers[:,0],centers[:,1],c=radii,s=radii*5,
alpha=0.6,label=label,cmap=f'Blues',edgecolors='black',linewidth=0.5)

axes[0,0].set_xlabel('Void Radius [Mpc]')
axes[0,0].set_ylabel('Density')
axes[0,0].set_title('Void Size Distributions')
axes[0,0].legend()
axes[0,0].grid(True,alpha=0.3)

axes[0,1].set_xlabel('X [Mpc]')
axes[0,1].set_ylabel('Y [Mpc]')
axes[0,1].set_title('Void Positions (Color = Radius)')
axes[0,1].set_aspect('equal')
axes[0,1].grid(True,alpha=0.3)

ax=axes[0,2]
fori,(alg,data)inenumerate(results.items()):
            radii=data['radii']
sorted_radii=np.sort(radii)
cdf=np.arange(1,len(sorted_radii)+1)/len(sorted_radii)
ax.plot(sorted_radii,cdf,label=data['label'],color=colors[i],linewidth=2)

ax.set_xlabel('Void Radius [Mpc]')
ax.set_ylabel('Cumulative Fraction')
ax.set_title('Void Size CDF')
ax.legend()
ax.grid(True,alpha=0.3)

ax=axes[1,0]
stats_data=[]
labels=[]
foralg,datainresults.items():
            stats_data.append([data['n_voids'],data['mean_radius'],data['max_radius']])
labels.append(data['label'])

stats_array=np.array(stats_data)
x_pos=np.arange(len(labels))

bars1=ax.bar(x_pos-0.2,stats_array[:,0],0.2,label='N Voids',alpha=0.8,color=COLORS[0])
bars2=ax.bar(x_pos,stats_array[:,1],0.2,label='Mean Radius',alpha=0.8,color=COLORS[1])
bars3=ax.bar(x_pos+0.2,stats_array[:,2],0.2,label='Max Radius',alpha=0.8,color=COLORS[2])

ax.set_xlabel('Algorithm')
ax.set_ylabel('Value')
ax.set_title('Algorithm Comparison')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels,rotation=45)
ax.legend()
ax.grid(True,alpha=0.3,axis='y')

ax=axes[1,1]
fori,(alg,data)inenumerate(results.items()):
            radii=data['radii']
volumes=(4/3)*np.pi*radii**3
ax.scatter(radii,volumes,alpha=0.7,s=50,label=data['label'],color=colors[i])

ax.set_xlabel('Void Radius [Mpc]')
ax.set_ylabel(r'Void Volume [Mpc³]')
ax.set_title('Radius-Volume Relation')
ax.set_yscale('log')
ax.legend()
ax.grid(True,alpha=0.3)

ax=axes[1,2]
ax.axis('off')
summary_text="Void Finder Comparison Summary:\n\n"
foralg,datainresults.items():
            summary_text+=f"{data['label']}:\n"
summary_text+=f"  • {data['n_voids']} voids\n"
summary_text+=".1f"
summary_text+=".1f"
summary_text+="\n"

summary_text+="\nRecommendation:\n• VIDE for survey geometries\n• ZOBOV for irregular distributions\n• kNN for quick analysis"
ax.text(0.05,0.95,summary_text,transform=ax.transAxes,
fontsize=11,verticalalignment='top',fontfamily='monospace')

plt.suptitle('Void Finder Algorithm Comparison',fontsize=18,fontweight='bold',y=0.98)
plt.tight_layout()
plt.savefig(self.output_dir/'void_algorithm_comparison.pdf',dpi=300,bbox_inches='tight')
plt.savefig(self.output_dir/'void_algorithm_comparison.png',dpi=300,bbox_inches='tight')
plt.close()

print(f"Comparison plots saved to: {self.output_dir}")

defrun_neutrino_analysis_comparison(self,results:Dict):
        """Compare neutrino mass constraints from different void catalogs."""
print("Comparing neutrino mass constraints...")

foralg,datainresults.items():
            centers,radii=data['centers'],data['radii']

catalog_file=self.output_dir/f"{alg}_void_catalog.csv"
df=pd.DataFrame({
'x':centers[:,0],
'y':centers[:,1],
'z':centers[:,2],
'radius_mpc':radii
})
df.to_csv(catalog_file,index=False)

print(f"Saved {alg} catalog: {catalog_file} ({len(centers)} voids)")

print("\nTo run neutrino analysis with different catalogs:")
print("1. Copy desired catalog to data/desi/edr/processed/voidfinder/catalog.csv")
print("2. Run: python scripts/vgcf.py --filter-stripes")
print("3. Run: python scripts/create_filtered_likelihood_npz.py")
print("4. Run: python scripts/run_cobaya_filtered.py")

if__name__=="__main__":
    importargparse

parser=argparse.ArgumentParser(description='Compare void finding algorithms for neutrino mass analysis')
parser.add_argument('--algorithms',nargs='+',default=['knn','vide','zobov'],
help='Algorithms to compare')
parser.add_argument('--max-voids',type=int,default=300,
help='Maximum voids per algorithm')
parser.add_argument('--neutrino-analysis',action='store_true',
help='Prepare catalogs for neutrino mass analysis')

args=parser.parse_args()

comparator=VoidFinderComparison()

results=comparator.compare_algorithms(args.algorithms,args.max_voids)

ifresults:
        comparator.create_comparison_plots(results)

ifargs.neutrino_analysis:
            comparator.run_neutrino_analysis_comparison(results)
else:
        print("No results to compare. Check algorithm implementations.")