utf-8frompathlibimportPath

importnumpyasnp

importmatplotlib.pyplotasplt

base=Path("data/desi/edr/processed/voidfinder")

a_path=base/"survey_aware_voids.txt"

b_path=base/"zobov_alternative_voids.txt"

defload_points(p):

    try:

        d=np.loadtxt(p)

exceptException:

        try:

            d_struct=np.genfromtxt(p,delimiter=None,names=True,dtype=None,encoding=None)

ifgetattr(d_struct,"dtype").names:

                cols=d_struct.dtype.names

d=np.column_stack([d_struct[n]fornincols])

else:

                d=np.genfromtxt(p,delimiter=None,skip_header=1)

exceptException:

            importpandasaspd

df=pd.read_csv(p,delim_whitespace=True,comment='#',header=0,engine="python")

df=df.apply(pd.to_numeric,errors='coerce')

df=df.dropna(axis=1,how='all')

d=df.values

d=np.array(d)

ifd.ndim==1:

        ifd.size%3==0:

            d=d.reshape(-1,3)

elifd.size>=2:

            d=d.reshape(-1,2)

ifd.shape[1]>=3:

        returnd[:,:3]

elifd.shape[1]==2:

        returnnp.hstack((d,np.zeros((d.shape[0],1))))

else:

        raiseValueError(f"cannot interpret file {p}")

a=load_points(a_path)

b=load_points(b_path)

print(a.min(axis=0),a.max(axis=0))

print(b.min(axis=0),b.max(axis=0))

plt.figure(figsize=(10,10))

plt.scatter(

b[:,0],b[:,1],

s=6,c="C1",alpha=0.6,

label="zobov_alt"

)

plt.scatter(

a[:,0],a[:,1],

s=60,marker="x",linewidths=2,

c="C0",label="survey_aware"

)

xmin=min(a[:,0].min(),b[:,0].min())

xmax=max(a[:,0].max(),b[:,0].max())

ymin=min(a[:,1].min(),b[:,1].min())

ymax=max(a[:,1].max(),b[:,1].max())

plt.xlim(xmin,xmax)

plt.ylim(ymin,ymax)

plt.gca().set_aspect("equal")

plt.legend()

plt.show()