import numpy as np
sig = np.array([0.05, 0.01, 0.01])
cov = np.diag(sig**2)
with open("initial_cov.covmat", "w") as f:
    f.write("# A beta C\n")
    for row in cov:
        f.write(" ".join(f"{x:.6f}" for x in row) + "\n")