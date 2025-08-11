import numpy as np
import os

"""
Generate a disbatch file
"""
# indexes of the conditions
arg_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]

# number of runs per arg
n_runs_per = 20

# make an output directory
disbatch_outdir = "./disbatch_output"
os.makedirs(disbatch_outdir, exist_ok=True)

# write the disBatch file
f = open("Tasks", "w")
for jj in range(n_runs_per):
    for ii, ag in enumerate(arg_list):
        line = f"( cd /mnt/home/mpadidar/code/ml/diffusion_for_fusion/experiments/out_of_sample ; ml python ; source ../../../env_diffusion_for_fusion/bin/activate ; python evaluate_model.py {ag} ) &> ./disbatch_output/task_{ii}_{jj}.log"
        print(line)
        f.write(line)
        f.write("\n")

