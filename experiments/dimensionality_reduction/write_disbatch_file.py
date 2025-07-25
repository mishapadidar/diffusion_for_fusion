import numpy as np
import os

"""
Generate a disbatch file
"""
# total number of jobs
n_runs = 100
# number of samples in each job
n_samples = 100

# make an output directory
disbatch_outdir = "./disbatch_output"
os.makedirs(disbatch_outdir, exist_ok=True)

# write the disBatch file
f = open("Tasks", "w")
for ii in range(n_runs):
  line = f"( cd /mnt/home/mpadidar/code/ml/diffusion_for_fusion/experiments/dimensionality_reduction ; ml python ; source ../../../env_diffusion_for_fusion/bin/activate ; python generate_plot_data.py {n_samples} ) &> ./disbatch_output/task_{ii}.log"
  print(line)
  f.write(line)
  f.write("\n")

