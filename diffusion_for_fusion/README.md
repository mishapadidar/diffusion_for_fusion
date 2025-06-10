In this directory we provide modules to do the heavy computations such as 
building a diffusion model and running the vacuum solver.

The conditional and vanilla (unconditional) diffusion models are stored in `ddpm_conditional_diffusion.py` and `ddpm_fusion.py`.

To evaluate the properties of stellarators we use `evaluate_configuration.py` which calls upon the vacuum solver `sheet_current.py`.

`autoencoder.py` contains code to train an autoencoder.
