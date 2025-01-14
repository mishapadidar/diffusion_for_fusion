import torch
from diffusion_for_fusion.ddpm_fusion import (init_diffusion_model_from_config)
# from diffusion_for_fusion.evaluate_configuration import evaluate_configuration
import pickle

def load_model(config_pickle, model_path):
    """Load a trained Gaussian diffusion model.

    Args:
        config_pickle (str): pickle file containing the config and input_dim.
        model_path (str): path to the statedict

    Returns:
        tuple: (GaussianDiffusion, NN Model)
    """
    data = pickle.load(open(config_pickle, "rb"))
    state_dict = torch.load(model_path)

    diffusion, model = init_diffusion_model_from_config(data['config'], data['input_dim'])
    diffusion.load_state_dict(state_dict)
    diffusion.eval()  # Set to evaluation mode
    model.eval()  # Set to evaluation mode
    return diffusion, model

indir = "./ddpm_tiny_exps/fig11_200_hidden=256_layer=3_schedule=linear_epoch=150"
config_pickle = indir+"/config.pickle"
model_path = indir+"/model.pth"

diffusion, model = load_model(config_pickle, model_path)

# now call VMEC
# evaluate_configuration(x,
#                        nfp,
#                        mpol,
#                        ntor,
#                        helicity_m=1,
#                        helicity_n=0,
#                        vmec_input=None)