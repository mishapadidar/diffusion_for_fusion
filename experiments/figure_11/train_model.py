import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import pickle
from uuid import uuid4
#from denoising_diffusion_pytorch import Unet
from diffusion_for_fusion.ddpm_fusion import (GaussianDiffusion,
                                              plot_image_denoising,
                                              set_seed,
                                              InputDataset,
                                              MLP,
                                              init_diffusion_model_from_config
                                              )
from load_quasr_data import figure_11_data
#src: https://github.com/tanelp/tiny-diffusion/blob/master
#SOTA models: https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main


def get_dataset(dataset='fig11', return_pca=None):
    '''
    return Fig.11 dataset of https://arxiv.org/pdf/2409.04826
    '''
    if dataset == 'fig11':
      X = figure_11_data(return_pca=return_pca)
    else:
      raise NotImplementedError

    return InputDataset(torch.from_numpy(X.astype(np.float32))), X

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="MLP", choices=["MLP",'Unet'])
parser.add_argument("--dataset", type=str, default="fig11", choices=["fig11","full"])
parser.add_argument("--train_batch_size", type=int, default=128)
parser.add_argument("--eval_batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--num_timesteps", type=int, default=50)
parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
parser.add_argument("--embedding_size", type=int, default=64)
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--hidden_layers", type=int, default=3)
parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])
parser.add_argument("--save_images_step", type=int, default=1000)
parser.add_argument("--seed", type=int, default=999, help='set seed of sampling')
parser.add_argument('--return_pca', action='store_true', help='generate the pca coefficient')

config = parser.parse_args()
# log experiment configuration
args_dict = vars(config)

# tag for run
run_uuid = uuid4()

# logging
if config.model_type == 'MLP':
    outdir = f"output/{config.dataset}_{config.num_timesteps}_hidden={config.hidden_size}_layer={config.hidden_layers}_schedule={config.beta_schedule}_epoch={config.num_epochs}"
else:
    outdir = f"output/{config.dataset}_{config.num_timesteps}_Unet"
outdir += f"/run_uuid_{run_uuid}"

# logdir = f"{outdir}/logs"
# os.makedirs(logdir, exist_ok=True)

# old_stdout = sys.stdout
# log_file = open(f"{logdir}/r={config.r}_lr={config.learning_rate}_hidden={config.hidden_size}.log","w")
# sys.stdout = log_file
# print configs
print('Experiment Setting:')
for key, value in args_dict.items():
    print(f"| {key}: {value}")

dataset, X = get_dataset(config.dataset, return_pca=config.return_pca)    
dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)
input_dim = 2 if config.return_pca else 661
# if config.model_type == 'MLP':
#     model = MLP(
#         hidden_size=config.hidden_size,
#         hidden_layers=config.hidden_layers,
#         emb_size= config.embedding_size,
#         time_emb= config.time_embedding,
#         input_dim = input_dim)
# elif config.model_type == 'Unet': #TODO: more powerful backbone from the ddpm_pytorch repo
#     model = Unet(
#         dim = 64,
#         dim_mults = (1, 2, 4, 8),
#         channels = 1,
#         flash_attn = True
#     )
# print(model)

# diffusion = GaussianDiffusion(
#     model, 
#     num_timesteps=config.num_timesteps,
#     beta_schedule=config.beta_schedule)

diffusion, model = init_diffusion_model_from_config(config, input_dim)
print(model)

if torch.cuda.device_count() > 0:
    device = 'cuda:0'
else:
    device = 'cpu'
diffusion = diffusion.to(device)

optimizer = torch.optim.AdamW(
    diffusion.model.parameters(),
    lr=config.learning_rate,
)

global_step = 0
frames = []
losses = []
print("Training model...")
for epoch in range(config.num_epochs):
    diffusion.model.train()

    for step, batch in enumerate(dataloader):
        batch = batch.to(device)
        noise = torch.randn(batch.shape).to(device)
        timesteps = torch.randint(
            0, diffusion.num_timesteps, (batch.shape[0],)
        ).long().to(device)
        #noisy = noise_scheduler.add_noise(batch, noise, timesteps)
        #noise_pred = model(noisy, timesteps)
        #print(batch.shape, noise.shape, timesteps.shape)
        noisy = diffusion(batch, noise, timesteps)
        noise_pred = diffusion.backward(noisy, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()

        nn.utils.clip_grad_norm_(diffusion.model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        logs = {"loss": loss.detach().item(), "step": global_step}
        losses.append(loss.detach().item())
        global_step += 1
        if (epoch % 100 == 0) :
            print(f'epoch = {epoch}, step = {global_step}, loss = {losses[-1]}')

    if epoch % config.save_images_step == 0 or epoch == config.num_epochs - 1:
        # generate data with the model to later visualize the learning process
        diffusion.model.eval()
        set_seed(config.seed)
        input_shape = list(batch.shape)
        input_shape[0] = config.eval_batch_size
        sample = torch.randn(input_shape).to(device)
        timesteps = list(range(diffusion.num_timesteps))[::-1] #reverse sampling 
        for i, t in enumerate(timesteps):
            t = torch.from_numpy(np.repeat(t, config.eval_batch_size)).long().to(device)
            with torch.no_grad():
                residual = diffusion.backward(sample, t)
            sample = diffusion.step(residual, t[0], sample)
        if device != 'cpu':
            sample = sample.to('cpu')
        frames.append(sample) #global epoch changes

print("")
print("Output directory:", outdir)

outfilename = f"{outdir}/model.pth"
print("Saving model to:", outfilename)
os.makedirs(outdir, exist_ok=True)
torch.save(diffusion.state_dict(), outfilename)

imgdir = f"{outdir}/images"
print("Saving images to:", imgdir)
os.makedirs(imgdir, exist_ok=True)
figure_11_data(return_pca=config.return_pca, plot=True, X_new=sample, save_path=f"{imgdir}/generated.png")
plot_image_denoising(imgdir, frames, basis=None, seed=99, img_train=X)

outfilename = f"{outdir}/loss.npy"
print("Saving loss as numpy array to:", outfilename)
np.save(outfilename, np.array(losses))

outfilename = f"{outdir}/config.pickle"
print("Saving config as pickle file to:", outfilename)
data = {'config': config, 'input_dim':input_dim}
pickle.dump(data, open(outfilename,"wb"))
