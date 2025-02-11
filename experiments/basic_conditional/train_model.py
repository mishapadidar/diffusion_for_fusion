import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import argparse
import os
import pickle
from uuid import uuid4
from sklearn.decomposition import PCA
from diffusion_for_fusion.ddpm_fusion import (plot_image_denoising,
                                              set_seed,
                                              to_standard,
                                              )
from diffusion_for_fusion.ddpm_conditional_diffusion import init_conditional_diffusion_model_from_config, generate_conditions_for_eval
from load_quasr_data import plot_pca_data, prepare_data_from_config


parser = argparse.ArgumentParser()
parser.add_argument("--conditions", type=str, nargs="+", default=["mean_iota"])
parser.add_argument("--train_batch_size", type=int, default=128)
parser.add_argument("--eval_batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--num_timesteps", type=int, default=50)
parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])

parser.add_argument("--input_emb_size", type=int, default=64)
parser.add_argument("--time_emb_size", type=int, default=64)
parser.add_argument("--cond_emb_size", type=int, default=64)

parser.add_argument("--input_head", action='store_true') # bool
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--hidden_layers", type=int, default=3)
parser.add_argument("--time_emb_type", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])

parser.add_argument('--use_pca', action='store_true', help='use the PCA data') # bool
parser.add_argument('--pca_size', type=int, default=10, help='number of PCA dimensions') # bool

parser.add_argument("--save_images_step", type=int, default=1000)
parser.add_argument("--seed", type=int, default=999, help='set seed of sampling')

config = parser.parse_args()
args_dict = vars(config)

# tag for run
run_uuid = uuid4()

# logging
outdir = f"output/{'_'.join(config.conditions)}"
outdir += f"/run_uuid_{run_uuid}"

print('Experiment Setting:')
for key, value in args_dict.items():
    print(f"| {key}: {value}")

# load, PCA, and standardize data
X_train, _, _, Y_train, _, _, _ = prepare_data_from_config(config)

# PCA for plotting in 2D data
pca_plot = PCA(n_components=2, svd_solver='full')
pca_plot = pca_plot.fit(X_train)

input_dim = np.shape(X_train)[1]
dataset = TensorDataset(torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(Y_train.astype(np.float32)))
dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)

diffusion, model = init_conditional_diffusion_model_from_config(config, input_dim)
print(model)

if torch.cuda.device_count() > 0:
    device = 'cuda:0'
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = 'cpu'
diffusion = diffusion.to(device)

print("")
print("Using device:",device)

optimizer = torch.optim.AdamW(
    diffusion.model.parameters(),
    lr=config.learning_rate,
)

global_step = 0
frames = []
losses = []
print("Training model...")
for epoch in range(config.num_epochs):
    t0 = time.time()
    diffusion.model.train()

    for step, (X_batch, cond_batch) in enumerate(dataloader):
        
        X_batch = X_batch.to(device)
        cond_batch = cond_batch.to(device)

        timesteps = torch.randint(
            0, diffusion.num_timesteps, (X_batch.shape[0],)
        ).long().to(device)

        # generate x_T
        noise = torch.randn(X_batch.shape).to(device)

        # generate noisy data
        noisy = diffusion(X_batch, noise, timesteps)

        # predict the noise
        noise_pred = diffusion.backward(noisy, timesteps, cond_batch)

        # compute loss
        loss = F.mse_loss(noise_pred, noise)

        # grad
        loss.backward()

        nn.utils.clip_grad_norm_(diffusion.model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        # print
        losses.append(loss.detach().item())
        global_step += 1
        if (global_step % 100 == 0) :
            print(f'epoch = {epoch}, step = {global_step}, loss = {losses[-1]}')

    # visualization
    if epoch % config.save_images_step == 0 or epoch == config.num_epochs - 1:
        diffusion.model.eval()
        set_seed(config.seed)

        # TODO: get from_train option from config
        # generate the conditions for sampling (config.eval_batch_size, cond_input_dim)
        cond_eval = generate_conditions_for_eval(Y_train, batch_size = config.eval_batch_size, from_train=True,
                                                 seed=config.seed, as_tensor = True, device=device)

        # sample diffusion process
        sample = diffusion.sample(cond_eval)
        if device != 'cpu':
            sample = sample.to('cpu')

        # project onto 2 dimensions for plotting
        sample = torch.tensor(pca_plot.transform(sample))

        frames.append(sample)
    t1 = time.time()
    print(f"--> time: {t1 - t0:.2f} sec")

# sample some data for plotting
cond_eval = generate_conditions_for_eval(Y_train, batch_size = config.eval_batch_size, from_train=True,
                                         seed=config.seed, as_tensor = True, device=device)
sample = diffusion.sample(cond_eval)
if device != 'cpu':
    sample = sample.to('cpu')

print("")
print("Output directory:", outdir)

outfilename = f"{outdir}/model.pth"
print("Saving model to:", outfilename)
os.makedirs(outdir, exist_ok=True)
torch.save(diffusion.state_dict(), outfilename)

imgdir = f"{outdir}/images"
print("Saving images to:", imgdir)
os.makedirs(imgdir, exist_ok=True)

# plot the sampled data in the PCA plane
plot_pca_data(X_train, X_new=sample.numpy(), is_pca=False, save_path=f"{imgdir}/generated.png")

# plot the movie in the PCA plane
if not config.use_pca:
    X_train = pca_plot.transform(X_train)
plot_image_denoising(imgdir, frames, basis=None, seed=99, img_train=X_train)

outfilename = f"{outdir}/loss.npy"
print("Saving loss as numpy array to:", outfilename)
np.save(outfilename, np.array(losses))

outfilename = f"{outdir}/config.pickle"
print("Saving config as pickle file to:", outfilename)
data = {'config': config, 'input_dim':input_dim}
pickle.dump(data, open(outfilename,"wb"))
