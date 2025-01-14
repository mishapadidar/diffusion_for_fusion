#src: https://github.com/tanelp/tiny-diffusion/blob/master
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset, ConcatDataset
import torchvision

import numpy as np
import pandas as pd
#from sklearn.datasets import make_moons

import argparse
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pickle
import sys
#SOTA models: https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main
#from denoising_diffusion_pytorch import Unet
import random
from sklearn.decomposition import PCA
from celluloid import Camera


def set_seed(seed):
    random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class InputDataset(Dataset):
    def __init__(self, data: torch.tensor):
        self.data = data
        
    def __getitem__(self, index):
        x = self.data[index]
        return x
    
    def __len__(self):
        return len(self.data)


def figure_11_data(return_pca=False, plot=False, X_new=None, save_path=None):
    # to load the data set
    Y_init = pd.read_pickle('QUASR.pkl') # y-values
    X_init = np.load('dofs.npy') # x-values

    Y_init = Y_init.reset_index(drop=True)

    # subset figure 11 dat
    idx = ((np.abs(Y_init.mean_iota - 2.30)/2.30 < 0.001) & (np.abs(Y_init.aspect_ratio - 12)/12 < 0.1)
        & (Y_init.nfp == 4) & (Y_init.helicity == 1))
    Y = Y_init[idx].mean_iota.values
    X = X_init[idx]

    pca = PCA(n_components=2, svd_solver='full')
    X_pca = pca.fit_transform(X)

    if plot:
        fig,ax = plt.subplots()
        plt.scatter(X_pca[:,0], X_pca[:,1], color='tab:blue')
        if X_new is not None:
          X_new = X_new.numpy()
          plt.scatter(X_new[:, 0], X_new[:, 1], color='tab:orange')
        ax.set_aspect('equal')
        #plt.show()
        fig.savefig(save_path) 

    if return_pca:
        return X_pca
    else:
        return X

def get_dataset(dataset='fig11', return_pca=None):
    '''
    return Fig.11 dataset of https://arxiv.org/pdf/2409.04826
    '''
    if dataset == 'fig11':
      X = figure_11_data(return_pca=return_pca)
    else:
      raise NotImplementedError
    return InputDataset(torch.from_numpy(X.astype(np.float32))), X


class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0])) / (half_size - 1)
        emb = (torch.exp(-emb * torch.arange(half_size))).to(x.device)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size


class LinearEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x / self.size * self.scale
        return x.unsqueeze(-1)

    def __len__(self):
        return 1


class LearnableEmbedding(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.linear = nn.Linear(1, size)

    def forward(self, x: torch.Tensor):
        return self.linear(x.unsqueeze(-1).float() / self.size)

    def __len__(self):
        return self.size


class IdentityEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(-1)

    def __len__(self):
        return 1


class ZeroEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(-1) * 0

    def __len__(self):
        return 1


class PositionalEmbedding(nn.Module):
    def __init__(self, size: int, type: str, **kwargs):
        super().__init__()

        if type == "sinusoidal":
            self.layer = SinusoidalEmbedding(size, **kwargs)
        elif type == "linear":
            self.layer = LinearEmbedding(size, **kwargs)
        elif type == "learnable":
            self.layer = LearnableEmbedding(size)
        elif type == "zero":
            self.layer = ZeroEmbedding()
        elif type == "identity":
            self.layer = IdentityEmbedding()
        else:
            raise ValueError(f"Unknown positional embedding type: {type}")

    def forward(self, x: torch.Tensor):
        return self.layer(x)
    

class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                 time_emb: str = "sinusoidal", input_dim: int = 64, input_head = False):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_head = input_head
        if input_head:
          self.input_proj = nn.Linear(input_dim, emb_size, bias=False) #input projection head
          concat_size = 2*emb_size
        else:
          concat_size = emb_size + input_dim
        output_dim = input_dim

        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, output_dim))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        #print(x.shape, t_emb.shape)
        if self.input_head:
          x = self.input_proj(x)
        
        x = torch.cat((x, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x


class GaussianDiffusion(nn.Module): #rewrite it into a nn module class
    def __init__(self,
                 model, #take in the backbone denoiser
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear"):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            scale = 1000 / num_timesteps  ##TODO: check this works!
            betas = scale * torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2
        self.register_buffer('betas', betas)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # required for self.add_noise
        sqrt_alphas_cumprod = alphas_cumprod ** 0.5
        sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod) ** 0.5
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)


        # required for reconstruct_x0
        sqrt_inv_alphas_cumprod = torch.sqrt(1 / alphas_cumprod)
        sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / alphas_cumprod - 1)
        self.register_buffer('sqrt_inv_alphas_cumprod', sqrt_inv_alphas_cumprod)
        self.register_buffer('sqrt_inv_alphas_cumprod_minus_one', sqrt_inv_alphas_cumprod_minus_one)

        # required for q_posterior
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)

    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, *((1,) *(len(x_t.shape) - 1)))
        s2 = s2.reshape(-1, *((1,) *(len(noise.shape) - 1)))
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, *((1,) *(len(x_0.shape) - 1)))
        s2 = s2.reshape(-1, *((1,) *(len(x_t.shape) - 1)))
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        s1 = s1.reshape(-1, *((1,) *(len(x_start.shape) - 1)))
        s2 = s2.reshape(-1, *((1,) *(len(x_noise.shape) - 1)))
            
        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps
    
    def forward(self, batch, noise, timesteps):
        noisy = self.add_noise(batch, noise, timesteps)
        return noisy
    
    def backward(self, noisy, timesteps):
        noise_pred = self.model(noisy, timesteps)
        return noise_pred

def count_num_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def plot_image_denoising(imgdir, imgs, basis=None, seed=99, img_train=None):
    imgs = torch.stack(imgs) #(T, batch_size, r)
    fig, ax = plt.subplots()
    camera = Camera(fig)
    for i, img in enumerate(imgs):
        if img_train is not None:
          ax.scatter(img_train[:,0], img_train[:,1], color='tab:blue')
        ax.scatter(img[:,0], img[:,1], color='tab:orange')  
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        camera.snap()
    animation = camera.animate()
    animation.save(f'{imgdir}/animation_seed={seed}.gif', writer='PillowWriter', fps=20)   


if __name__ == "__main__":
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

    # logging
    if config.model_type == 'MLP':
        outdir = f"ddpm_tiny_exps/{config.dataset}_{config.num_timesteps}_hidden={config.hidden_size}_layer={config.hidden_layers}_schedule={config.beta_schedule}_epoch={config.num_epochs}"
    else:
        outdir = f"ddpm_tiny_exps/{config.dataset}_{config.num_timesteps}_Unet"

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
    if config.model_type == 'MLP':
        model = MLP(
            hidden_size=config.hidden_size,
            hidden_layers=config.hidden_layers,
            emb_size= config.embedding_size,
            time_emb= config.time_embedding,
            input_dim = input_dim)
    elif config.model_type == 'Unet': #TODO: more powerful backbone from the ddpm_pytorch repo
        model = Unet(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            channels = 1,
            flash_attn = True
        )
    print(model)
    
    diffusion = GaussianDiffusion(
        model, 
        num_timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule)

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

    print("Saving model...")
    os.makedirs(outdir, exist_ok=True)
    torch.save(diffusion.state_dict(), f"{outdir}/model.pth")

    print("Saving images...")
    imgdir = f"{outdir}/images"
    os.makedirs(imgdir, exist_ok=True)
    figure_11_data(return_pca=config.return_pca, plot=True, X_new=sample, save_path=f"{imgdir}/generated.png")
    plot_image_denoising(imgdir, frames, basis=None, seed=99, img_train=X)

    print("Saving loss as numpy array...")
    np.save(f"{outdir}/loss.npy", np.array(losses))
