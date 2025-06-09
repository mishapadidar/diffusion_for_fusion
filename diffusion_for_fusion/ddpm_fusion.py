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

def to_standard(X, mean=None, std=None):
    """Standardize a dataset without affecting relative scaling.
        return (X - mean) / std
    std is the largest standard deviation and mean is the mean
    of the features.

    Args:
        X (array, tensor): array or tensor of shape (n points, dim)

    Returns: tuple (X, mean, std)
        X (array, tensor): array or tensor of shape (n points, dim)
        mean (array, tensor): array or tensor of shape (dim, )
        std (float): standard deviation
    """
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0).max()
    X = (X - mean) / std
    return X, mean, std

def to_unit_cube(X, lb=None, ub=None):
    """Map a dataset to the unit cube.
        return (X - lb) / (ub - lb)

    Args:
        X (array, tensor): array or tensor of shape (n points, dim)
        lb (array, tensor, optional): array or tensor of shape (dim, )
        ub (array, tensor, optional): array or tensor of shape (dim, )
    
    Returns: tuple (X, mean, std)
        X (array, tensor): array or tensor of shape (n points, dim)
        lb (array, tensor): array or tensor of shape (dim, )
        ub (array, tensor): array or tensor of shape (dim, )
    """
    if lb is None:
        lb = X.min(axis=0)
        ub = X.max(axis=0)
    X = (X - lb) / (ub - lb)
    return X, lb, ub

def from_unit_cube(X, lb, ub):
    """Un-map a dataset from the unit cube.
        return X * (ub - lb) + lb

    Args:
        X (array, tensor): array or tensor of shape (n points, dim)
        lb (array, tensor): array or tensor of shape (dim, )
        ub (array, tensor): array or tensor of shape (dim, )

    Returns:
        (array, tensor): array or tensor of shape (n points, dim)
    """
    return X * (ub - lb) + lb

def from_standard(X, mean, std):
    """Un-standardize a dataset without affecting relative scaling.
        return X * std + mean
    This inverts the to_standard() function

    Args:
        X (array, tensor): array or tensor of shape (n points, dim)
        mean (array, tensor): array or tensor of shape (dim, )
        std (float): standard deviation

    Returns:
        (array, tensor): array or tensor of shape (n points, dim)
    """
    return X * std + mean

class InputDataset(Dataset):
    def __init__(self, data: torch.tensor):
        self.data = data
        
    def __getitem__(self, index):
        x = self.data[index]
        return x
    
    def __len__(self):
        return len(self.data)

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
    """
    This class implements the Gaussian diffusion process for denoising. The class cannot
    condition on any inputs like the ConditionalGaussianDiffusion class.
    """
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
    
    def sample(self, eval_batch_size, input_dim):
        """Sample x_0 from the diffusion process.

        Args:
            eval_batch_size (int): number of points to sample
            input_dim (int): dimension of input i.e. dim of x_T

        Returns:
            sample: (eval_batch_size, input_dim) array of samples
        """
        # TODO: push all arrays to correct device
        # input_shape = list(batch.shape)
        # input_shape[0] = eval_batch_size
        input_shape = [eval_batch_size, input_dim]
        sample = torch.randn(input_shape)#.to(device)
        timesteps = list(range(self.num_timesteps))[::-1] #reverse sampling 
        for i, t in enumerate(timesteps):
            t = torch.from_numpy(np.repeat(t, eval_batch_size)).long()#.to(device)
            with torch.no_grad():
                residual = self.backward(sample, t)
            sample = self.step(residual, t[0], sample)
        return sample

def init_diffusion_model_from_config(config, input_dim):
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
        
    diffusion = GaussianDiffusion(
        model, 
        num_timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule)

    return diffusion, model

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
        ax.set_xlim([np.min(img_train[:,0]),np.max(img_train[:,0])])
        ax.set_ylim([np.min(img_train[:,1]),np.max(img_train[:,1])])
        # ax.set_aspect('equal')
        camera.snap()
    animation = camera.animate()
    animation.save(f'{imgdir}/animation_seed={seed}.gif', writer='PillowWriter', fps=20)   
