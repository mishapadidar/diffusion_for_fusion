import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import argparse
import os
import pickle
from uuid import uuid4
from diffusion_for_fusion.autoencoder import ConditionalAutoencoder
from diffusion_for_fusion.ddpm_fusion import set_seed, to_standard
from experiments.conditional_diffusion.load_quasr_data import load_quasr_data

parser = argparse.ArgumentParser()
parser.add_argument("--conditions", type=str, nargs="+", default=["mean_iota"])
parser.add_argument("--train_batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--encoding_size", type=int, default=512)
parser.add_argument("--encoding_base", type=int, default=2)
parser.add_argument("--cond_emb_size", type=int, default=2)
parser.add_argument("--seed", type=int, default=999, help='set seed of sampling')

config = parser.parse_args()
args_dict = vars(config)

# tag for run
run_uuid = uuid4()

# logging
outdir = f"output"
outdir += f"/run_{config.encoding_size}_uuid_{run_uuid}"

print('Experiment Setting:')
for key, value in args_dict.items():
    print(f"| {key}: {value}")

# load and standardize data
X_train, Y_train = load_quasr_data(config.conditions)
X_train, X_mean, X_std = to_standard(X_train)
Y_train, Y_mean, Y_std = to_standard(Y_train)


dataset = TensorDataset(torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(Y_train.astype(np.float32)))
dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)

print("")
print("Dataset shape", np.shape(X_train))

input_size = X_train.shape[1]  # Number of input features
condition_size = Y_train.shape[1]  # Number of condition features
print("Input size:", input_size)
print("Condition size:", condition_size)
print("Encoding size:", config.encoding_size)
autoenc = ConditionalAutoencoder(input_size, condition_size=condition_size,
                                 encoding_size=config.encoding_size, cond_emb_size = config.cond_emb_size, 
                                 base=config.encoding_base)

print("")
print(autoenc)

if torch.cuda.device_count() > 0:
    device = 'cuda:0'
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = 'cpu'
autoenc = autoenc.to(device)

print("")
print("Using device:",device)

optimizer = torch.optim.AdamW(
    autoenc.parameters(),
    lr=config.learning_rate,
)

global_step = 0
losses = []
print("Training model...")
for epoch in range(config.num_epochs):
    t0 = time.time()
    autoenc.train()
    set_seed(config.seed)

    for step, (X_batch, Y_batch) in enumerate(dataloader):

        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        # encode and decode
        decoded_preds = autoenc(X_batch, Y_batch)

        # compute loss
        loss = functional.mse_loss(decoded_preds, X_batch)

        # grad
        loss.backward()
        nn.utils.clip_grad_norm_(autoenc.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        # print
        losses.append(loss.detach().item())
        global_step += 1
        if (global_step % 100 == 0):
            print(f'epoch = {epoch}, step = {global_step}, loss = {losses[-1]}')


# # encode the data; it can be plugged direclty into the decoder.
# X_train_encoded = autoenc.encoder(X_train).detach().numpy()

print("")
print("Output directory:", outdir)

outfilename = f"{outdir}/model.pth"
print("Saving model to:", outfilename)
os.makedirs(outdir, exist_ok=True)
torch.save(autoenc.state_dict(), outfilename)

# outfilename = f"{outdir}/data_encoded.npy"
# print("Saving encoded data to:", outfilename)
# data_encoded = {'X_train_encoded':X_train_encoded, "Y_train":Y_train}
# pickle.dump(data_encoded, open(outfilename, "wb"))

outfilename = f"{outdir}/loss.npy"
print("Saving loss as numpy array to:", outfilename)
np.save(outfilename, np.array(losses))

outfilename = f"{outdir}/config.pickle"
print("Saving config as pickle file to:", outfilename)
data = {'config': config, 'input_dim':input_size, 'condition_dim':condition_size}
pickle.dump(data, open(outfilename,"wb"))
