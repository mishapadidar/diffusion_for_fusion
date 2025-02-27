import torch
import torch.nn as nn
from math import log2

class EncoderBlock(nn.Module):
    def __init__(self, size_in: int, size_out: int):
        super().__init__()

        self.ff = nn.Linear(size_in, size_out)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return self.act(self.ff(x))

# Defining Autoencoder model
class Autoencoder(nn.Module):
   def __init__(self, input_size, encoding_size):
        super().__init__()

        hidden_size = pow(2, int(log2(input_size)))
        layers = [EncoderBlock(input_size, hidden_size)]
        while hidden_size / 2 > encoding_size:
            layers.append(EncoderBlock(hidden_size, int(hidden_size / 2)))
            hidden_size = int(hidden_size / 2)
        layers.append(EncoderBlock(hidden_size, encoding_size))
        self.encoder = nn.Sequential(*layers)

        layers = [EncoderBlock(encoding_size, hidden_size)]
        while hidden_size * 2 < input_size:
            layers.append(EncoderBlock(hidden_size, int(hidden_size * 2)))
            hidden_size = int(hidden_size * 2)
        layers.append(EncoderBlock(hidden_size, input_size))
        self.decoder = nn.Sequential(*layers)


   def forward(self, x):
       x = self.encoder(x)
       x = self.decoder(x)
       return x