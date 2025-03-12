import torch
import torch.nn as nn
from math import log, floor

class EncoderBlock(nn.Module):
    def __init__(self, size_in: int, size_out: int):
        super().__init__()

        self.ff = nn.Linear(size_in, size_out)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return self.act(self.ff(x))

# Defining Autoencoder model
class Autoencoder(nn.Module):
   def __init__(self, input_size, encoding_size, base=2):
        super().__init__()

        # layer sizes
        sizes = base_layer_sequence(input_size, encoding_size, base=base)
        print(sizes)
        layers = [EncoderBlock(sizes[ii], sizes[ii-1]) for ii in range(len(sizes)-1,0,-1)]
        self.encoder = nn.Sequential(*layers)

        layers = [EncoderBlock(sizes[ii], sizes[ii+1]) for ii in range(len(sizes)-1)]
        self.decoder = nn.Sequential(*layers)

   def forward(self, x):
       x = self.encoder(x)
       x = self.decoder(x)
       return x
   
def base_layer_sequence(input_size, encoding_size, base=2):
    """Makes a base-b list of layer sizes.
        [encoding_size, b**k, b**(k+1), ..., b**(k+n), input_size]

    Args:
        input_size (int): size of the input.
        encoding_size (int): size of the encoding.

    Returns:
        _type_: _description_
    """
    kmin = floor(log(encoding_size, base)) + 1
    kmax = floor(log(input_size, base))
    layers = [encoding_size, *[base**k for k in range(kmin, kmax+1)], input_size]
    return layers