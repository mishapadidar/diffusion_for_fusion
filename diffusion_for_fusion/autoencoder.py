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
    
class ResidualBlock(nn.Module):
    def __init__(self, size_in: int, size_out: int):
        super().__init__()

        self.ff = nn.Linear(size_in, size_out)
        self.act = nn.GELU()
        self.residual = nn.Linear(size_in, size_out)

    def forward(self, x: torch.Tensor):
        return self.act(self.ff(x)) + self.residual(x)

# Defining Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_size, base=2):
        super().__init__()

        # layer sizes
        sizes = base_layer_sequence(input_size, encoding_size, base=base)
        layers = [EncoderBlock(sizes[ii], sizes[ii-1]) for ii in range(len(sizes)-1,0,-1)]
        self.encoder = nn.Sequential(*layers)

        layers = [EncoderBlock(sizes[ii], sizes[ii+1]) for ii in range(len(sizes)-1)]
        self.decoder = nn.Sequential(*layers)

        self.encoder_residual = nn.Linear(input_size, encoding_size)
        self.decoder_residual = nn.Linear(encoding_size, input_size)

    def encode(self, x0):
        return self.encoder(x0) + self.encoder_residual(x0)
    
    def decode(self, x1):
        return self.decoder(x1) + self.decoder_residual(x1)

    def forward(self, x0):
        x1 = self.encode(x0)
        x2 = self.decode(x1)
        return x2
   
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

def init_autoencoder_from_config(config, input_dim):
    """Initialize an autoencoder from a config file.

    Args:
        config (argparse.Namespace): A config file that has been ingested by argparse,
            config = parser.parse_args()
        input_dim (int): dimension of X_train

    Returns:
        autoenc: initialized autoencoder
    """
    autoenc = Autoencoder(input_dim, config.encoding_size, base=config.encoding_base)
    return autoenc

