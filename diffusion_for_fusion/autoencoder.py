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

# Defining Autoencoder model
class ConditionalAutoencoder(nn.Module):
    """Conditional autoencoder. 
    Take a data point x and a condition vector c, and encode them to product
    a latent representation z. Then, decode z and c to produce xhat. The net mapping
    is 
        xhat = decode(encode(x,c), c)
    """
    def __init__(self, input_size, condition_size, encoding_size, cond_emb_size=128, base=2):
        super().__init__()

        # embedding for conditional
        self.cond_mlp = nn.Linear(condition_size, cond_emb_size, bias=False)

        # encoder
        # TODO: revert
        sizes = base_layer_sequence(input_size + cond_emb_size, encoding_size, base=base)
        # sizes = [encoding_size, input_size + cond_emb_size]

        layers = [EncoderBlock(sizes[ii], sizes[ii-1]) for ii in range(len(sizes)-1,0,-1)]
        self.encoder_net = nn.Sequential(*layers)
        self.encoder_residual = nn.Linear(input_size + cond_emb_size, encoding_size)
                
        # decoder
        # TODO: revert
        sizes = base_layer_sequence(input_size, encoding_size + cond_emb_size, base=base)
        # sizes = [encoding_size + cond_emb_size, input_size]
        
        layers = [EncoderBlock(sizes[ii], sizes[ii+1]) for ii in range(len(sizes)-1)]
        self.decoder_net = nn.Sequential(*layers)
        self.decoder_residual = nn.Linear(encoding_size + cond_emb_size, input_size)

        # Initialize encoder network weights with Xavier initialization
        for block in self.encoder_net:
            layer = block.ff
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Initialize decoder network weights with Xavier initialization
        for block in self.decoder_net:
            layer = block.ff
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def encode(self, x, c):
        xc = torch.cat((x,c), dim=-1)
        return self.encoder_net(xc) #+ self.encoder_residual(xc)
    
    def decode(self, z, c):
        zc = torch.cat((z,c), dim=-1)
        return self.decoder_net(zc) + self.decoder_residual(zc)

    def forward(self, x, c):
        # embed the condition
        c_emb = self.cond_mlp(c)
        # encode
        z = self.encode(x, c_emb)
        # decode
        xhat = self.decode(z, c_emb)
        return xhat

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

def init_conditional_autoencoder_from_config(config, input_dim):
    """Initialize an autoencoder from a config file.

    Args:
        config (argparse.Namespace): A config file that has been ingested by argparse,
            config = parser.parse_args()
        input_dim (int): dimension of X_train

    Returns:
        autoenc: initialized autoencoder
    """
    autoenc = ConditionalAutoencoder(input_dim, len(config.conditions), config.encoding_size,
                                     config.cond_emb_size, base=config.encoding_base)
    return autoenc
