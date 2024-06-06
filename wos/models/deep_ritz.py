from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# define a new activation function
class RitzActivation(nn.Module):
    """
    Ritz activation function for Deep Ritz Net
    """

    def __init__(self):
        super(RitzActivation, self).__init__()

    def forward(ctx, x):
        result = torch.pow(F.relu(x), 3)
        return result


class RitzLayer(nn.Module):
    dim: int
    n_layers: int
    act_fn: Callable

    def __init__(self, dim: int, n_layers: int, act_fn: Callable = nn.GELU):
        super(RitzLayer, self).__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.act_fn = act_fn
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Linear(dim, dim))
            self.layers.append(self.act_fn)

    def forward(self, x):
        x0 = x.clone()
        for layer in self.layers:
            x = layer(x)
        x = x + x0  # resnet
        return x


class ZeroPadLayer(nn.Module):
    in_dim: int
    out_dim: int

    def __init__(self, in_dim: int, out_dim: int):
        super(ZeroPadLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        assert out_dim > in_dim

    def forward(self, x):
        return torch.cat(
            [x, torch.zeros(x.shape[0], self.out_dim - self.in_dim)], dim=1
        )  # attach zeros to the end of the vector


class DeepRitzNet(nn.Module):
    in_dim: int  # number of input dimensions
    hid_dim: int  # number of hidden dimensions
    out_dim: int  # number of output dimensions
    n_blocks: int  # number of blocks
    n_layers: int  # number of layers in each block
    is_original: bool  # whether to use the original Deep Ritz Net
    act_fn: Callable  # activation function

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hid_dim: int,
        n_blocks: int,
        n_layers: int,
        is_original: bool,
        act_fn: Callable = nn.GELU,
    ):
        super(DeepRitzNet, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.is_original = is_original  # boolean indicator to see if we want to use the original processing for the input
        self.act_fn = act_fn

        if self.is_original:
            if self.in_dim < self.hid_dim:
                self.in_layer = ZeroPadLayer(self.in_dim, self.hid_dim)
            elif self.in_dim > self.hid_dim:
                self.in_layer = nn.Linear(self.in_dim, self.hid_dim)
            else:
                self.in_layer = nn.Identity()
            self.act_fn = RitzActivation()

        else:
            self.in_layer = nn.Linear(self.in_dim, self.hid_dim)
            self.act_fn = self.act_fn

        self.block_layers = nn.ModuleList()
        for _ in range(self.n_blocks):
            self.block_layers.append(
                RitzLayer(dim=self.hid_dim, n_layers=self.n_layers, act_fn=self.act_fn)
            )

        self.out_layer = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, x):
        x = self.in_layer(x)
        for layer in self.block_layers:
            x = layer(x)
        x = self.out_layer(x)
        return x
