"""
    MLP model
"""

from typing import Callable, List

import torch
import torch.nn as nn


class Model(nn.Module):
    in_dim: int  # number of input dimensions
    hid_dim: int or List[int]  # number of hidden dimensions
    out_dim: int  # number of output dimensions
    n_hidden: int  # number of hidden layers
    act_fn: Callable  # activation function

    def __init__(
        self,
        in_dim: int,
        hid_dim: int or List[int],
        out_dim: int,
        n_hidden: int,
        act_fn: Callable,
        use_resnet: bool = False,
    ):
        super(Model, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.n_hidden = n_hidden
        self.act_fn = act_fn
        self.use_resnet = use_resnet

    @staticmethod
    def init_linear(
        layer: nn.Linear,
        bias_init: Callable | None = None,
        weight_init: Callable | None = None,
    ) -> None:
        """
        Initialize linear layer
        """
        if bias_init is not None:
            bias_init(layer.bias)
        if weight_init is not None:
            weight_init(layer.weight)

    def flatten(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Combine t (time domain) and x (spatial domain)

        Not necessary for our time-independent problems
        """
        if t.dim() == 0 or t.shape[0] == 1:
            t = t.expand(x.shape[0], 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        assert x.shape[-1] == self.dim
        assert t.shape == (x.shape[0], 1)
        return torch.cat([t, x], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        raise NotImplementedError


class MLP(Model):
    """
    A layer of dense neural network
    """

    normalization_factory: Callable | None
    normalization_kwargs: dict | None
    last_bias_init: Callable | None
    last_weight_init: Callable | None

    def __init__(
        self,
        in_dim: int,
        hid_dim: int or List[int],
        out_dim: int,
        n_hidden: int,
        act_fn: Callable,
        use_resnet: bool = False,
        normalization_factory=None,
        normalization_kwargs=None,
        last_bias_init: Callable | None = None,
        last_weight_init: Callable | None = None,
    ):
        super().__init__(
            in_dim=in_dim,
            hid_dim=hid_dim,
            out_dim=out_dim,
            n_hidden=n_hidden,
            act_fn=act_fn,
            use_resnet=use_resnet,
        )

        if isinstance(self.hid_dim, int):
            self.hid_dim = [self.hid_dim] * (self.n_hidden)
        # use bias when no normalization is used
        bias = normalization_factory is None
        self.dense_layers = nn.ModuleList(
            [nn.Linear(self.in_dim, self.hid_dim[0], bias=bias)]
        )

        self.dense_layers += [
            nn.Linear(self.hid_dim[i], self.hid_dim[i + 1], bias=bias)
            for i in range(self.n_hidden - 1)
        ]

        self.dense_layers.append(nn.Linear(self.hid_dim[-1], self.out_dim))

        Model.init_linear(
            self.dense_layers[-1],
            bias_init=last_bias_init,
            weight_init=last_weight_init,
        )  # type: ignore
        if normalization_factory is None:
            self.norm_layers = None
        else:
            normalization_kwargs = normalization_kwargs or {}
            print(f"normalization_kwargs: {normalization_kwargs}")
            self.norm_layers = nn.ModuleList(
                [  
                    normalization_factory  # (self.hid_dim[i], **normalization_kwargs)
                    for i in range(self.n_hidden)
                ]
            )

    def forward(
        self, x: torch.Tensor, params: torch.Tensor | None = None
    ) -> torch.Tensor:
        if params is not None:
            if params.shape == (1, 3):
                params = params.expand(x.shape[0], -1)
            x = torch.cat([x, params], dim=1)
        tensor = x
        res_tensors = []
        tensor = self.dense_layers[0](tensor)  # embed to the hidden dimension
        for i, dense in enumerate(self.dense_layers[1:-1]):
            if self.use_resnet:
                res_tensors.append(tensor)
            if self.norm_layers is not None:
                tensor = self.norm_layers[i](tensor)
            tensor = self.act_fn(tensor)
            tensor = dense(tensor)
            if self.use_resnet:
                tensor = tensor + res_tensors[i]
        if self.norm_layers is not None:
            tensor = self.norm_layers[-1](tensor)
        tensor = self.dense_layers[-1](
            tensor
        )  # embed to the output dimension while avoiding
        # resnet
        return tensor

    def reset(self, last_layers: int) -> None:
        """
        Reset the weights of the layer
        """
        assert last_layers <= (self.n_hidden + 2)
        if isinstance(layer_index, int):
            layer_index = [-i for i in range(1, layer_index + 1)]
        for i in layer_index:
            if self.norm_layers is not None:
                self.norm_layers[i].reset_parameters()
            self.dense_layers[i].reset_parameters()

class DenseNet(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        arch: list[int],
        act_fn: Callable,
    ):
        super().__init__()
        self.nn_dims = [in_dim] + arch
        self.hidden_layer = nn.ModuleList(
            [
                nn.Linear(sum(self.nn_dims[: i + 1]), self.nn_dims[i + 1])
                for i in range(len(self.nn_dims) - 1)
            ]
        )
        self.out_layer = nn.Linear(sum(self.nn_dims), 1)
        self.act_fn = act_fn

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        tensor = self.flatten(t, x)
        for layer in self.hidden_layer:
            tensor = torch.cat([tensor, self.act_fn(layer(tensor))], dim=1)
        return self.out_layer(tensor)
