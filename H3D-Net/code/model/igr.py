import numpy as np
import torch.nn as nn
import torch
from torch.autograd import grad
from model.embedder import *


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0][:, -3:]
    return points_grad


class GeometryNetwork(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        dims,
        weight_norm,
        bias=1.0,
        skip_in=(),
        geometric_init=True,
        beta=100,
        latent_size = 256,
        multires=0
    ):
        super().__init__()

        dims = [d_in + latent_size] + dims + [d_out]
        print(dims)
        self.num_layers = len(dims)
        self.skip_in = skip_in

        # CHANGED: Add embedding from IDR with multires = 6
        # Changes the input dims to the embedding dims
        self.embed_fn = None
        if (multires > 0):
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = dims[0] - 3 + input_ch 
            print(f'emInput:{input_ch} - {dims[0]}')
            d_in = dims[0]

        print(dims)

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)

            # if true preform preform geometric initialization
            if geometric_init:

                if layer == self.num_layers - 2:

                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -bias)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)

                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, "lin" + str(layer), lin)
            
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)

        # vanilla relu
        else:
            self.activation = nn.ReLU()

    def forward(self, input):

        # CHANGE: Transform the input into the NeRF positional encoding
        # Then feed it through the net.q
        if (self.embed_fn is not None):
            x_em = input[:, -3:] 
            x_em = self.embed_fn(x_em)
            input = torch.cat((input[:, :-3], x_em), 1)

        x = input

        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                # print(f'preskip: {x.shape} - {input.shape}')
                x = torch.cat([x, input], -1) / np.sqrt(2)
                # print('skip')
            # print(x.shape)

            x = lin(x)

            if layer < self.num_layers - 2:
                x = self.activation(x)

        return x