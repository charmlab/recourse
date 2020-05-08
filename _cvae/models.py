import torch
import torch.nn as nn
import pandas as pd

from debug import ipsh


class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, num_labels=0):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels)

    def forward(self, x, pa=None):

        batch_size = x.size(0)

        means, log_var = self.encoder(x, pa)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size])
        z = eps * std + means

        recon_x = self.decoder(z, pa)

        return recon_x, means, log_var, z

    def reconstructUsingPrior(self, n=1, pa=None):

        if isinstance(pa, pd.DataFrame):
            pa = torch.tensor(pa.values).float()

        batch_size = n
        z = torch.randn([batch_size, self.latent_size])

        recon_x = self.decoder(z, pa)

        return pd.DataFrame(recon_x.detach().numpy())

    # def reconstructUsingPosterior(self, x, pa=None):

    #     batch_size = x.size(0)

    #     means, log_var = self.encoder(x, pa)

    #     std = torch.exp(0.5 * log_var)
    #     eps = torch.randn([batch_size, self.latent_size])
    #     z = eps * std + means

    #     recon_x = self.decoder(z, pa)

    #     return recon_x, means, log_var, z


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, pa=None):

        if self.conditional:
            x = torch.cat((x, pa), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            # TODO: add back for binary / categorical variables
            # else:
            #     self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, pa):

        if self.conditional:
            z = torch.cat((z, pa), dim=-1)

        x = self.MLP(z)

        return x
