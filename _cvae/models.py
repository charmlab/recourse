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

    def forward(self, x, pa):

        batch_size = x.size(0)

        means, log_var = self.encoder(x, pa)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size])
        z = eps * std + means

        recon_x = self.decoder(z, pa)

        return recon_x, means, log_var, z

    def reconstruct(self, x_factual, pa_factual, pa_counter, sample_from):


        if isinstance(x_factual, pd.DataFrame):
            x_factual = torch.tensor(x_factual.values).float()
        if isinstance(pa_factual, pd.DataFrame):
            pa_factual = torch.tensor(pa_factual.values).float()
        if isinstance(pa_counter, pd.DataFrame):
            pa_counter = torch.tensor(pa_counter.values).float()

        batch_size = x_factual.size(0)

        if sample_from == 'prior':
            z = torch.randn([batch_size, self.latent_size])
        elif sample_from == 'reweighted_prior':
            raise NotImplementedError
        elif sample_from == 'posterior':
            means, log_var = self.encoder(x_factual, pa_factual) # noise is computed in factual world
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([batch_size, self.latent_size])
            z = eps * std + means
        else:
            raise Exception(f'{sample_from} not recognized.')

        recon_x = self.decoder(z, pa_counter)

        return pd.DataFrame(recon_x.detach().numpy())


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

    def forward(self, x, pa):

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
