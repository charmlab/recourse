import torch
import torch.nn as nn
import pandas as pd
from torch.distributions.normal import Normal
from scipy.stats import multivariate_normal

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

        means, log_vars = self.encoder(x, pa)

        stds = torch.exp(0.5 * log_vars)
        eps = torch.randn([batch_size, self.latent_size])
        z = eps * stds + means

        recon_x = self.decoder(z, pa)

        return recon_x, means, log_vars, z

    def reconstruct(self, x_factual, pa_factual, pa_counter, sample_from):

        with torch.no_grad():

            if isinstance(x_factual, pd.DataFrame):
                x_factual = torch.tensor(x_factual.values).float()
            if isinstance(pa_factual, pd.DataFrame):
                pa_factual = torch.tensor(pa_factual.values).float()
            if isinstance(pa_counter, pd.DataFrame):
                pa_counter = torch.tensor(pa_counter.values).float()

            batch_size = x_factual.size(0)


            samples_pz = torch.randn([batch_size, self.latent_size])
            means, log_vars = self.encoder(x_factual, pa_factual) # noise is computed in factual world
            stds = torch.exp(0.5 * log_vars)
            eps = torch.randn([batch_size, self.latent_size])
            samples_qz = eps * stds + means

            # print(f'KL between posterior and prior: {-0.5 * torch.sum(1 + log_vars - means.pow(2) - log_vars.exp())}')

            if sample_from == 'prior':

                recon_x = self.decoder(samples_pz, pa_counter)

            elif sample_from == 'reweighted_prior':

                means_pz, stds_pz = torch.zeros([batch_size, self.latent_size]), torch.ones([batch_size, self.latent_size])
                means_qz, stds_qz = means.detach(), stds.detach()

                assert means_pz.shape == stds_pz.shape == means_qz.shape == stds_qz.shape

                pdf_pz = [
                    multivariate_normal(mean_pz, torch.diag(std_pz)).pdf(sample_qz)
                    for mean_pz, std_pz, sample_qz in zip(means_pz, stds_pz, samples_qz) # IMP: note this is sample qz not pz
                ]
                pdf_qz = [
                    multivariate_normal(mean_qz, torch.diag(std_qz)).pdf(sample_qz)
                    for mean_qz, std_qz, sample_qz in zip(means_qz, stds_qz, samples_qz)
                ]

                pz_over_qz = torch.div(torch.tensor(pdf_pz), torch.tensor(pdf_qz)).reshape(-1, 1)
                recon_x = self.decoder(samples_qz, pa_counter) * pz_over_qz
                recon_x = torch.mul(recon_x, pz_over_qz)

            elif sample_from == 'posterior':

                recon_x = self.decoder(samples_qz, pa_counter)

            else:
                raise Exception(f'{sample_from} not recognized.')

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
        self.linear_log_vars = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, pa):

        if self.conditional:
            x = torch.cat((x, pa), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_vars(x)

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
