import os
import copy
import time
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from .models import VAE

from debug import ipsh


class Dataset(torch.utils.data.Dataset):
  def __init__(self, data, target, transform=None):
        self.data = torch.tensor(data.values).float()
        self.target = torch.tensor(target.values).float() # this was long before... changed to float to replace all y.float()s in the trainig loop
        self.transform = transform

  def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

  def __len__(self):
        return len(self.data)


def train_cvae(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # writer = SummaryWriter(log_dir = args.debug_folder)

    ts = time.time()

    dataset = Dataset(args.node_train, args.parents_train) # order is important: x=node, pa=parents (the class the cvae is conditioned on)
    data_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True)

    def loss_fn(recon_x, x, mean, log_var):
        # TODO: add back for binary / categorical variables
        # BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        MSE = torch.nn.functional.mse_loss(recon_x, x, reduction='mean')
        KLD = -0.5 * torch.mean(torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), axis=1))
        return MSE / args.lambda_kld + KLD

    vae = VAE(
        encoder_layer_sizes=list(args.encoder_layer_sizes), # bug in AttrDict package: https://github.com/bcj/AttrDict/issues/34#issuecomment-202920540
        latent_size=args.latent_size,
        decoder_layer_sizes=list(args.decoder_layer_sizes), # bug in AttrDict package: https://github.com/bcj/AttrDict/issues/34#issuecomment-202920540
        conditional=args.conditional,
        num_labels=args.parents_train.shape[1] if args.conditional else 0).to(device)


    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)
    best_cvae = None
    early_stop_counter = 0
    best_total_loss = float('inf')

    all_mse_validation_losses = []
    stopped_early = False
    iterator = tqdm(range(args.epochs))
    for epoch in iterator:
        vae.train()

        # tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, (x, pa) in enumerate(data_loader):

            x, pa = x.to(device), pa.to(device)

            if args.conditional:
                recon_x, mean, log_var, z = vae(x, pa)
            else:
                recon_x, mean, log_var, z = vae(x)

            loss = loss_fn(recon_x, x, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

        # train set
        x = torch.tensor(args.node_train.to_numpy()).float()
        pa = torch.tensor(args.parents_train.to_numpy()).float()
        recon_x, mean, log_var, z = vae(x, pa)
        recon_x, mean, log_var, z = recon_x.detach(), mean.detach(), log_var.detach(), z.detach()
        x_train, pa_train, recon_x_train = x, pa, recon_x
        MSE_train = torch.nn.functional.mse_loss(recon_x, x, reduction='mean')
        KLD_train = -0.5 * torch.mean(torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), axis=1))
        total_loss_train = MSE_train / args.lambda_kld + KLD_train

        # validation set
        x = torch.tensor(args.node_validation.to_numpy()).float()
        pa = torch.tensor(args.parents_validation.to_numpy()).float()
        recon_x, mean, log_var, z = vae(x, pa)
        recon_x, mean, log_var, z = recon_x.detach(), mean.detach(), log_var.detach(), z.detach()
        x_validation, pa_validation, recon_x_validation = x, pa, recon_x
        MSE_validation = torch.nn.functional.mse_loss(recon_x, x, reduction='mean')
        KLD_validation = -0.5 * torch.mean(torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), axis=1))
        total_loss_validation = MSE_validation / args.lambda_kld + KLD_validation

        # writer.add_scalars(f'loss/{args.name}', {
        #     'MSE_train': MSE_train,
        #     'KLD_train': KLD_train,
        #     # 'sum_train': MSE_train + KLD_train,
        #     'MSE_validation': MSE_validation,
        #     'KLD_validation': KLD_validation,
        #     # 'sum_validation': MSE_validation + KLD_validation,
        # }, epoch)

        if total_loss_validation < best_total_loss:
            best_total_loss = total_loss_validation
            best_cvae = copy.deepcopy(vae)
            early_stop_counter = 0

        if total_loss_validation > best_total_loss:
            early_stop_counter += 1

        if early_stop_counter > 25:
            stopped_early = True
            iterator.close()
            break

    if stopped_early: print(f'\t\t[INFO] Early stopping at epoch {epoch}')

    assert best_cvae is not None
    return best_cvae, recon_x_train, recon_x_validation


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[784, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 784])
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')

    args = parser.parse_args()

    main(args)
