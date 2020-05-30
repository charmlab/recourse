import torch
import utils

from debug import ipsh

@utils.Memoize
def computeRBFKernel(x1, x2, lengthscale):
    return torch.exp(
        -0.5 * torch.sum(
            torch.pow(
                (x1[:, None] - x2) / lengthscale,
                2, # power exponent
            ),
            2, # sum over second dimension
        )
    ) # TODO: why does this give 1 on the diagonals??

@utils.Memoize
def get_inverse_covariance(K, noise_var):
    return torch.inverse(K + noise_var * torch.eye(K.shape[0]))

def sample_from_KRR_model(model, X_new):
    if not isinstance(X_new, torch.Tensor):
        X_new = torch.tensor(X_new.copy())

    X_train = torch.tensor(model.X_fit_)
    Y_train = torch.tensor(model.Y_fit_)
    lengthscale = torch.tensor(model.kernel.length_scale)
    lamdba = torch.tensor(model.alpha)

    assert X_new.shape[1] == X_train.shape[1]

    K = computeRBFKernel(X_train, X_train, lengthscale)
    inv_cov = get_inverse_covariance(K, lamdba)
    K_new = computeRBFKernel(X_train, X_new, lengthscale)

    Y_new = torch.matmul(
        Y_train.T,
        torch.matmul(inv_cov, K_new),
    )
    return Y_new.float()


def sample_from_LIN_model(model, X_new):
    coef_ = torch.tensor(model.coef_)
    intercept_ = torch.tensor(model.intercept_)
    Y_new = torch.matmul(coef_, X_new.T) + intercept_
    return Y_new.float()

