import torch
import utils

from debug import ipsh

class RBFKernel():
    def __init__(self, input_dim, signal_var, lengthscales):
        assert lengthscales.shape[0] == input_dim
        self.input_dim  = input_dim
        self.signal_var = signal_var
        self.lengthscales = lengthscales

    # def eval_sample(self, x1, x2):
    #     return self.signal_var * torch.exp(-0.5*torch.sum(((x1-x2)/self.lengthscales)**2))

    # https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/10
    # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3598
    def eval_sample(self, x1, x2):
        return self.signal_var * torch.exp(
            -0.5 * torch.sum(
                torch.pow(
                    (x1[:, None] - x2) / self.lengthscales,
                    2, # power exponent
                ),
                2, # sum over second dimension
            )
        )

    def eval(self, x1, x2):
        n1, d1 = x1.shape
        n2, d2 = x2.shape
        assert d1 == d2
        assert isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor)
        return self.eval_sample(x1, x2)


def get_optimized_rbf_kernel(m):
    input_dim = torch.tensor(m.input_dim)
    signal_var = torch.tensor(m.kern.variance)
    lengthscales = torch.tensor(m.kern.lengthscale)
    return RBFKernel(input_dim, signal_var, lengthscales)

def get_inverse_covariance(K, noise_var):
    return torch.inverse(K + noise_var * torch.eye(K.shape[0]))

@utils.Memoize
def get_manual_GP_model(m):
    kernel = get_optimized_rbf_kernel(m)
    noise_var = torch.tensor(m.Gaussian_noise.variance)
    X_train = torch.tensor(m.X)
    Y_train = torch.tensor(m.Y)
    K = kernel.eval(X_train, X_train)
    inv_cov = get_inverse_covariance(K, noise_var)
    return kernel, noise_var, X_train, Y_train, K, inv_cov

def noise_posterior_mean(noise_var, inv_cov, Y_train):
    return noise_var * torch.matmul(inv_cov, Y_train)

def noise_posterior_covariance(noise_var, inv_cov):
    N = inv_cov.shape[0]
    return noise_var * (torch.eye(N) - noise_var * inv_cov)

def noise_posterior_variance(noise_var, inv_cov):
    return torch.diag(noise_posterior_covariance(noise_var, inv_cov))

def get_pred_post_noiseless(X_new, X_train, Y_train, kernel, inv_cov):
    k_new = kernel.eval(X_new, X_new)
    K_new = kernel.eval(X_new, X_train)
    pred_mean = torch.matmul(torch.matmul(K_new, inv_cov), Y_train)
    pred_cov = k_new - torch.matmul(torch.matmul(K_new, inv_cov), torch.transpose(K_new,0,1)) # np.transpose without (.,0,1)
    pred_var = torch.diag(pred_cov).reshape(pred_mean.shape)
    return pred_mean, pred_var

def get_predictive_distribution(pred_mean, pred_var, noise_var,
                                distribution_type='iv', inv_cov=None,
                                Y_train=None, factual_instance=None):
    if distribution_type == 'iv':
        mean = pred_mean + 0.
        var = pred_var + noise_var

    elif distribution_type == 'cf':
        assert inv_cov is not None and Y_train is not None and factual_instance is not None
        noise_post_means = noise_posterior_mean(noise_var, inv_cov, Y_train)
        noise_post_vars = noise_posterior_variance(noise_var, inv_cov)
        mean = pred_mean + noise_post_means[factual_instance]
        var = pred_var + noise_post_vars[factual_instance]

    else:
        print('Distribution type not recognised. \
        Please choose between "iv" (interventional), "cf" (counterfactual)')

    return mean, var

def sample_from_Gaussian_with_reparametrisation_trick(mean, var):
    assert mean.shape == var.shape
    if torch.any(torch.isnan(torch.sqrt(var))):
        # investigate why are we seeing nan samples for 10 training samples?
        raise Exception(f'Sqrt of variance is -1, why?!.')
    return mean + torch.randn(mean.shape[0],1) * torch.sqrt(var) # np.random.randn

def sample_from_GP_model(m, X_new, distribution_type='iv', factual_instance=None):
    if not isinstance(X_new, torch.Tensor):
        X_new = torch.tensor(X_new.copy())
    assert X_new.shape[1] == m.input_dim
    kernel, noise_var, X_train, Y_train, K, inv_cov = get_manual_GP_model(m)
    pred_mean, pred_var = get_pred_post_noiseless(X_new, X_train, Y_train, kernel, inv_cov)
    mean, var = get_predictive_distribution(pred_mean, pred_var, noise_var, distribution_type, inv_cov, Y_train, factual_instance)
    Y_new = sample_from_Gaussian_with_reparametrisation_trick(mean, var)
    return Y_new
