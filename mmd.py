try:
    from tqdm import tqdm_notebooks as tqdm
except ImportError:
    tqdm = lambda x: x

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform, pdist, cdist

def two_sample_permutation_test(test_statistic, X, Y, num_permutations, prog_bar=True):
    assert X.ndim == Y.ndim

    statistics = np.zeros(num_permutations)

    range_ = range(num_permutations)
    if prog_bar:
        range_ = tqdm(range_)
    for i in range_:
        # concatenate samples
        if X.ndim == 1:
            Z = np.hstack((X,Y))
        elif X.ndim == 2:
            Z = np.vstack((X,Y))

        # permute samples and compute test statistic
        perm_inds = np.random.permutation(len(Z))
        Z = Z[perm_inds]
        X_ = Z[:len(X)]
        Y_ = Z[len(X):]
        my_test_statistic = test_statistic(X_, Y_)
        statistics[i] = my_test_statistic
    return statistics

def plot_permutation_samples(null_samples, statistic=None):
    plt.hist(null_samples)
    plt.axvline(x=np.percentile(null_samples, 2.5), c='b')
    legend = ["95% quantiles"]
    if statistic is not None:
        plt.axvline(x=statistic, c='r')
        legend += ["Actual test statistic"]
    plt.legend(legend)
    plt.axvline(x=np.percentile(null_samples, 97.5), c='b')
    plt.xlabel("Test statistic value")
    plt.ylabel("Counts")
    plt.show()

def sq_distances(X,Y=None):
    """
    If Y=None, then this computes the distance between X and itself
    """
    assert(X.ndim==2)

    if Y is None:
        sq_dists = squareform(pdist(X, 'sqeuclidean'))
    else:
        assert(Y.ndim==2)
        assert(X.shape[1]==Y.shape[1])
        sq_dists = cdist(X, Y, 'sqeuclidean')

    return sq_dists

def gauss_kernel(X, Y=None, sigma=1.0):
    """
    Computes the standard Gaussian kernel k(x,y)=exp(- ||x-y||**2 / (2 * sigma**2))

    X - 2d array, samples on left hand side
    Y - 2d array, samples on right hand side, can be None in which case they are replaced by X

    returns: kernel matrix
    """
    sq_dists = sq_distances(X,Y)
    K = np.exp(-sq_dists / (2 * sigma**2))
    return K

def quadratic_time_mmd(X,Y,kernel):
    assert X.ndim == Y.ndim == 2
    K_XX = kernel(X,X)
    K_XY = kernel(X,Y)
    K_YY = kernel(Y,Y)

    n = len(K_XX)
    m = len(K_YY)

    # unbiased MMD statistic (could also use biased, doesn't matter if we use permutation tests)
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    mmd = np.sum(K_XX) / (n*(n-1))  + np.sum(K_YY) / (m*(m-1))  - 2*np.sum(K_XY)/(n*m)
    return mmd

def gaussian_kernel_median_heuristic(Z):
    # compute the median of the pairwise distances in Z
    # (not taking zero distance between identical samples (diagonal) into account)
    sq_dists = sq_distances(Z)
    np.fill_diagonal(sq_dists, np.nan)
    sq_dists = np.ravel(sq_dists)
    sq_dists = sq_dists[~np.isnan(sq_dists)]
    median_dist = np.median(np.sqrt(sq_dists))

    return np.sqrt(median_dist/2.0) # our kernel uses a bandwidth of 2*(sigma**2)

def mmd_with_median_heuristic(X, Y):
  sigma_median = gaussian_kernel_median_heuristic(np.vstack((X,Y)))

  my_kernel = lambda X,Y : gauss_kernel(X,Y,sigma=sigma_median)
  my_mmd = lambda X,Y : quadratic_time_mmd(X, Y, my_kernel)
  statistics = two_sample_permutation_test(my_mmd, X, Y, num_permutations=200)
  my_statistic = my_mmd(X,Y)
  # plot_permutation_samples(statistics, my_statistic)
  return my_statistic, statistics, sigma_median




# print('INVESTIGATING p(x_2|x_1)')

# k_2 = GPy.kern.RBF(input_dim=1)
# m_2 = GPy.models.GPRegression(X_1, X_2, k_2)
# X_12_true = np.concatenate((X_1, X_2), axis=1)

# print('GP fit BEFORE hyper-parameter optimisation:')
# # display(m_2)
# m_2.plot()
# plt.show()

# # check model fit with MMD
# for model_type in ['iv']:
#   print('Checking fit of distribution:', model_type)
#   if model_type == 'iv':
#     X_2_samples = sample_from_GP_model(m_2, X_1, distribution_type=model_type)

#   elif model_type == 'cf':
#     X_2_samples = sample_from_GP_model(m_2, X_1, distribution_type=model_type, factual_instance=1)

#   X_12_approximate = np.concatenate((X_1, X_2_samples), axis=1)

#   my_statistic, statistics, sigma_median = mmd_with_median_heuristic(X_12_true, X_12_approximate)
#   print('using median of ', sigma_median, 'as bandwith')
#   print('test-statistic = ', my_statistic)

# # optimise hyperparams
# m_2.optimize_restarts(parallel=True, num_restarts = 5)
# print('GP fit AFTER hyper-parameter optimisation:')
# # display(m_2)
# m_2.plot()
# plt.show()

# # check model fit with MMD
# for model_type in ['iv']:
#   # print('Checking fit of distribution:', model_type)
#   if model_type == 'iv':
#     X_2_samples = sample_from_GP_model(m_2, X_1, distribution_type=model_type)

#   elif model_type == 'cf':
#     X_2_samples = sample_from_GP_model(m_2, X_1, distribution_type=model_type, factual_instance=1)

#   X_12_approximate = np.concatenate((X_1, X_2_samples), axis=1)

#   my_statistic, statistics, sigma_median = mmd_with_median_heuristic(X_12_true, X_12_approximate)
#   print('using median of ', sigma_median, 'as bandwith')
#   print('test-statistic = ', my_statistic)

# print('INVESTIGATING p(x_3|x_1, x_2)')
# X_123_true = np.concatenate((X_1, X_2, X_3), axis=1)
# k_3 = GPy.kern.RBF(input_dim=2, ARD=True)
# X_pa_3 = np.concatenate((X_1, X_2), axis=1)
# m_3 = GPy.models.GPRegression(X_pa_3, X_3, k_3)

# print('GP fit BEFORE hyper-parameter optimisation:')
# # display(m_3)
# m_3.plot()
# plt.show()

# # check model fit with MMD
# for model_type in ['iv']:
#   # print('Checking fit of distribution:', model_type)
#   if model_type == 'iv':
#     X_3_samples = sample_from_GP_model(m_3, X_pa_3, distribution_type=model_type)

#   elif model_type == 'cf':
#     X_3_samples = sample_from_GP_model(m_3, X_pa_3, distribution_type=model_type, factual_instance=1)

#   X_123_approximate = np.concatenate((X_1, X_2, X_3_samples), axis=1)

#   my_statistic, statistics, sigma_median = mmd_with_median_heuristic(X_123_true, X_123_approximate)
#   print('using median of ', sigma_median, 'as bandwith')
#   print('test-statistic = ', my_statistic)

# # optimise hyperparams
# m_3.optimize_restarts(parallel=True, num_restarts = 5)
# print('GP fit AFTER hyper-parameter optimisation:')
# # display(m_3)
# m_3.plot()
# plt.show()

# # check model fit with MMD
# for model_type in ['iv']:
#   print('Checking fit of distribution:', model_type)
#   if model_type == 'iv':
#     X_3_samples = sample_from_GP_model(m_3, X_pa_3, distribution_type=model_type)

#   elif model_type == 'cf':
#     X_3_samples = sample_from_GP_model(m_3, X_pa_3, distribution_type=model_type, factual_instance=1)

#   X_123_approximate = np.concatenate((X_1, X_2, X_3_samples), axis=1)

#   my_statistic, statistics, sigma_median = mmd_with_median_heuristic(X_123_true, X_123_approximate)
#   print('using median of ', sigma_median, 'as bandwith')
#   print('test-statistic = ', my_statistic)
