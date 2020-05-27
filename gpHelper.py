import GPy
import autograd.numpy as np
# from autograd import grad


class RBFKernel():
    def __init__(self, input_dim, signal_var, lengthscales):
        assert lengthscales.shape[0] == input_dim
        self.input_dim  = input_dim
        self.signal_var = signal_var
        self.lengthscales = lengthscales

    def eval_sample(self, x1, x2):
        return self.signal_var * np.exp(-0.5*np.sum(((x1-x2)/self.lengthscales)**2))

    def eval(self, x1, x2):
        n1, d1 = x1.shape
        n2, d2 = x2.shape
        assert d1 == d2
        K = []
        for i in range(n1):
            K.append(np.concatenate([self.eval_sample(x1[i], x2[j]) for j in range(n2)]))

        return np.array(K)

def get_optimized_rbf_kernel(m):
    input_dim = m.input_dim
    signal_var = np.array(m.kern.variance)
    lengthscales = np.array(m.kern.lengthscale)
    return RBFKernel(input_dim, signal_var, lengthscales)

def get_inverse_covariance(K, noise_var):
    return np.linalg.inv(K + noise_var*np.eye(K.shape[0]))

def get_manual_GP_model(m):
    kernel = get_optimized_rbf_kernel(m)
    noise_var = np.array(m.Gaussian_noise.variance)
    X_train = m.X
    Y_train = m.Y
    K = kernel.eval(X_train, X_train)
    inv_cov = get_inverse_covariance(K, noise_var)
    return kernel, noise_var, X_train, Y_train, K, inv_cov

def noise_posterior_mean(noise_var, inv_cov, Y_train):
    return noise_var * np.dot(inv_cov, Y_train)

def noise_posterior_covariance(noise_var, inv_cov):
    N = inv_cov.shape[0]
    return  noise_var * (np.eye(N) - noise_var * inv_cov)

def noise_posterior_variance(noise_var, inv_cov):
    C = noise_posterior_covariance(noise_var, inv_cov)
    return np.array([C[i,i] for i in range(C.shape[0])])

def get_pred_post_noiseless(x_new, X_train, Y_train, kernel, inv_cov):
    k_new = kernel.eval(x_new, x_new)
    K_new = kernel.eval(x_new, X_train)
    pred_mean = np.matmul(np.matmul(K_new, inv_cov), Y_train)
    pred_cov = k_new - np.matmul(np.matmul(K_new, inv_cov), np.transpose(K_new))
    pred_var = np.diag(pred_cov).reshape(pred_mean.shape)
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
    return mean + np.random.randn(mean.shape[0],1) * np.sqrt(var)

def sample_from_GP_model(m, X_new, distribution_type='iv', factual_instance=None):
    assert X_new.shape[1] == m.input_dim
    kernel, noise_var, X_train, Y_train, K, inv_cov = get_manual_GP_model(m)
    pred_mean, pred_var = get_pred_post_noiseless(X_new, X_train, Y_train, kernel, inv_cov)
    mean, var = get_predictive_distribution(pred_mean, pred_var, noise_var, distribution_type, inv_cov, Y_train, factual_instance)
    Y_new = sample_from_Gaussian_with_reparametrisation_trick(mean, var)
    return Y_new

def plot_predicted_outcomes(theta, F, M, m_2, m_3, f_2, f_3, U_2, U_3, sigma_2, sigma_3):
    X_SCF_2 = f_2(theta) + U_2[F]
    X_SCF_3 = f_3(theta, X_SCF_2) + U_3[F]

    X_1_new = theta * np.ones((M, 1))  # fixed by intervention, same as X_1_cf

    X_2_new_iv = sample_from_GP_model(m_2, X_1_new, 'iv', F)
    X_2_new_cf = sample_from_GP_model(m_2, X_1_new, 'cf', F)
    X_2_new_true = f_2(X_1_new) + np.sqrt(sigma_2) * np.random.randn(M, 1)

    X_pa_3_new_iv = np.concatenate((X_1_new, X_2_new_iv), axis=1)
    X_pa_3_new_cf = np.concatenate((X_1_new, X_2_new_cf), axis=1)
    X_3_new_iv = sample_from_GP_model(m_3, X_pa_3_new_iv, 'iv', F)
    X_3_new_cf = sample_from_GP_model(m_3, X_pa_3_new_cf, 'cf', F)
    X_3_new_true = f_3(X_1_new, X_2_new_true) + np.sqrt(sigma_3) * np.random.randn(M, 1)

    plt.plot(X_SCF_2, X_SCF_3, 'ko', label='M0-oracle')
    plt.plot(X_2_new_cf, X_3_new_cf, 'b+', label='M1-GP (cf.)')
    plt.plot(X_2_new_iv, X_3_new_iv, 'rx', label='M2-GP (iv.)')
    plt.plot(X_2_new_true, X_3_new_true, 'g.', label='M2-oracle (iv.)' )
    plt.legend()
    plt.title('do(X_1=%.2f)' % (theta))
    plt.xlabel('$X_2$')
    plt.ylabel('$X_3$')
    plt.show()

    # print('Factual observation no.: ', F)
    # print('Intervention: do(X_1 =', theta.squeeze(), ')')
    # print('True (oracle) counterfactual:')
    # print('X_2_CF =', X_SCF_2.squeeze())
    # print('X_3_CF =', X_SCF_3.squeeze())
    # print('U_2[', F,'] = ', U_2[F].squeeze())
    # print('U_3[', F,'] = ', U_3[F].squeeze())
    # print('Noise prior CI for U_2: 0 +/- ', 1.96*np.sqrt(noise_var_2.squeeze()))
    # print('Noise prior CI for U_3: 0 +/- ', 1.96*np.sqrt(noise_var_3.squeeze()))
    # print('Noise posterior CI for U_2:', noise_post_means_2[F].squeeze(),
    #       '+/-', 1.96*np.sqrt(noise_post_vars_2[F]))
    # print('Noise posterior CI for U_3:', noise_post_means_3[F].squeeze(),
    #       '+/-', 1.96*np.sqrt(noise_post_vars_3[F]))
    # print('Interventional CI for X_2:', iv_mean_2.squeeze()[0],
    #       '+/-', 1.96*np.sqrt(iv_var_2.squeeze()[0]))
    # print('Counterfactual CI for X_2:', cf_mean_2.squeeze()[0],
    #       '+/-', 1.96*np.sqrt(cf_var_2.squeeze()[0]))

def objective_for_intervention_on_x1(theta, M, m_2, m_3, distribution_type, factual_instance, lambda_LCB=0):
    X_1_new = theta * np.ones((M, 1))

    # sample X_2|do(X_1=X_1_new)
    X_2_new = sample_from_GP_model(m_2, X_1_new, distribution_type, factual_instance)

    # sample X_3|do(X_1=X_1_new), X_2=X_2_new
    X_pa_3_new = np.concatenate((X_1_new, X_2_new), axis=1)
    X_3_new = sample_from_GP_model(m_3, X_pa_3_new, distribution_type, factual_instance)

    # compute objective from samples
    classified_samples = h(X_1_new, X_2_new, X_3_new)
    return np.mean(classified_samples) - lambda_LCB * np.std(classified_samples)

def oracle_objective(theta, M, f_2, f_3, U_2, U_3, sigma_2, sigma_3, F, distribution_type, lambda_LCB=0):
    X_1_new = theta * np.ones((M, 1))
    if distribution_type == 'cf':
        X_2_new = f_2(X_1_new) + U_2[F]
        X_3_new = f_3(X_1_new, X_2_new) + U_3[F]

    elif distribution_type == 'iv':
        X_2_new = f_2(X_1_new) + np.sqrt(sigma_2) * np.random.randn(M,1)
        X_3_new = f_3(X_1_new, X_2_new) + np.sqrt(sigma_3) * np.random.randn(M,1)

    classified_samples = h(X_1_new, X_2_new, X_3_new)
    return np.mean(classified_samples) - lambda_LCB * np.std(classified_samples)

# grad_objective_for_intervention_on_x1 = grad(objective_for_intervention_on_x1)
