import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm # norm for univariate; use multivariate_normal otherwise
from scipy.stats import bernoulli

# univariate distributions
class BaseDistribution(object):
  def __init__(self):
    pass

  def sample(self, size=1):
    raise NotImplementedError

  def pdf(self):
    raise NotImplementedError

  def visualize(self):
    plt.hist(self.sample(500), 50, facecolor='green', alpha=0.75)
    plt.ylabel('Count')
    plt.title(fr'Histogram of {self.name}')
    plt.grid(True)
    plt.show()


class Normal(BaseDistribution):
  def __init__(self, mean, std):
    assert isinstance(mean, int) or isinstance(mean, float), 'Expected `mean` to be an int or float.'
    assert isinstance(std, int) or isinstance(std, float), 'Expected `std` to be an int or float.'
    self.mean = mean
    self.std = std
    self.name = f'Normal\t mean={self.mean}, std={self.std}'

  def sample(self, size=1):
    tmp = [np.random.normal(self.mean, np.sqrt(self.std)) for _ in range(size)]
    return tmp[0] if size == 1 else tmp

  def pdf(self, value):
    return norm(self.mean, self.std).pdf(value)


class MixtureOfGaussians(BaseDistribution):

  def __init__(self, probs, means, stds):
    assert sum(probs) == 1, 'Mixture probabilities must sum to 1.'
    assert len(probs) == len(means) == len(stds), 'Length mismatch.'
    self.probs = probs
    self.means = means
    self.stds = stds
    self.name = f'MoG\t probs={self.probs}, means={self.means}, stds={self.stds}'

  def sample(self, size=1):
    tmp = [
      np.random.normal(self.means[mixture_idx], np.sqrt(self.stds[mixture_idx]))
      for mixture_idx in np.random.choice(len(self.probs), size=size, p=self.probs)
    ]
    return tmp[0] if size == 1 else tmp

  def pdf(self, value):
    return np.sum([
      prob * norm(mean, std).pdf(value)
      for (prob, mean, std) in zip(self.probs, self.means, self.stds)
    ])

class Bernoulli(BaseDistribution):

  def __init__(self, prob):
    assert isinstance(prob, int) or isinstance(prob, float), 'Expected `prob` to be an int or float.'
    assert prob >= 0 and prob <= 1

    self.prob = prob
    self.name = f'Bernoulli\t prob={self.prob}'

  def sample(self, size=1):
    tmp = bernoulli.rvs(self.prob, size=size)
    return tmp[0] if size == 1 else list(tmp)

  def pdf(self, value):
    raise Exception(f'not supported yet; code should not come here.')
    # return norm(self.mean, self.std).pdf(value)

class Poisson(BaseDistribution):

  def __init__(self, p_lambda):
    assert isinstance(p_lambda, int) or isinstance(p_lambda, float), 'Expected `p_lambda` to be an int or float.'
    assert p_lambda >= 0
    self.p_lambda = p_lambda
    self.name = f'RoundedPoisson\t prob={self.p_lambda}'

  def sample(self, size=1):
    tmp = np.random.poisson(self.p_lambda, size)
    return tmp[0] if size == 1 else list(tmp)

  def pdf(self, value):
    raise Exception(f'not supported yet; code should not come here.')
    # return norm(self.mean, self.std).pdf(value)


# # test
# Normal(0,1).sample()
# Normal(0,1).sample(10)
# Normal(0,1).pdf(0)
# Normal(0,1).visualize()

# MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]).sample()
# MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]).sample(10)
# MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]).pdf(0)
# MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]).pdf(+2)
# MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]).pdf(-2)
# MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]).visualize()
