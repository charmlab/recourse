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
  def __init__(self, mean, var):
    assert isinstance(mean, int) or isinstance(mean, float), 'Expected `mean` to be an int or float.'
    assert isinstance(var, int) or isinstance(var, float), 'Expected `var` to be an int or float.'
    self.mean = mean
    self.var = var
    self.name = f'Normal\t mean={self.mean}, var={self.var}'

  def sample(self, size=1):
    tmp = [np.random.normal(self.mean, np.sqrt(self.var)) for _ in range(size)]
    return tmp[0] if size == 1 else tmp

  def pdf(self, value):
    return norm(self.mean, self.var).pdf(value)


class MixtureOfGaussians(BaseDistribution):

  def __init__(self, probs, means, vars):
    assert sum(probs) == 1, 'Mixture probabilities must sum to 1.'
    assert len(probs) == len(means) == len(vars), 'Length mismatch.'
    self.probs = probs
    self.means = means
    self.vars = vars
    self.name = f'MoG\t probs={self.probs}, means={self.means}, vars={self.vars}'

  def sample(self, size=1):
    tmp = [
      np.random.normal(self.means[mixture_idx], np.sqrt(self.vars[mixture_idx]))
      for mixture_idx in np.random.choice(len(self.probs), size=size, p=self.probs)
    ]
    return tmp[0] if size == 1 else tmp

  def pdf(self, value):
    return np.sum([
      prob * norm(mean, var).pdf(value)
      for (prob, mean, var) in zip(self.probs, self.means, self.vars)
    ])


class Uniform(BaseDistribution):

  def __init__(self, lower, upper):
    assert isinstance(lower, int) or isinstance(lower, float), 'Expected `lower` to be an int or float.'
    assert isinstance(upper, int) or isinstance(upper, float), 'Expected `upper` to be an int or float.'
    assert lower < upper

    self.lower = lower
    self.upper = upper
    self.name = f'Uniform\t lower={self.lower}\t upper={self.upper}'

  def sample(self, size=1):
    tmp = np.random.uniform(self.lower, self.upper, size=size)
    return tmp[0] if size == 1 else list(tmp)

  def pdf(self, value):
    raise 1 / (self.upper - self.lower)


class Bernoulli(BaseDistribution):

  def __init__(self, prob, btype='01'):
    assert isinstance(prob, int) or isinstance(prob, float), 'Expected `prob` to be an int or float.'
    assert prob >= 0 and prob <= 1

    self.prob = prob
    self.name = f'Bernoulli\t prob={self.prob}'
    self.btype = btype # '01' is standard, '-11' also supported here

  def sample(self, size=1):
    tmp = bernoulli.rvs(self.prob, size=size)
    if self.btype == '-11': tmp = tmp * 2 - 1
    return tmp[0] if size == 1 else list(tmp)

  def pdf(self, value):
    raise Exception(f'not supported yet; code should not come here.')


class Poisson(BaseDistribution):

  def __init__(self, p_lambda):
    assert isinstance(p_lambda, int) or isinstance(p_lambda, float), 'Expected `p_lambda` to be an int or float.'
    assert p_lambda > 0
    self.p_lambda = p_lambda
    self.name = f'Poisson\t prob={self.p_lambda}'

  def sample(self, size=1):
    tmp = np.random.poisson(self.p_lambda, size)
    return tmp[0] if size == 1 else list(tmp)

  def pdf(self, value):
    raise Exception(f'not supported yet; code should not come here.')


class Gamma(BaseDistribution):

  def __init__(self, shape, scale):
    assert isinstance(shape, int) or isinstance(shape, float), 'Expected `shape` to be an int or float.'
    assert isinstance(scale, int) or isinstance(scale, float), 'Expected `scale` to be an int or float.'
    assert shape > 0
    assert scale > 0
    self.shape = shape
    self.scale = scale
    self.name = f'Gamma\t shape={self.shape}, scale={self.scale}'

  def sample(self, size=1):
    tmp = np.random.gamma(self.shape, self.scale, size)
    return tmp[0] if size == 1 else list(tmp)

  def pdf(self, value):
    raise Exception(f'not supported yet; code should not come here.')


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
