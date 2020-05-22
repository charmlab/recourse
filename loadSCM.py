import sys
import numpy as np

import utils
from scm import CausalModel
from distributions import *
from main import getNoiseStringForNode

from debug import ipsh

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)


@utils.Memoize
def loadSCM(scm_class, experiment_folder_name = None):

  structural_equations = {
    'x1': lambda n_samples, : n_samples,
    # 'x2': TBD
  }
  noises_distributions = {
    'u1': MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]),
    'u2': Normal(0, 1),
  }

  if scm_class == 'sanity-2-add':
    structural_equations['x2'] = lambda n_samples, x1 : 2 * x1 + n_samples
  elif scm_class == 'sanity-2-mult':
    structural_equations['x2'] = lambda n_samples, x1 : np.array(x1) * n_samples
  elif scm_class == 'sanity-2-add-pow':
    structural_equations['x2'] = lambda n_samples, x1 : (x1 + n_samples) ** 2
  elif scm_class == 'sanity-2-add-sig':
    structural_equations['x2'] = lambda n_samples, x1 : 5 / (1 + np.exp(- x1 - n_samples))
  elif scm_class == 'sanity-2-sig-add':
    structural_equations['x2'] = lambda n_samples, x1 : 5 / (1 + np.exp(-x1)) + n_samples
  elif scm_class == 'sanity-2-pow-add':
    structural_equations['x2'] = lambda n_samples, x1 : x1 ** 2 + n_samples
  elif scm_class == 'sanity-2-sin-add':
    structural_equations['x2'] = lambda n_samples, x1 : np.sin(x1 + 2 * x2) + n_samples
  elif scm_class == 'sanity-2-cos-exp-add':
    structural_equations['x2'] = lambda n_samples, x1 : 2 * np.cos(3 * x1) * np.exp(-0.3 * x1**2) + n_samples

  if scm_class == 'sanity-3-add':

    structural_equations = {
      'x1': lambda n_samples,        :           n_samples,
      'x2': lambda n_samples, x1     :  2 * x1 + n_samples,
      'x3': lambda n_samples, x1, x2 : x1 + x2 + n_samples,
    }
    noises_distributions = {
      'u1': MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]),
      'u2': Normal(0, 1),
      'u3': Normal(0, 1),
    }

  elif scm_class == 'sanity-3-mult':

    structural_equations = {
      'x1': lambda n_samples,        :           n_samples,
      'x2': lambda n_samples, x1     :  2 * x1 + n_samples,
      'x3': lambda n_samples, x1, x2 : x1 * x2 + n_samples,
    }
    noises_distributions = {
      'u1': MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]),
      'u2': Normal(0, 1),
      'u3': Normal(0, 1),
    }

  elif scm_class == 'sanity-3-power':

    structural_equations = {
      'x1': lambda n_samples,        :                  n_samples,
      'x2': lambda n_samples, x1     :         2 * x1 + n_samples,
      'x3': lambda n_samples, x1, x2 : (x1 + x2 + n_samples) ** 2,
    }
    noises_distributions = {
      'u1': MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]),
      'u2': Normal(0, 1),
      'u3': Normal(0, 1),
    }

  assert \
    set([getNoiseStringForNode(node) for node in structural_equations.keys()]) == \
    set(noises_distributions.keys()), \
    'structural_equations & noises_distributions should have identical keys.'

  scm = CausalModel(scm_class, structural_equations, noises_distributions)
  if experiment_folder_name is not None:
    scm.visualizeGraph(experiment_folder_name)
  return scm
