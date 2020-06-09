import os
import sys
import copy
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from sklearn.model_selection import train_test_split

import utils
from distributions import *
from main import getNoiseStringForNode

from causalgraphicalmodels import CausalGraphicalModel
from causalgraphicalmodels import StructuralCausalModel
import networkx as nx

from debug import ipsh

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)


class CausalModel(object):

  def __init__(self, *args, **kwargs):

    self.scm_class = args[0]
    self.structural_equations_np = args[1]
    self.structural_equations_ts = args[2]
    self.noises_distributions = args[3]

    self._scm = StructuralCausalModel(self.structural_equations_np) # may be redundant, can simply call CausalGraphicalModel...
    self._cgm = self._scm.cgm

  def getTopologicalOrdering(self, node_type = 'endogenous'):
    tmp = nx.topological_sort(self._cgm.dag)
    if node_type == 'endogenous':
      return tmp
    elif node_type == 'exogenous':
      return ['u'+node[1:] for node in tmp]
    else:
      raise Exception(f'{node_type} not recognized.')

  def getChildrenForNode(self, node):
    return set(self._cgm.dag.successors(node))

  def getDescendentsForNode(self, node):
    return nx.descendants(self._cgm.dag, node)

  def getParentsForNode(self, node, return_sorted = True):
    tmp = set(self._cgm.dag.predecessors(node))
    return sorted(tmp) if return_sorted else tmp

  def getAncestorsForNode(self, node):
    return nx.ancestors(self._cgm.dag, node)

  def getNonDescendentsForNode(self, node):
    return set(nx.topological_sort(self._cgm.dag)) \
      .difference(self.getDescendentsForNode(node)) \
      .symmetric_difference(set([node]))

  def getStructuralEquationForNode(self, node):
    # self._scm.assignment[node]
    raise NotImplementedError

  def visualizeGraph(self, experiment_folder_name = None):
    if experiment_folder_name:
      save_path = f'{experiment_folder_name}/_causal_graph'
      view_flag = False
    else:
      save_path = '_tmp/_causal_graph'
      view_flag = True
    self._cgm.draw().render(save_path, view=view_flag)

  def printSCM(self):
    raise NotImplementedError



@utils.Memoize
def loadSCM(scm_class, experiment_folder_name = None):
  print(f'loadSCM.loadSCM: scm_class: {scm_class}')

  # structural_equations_np = {
  #   'x1': lambda n_samples, : n_samples,
  #   # 'x2': TBD
  # }
  # structural_equations_ts = {
  #   'x1': lambda n_samples, : n_samples,
  #   # 'x2': TBD
  # }
  # noises_distributions = {
  #   'u1': MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]),
  #   'u2': Normal(0, 1),
  # }

  # if scm_class == 'sanity-2-add':
  #   structural_equations_np['x2'] = lambda n_samples, x1 : 2 * x1 + n_samples
  #   structural_equations_ts['x2'] = lambda n_samples, x1 : 2 * x1 + n_samples
  # elif scm_class == 'sanity-2-mult':
  #   structural_equations_np['x2'] = lambda n_samples, x1 : x1 * n_samples
  #   structural_equations_ts['x2'] = lambda n_samples, x1 : x1 * n_samples
  # elif scm_class == 'sanity-2-add-pow':
  #   structural_equations_np['x2'] = lambda n_samples, x1 : (x1 + n_samples) ** 2
  #   structural_equations_ts['x2'] = lambda n_samples, x1 : (x1 + n_samples) ** 2
  # elif scm_class == 'sanity-2-add-sig':
  #   structural_equations_np['x2'] = lambda n_samples, x1 : 5 / (1 + np.exp(- x1 - n_samples))
  #   structural_equations_ts['x2'] = lambda n_samples, x1 : 5 / (1 + torch.exp(- x1 - n_samples))
  # elif scm_class == 'sanity-2-sig-add':
  #   structural_equations_np['x2'] = lambda n_samples, x1 : 5 / (1 + np.exp(-x1)) + n_samples
  #   structural_equations_ts['x2'] = lambda n_samples, x1 : 5 / (1 + torch.exp(-x1)) + n_samples
  # elif scm_class == 'sanity-2-pow-add':
  #   structural_equations_np['x2'] = lambda n_samples, x1 : x1 ** 2 + n_samples
  #   structural_equations_ts['x2'] = lambda n_samples, x1 : x1 ** 2 + n_samples
  # elif scm_class == 'sanity-2-sin-add':
  #   structural_equations_np['x2'] = lambda n_samples, x1 : np.sin(x1) + n_samples
  #   structural_equations_ts['x2'] = lambda n_samples, x1 : torch.sin(x1) + n_samples
  # elif scm_class == 'sanity-2-cos-exp-add':
  #   structural_equations_np['x2'] = lambda n_samples, x1 : 2 * np.cos(3 * x1) * np.exp(-0.3 * x1**2) + n_samples
  #   structural_equations_ts['x2'] = lambda n_samples, x1 : 2 * torch.cos(3 * x1) * torch.exp(-0.3 * x1**2) + n_samples

  # ============================================================================
  # ABOVE: 2-variable sanity models used for checking cond. dist. fit
  # BELOW: 3+variable sanity models used in paper
  # ============================================================================

  if scm_class == 'sanity-3-lin':

    structural_equations_np = {
      'x1': lambda n_samples,        :                               n_samples,
      'x2': lambda n_samples, x1     :                        - x1 + n_samples,
      'x3': lambda n_samples, x1, x2 : 0.5 * (0.1 * x1 + 0.5 * x2) + n_samples,
    }
    structural_equations_ts = structural_equations_np
    noises_distributions = {
      'u1': MixtureOfGaussians([0.5, 0.5], [-2, +1], [1.5, 1]),
      'u2': Normal(0, 1),
      'u3': Normal(0, 1),
    }

  elif scm_class == 'sanity-3-anm':

    structural_equations_np = {
      'x1': lambda n_samples,        :                                    n_samples,
      'x2': lambda n_samples, x1     : 3 / (1 + np.exp(- 2.0 * x1 )) -1 + n_samples,
      'x3': lambda n_samples, x1, x2 : - 0.5 * (0.1 * x1 + 0.5 * x2**2) + n_samples,
    }
    structural_equations_ts = {
      'x1': lambda n_samples,        :                                       n_samples,
      'x2': lambda n_samples, x1     : 3 / (1 + torch.exp(- 2.0 * x1 )) -1 + n_samples,
      'x3': lambda n_samples, x1, x2 :    - 0.5 * (0.1 * x1 + 0.5 * x2**2) + n_samples,
    }
    noises_distributions = {
      'u1': MixtureOfGaussians([0.5, 0.5], [-2, +1], [1.5, 1]),
      'u2': Normal(0, 0.1),
      'u3': Normal(0, 1),
    }

  elif scm_class == '_bu_sanity-3-gen':

    structural_equations_np = {
      'x1': lambda n_samples,        :                                               n_samples,
      'x2': lambda n_samples, x1     : - 3 * (1 / (1 + np.exp(- 2.0 * x1  + n_samples)) - 0.4),
      'x3': lambda n_samples, x1, x2 :    - 0.5 * (0.1 * x1 + 0.5 * (x2 - 0.0)**2 * n_samples),
    }
    structural_equations_ts = {
      'x1': lambda n_samples,        :                                                 n_samples,
      'x2': lambda n_samples, x1     : - 3 * (1 / (1 + torch.exp(- 2.0 * x1 + n_samples)) - 0.4),
      'x3': lambda n_samples, x1, x2 :      - 0.5 * (0.1 * x1 + 0.5 * (x2 - 0.0)**2 * n_samples),
    }
    noises_distributions = {
      'u1': MixtureOfGaussians([0.5, 0.5], [-2, +1], [1.5, 1]),
      'u2': Normal(0, 1),
      'u3': Normal(0, 1),
    }

  elif scm_class == 'sanity-3-gen-OLD':

    structural_equations_np = {
      'x1': lambda n_samples,        :                                                       n_samples,
      'x2': lambda n_samples, x1     :        - 3 * (1 / (1 + np.exp(- 1 * x1**2  + n_samples)) - 0.4),
      # 'x3': lambda n_samples, x1, x2 : np.sin(x1) * np.exp(-(x1+n_samples)**2) + x2**2 * (n_samples-2),
      'x3': lambda n_samples, x1, x2 : np.sin(x1) * np.exp(-(x1+n_samples)**2) + x2**2 * (n_samples-2),
    }
    structural_equations_ts = {
      'x1': lambda n_samples,        :                                                             n_samples,
      'x2': lambda n_samples, x1     :           - 3 * (1 / (1 + torch.exp(- 1 * x1**2  + n_samples)) - 0.4),
      'x3': lambda n_samples, x1, x2 : torch.sin(x1) * torch.exp(-(x1+n_samples)**2) + x2**2 * (n_samples-2),
    }
    noises_distributions = {
      'u1': MixtureOfGaussians([0.5, 0.5], [-2, +1], [1.5, 1]),
      'u2': Normal(0, 1),
      'u3': Normal(0, 1),
    }

  elif scm_class == 'sanity-3-gen-NEW':

    structural_equations_np = {
      'x1': lambda n_samples,        :                                   n_samples,
      # 'x2': lambda n_samples, x1     :      - 2 * np.sign(n_samples) * (x1 * (1 + n_samples)) - 1,
      'x2': lambda n_samples, x1     :      np.sign(n_samples) * (x1 ** 2 + n_samples) / 5,
      'x3': lambda n_samples, x1, x2 : -1 * np.sqrt(x1**2 + x2**2) + n_samples,
      # 'x3': lambda n_samples, x1, x2 : np.sin(x1) * np.exp(-(x1+n_samples)**2) + x2**2 * (n_samples-2),
    }
    structural_equations_ts = structural_equations_np
    # structural_equations_ts = {
    #   'x1': lambda n_samples,        :                                                             n_samples,
    #   'x2': lambda n_samples, x1     :           - 3 * (1 / (1 + torch.exp(- 1 * x1**2  + n_samples)) - 0.4),
    #   'x3': lambda n_samples, x1, x2 : torch.sin(x1) * torch.exp(-(x1+n_samples)**2) + x2**2 * (n_samples-2),
    # }
    noises_distributions = {
      'u1': MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]),
      'u2': Normal(0, .3),
      'u3': Normal(0, 1),
    }

  elif scm_class == 'sanity-3-gen':
    a0 = 0.25
    b = -1
    b0 = .1
    b1 = 1
    b2 = 1

    structural_equations_np = {
      'x1': lambda n_samples,        :                                   n_samples,
      'x2': lambda n_samples, x1     :      a0 * np.sign(n_samples) * (x1 ** 2) * (1 + n_samples**2),
      'x3': lambda n_samples, x1, x2 : b + b0 * (b1 * x1**2 + b2 * x2**2) + n_samples,
    }
    structural_equations_ts = {
      'x1': lambda n_samples,        :                                   n_samples,
      'x2': lambda n_samples, x1     :      a0 * torch.sign(torch.tensor(n_samples)) * (x1 ** 2) * (1 + n_samples**2),
      'x3': lambda n_samples, x1, x2 : b + b0 * (b1 * x1**2 + b2 * x2**2) + n_samples,
    }
    noises_distributions = {
      'u1': MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]),
      'u2': Normal(0, .5**2),
      'u3': Normal(0, .25**2),
    }

  elif scm_class == 'sanity-6-lin':

    structural_equations_np = {
      'x1': lambda n_samples,: n_samples,
      'x2': lambda n_samples,: n_samples,
      'x3': lambda n_samples,: n_samples,
      'x4': lambda n_samples, x1, x2: x1 + 2 * x2 + n_samples,
      'x5': lambda n_samples, x2, x3, x4: x2 - x4 + 2 * x3 + n_samples,
      'x6': lambda n_samples, x1, x3, x4, x5: x3 + x4 - x5 + x1 + n_samples,
    }
    structural_equations_ts = structural_equations_np
    noises_distributions = {
      'u1': MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]),
      'u2': MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]),
      'u3': MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]),
      'u4': Normal(0, 2),
      'u5': Normal(0, 2),
      'u6': Normal(0, 2),
    }

  elif scm_class == 'german-credit':

    e_0 = -1
    e_G = 0.5
    e_A = 1

    l_0 = 1
    l_A = .01
    l_G = 1

    d_0 = -1
    d_A = .1
    d_G = 2
    d_L = 1

    i_0 = -4
    i_A = .1
    i_G = 2
    i_E = 10
    i_GE = 1

    s_0 = -4
    s_I = 1.5

    structural_equations_np = {
      # Gender
      'x1': lambda n_samples,: n_samples,
      # Age
      'x2': lambda n_samples,: -35 + n_samples,
      # Education
      'x3': lambda n_samples, x1, x2 : -0.5 + (1 + np.exp(-(e_0 + e_G * x1 + e_A * (1 + np.exp(- .1 * (x2)))**(-1) + n_samples)))**(-1),
      # Loan amount
      'x4': lambda n_samples, x1, x2 :  l_0 + l_A * (x2 - 5) * (5 - x2) + l_G * x1 + n_samples,
      # Loan duration
      'x5': lambda n_samples, x1, x2, x4 : d_0 + d_A * x2 + d_G * x1 + d_L * x4 + n_samples,
      # Income
      'x6': lambda n_samples, x1, x2, x3 : i_0 + i_A * (x2 + 35) + i_G * x1 + i_GE * x1 * x3 + n_samples,
      # Savings
      'x7': lambda n_samples, x6 : s_0 + s_I * (x6 > 0) * x6 + n_samples,
    }
    structural_equations_ts = {
      # Gender
      'x1': lambda n_samples,: n_samples,
      # Age
      'x2': lambda n_samples,: -35 + n_samples,
      # Education
      'x3': lambda n_samples, x1, x2 : -0.5 + (1 + torch.exp(-(e_0 + e_G * x1 + e_A * (1 + torch.exp(- .1 * (x2)))**(-1) + n_samples)))**(-1),
      # Loan amount
      'x4': lambda n_samples, x1, x2 :  l_0 + l_A * (x2 - 5) * (5 - x2) + l_G * x1 + n_samples,
      # Loan duration
      'x5': lambda n_samples, x1, x2, x4 : d_0 + d_A * x2 + d_G * x1 + d_L * x4 + n_samples,
      # Income
      'x6': lambda n_samples, x1, x2, x3 : i_0 + i_A * (x2 + 35) + i_G * x1 + i_GE * x1 * x3 + n_samples,
      # Savings
      'x7': lambda n_samples, x6 : s_0 + s_I * (x6 > 0) * x6 + n_samples,
    }
    noises_distributions = {
      # Gender
      'u1': Bernoulli(0.5),
      # Age
      'u2': Gamma(10, 3.5),
      # Education
      'u3': Normal(0, 0.5**2),
      # Loan amount
      'u4': Normal(0, 2**2),
      # Loan duration
      'u5': Normal(0, 3**2),
      # Income
      'u6': Normal(0, 2**2),
      # Savings
      'u7': Normal(0, 5**2),
    }

  else:
    raise Exception(f'scm_class `{scm_class}` not recognized.')

  assert \
    list([getNoiseStringForNode(node) for node in structural_equations_np.keys()]) == \
    list([getNoiseStringForNode(node) for node in structural_equations_ts.keys()]) == \
    list(noises_distributions.keys()), \
    'structural_equations_np & structural_equations_ts & noises_distributions should have identical keys.'

  scm = CausalModel(scm_class, structural_equations_np, structural_equations_ts, noises_distributions)
  if experiment_folder_name is not None:
    scm.visualizeGraph(experiment_folder_name)
  return scm

