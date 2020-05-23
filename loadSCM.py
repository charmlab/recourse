import os
import sys
import copy
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
    self.structural_equations = args[1]
    self.noises_distributions = args[2]

    self._scm = StructuralCausalModel(self.structural_equations) # may be redundant, can simply call CausalGraphicalModel...
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
      save_path = f'{experiment_folder_name}/causal_graph'
      view_flag = False
    else:
      save_path = '_tmp/causal_graph'
      view_flag = True
    self._cgm.draw().render(save_path, view=view_flag)

  def printSCM(self):
    raise NotImplementedError



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
    structural_equations['x2'] = lambda n_samples, x1 : np.sin(x1) + n_samples
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
      'u3': Normal(0, 5),
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
