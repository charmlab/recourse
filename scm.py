import os
import sys
import copy
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from sklearn.model_selection import train_test_split

from debug import ipsh

from causalgraphicalmodels import CausalGraphicalModel
from causalgraphicalmodels import StructuralCausalModel
import networkx as nx


class CausalModel(object):

  def __init__(self, *args, **kwargs):

    self.structural_equations = args[0]
    self.noises_distributions = args[1]

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





# print(f"getChildrenForNode('x1'): \t {self.getChildrenForNode('x1')}")
# print(f"getChildrenForNode('x2'): \t {self.getChildrenForNode('x2')}")
# print(f"getChildrenForNode('x3'): \t {self.getChildrenForNode('x3')}")
# print(f"getChildrenForNode('x4'): \t {self.getChildrenForNode('x4')}")
# print(f"getChildrenForNode('x5'): \t {self.getChildrenForNode('x5')}")

# print(f"getDescendentsForNode('x1'): \t {self.getDescendentsForNode('x1')}")
# print(f"getDescendentsForNode('x2'): \t {self.getDescendentsForNode('x2')}")
# print(f"getDescendentsForNode('x3'): \t {self.getDescendentsForNode('x3')}")
# print(f"getDescendentsForNode('x4'): \t {self.getDescendentsForNode('x4')}")
# print(f"getDescendentsForNode('x5'): \t {self.getDescendentsForNode('x5')}")

# print(f"getParentsForNode('x1'): \t {self.getParentsForNode('x1')}")
# print(f"getParentsForNode('x2'): \t {self.getParentsForNode('x2')}")
# print(f"getParentsForNode('x3'): \t {self.getParentsForNode('x3')}")
# print(f"getParentsForNode('x4'): \t {self.getParentsForNode('x4')}")
# print(f"getParentsForNode('x5'): \t {self.getParentsForNode('x5')}")

# print(f"getAncestorsForNode('x1'): \t {self.getAncestorsForNode('x1')}")
# print(f"getAncestorsForNode('x2'): \t {self.getAncestorsForNode('x2')}")
# print(f"getAncestorsForNode('x3'): \t {self.getAncestorsForNode('x3')}")
# print(f"getAncestorsForNode('x4'): \t {self.getAncestorsForNode('x4')}")
# print(f"getAncestorsForNode('x5'): \t {self.getAncestorsForNode('x5')}")

# print(f"getNonDescendentsForNode('x1'): \t {self.getNonDescendentsForNode('x1')}")
# print(f"getNonDescendentsForNode('x2'): \t {self.getNonDescendentsForNode('x2')}")
# print(f"getNonDescendentsForNode('x3'): \t {self.getNonDescendentsForNode('x3')}")
# print(f"getNonDescendentsForNode('x4'): \t {self.getNonDescendentsForNode('x4')}")
# print(f"getNonDescendentsForNode('x5'): \t {self.getNonDescendentsForNode('x5')}")
