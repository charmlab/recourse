import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

import loadSCM

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

from debug import ipsh

mu_x, sigma_x = 0, 1 # mean and standard deviation for data
mu_w, sigma_w = 0, 1 # mean and standard deviation for weights
n = 2500

# from main import getNoiseStringForNode $ TODO: should be from ../main??
def getNoiseStringForNode(node):
  assert node[0] == 'x'
  return 'u' + node[1:]

def load_random_data(scm_class, variable_type = 'real'):

  scm_obj = loadSCM.loadSCM(scm_class)

  d = len(scm_obj.structural_equations.keys())

  print(f'\n\n[INFO] Creating dataset using scm class `{scm_class}`...')

  # sample exogenous U variables
  print(f'\t[INFO] Sampling {n} exogenous U variables (d = {d})...')
  U = np.concatenate(
    [
      np.array(
        [scm_obj.noises_distributions[node].sample() for _ in range(n)]
      ).reshape(-1,1)
      for node in scm_obj.getTopologicalOrdering('exogenous')
    ],
    axis = 1,
  )
  U = pd.DataFrame(U, columns=scm_obj.getTopologicalOrdering('exogenous'))
  print(f'\t[INFO] done.')

  # sample endogenous X variables
  print(f'\t[INFO] Sampling {n} endogenous X variables (d = {d})...') # (i.e., processDataAccordingToGraph)
  X = U.copy()
  X = X.rename(columns=dict(zip(
    scm_obj.getTopologicalOrdering('exogenous'),
    scm_obj.getTopologicalOrdering('endogenous')
  )))
  X.loc[:] = np.nan # used later as an assertion to make sure parents are populated when computing children
  for node in scm_obj.getTopologicalOrdering('endogenous'):
    parents = scm_obj.getParentsForNode(node)
    # assuming we're iterating in topological order, parents in X should already be occupied
    assert not X.loc[:,list(parents)].isnull().values.any()
    for row_idx, row in tqdm(X.iterrows(), total=X.shape[0]):
      X.loc[row_idx, node] = scm_obj.structural_equations[node](
        # scm_obj.noises_distributions[node].sample(), # this is more elegant, but we also want to save the U's, so we do the above first
        U.loc[row_idx, getNoiseStringForNode(node)],
        *X.loc[row_idx, parents].to_numpy(),
      )
  print(f'\t[INFO] done.')
  print(f'[INFO] done.')

  if variable_type == 'integer':
    X = np.round(4 * X)

  # to create a more balanced dataset, do not set b to 0.
  w = np.random.normal(mu_w, sigma_w, (d, 1))
  # b = 0 # see below.
  b = - np.mean(np.dot(X, w))
  y = (np.sign(np.sign(np.dot(X, w) + b) + 1e-6) + 1) / 2 # add 1e-3 to prevent label 0.5
  y = pd.DataFrame(data=y, columns={'label'})

  data_frame_non_hot = pd.concat([y,X,U], axis=1)
  return data_frame_non_hot.astype('float64')
