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
mu_w, sigma_w = 0, 0.5 # mean and standard deviation for weights
n = 2500

# from main import getNoiseStringForNode $ TODO: should be from ../main??
def getNoiseStringForNode(node):
  assert node[0] == 'x'
  return 'u' + node[1:]

def load_random_data(scm_class, variable_type = 'real'):

  scm_obj = loadSCM.loadSCM(scm_class)

  d = len(scm_obj.structural_equations_np.keys())

  print(f'\n\n[INFO] Creating dataset using scm class `{scm_class}`...')

  # sample exogenous U variables
  print(f'\t[INFO] Sampling {n} exogenous U variables (d = {d})...')
  U = np.concatenate(
    [
      np.array(scm_obj.noises_distributions[node].sample(n)).reshape(-1,1)
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
    X[node] = scm_obj.structural_equations_np[node](
      U[getNoiseStringForNode(node)],
      *[X[parent] for parent in parents]
    )
  print(f'\t[INFO] done.')
  print(f'[INFO] done.')

  if variable_type == 'integer':
    X = np.round(4 * X)

  # sample a random hyperplane through the origin
  w = np.random.rand(d, 1)

  # get the average scale of (w^T)*X (this depends on the scale of the data)
  scale = 2/np.mean(np.absolute(np.dot(X, w)))

  predictions = 1/(1+np.exp(-scale * np.dot(X, w)))
  # check that labels are not all 0 or 1
  assert np.std(predictions) < 0.4

  # sample labels from class probabilities in predictions
  uniform_rv = np.random.rand(X.shape[0], 1)
  y = uniform_rv < predictions  # add 1e-3 to prevent label 0.5

  y = pd.DataFrame(data=y, columns={'label'})


  data_frame_non_hot = pd.concat([y,X,U], axis=1)
  return data_frame_non_hot.astype('float64')
