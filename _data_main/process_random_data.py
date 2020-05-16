import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from main import loadCausalModel

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

from debug import ipsh

mu_x, sigma_x = 0, 1 # mean and standard deviation for data
mu_w, sigma_w = 0, 1 # mean and standard deviation for weights
n = 1000

def load_random_data(variable_type = 'real'):

  causal_model_obj = loadCausalModel()

  d = len(causal_model_obj.structural_equations.keys())

  print(f'\n\n[INFO] Creating dataset using scm class `???`...')

  print(f'\t[INFO] Sampling {n} exogenous U variables (d = {d})...')
  U = np.concatenate(
    [
      np.array(
        [causal_model_obj.noises_distributions[node]() for _ in range(n)]
      ).reshape(-1,1)
      for node in causal_model_obj.getTopologicalOrdering()
    ],
    axis = 1,
  )
  U = pd.DataFrame(U, columns=causal_model_obj.getTopologicalOrdering())
  print(f'\t[INFO] done.')


  # sample endogenous X variables
  print(f'\t[INFO] Sampling {n} endogenous X variables (d = {d})...') # (i.e., processDataAccordingToGraph)
  X = U.copy()
  X.loc[:] = np.nan # used later as an assertion to make sure parents are populated when computing children
  for node in causal_model_obj.getTopologicalOrdering():
    parents = causal_model_obj.getParentsForNode(node)
    # assuming we're iterating in topological order, parents in X should already be occupied
    assert not X.loc[:,list(parents)].isnull().values.any()
    for row_idx, row in tqdm(X.iterrows(), total=X.shape[0]):
      X.loc[row_idx, node] = causal_model_obj.structural_equations[node](
        # causal_model_obj.noises_distributions[node](), # this is more elegant, but we also want to save the U's, so we do the above first
        U.loc[row_idx, node],
        *X.loc[row_idx, parents].to_numpy(),
      )
  X = X.to_numpy()
  print(f'\t[INFO] done.')
  print(f'[INFO] done.')

  if variable_type == 'integer':
    X = np.round(4 * X)
  np.random.shuffle(X)

  # to create a more balanced dataset, do not set b to 0.
  w = np.random.normal(mu_w, sigma_w, (d, 1))
  # b = 0 # see below.
  b = - np.mean(np.dot(X, w))
  y = (np.sign(np.sign(np.dot(X, w) + b) + 1e-6) + 1) / 2 # add 1e-3 to prevent label 0.5

  X_train = X[ : n // 2, :]
  X_test = X[n // 2 : , :]
  y_train = y[ : n // 2, :]
  y_test = y[n // 2 : , :]

  data_frame_non_hot = pd.DataFrame(
      np.concatenate((
        np.concatenate((y_train, X_train), axis = 1), # importantly, labels have to go first, else Dataset.__init__ messes up kurz column names
        np.concatenate((y_test, X_test), axis = 1), # importantly, labels have to go first, else Dataset.__init__ messes up kurz column names
      ),
      axis = 0,
    ),
    columns=['label'] + [f'x{i}' for i in range(X.shape[1])]
  )
  return data_frame_non_hot.astype('float64')
