import numpy as np
import pandas as pd

import loadData
from sklearn.model_selection import train_test_split

from debug import ipsh

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

def loadFactualInstance():
  dataset_obj = loadData.loadDataset('random', return_one_hot = False, load_from_cache = False)
  balanced_data_frame, input_cols, output_col = loadData.getBalancedDataFrame(dataset_obj)

  # get train / test splits
  all_data = balanced_data_frame.loc[:,input_cols]
  all_true_labels = balanced_data_frame.loc[:,output_col]
  X_train, X_test, y_train, y_test = train_test_split(
    all_data,
    all_true_labels,
    train_size=.7,
    random_state = RANDOM_SEED)

  # choose a random element
  factual_instance = np.array(X_train.iloc[1])
  return factual_instance

def getStructuralEquation(variable_index, scm_type):
  # TODO: write using lambda functions with unknown / multiple inputs
  if variable_index == 'x0':
    return lambda: 0
  elif variable_index == 'x1':
    if scm_type == 'true':
      return lambda x0: x0 ** 3 + 1
    elif scm_type == 'approx':
      return lambda x0: 3 * x0 + 1
  elif variable_index == 'x2':
    if scm_type == 'true':
      return lambda x0, x1: x0 / 4 + np.sqrt(3) * np.sin(x1) - 1 / 4
    elif scm_type == 'approx':
      return lambda x0, x1: 0.425 * x0 + 0.0003 * x1 - 1 / 4

def computeCounterfactual(factual_instance, action_set, scm_type):
  # we have true knowledge of the SCM

  structural_equations = []
  structural_equations.append(getStructuralEquation('x0', scm_type))
  structural_equations.append(getStructuralEquation('x1', scm_type))
  structural_equations.append(getStructuralEquation('x2', scm_type))

  # Step 1. abduction: get value of noise variables
  noise_variables = []
  noise_variables.append(factual_instance[0] - structural_equations[0]())
  noise_variables.append(factual_instance[1] - structural_equations[1](factual_instance[0]))
  noise_variables.append(factual_instance[2] - structural_equations[2](factual_instance[0], factual_instance[1]))

  # Step 2. action: update structural equations
  for key, value in action_set.items():
    variable_index_string = key
    intervention_value = value
    variable_index = int(variable_index_string[-1]) # doens't work for > 10 variables
    structural_equations[variable_index] = lambda *args: intervention_value
    # *args is used to allow for ignoring arguments that may be passed into this
    # function (consider, for example, an intervention on x2 which then requires
    # no inputs to call the second structural equation function, but we still pass
    # in the arugments a few lines down)

  # Step 3. prediction: compute counterfactual values starting from root node
  counterfactual_instance = np.zeros(factual_instance.shape)
  counterfactual_instance[0] = noise_variables[0] + structural_equations[0]()
  counterfactual_instance[1] = noise_variables[1] + structural_equations[1](counterfactual_instance[0])
  counterfactual_instance[2] = noise_variables[2] + structural_equations[2](counterfactual_instance[0], counterfactual_instance[1])
  # TODO: if intervened, remove the noise variable..

  return counterfactual_instance



if __name__ == "__main__":
  factual_instance = loadFactualInstance()

  # iterative over a number of action sets and compare the three counterfactuals
  action_set = {'x0': +1}

  oracle_counterfactual_instance = computeCounterfactual(factual_instance, action_set, 'true')
  approx_counterfactual_instance = computeCounterfactual(factual_instance, action_set, 'approx')

  print(oracle_counterfactual_instance)
  print(approx_counterfactual_instance)

  # # compute counterfactuals accordingly
  # oracle_counterfactual_instance =
  # approx_counterfactual_instance =
  # cate_counterfactual_instance =

