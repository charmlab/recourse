import numpy as np
import pandas as pd
from matplotlib import pyplot

import loadData
import loadModel
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity

from scipy import stats

from debug import ipsh

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

def loadDataset():
  dataset_obj = loadData.loadDataset('random', return_one_hot = False, load_from_cache = False)
  dataset_obj.data_frame_long = dataset_obj.data_frame_long.rename(columns={'x0': 'x1', 'x1': 'x2', 'x2': 'x3'})
  dataset_obj.data_frame_kurz = dataset_obj.data_frame_kurz.rename(columns={'x0': 'x1', 'x1': 'x2', 'x2': 'x3'})
  balanced_data_frame, input_cols, output_col = loadData.getBalancedDataFrame(dataset_obj)

  # get train / test splits
  all_data = balanced_data_frame.loc[:,input_cols]
  all_true_labels = balanced_data_frame.loc[:,output_col]
  X_train, X_test, y_train, y_test = train_test_split(
    all_data,
    all_true_labels,
    train_size=.7,
    random_state = RANDOM_SEED)

  return X_train, X_test, y_train, y_test


def loadRandomFactualInstance(X_train):
  # choose a random element
  # return X_train.iloc[0].round(decimals=4).to_dict()
  return X_train.iloc[np.random.randint(0,10)].to_dict()


# TODO: should have a class of defining SCM with this as a method
def getStructuralEquation(variable_index, scm_type):

  if scm_type == 'true':

    if variable_index == 'x1':
      return lambda n1: n1
    elif variable_index == 'x2':
      return lambda x1, n2: (x1 + 1) ** 3 + 1 + n2
    elif variable_index == 'x3':
      return lambda x1, x2, n3: x1 * 2 + np.sqrt(3) * np.sin(x2) - 1 / 4 + n3

  elif scm_type == 'approx':

    if variable_index == 'x1':
      return lambda n1: n1
    elif variable_index == 'x2':
      return lambda x1, n2: 7.43 * x1 + 4.88 + n2
    elif variable_index == 'x3':
      return lambda x1, x2, n3: 1.6 * x1 + 0.015 * x2 - 0.01 + n3


# TODO: write recursively?
def getAncestors(node):
  if node == 'x1':
    return {}
  elif node == 'x2':
    return {'x1'}
  elif node == 'x3':
    return {'x1', 'x2'}


def getParents(node):
  if node == 'x1':
    return {}
  elif node == 'x2':
    return {'x1'}
  elif node == 'x3':
    return {'x1', 'x2'}


def getRandomM2Sample(factual_instance, action_set):
  # intervening_set = {'x2'}, conditioning_set = {'x1'}):
  # TODO: we're using the SCM to get the samples for now; later we need to use conditional KDE

  structural_equations = {}
  structural_equations['x1'] = getStructuralEquation('x1', 'true')
  structural_equations['x2'] = getStructuralEquation('x2', 'true')
  structural_equations['x3'] = getStructuralEquation('x3', 'true')

  intervening_set = set(action_set.keys())
  conditioning_set = set()

  for node in action_set.keys():
    for ancestor in getAncestors(node):
      conditioning_set.add(ancestor)

  # intervening takes precedence over conditioning; order them accordingly
  for node in conditioning_set:
    intervention_value = factual_instance[node]
    structural_equations[node] = lambdaWrapper(intervention_value)

  for node in intervening_set:
    intervention_value = action_set[node]
    structural_equations[node] = lambdaWrapper(intervention_value)

  # ipsh()
  # generate random noise and pass in through structural_equations from the top!
  noise_variables = {}
  noise_variables['x1'] = np.random.normal(0,1)
  noise_variables['x2'] = np.random.normal(0,1)
  noise_variables['x3'] = np.random.normal(0,1)

  counterfactual_instance = {}
  counterfactual_instance['x1'] = structural_equations['x1'](noise_variables['x1'])
  counterfactual_instance['x2'] = structural_equations['x2'](counterfactual_instance['x1'], noise_variables['x2'])
  counterfactual_instance['x3'] = structural_equations['x3'](counterfactual_instance['x1'], counterfactual_instance['x2'], noise_variables['x3'])

  return counterfactual_instance


# See https://stackoverflow.com/a/25670697
def lambdaWrapper(intervention_value):
  return lambda *args: intervention_value


def computeCounterfactual(factual_instance, action_set, scm_type):

  structural_equations = {}
  structural_equations['x1'] = getStructuralEquation('x1', scm_type)
  structural_equations['x2'] = getStructuralEquation('x2', scm_type)
  structural_equations['x3'] = getStructuralEquation('x3', scm_type)

  # Step 1. abduction: get value of noise variables
  # tip: pass in n* = 0 to structural_equations (lambda functions)
  noise_variables = {}
  noise_variables['x1'] = factual_instance['x1'] - structural_equations['x1'](0)
  noise_variables['x2'] = factual_instance['x2'] - structural_equations['x2'](factual_instance['x1'], 0)
  noise_variables['x3'] = factual_instance['x3'] - structural_equations['x3'](factual_instance['x1'], factual_instance['x2'], 0)

  # Step 2. action: update structural equations
  for key, value in action_set.items():
    node = key
    intervention_value = value
    structural_equations[node] = lambdaWrapper(intervention_value)
    # *args is used to allow for ignoring arguments that may be passed into this
    # function (consider, for example, an intervention on x2 which then requires
    # no inputs to call the second structural equation function, but we still pass
    # in the arugments a few lines down)

  # Step 3. prediction: compute counterfactual values starting from root node
  counterfactual_instance = {}
  counterfactual_instance['x1'] = structural_equations['x1'](noise_variables['x1'])
  counterfactual_instance['x2'] = structural_equations['x2'](counterfactual_instance['x1'], noise_variables['x2'])
  counterfactual_instance['x3'] = structural_equations['x3'](counterfactual_instance['x1'], counterfactual_instance['x2'], noise_variables['x3'])

  return counterfactual_instance


def scatter3counterfactuals(factual_instance, action_set, ax):

  fc = factual_instance
  m0 = computeCounterfactual(factual_instance, action_set, 'true')
  m1 = computeCounterfactual(factual_instance, action_set, 'approx')

  ax.scatter(fc['x1'], fc['x2'], fc['x3'], marker='o', color='gray', s=100)
  ax.scatter(m0['x1'], m0['x2'], m0['x3'], marker='o', color='green', s=100)
  ax.scatter(m1['x1'], m1['x2'], m1['x3'], marker='s', color='red', s=100)
  list_m2 = []
  for idx in range(100):
    m2 = getRandomM2Sample(factual_instance, action_set)
    list_m2.append(m2)
    ax.scatter(m2['x1'], m2['x2'], m2['x3'], marker = '^', color='blue', alpha=0.1)
  list_m2_x1 = [elem['x1'] for elem in list_m2]
  list_m2_x2 = [elem['x2'] for elem in list_m2]
  list_m2_x3 = [elem['x3'] for elem in list_m2]

  ax.scatter(np.mean(list_m2_x1), np.mean(list_m2_x2), np.mean(list_m2_x3), marker = 'o', color='blue', alpha=0.5, s=100)
  ax.scatter(np.median(list_m2_x1), np.median(list_m2_x2), np.median(list_m2_x3), marker = 's', color='blue', alpha=0.5, s=100)

  ax.set_xlabel('x1')
  ax.set_ylabel('x2')
  ax.set_zlabel('x3')
  ax.set_title(f'do({action_set})')
  ax.view_init(elev=15, azim=10)

  # for angle in range(0, 360):
  #   ax.view_init(30, angle)
  #   pyplot.draw()
  #   pyplot.pause(.001)


def scatterDecisionBoundary(ax):
  prev_xlim = ax.get_xlim()
  prev_ylim = ax.get_ylim()
  prev_zlim = ax.get_zlim()

  sklearn_model = loadModel.loadModelForDataset('lr', 'random')
  ipsh()
  fixed_model_w = sklearn_model.coef_
  fixed_model_b = sklearn_model.intercept_

  X = np.linspace(prev_xlim[0], prev_xlim[1], 10)
  Y = np.linspace(prev_ylim[0], prev_ylim[1], 10)
  X, Y = np.meshgrid(X, Y)
  Z = - (fixed_model_w[0][0] * X + fixed_model_w[0][1] * Y + fixed_model_b) / fixed_model_w[0][2]

  surf = ax.plot_wireframe(X, Y, Z, alpha=0.5)


def experiment1(X_train, X_test, y_train, y_test):
  ''' compare M0, M1, M2 on one factual sample and one action set '''
  factual_instance = loadRandomFactualInstance(X_train)

  # iterative over a number of action sets and compare the three counterfactuals
  action_set = {'x1': -3}
  # action_set = {'x2': +1}
  # action_set = {'x3': +1}
  # action_set = {'x1': +2, 'x2': +1}

  oracle_counterfactual_instance = computeCounterfactual(factual_instance, action_set, 'true')
  approx_counterfactual_instance = computeCounterfactual(factual_instance, action_set, 'approx')
  cate_counterfactual_instance = getRandomM2Sample(factual_instance, action_set)

  print(f'Factual instance: \t\t{factual_instance}')
  print(f'M0 structural counterfactual: \t{oracle_counterfactual_instance}')
  print(f'M1 structural counterfactual: \t{approx_counterfactual_instance}')
  print(f'M2 cate counterfactual: \t{cate_counterfactual_instance}')


def experiment2(X_train, X_test, y_train, y_test):
  ''' compare M0, M1, M2 on three factual sample and one action set '''
  fig = pyplot.figure()

  factual_instances = [loadRandomFactualInstance(X_train) for i in range(1)]
  action_sets = [ \
    {'x1': 1}, \
    {'x2': 1}, \
    {'x3': 1}, \
  ]
  for idx_sample, factual_instance in enumerate(factual_instances):
    for idx_action, action_set in enumerate(action_sets):
      ax = pyplot.subplot(
        len(factual_instances),
        len(action_sets),
        idx_sample * len(action_sets) + idx_action + 1,
        projection = '3d')
      scatter3counterfactuals(factual_instance, action_set, ax)
      scatterDecisionBoundary(ax)

  pyplot.show()


def experiment3(X_train, X_test, y_train, y_test):
  ipsh()
  for idx in range(3):
    factual_instance = X_test.iloc[idx].to_dict()
     # TODO: compute


def visualizeDatasetAndFixedModel(X_train, X_test, y_train, y_test):
  X_train_numpy = X_train.to_numpy()
  X_test_numpy = X_test.to_numpy()
  number_of_samples_to_plot = 100

  fig = pyplot.figure()
  ax = pyplot.subplot(1,1,1, projection='3d')

  for idx in range(number_of_samples_to_plot):
    color_train = 'blue' if y_train.to_numpy()[idx] == 1 else 'green'
    color_test = 'blue' if y_test.to_numpy()[idx] == 1 else 'green'
    ax.scatter(X_train_numpy[idx, 0], X_train_numpy[idx, 1], X_train_numpy[idx, 2], marker='s', color=color_train, alpha=0.2, s=10)
    ax.scatter(X_test_numpy[idx, 0], X_test_numpy[idx, 1], X_test_numpy[idx, 2], marker='o', color=color_test, alpha=0.2, s=15)

  sklearn_model = loadModel.loadModelForDataset('lr', 'random')
  fixed_model_w = sklearn_model.coef_
  fixed_model_b = sklearn_model.intercept_

  X = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 10)
  Y = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 10)
  X, Y = np.meshgrid(X, Y)
  Z = - (fixed_model_w[0][0] * X + fixed_model_w[0][1] * Y + fixed_model_b) / fixed_model_w[0][2]

  surf = ax.plot_wireframe(X, Y, Z, alpha=0.5)

  ax.set_xlabel('x1')
  # ax.set_xlim(())
  ax.set_ylabel('x2')
  # ax.set_ylim(())
  ax.set_zlabel('x3')
  ax.set_zlim((-5,5))
  ax.set_title(f'datatset')
  # ax.legend()
  ax.grid(True)

  pyplot.show()




if __name__ == "__main__":

  # only load once so shuffling order is the same
  X_train, X_test, y_train, y_test = loadDataset()

  experiment1(X_train, X_test, y_train, y_test)
  experiment2(X_train, X_test, y_train, y_test)
  # visualizeDatasetAndFixedModel(X_train, X_test, y_train, y_test)















