import pickle
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

SAVED_MACE_RESULTS_PATH_M0 = '/Users/a6karimi/dev/recourse/_minimum_distances_m0'
SAVED_MACE_RESULTS_PATH_M1 = '/Users/a6karimi/dev/recourse/_minimum_distances_m1'

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


def incrementIndices(tmp_dict):
  new_dict = {}
  for key, value in tmp_dict.items():
    # TODO: only works for single digit x variables; use find
    new_dict['x' + str(int(key[-1]) + 1)] = value
  return new_dict


def prettyPrintActionSet(action_set):
  for key, value in action_set.items():
    action_set[key] = np.around(value, 2)
  return action_set


# TODO: should have a class of defining SCM with this as a method
def getStructuralEquation(variable_index, scm_type):

  if scm_type == 'true':

    if variable_index == 'x1':
      return lambda n1: n1
    elif variable_index == 'x2':
      return lambda x1, n2: x1 + 1 + n2
    elif variable_index == 'x3':
      return lambda x1, x2, n3: np.sqrt(3) * x1 * x2 * x2 + n3

  elif scm_type == 'approx':

    if variable_index == 'x1':
      return lambda n1: n1
    elif variable_index == 'x2':
      return lambda x1, n2: 1 * x1 + 1 + n2
    elif variable_index == 'x3':
      return lambda x1, x2, n3: 5.5 * x1 + 3.5 * x2 - 0.1 + n3


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


def didFlip(factual_instance, counterfactual_instance):
  sklearn_model = loadModel.loadModelForDataset('lr', 'random')
  factual_prediction = sklearn_model.predict(np.array(list(factual_instance.values())).reshape(1,-1))
  counterfactual_prediction = sklearn_model.predict(np.array(list(counterfactual_instance.values())).reshape(1,-1))
  return factual_prediction != counterfactual_prediction


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


def scatterDecisionBoundary(ax):
  sklearn_model = loadModel.loadModelForDataset('lr', 'random')
  fixed_model_w = sklearn_model.coef_
  fixed_model_b = sklearn_model.intercept_

  x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
  y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
  X = np.linspace(ax.get_xlim()[0] - x_range / 10, ax.get_xlim()[1] + x_range / 10, 10)
  Y = np.linspace(ax.get_ylim()[0] - y_range / 10, ax.get_ylim()[1] + y_range / 10, 10)
  X, Y = np.meshgrid(X, Y)
  Z = - (fixed_model_w[0][0] * X + fixed_model_w[0][1] * Y + fixed_model_b) / fixed_model_w[0][2]

  surf = ax.plot_wireframe(X, Y, Z, alpha=0.3)


def scatterDataset(X_train, X_test, y_train, y_test, ax):
  X_train_numpy = X_train.to_numpy()
  X_test_numpy = X_test.to_numpy()
  number_of_samples_to_plot = 200
  for idx in range(number_of_samples_to_plot):
    color_train = 'black' if y_train.to_numpy()[idx] == 1 else 'magenta'
    color_test = 'black' if y_test.to_numpy()[idx] == 1 else 'magenta'
    ax.scatter(X_train_numpy[idx, 0], X_train_numpy[idx, 1], X_train_numpy[idx, 2], marker='s', color=color_train, alpha=0.2, s=10)
    ax.scatter(X_test_numpy[idx, 0], X_test_numpy[idx, 1], X_test_numpy[idx, 2], marker='o', color=color_test, alpha=0.2, s=15)


def scatterFactual(factual_instance, ax):
  fc = factual_instance
  ax.scatter(fc['x1'], fc['x2'], fc['x3'], marker='P', color='black', s=70)


def scatterCounterfactuals(factual_instance, action_set, counterfactual_type, scm_type, ax):

  if counterfactual_type == 'm0':

    m0 = computeCounterfactual(factual_instance, action_set, scm_type)
    color_string = 'green' if didFlip(factual_instance, m0) else 'red'
    ax.scatter(m0['x1'], m0['x2'], m0['x3'], marker='o', color=color_string, s=70)

  elif counterfactual_type == 'm1':

    # TODO: do something better here... should not have to manually handle this!
    # m1 = computeCounterfactual(factual_instance, action_set, 'approx')
    m1 = computeCounterfactual(factual_instance, action_set, scm_type)
    color_string = 'green' if didFlip(factual_instance, m1) else 'red'
    ax.scatter(m1['x1'], m1['x2'], m1['x3'], marker='s', color=color_string, s=70)

  elif counterfactual_type == 'm2':
    list_m2 = []

    for idx in range(100):
      m2 = getRandomM2Sample(factual_instance, action_set)
      list_m2.append(m2)
      color_string = 'green' if didFlip(factual_instance, m2) else 'red'
      ax.scatter(m2['x1'], m2['x2'], m2['x3'], marker = '^', color=color_string, alpha=0.1, s=40)

    list_m2_x1 = [elem['x1'] for elem in list_m2]
    list_m2_x2 = [elem['x2'] for elem in list_m2]
    list_m2_x3 = [elem['x3'] for elem in list_m2]

    # TODO: choose 1 shape for each of m0, m1, m2 and choose color based on did_flip
    # TODO: choose whether to show mean / median or not.. (conflicts with above)
    # ax.scatter(np.mean(list_m2_x1), np.mean(list_m2_x2), np.mean(list_m2_x3), marker = 'o', color='blue', alpha=0.5, s=70)
    # ax.scatter(np.median(list_m2_x1), np.median(list_m2_x2), np.median(list_m2_x3), marker = 's', color='blue', alpha=0.5, s=70)

  else:

    raise Exception(f'{counterfactual_type} not recognized.')



def experiment1(X_train, X_test, y_train, y_test):
  ''' compare M0, M1, M2 on one factual samples and one **fixed** action sets '''
  factual_instance = X_test.iloc[0].T.to_dict()

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
  ''' compare M0, M1, M2 on <n> factual samples and <n> **fixed** action sets '''

  factual_instances_dict = X_test.iloc[:3].T.to_dict()
  action_sets = [ \
    {'x1': 1}, \
    {'x2': 1}, \
    {'x3': 1}, \
  ]

  fig = pyplot.figure()

  for index, (key, value) in enumerate(factual_instances_dict.items()):
    idx_sample = index
    factual_instance_idx = key
    factual_instance = value
    for idx_action, action_set in enumerate(action_sets):
      ax = pyplot.subplot(
        len(factual_instances_dict),
        len(action_sets),
        idx_sample * len(action_sets) + idx_action + 1,
        projection = '3d')
      scatterFactual(factual_instance, ax)
      scatterCounterfactuals(factual_instance, action_set, 'm0', 'true', ax)
      scatterCounterfactuals(factual_instance, action_set, 'm1', 'approx', ax)
      scatterCounterfactuals(factual_instance, action_set, 'm2', 'irrelevant', ax)
      scatterDecisionBoundary(ax)
      ax.set_xlabel('x1')
      ax.set_ylabel('x2')
      ax.set_zlabel('x3')
      ax.set_title(f'sample_{factual_instance_idx} \n do({prettyPrintActionSet(action_set)})', fontsize=8)
      ax.view_init(elev=15, azim=10)


      # for angle in range(0, 360):
      #   ax.view_init(30, angle)
      #   pyplot.draw()
      #   pyplot.pause(.001)

  pyplot.suptitle("Compare M0, M1, M2 on <n> factual samples and <n> **fixed** action sets.", fontsize=14)
  pyplot.show()


def experiment3(X_train, X_test, y_train, y_test):
  ''' compare M0, M1, M2 on <n> factual samples and <n> **computed** action sets '''

  mace_results_m0 = pickle.load(open(SAVED_MACE_RESULTS_PATH_M0, 'rb'))
  mace_results_m1 = pickle.load(open(SAVED_MACE_RESULTS_PATH_M1, 'rb'))

  # for

  # for factual_instance_idx, mace_result in mace_results:

  # factual_instances_df = X_test.iloc[:3].copy()
  # factual_instances_df['m0_action_set'] = 0 # default values
  # factual_instances_df['m1_action_set'] = 0 # default values
  # factual_instances_df['m2_action_set'] = 0 # default values

  # for factual_instance_idx in factual_instances_df.index:
  #   factual_instance_idx_string = f'sample_{factual_instance_idx}'
  #   tmp = mace_results[factual_instance_idx_string]
  #   factual_instances_df.loc[factual_instance_idx]['m0_action_set'] = incrementIndices(mace_results[factual_instance_idx_string]['action_set'])
  #   factual_instances_df.loc[factual_instance_idx]['m1_action_set'] = incrementIndices(mace_results[factual_instance_idx_string]['action_set'])
  #   factual_instances_df.loc[factual_instance_idx]['m2_action_set'] = incrementIndices(mace_results[factual_instance_idx_string]['action_set'])

  # ipsh()

  NUM_SAMPLES = 16
  NUM_PLOT_ROWS = 4

  factual_instances_df = X_test.iloc[:NUM_SAMPLES].copy()
  factual_instances_dict = factual_instances_df.T.to_dict()

  fig = pyplot.figure()

  for index, (key, value) in enumerate(factual_instances_dict.items()):
    idx_sample = index
    factual_instance_idx = key
    factual_instance_idx_mace = f'sample_{factual_instance_idx}'
    factual_instance = value
    assert factual_instance_idx_mace in mace_results_m0.keys(), f'missing results for `{factual_instance_idx_mace}` in mace_results.'
    assert factual_instance_idx_mace in mace_results_m0.keys(), f'missing results for `{factual_instance_idx_mace}` in mace_results.'
    ax = pyplot.subplot(
      len(factual_instances_dict) // NUM_PLOT_ROWS + int(not(len(factual_instances_dict) % NUM_PLOT_ROWS == 0)),
      NUM_PLOT_ROWS,
      idx_sample + 1,
      projection = '3d')
    # scatterDataset(X_train, X_test, y_train, y_test, ax)
    scatterFactual(factual_instance, ax)
    m0_action_set = incrementIndices(mace_results_m0[factual_instance_idx_mace]['action_set'])
    m1_action_set = incrementIndices(mace_results_m1[factual_instance_idx_mace]['action_set'])
    scatterCounterfactuals(factual_instance, m0_action_set, 'm0', 'true', ax)
    scatterCounterfactuals(factual_instance, m1_action_set, 'm1', 'true', ax)
    scatterDecisionBoundary(ax)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.set_title(
      f'sample_{factual_instance_idx}'
      f'\n m0 action set: do({prettyPrintActionSet(m0_action_set)})'
      f'\n m1 action set: do({prettyPrintActionSet(m1_action_set)})'
    , fontsize=8)
    ax.view_init(elev=20, azim=-30)

    # for angle in range(0, 360):
    #   ax.view_init(30, angle)
    #   pyplot.draw()
    #   pyplot.pause(.001)

  # TODO: fix
  # handles, labels = ax.get_legend_handles_labels()
  # fig.legend(handles, labels, loc='upper center')
  # ipsh()

  pyplot.suptitle("Compare M0, M1, M2 on <n> factual samples and <n> **computed** action sets.", fontsize=14)
  pyplot.show()


def visualizeDatasetAndFixedModel(X_train, X_test, y_train, y_test):

  fig = pyplot.figure()
  ax = pyplot.subplot(1, 1, 1, projection='3d')

  scatterDataset(X_train, X_test, y_train, y_test, ax)
  scatterDecisionBoundary(ax)

  ax.set_xlabel('x1')
  ax.set_ylabel('x2')
  ax.set_zlabel('x3')
  ax.set_title(f'datatset')
  # ax.legend()
  ax.grid(True)

  pyplot.show()



if __name__ == "__main__":

  # only load once so shuffling order is the same
  X_train, X_test, y_train, y_test = loadDataset()

  ''' compare M0, M1, M2 on one factual samples and one **fixed** action sets '''
  # experiment1(X_train, X_test, y_train, y_test)
  ''' compare M0, M1, M2 on <n> factual samples and <n> **fixed** action sets '''
  # experiment2(X_train, X_test, y_train, y_test)
  ''' compare M0, M1, M2 on <n> factual samples and <n> **computed** action sets '''
  experiment3(X_train, X_test, y_train, y_test)

  # sanity check
  # visualizeDatasetAndFixedModel(X_train, X_test, y_train, y_test)















