import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot

import utils
import loadData
import loadModel

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared

import GPy

from debug import ipsh

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

NUM_TEST_SAMPLES = 4
NORM_TYPE = 2
GRID_SEARCH_BOUND = 2
GRID_SEARCH_BINS = 5
NUMBER_OF_MONTE_CARLO_SAMPLES = 100
LAMBDA_LCB = 1
SAVED_MACE_RESULTS_PATH_M0 = '/Users/a6karimi/dev/recourse/_minimum_distances_m0'
SAVED_MACE_RESULTS_PATH_M1 = '/Users/a6karimi/dev/recourse/_minimum_distances_m1'


@utils.Memoize
def loadDataset():
  dataset_obj = loadData.loadDataset('random', return_one_hot = True, load_from_cache = False)
  dataset_obj.data_frame_long = dataset_obj.data_frame_long.rename(columns={'x0': 'x1', 'x1': 'x2', 'x2': 'x3'})
  dataset_obj.data_frame_kurz = dataset_obj.data_frame_kurz.rename(columns={'x0': 'x1', 'x1': 'x2', 'x2': 'x3'})
  X_train, X_test, y_train, y_test = loadData.getTrainTestData(dataset_obj, RANDOM_SEED, standardize_data = False)
  return X_train, X_test, y_train, y_test


def measureActionSetCost(factual_instance, action_set, norm_type):
  deltas = []
  for key in action_set.keys():
    deltas.append(action_set[key] - factual_instance[key])
  return np.linalg.norm(deltas, norm_type)


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


def prettyPrintInstance(instance):
  for key, value in instance.items():
    instance[key] = np.around(value, 2)
  return instance


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


def getPredictionBatch(instances_df):
  sklearn_model = loadModel.loadModelForDataset('lr', 'random')
  return sklearn_model.predict(instances_df)


def getPrediction(instance):
  sklearn_model = loadModel.loadModelForDataset('lr', 'random')
  prediction = sklearn_model.predict(np.array(list(instance.values())).reshape(1,-1))[0]
  assert prediction in {0, 1}, f'Expected prediction in {0,1}; got {prediction}'
  return prediction


def didFlip(factual_instance, counterfactual_instance):
  return getPrediction(factual_instance) != getPrediction(counterfactual_instance)


# See https://stackoverflow.com/a/25670697
def lambdaWrapper(intervention_value):
  return lambda *args: intervention_value


# TODO: should have a class of defining SCM with this as a method
@utils.Memoize
def getStructuralEquation(variable_index, scm_type):

  if scm_type == 'true':

    if variable_index == 'x1':
      return lambda n1: n1
    elif variable_index == 'x2':
      return lambda x1, n2: x1 + 1 + n2
    elif variable_index == 'x3':
      return lambda x1, x2, n3: np.sqrt(3) * x1 * x2 * x2 + n3

  # elif scm_type == 'approx_deprecated':

  #   if variable_index == 'x1':
  #     return lambda n1: n1
  #   elif variable_index == 'x2':
  #     return lambda x1, n2: 1 * x1 + 1 + n2
  #   elif variable_index == 'x3':
  #     return lambda x1, x2, n3: 5.5 * x1 + 3.5 * x2 - 0.1 + n3

  elif scm_type == 'approx_lin':

    X_train, X_test, y_train, y_test = loadDataset()
    X_all = X_train.append(X_test)
    param_grid = {"alpha": np.linspace(0,10,11)}

    if variable_index == 'x1':

      return lambda n1: n1

    elif variable_index == 'x2':

      model = GridSearchCV(Ridge(), param_grid=param_grid)
      model.fit(X_all[['x1']], X_all[['x2']])
      return lambda x1, n2: model.predict([[x1]])[0][0] + n2

    elif variable_index == 'x3':

      model = GridSearchCV(Ridge(), param_grid=param_grid)
      model.fit(X_all[['x1', 'x2']], X_all[['x3']])
      return lambda x1, x2, n3: model.predict([[x1, x2]])[0][0] + n3

  elif scm_type == 'approx_krr':

    X_train, X_test, y_train, y_test = loadDataset()
    X_all = X_train.append(X_test)
    param_grid = {"alpha": [1e0, 1e-1, 1e-2, 1e-3],
                  "kernel": [ExpSineSquared(l, p)
                             for l in np.logspace(-2, 2, 5)
                             for p in np.logspace(0, 2, 5)]}

    if variable_index == 'x1':
      print(f'[INFO] Fitting KRR (parent: n/a; child: x1) may be very expensive, memoizing aftewards.')

      return lambda n1: n1

    elif variable_index == 'x2':
      print(f'[INFO] Fitting KRR (parent: x1; child: x2) may be very expensive, memoizing aftewards.')

      model = GridSearchCV(KernelRidge(), param_grid=param_grid)
      model.fit(X_all[['x1']], X_all[['x2']])
      return lambda x1, n2: model.predict([[x1]])[0][0] + n2

    elif variable_index == 'x3':
      print(f'[INFO] Fitting KRR (parent: x1, x2; child: x3) may be very expensive, memoizing aftewards.')

      model = GridSearchCV(KernelRidge(), param_grid=param_grid)
      model.fit(X_all[['x1', 'x2']], X_all[['x3']])
      return lambda x1, x2, n3: model.predict([[x1, x2]])[0][0] + n3


# TODO: should not memoize this function, should memoize the GP fit, but not the sampling
@utils.Memoize
def getGPSample(variable_index):
  X_train, X_test, y_train, y_test = loadDataset()
  X_all = X_train.append(X_test)
  X_all = X_all[:10]

  if variable_index == 'x1':

    return 'TODO: update'

  elif variable_index == 'x2':

    print(f'[INFO] Fitting GP (parent: x1; child: x2) may be very expensive, memoizing aftewards.')
    X = X_all[['x1']].to_numpy()
    Y = X_all[['x2']].to_numpy()
    kernel = GPy.kern.RBF(input_dim=X.shape[1], variance=1., lengthscale=1.)
    model = GPy.models.GPRegression(X, Y, kernel)
    model.optimize_restarts(parallel=True, num_restarts = 5)

  elif variable_index == 'x3':

    print(f'[INFO] Fitting GP (parent: x1, x2; child: x3) may be very expensive, memoizing aftewards.')
    X = X_all[['x1', 'x2']].to_numpy()
    Y = X_all[['x3']].to_numpy()
    kernel = GPy.kern.RBF(input_dim=X.shape[1], variance=1., lengthscale=1.)
    model = GPy.models.GPRegression(X, Y, kernel)
    model.optimize_restarts(parallel=True, num_restarts = 5)

  K = kernel.K(X)
  sigma_noise = np.array(model.Gaussian_noise.variance)

  N = K.shape[0]
  def noise_posterior_mean(K, sigma, Y):
    S = np.linalg.inv(K + sigma * np.eye(N))
    return sigma * np.dot(S, Y)

  def noise_posterior_covariance(K, sigma):
    S = np.linalg.inv(K + sigma * np.eye(N))
    return  sigma * (np.eye(N) - sigma * S)

  mu_post = noise_posterior_mean(K, sigma_noise, Y)
  cov_post = noise_posterior_covariance(K, sigma_noise)
  var_post = np.array([cov_post[i,i] for i in range(N)])
  conf_post = 1.96 * np.sqrt(var_post)

  mu_prior = np.zeros_like(mu_post)
  var_prior = sigma_noise * np.ones_like(var_post)
  conf_prior = 1.96 * np.sqrt(var_prior)


def computeCounterfactualInstance(factual_instance, action_set, scm_type):

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


# TODO: naming isn't 100% accurate, in case of true SCM and HI-VAE, this function
# returns interventional samples, but counterfactual samples for GP regression
# Also fix naming, because we are not returning a `counterfactual_instance` per se.
def getRandomInterventionalSample(factual_instance, action_set, scm_type):
  # intervening_set = {'x2'}, conditioning_set = {'x1'}):

  if scm_type == 'approx_gp':
    pass
    # intervening_set = set(action_set.keys())
    # conditioning_set = set()

    # for node in action_set.keys():
    #   for ancestor in getAncestors(node):
    #     conditioning_set.add(ancestor)

    # counterfactual_instance = {}

    # for node in conditioning_set:
    #   counterfactual_instance[node] = factual_instance[node]

    # for node in intervening_set:
    #   counterfactual_instance[node] = action_set[node]

    # ipsh()
    # # counterfactual_instance[x]  = for k,v in action_set
    # ipsh()

    # getGPSample()

  elif scm_type == 'cate_true':

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

    # generate random noise and pass in through structural_equations from the top!
    noise_variables = {}
    noise_variables['x1'] = np.random.normal(0,1)
    noise_variables['x2'] = np.random.normal(0,1)
    noise_variables['x3'] = np.random.normal(0,1)

    counterfactual_instance = {}
    counterfactual_instance['x1'] = structural_equations['x1'](noise_variables['x1'])
    counterfactual_instance['x2'] = structural_equations['x2'](counterfactual_instance['x1'], noise_variables['x2'])
    counterfactual_instance['x3'] = structural_equations['x3'](counterfactual_instance['x1'], counterfactual_instance['x2'], noise_variables['x3'])

  elif scm_type == 'cate_hivae':
    pass

  return counterfactual_instance


def isPointConstraintSatisfied(factual_instance, action_set, scm_type):
  return didFlip(factual_instance, computeCounterfactualInstance(factual_instance, action_set, scm_type))


def isDistrConstraintSatisfied(factual_instance, action_set, scm_type):
  monte_carlo_samples = []
  for i in range(NUMBER_OF_MONTE_CARLO_SAMPLES):
    monte_carlo_sample = getRandomInterventionalSample(factual_instance, action_set, scm_type)
    monte_carlo_samples.append(monte_carlo_sample)

  monte_carlo_predictions = getPredictionBatch(pd.DataFrame(monte_carlo_samples).to_numpy())

  # IMPORTANT... WE ARE CONSIDERING {0,1} LABELS AND FACTUAL SAMPLES MAY BE OF
  # EITHER CLASS. THEREFORE, THE CONSTRAINT IS SATISFIED WHEN SIGNIFICANTLY
  # > 0.5 OR < 0.5 FOR A FACTUAL SAMPLE WITH Y = 0 OR Y = 1, RESPECTIVELY.

  expectation = np.mean(monte_carlo_predictions)
  variance = np.sum(np.power(monte_carlo_predictions - expectation, 2)) / (len(monte_carlo_predictions) - 1)

  if getPrediction(factual_instance) == 0:
    return expectation - LAMBDA_LCB * np.sqrt(variance) > 0.5 # NOTE DIFFERNCE IN SIGN OF STD
  else: # factual_prediction == 1
    return expectation + LAMBDA_LCB * np.sqrt(variance) < 0.5 # NOTE DIFFERNCE IN SIGN OF STD


def getValidDiscretizedActionSets():
  x1_possible_actions = list(np.around(np.linspace(-GRID_SEARCH_BOUND, GRID_SEARCH_BOUND, GRID_SEARCH_BINS+1), 2))
  x2_possible_actions = list(np.around(np.linspace(-GRID_SEARCH_BOUND, GRID_SEARCH_BOUND, GRID_SEARCH_BINS+1), 2))
  x3_possible_actions = list(np.around(np.linspace(-GRID_SEARCH_BOUND, GRID_SEARCH_BOUND, GRID_SEARCH_BINS+1), 2))
  x1_possible_actions.append('n/a')
  x2_possible_actions.append('n/a')
  x3_possible_actions.append('n/a')

  all_action_sets = []
  for x1_action in x1_possible_actions:
    for x2_action in x2_possible_actions:
      for x3_action in x3_possible_actions:
        all_action_sets.append({
          'x1': x1_action,
          'x2': x2_action,
          'x3': x3_action,
        })

  # go through, and for any action_set that has a value = 'n/a', remove ONLY
  # THAT key, value pair, NOT THE ENTIRE action_set.
  valid_action_sets = []
  for action_set in all_action_sets:
    valid_action_sets.append({k:v for k,v in action_set.items() if v != 'n/a'})

  # # FOR TEST PURPOSES ONLY
  # valid_action_sets = [ \
  #   {'x1': 1}, \
  #   {'x2': 1}, \
  #   {'x3': 1}, \
  # ]
  # valid_action_sets = list(np.random.choice(valid_action_sets, 10))

  return valid_action_sets


def computeOptimalActionSet(factual_instance, recourse_type, scm_type):

  # TODO: add option to select computation: brute-force, using MACE/MINT, or SGD

  assert recourse_type in {'point', 'distr'}, f'{recourse_type} not recognized.'
  assert scm_type in {'true', 'approx_lin', 'approx_krr', 'approx_gp', 'cate_true', 'cate_hivae'}, f'{scm_type} not recognized.'

  if recourse_type == 'point':
    assert scm_type in {'true', 'approx_lin', 'approx_krr'}, f'SCM type `{scm_type}` not recognized for recourse_type `{recourse_type}`.'
    constraint_handle = isPointConstraintSatisfied
  elif recourse_type == 'distr':
    assert scm_type in {'approx_gp', 'cate_true', 'cate_hivae'}, f'SCM type `{scm_type}` not recognized for recourse_type `{recourse_type}`.'
    constraint_handle = isDistrConstraintSatisfied

  valid_action_sets = getValidDiscretizedActionSets()
  print(f'\t[INFO] Computing optimal {recourse_type}-{scm_type}: grid searching over {len(valid_action_sets)} action sets...')

  min_cost = 1e10
  min_cost_action_set = {}
  for action_set in tqdm(valid_action_sets):
    if constraint_handle(factual_instance, action_set, scm_type):
      cost_of_action_set = measureActionSetCost(factual_instance, action_set, NORM_TYPE)
      if cost_of_action_set < min_cost:
        min_cost = cost_of_action_set
        min_cost_action_set = action_set

  print(f'\t done.')

  return min_cost_action_set


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
  y_train = y_train.to_numpy()
  y_test = y_test.to_numpy()
  number_of_samples_to_plot = 200
  for idx in range(number_of_samples_to_plot):
    color_train = 'black' if y_train[idx] == 1 else 'magenta'
    color_test = 'black' if y_test[idx] == 1 else 'magenta'
    ax.scatter(X_train_numpy[idx, 0], X_train_numpy[idx, 1], X_train_numpy[idx, 2], marker='s', color=color_train, alpha=0.2, s=10)
    ax.scatter(X_test_numpy[idx, 0], X_test_numpy[idx, 1], X_test_numpy[idx, 2], marker='o', color=color_test, alpha=0.2, s=15)


def scatterFactual(factual_instance, ax):
  ax.scatter(
    factual_instance['x1'],
    factual_instance['x2'],
    factual_instance['x3'],
    marker='P',
    color='black',
    s=70
  )


def scatterRecourse(factual_instance, action_set, recourse_type, scm_type, marker_type, ax):

  if recourse_type == 'point':

    point = computeCounterfactualInstance(factual_instance, action_set, scm_type)
    color_string = 'green' if didFlip(factual_instance, point) else 'red'
    ax.scatter(point['x1'], point['x2'], point['x3'], marker = marker_type, color=color_string, s=70)

  elif recourse_type == 'distr':
    distr_samples = []

    for idx in range(100):
      # TODO: accept # of samples of distr
      sample = getRandomInterventionalSample(factual_instance, action_set, scm_type)
      distr_samples.append(sample)
      color_string = 'green' if didFlip(factual_instance, sample) else 'red'
      ax.scatter(sample['x1'], sample['x2'], sample['x3'], marker = marker_type, color=color_string, alpha=0.1, s=30)

    mean_distr_samples = {
      'x1': np.mean([elem['x1'] for elem in distr_samples]),
      'x2': np.mean([elem['x2'] for elem in distr_samples]),
      'x3': np.mean([elem['x3'] for elem in distr_samples]),
    }
    color_string = 'green' if didFlip(factual_instance, mean_distr_samples) else 'red'
    ax.scatter(mean_distr_samples['x1'], mean_distr_samples['x2'], mean_distr_samples['x3'], marker = marker_type, color=color_string, alpha=0.5, s=70)

  else:

    raise Exception(f'{recourse_type} not recognized.')


def experiment1(X_train, X_test, y_train, y_test):
  ''' compare M0, M1, M2 on one factual samples and one **fixed** action sets '''
  factual_instance = X_test.iloc[0].T.to_dict()

  # iterative over a number of action sets and compare the three counterfactuals
  # action_set = {'x1': -3}
  action_set = {'x2': +1}
  # action_set = {'x3': +1}
  # action_set = {'x1': +2, 'x2': +1}

  print(f'FC: \t{factual_instance}')
  print(f'M0: \t{computeCounterfactualInstance(factual_instance, action_set, "true")}')
  print(f'M11: \t{computeCounterfactualInstance(factual_instance, action_set, "approx_lin")}')
  # print(f'M12: \t{computeCounterfactualInstance(factual_instance, action_set, "approx_krr")}')
  print(f'M13: \t{getRandomInterventionalSample(factual_instance, action_set, "approx_gp")}')
  print(f'M21: \t{getRandomInterventionalSample(factual_instance, action_set, "cate_true")}')
  # print(f'M22: \t{getRandomInterventionalSample(factual_instance, action_set, "cate_hivae")}')


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
      scatterRecourse(factual_instance, action_set, 'point', 'true', '*', ax)
      scatterRecourse(factual_instance, action_set, 'point', 'approx_lin', 's', ax)
      scatterRecourse(factual_instance, action_set, 'distr', 'n/a', '^', ax)
      scatterDecisionBoundary(ax)
      ax.set_xlabel('x1')
      ax.set_ylabel('x2')
      ax.set_zlabel('x3')
      ax.set_title(f'sample_{factual_instance_idx} \n do({prettyPrintActionSet(action_set)})', fontsize=8, horizontalalignment='left')
      ax.view_init(elev=15, azim=10)


      # for angle in range(0, 360):
      #   ax.view_init(30, angle)
      #   pyplot.draw()
      #   pyplot.pause(.001)

  pyplot.suptitle('Compare M0, M1, M2 on <n> factual samples and <n> **fixed** action sets.', fontsize=14)
  pyplot.show()


def experiment3(X_train, X_test, y_train, y_test):
  ''' compare M0, M1, M2 on <n> factual samples and <n> **computed** action sets '''

  NUM_PLOT_ROWS = np.floor(np.sqrt(NUM_TEST_SAMPLES))
  NUM_PLOT_COLS = np.ceil(NUM_TEST_SAMPLES / NUM_PLOT_ROWS)

  factual_instances_dict = X_test.iloc[:NUM_TEST_SAMPLES].T.to_dict()

  fig = pyplot.figure()

  for index, (key, value) in enumerate(factual_instances_dict.items()):
    idx_sample = index
    factual_instance_idx = key
    factual_instance = value

    print(f'\n\n[INFO] Computing counterfactuals (M0, M1, M2) for factual instance #{index+1} / {NUM_TEST_SAMPLES} (id #{factual_instance_idx})...')

    ax = pyplot.subplot(
      NUM_PLOT_COLS,
      NUM_PLOT_ROWS,
      idx_sample + 1,
      projection = '3d')

    # scatterDataset(X_train, X_test, y_train, y_test, ax)

    scatterFactual(factual_instance, ax)

    m0_optimal_action_set = computeOptimalActionSet(factual_instance, 'point', 'true')
    m11_optimal_action_set = computeOptimalActionSet(factual_instance, 'point', 'approx_lin')
    m12_optimal_action_set = computeOptimalActionSet(factual_instance, 'point', 'approx_krr')
    m13_optimal_action_set = computeOptimalActionSet(factual_instance, 'distr', 'approx_gp')
    m21_optimal_action_set = computeOptimalActionSet(factual_instance, 'distr', 'cate_true')
    m22_optimal_action_set = computeOptimalActionSet(factual_instance, 'distr', 'cate_hivae')

    scatterRecourse(factual_instance, m0_optimal_action_set, 'point', 'true', '*', ax)
    scatterRecourse(factual_instance, m11_optimal_action_set, 'point', 'true', 's', ax) # show where the counterfactual will lie, when action set is computed using m1 but carried out in m0
    scatterRecourse(factual_instance, m12_optimal_action_set, 'point', 'true', 'D', ax) # show where the counterfactual will lie, when action set is computed using m1 but carried out in m0
    # scatterRecourse(factual_instance, m1_optimal_action_set, 'distr', 'n/a', 's', ax) # how many of m2 suggested by m1 would fail
    scatterRecourse(factual_instance, m13_optimal_action_set, 'distr', 'approx_gp', 'p', ax)
    scatterRecourse(factual_instance, m21_optimal_action_set, 'distr', 'cate_true', '^', ax)
    scatterRecourse(factual_instance, m22_optimal_action_set, 'distr', 'cate_hivae', 'v', ax)

    scatterDecisionBoundary(ax)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.set_title(
      f'sample_{factual_instance_idx} - {prettyPrintInstance(factual_instance)}'
      f'\n m0 action set: do({prettyPrintActionSet(m0_optimal_action_set)}); cost: {measureActionSetCost(factual_instance, m0_optimal_action_set, NORM_TYPE):.2f}'
      f'\n m11 action set: do({prettyPrintActionSet(m11_optimal_action_set)}); cost: {measureActionSetCost(factual_instance, m11_optimal_action_set, NORM_TYPE):.2f}'
      f'\n m12 action set: do({prettyPrintActionSet(m12_optimal_action_set)}); cost: {measureActionSetCost(factual_instance, m12_optimal_action_set, NORM_TYPE):.2f}'
      f'\n m13 action set: do({prettyPrintActionSet(m13_optimal_action_set)}); cost: {measureActionSetCost(factual_instance, m13_optimal_action_set, NORM_TYPE):.2f}'
      f'\n m21 action set: do({prettyPrintActionSet(m21_optimal_action_set)}); cost: {measureActionSetCost(factual_instance, m21_optimal_action_set, NORM_TYPE):.2f}'
      f'\n m12 action set: do({prettyPrintActionSet(m22_optimal_action_set)}); cost: {measureActionSetCost(factual_instance, m22_optimal_action_set, NORM_TYPE):.2f}'
    , fontsize=8, horizontalalignment='left')
    ax.view_init(elev=20, azim=-30)
    print('[INFO] done.')

    # for angle in range(0, 360):
    #   ax.view_init(30, angle)
    #   pyplot.draw()
    #   pyplot.pause(.001)

  pyplot.suptitle(f'Compare M0, M1, M2 on {NUM_TEST_SAMPLES} factual samples and **optimal** action sets.', fontsize=14)
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
  # TODO: create experiment folder and save model there

  ''' compare M0, M1, M2 on one factual samples and one **fixed** action sets '''
  experiment1(X_train, X_test, y_train, y_test)
  ''' compare M0, M1, M2 on <n> factual samples and <n> **fixed** action sets '''
  # experiment2(X_train, X_test, y_train, y_test)
  ''' compare M0, M1, M2 on <n> factual samples and <n> **computed** action sets '''
  # experiment3(X_train, X_test, y_train, y_test)

  # sanity check
  # visualizeDatasetAndFixedModel(X_train, X_test, y_train, y_test)















