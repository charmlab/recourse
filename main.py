import os
import time
import pickle
import inspect
import pathlib
import argparse
import itertools
import subprocess
import numpy as np
import pandas as pd

from tqdm import tqdm
from pprint import pprint
from matplotlib import pyplot
from datetime import datetime
from attrdict import AttrDict

import GPy
from scm import CausalModel
import utils
import loadData
import loadModel

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared

from _cvae.train import *

from debug import ipsh

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

DEBUG_FLAG = False
NORM_TYPE = 2
LAMBDA_LCB = 1
NUM_TRAIN_SAMPLES = 100
NUM_TEST_SAMPLES = 10
GRID_SEARCH_BINS = 3
NUMBER_OF_MONTE_CARLO_SAMPLES = 100
SAVED_MACE_RESULTS_PATH_M0 = '/Users/a6karimi/dev/recourse/_minimum_distances_m0'
SAVED_MACE_RESULTS_PATH_M1 = '/Users/a6karimi/dev/recourse/_minimum_distances_m1'

ACCEPTABLE_POINT_RECOURSE = {'m0_true', 'm1_alin', 'm1_akrr'}
ACCEPTABLE_DISTR_RECOURSE = {'m1_gaus', 'm1_cvae', 'm2_true', 'm2_gaus', 'm2_cvae', 'm2_cvae_ps'}


@utils.Memoize
def loadDataset(dataset_class, increment_indices = True):

  def incrementIndices(old_indices):
    new_indices = ['x' + str(int(index[1:]) + 1) for index in old_indices]
    return new_indices

  dataset_obj = loadData.loadDataset(dataset_class, return_one_hot = True, load_from_cache = False)

  if increment_indices:

    # only change data_frame_kurz/attributes_kurz (data_frame_long/attributes_long
    # may contain non-numeric columns)
    replacement_dict = dict(zip(
      dataset_obj.getInputAttributeNames(),
      incrementIndices(dataset_obj.getInputAttributeNames()),
    ))

    # update data_frame_kurz
    dataset_obj.data_frame_kurz = dataset_obj.data_frame_kurz.rename(columns=replacement_dict)

    # update attributes_kurz
    old_attributes_kurz = dataset_obj.attributes_kurz
    new_attributes_kurz = {}

    for old_key, old_value in old_attributes_kurz.items():
      if old_key in replacement_dict.keys():
        new_key = replacement_dict[old_key]
        new_value = old_value
        new_value.attr_name_kurz = replacement_dict[old_key]
      else:
        new_key = old_key
        new_value = old_value
      new_attributes_kurz[new_key] = new_value

    dataset_obj.attributes_kurz = new_attributes_kurz

  return dataset_obj


@utils.Memoize
def loadClassifier(dataset_class, model_class, experiment_folder_name):
  return loadModel.loadModelForDataset(model_class, dataset_class, experiment_folder_name)


@utils.Memoize
def loadCausalModel(dataset_class, experiment_folder_name):

  # structural_equations = {
  #   'x1': lambda n1,       :                 n1,
  #   'x2': lambda n2,     x1:        x1 + 1 + n2,
  #   'x3': lambda n3, x1, x2: 5 * (x1 + x2) + n3,
  # }
  # noise_distributions = {
  #   'x1': lambda n_samples: 1.00 * np.random.normal(size=n_samples),
  #   'x2': lambda n_samples: 0.25 * np.random.normal(size=n_samples),
  #   'x3': lambda n_samples: 0.25 * np.random.normal(size=n_samples),
  # }
  # new_dict = dict(zip(
  #   structural_equations.keys(),
  #   [lambda parents, n_samples: structural_equations[node](0, *parents) + np.random.normal(size=n_samples) for node in structural_equations.keys()]
  # ))
  # scm = CausalModel(new_dict)


  scm = CausalModel({
    'x1': lambda         n_samples:                                  np.random.normal(size=n_samples),
    'x2': lambda     x1, n_samples:                         x1 + 1 + np.random.normal(size=n_samples),
    'x3': lambda x1, x2, n_samples: x1 / 4 + np.sqrt(3) * x2 - 1/4 + np.random.normal(size=n_samples),
  })

  # scm = CausalModel({
  #   'x1': lambda         n_samples:                               np.random.normal(size=n_samples),
  #   'x2': lambda     x1, n_samples:                      x1 + 1 + np.random.normal(size=n_samples),
  #   'x3': lambda x1, x2, n_samples: np.sqrt(3) * x1 * (x2 ** 2) + np.random.normal(size=n_samples),
  # })

  # scm = CausalModel({
  #   'x1': lambda         n_samples:                 1.00 * np.random.normal(size=n_samples),
  #   'x2': lambda     x1, n_samples:        x1 + 1 + 0.25 * np.random.normal(size=n_samples),
  #   'x3': lambda x1, x2, n_samples: 5 * (x1 + x2) + 0.25 * np.random.normal(size=n_samples),
  #   'x4': lambda     x3, n_samples:            x3 + 0.25 * np.random.normal(size=n_samples),
  #   'x5': lambda     x4, n_samples:            x4 + 0.25 * np.random.normal(size=n_samples),
  #   'x6': lambda     x2, n_samples:            x2 + 0.25 * np.random.normal(size=n_samples),
  # })
  scm.visualizeGraph(experiment_folder_name)
  return scm


@utils.Memoize
def getStructuralEquation(dataset_obj, classifier_obj, causal_model_obj, node, recourse_type):

  if recourse_type == 'm0_true':

    # # TODO: use causal_model_obj to get this?
    if node == 'x1':
      return lambda n1: n1
    elif node == 'x2':
      return lambda n2, x1: x1 + 1 + n2
    elif node == 'x3':
      return lambda n3, x1, x2: x1 / 4 + np.sqrt(3) * x2 - 1/4 + n3

    # if node == 'x1':
    #   return lambda n1: n1
    # elif node == 'x2':
    #   return lambda n2, x1: x1 + 1 + n2
    # elif node == 'x3':
    #   return lambda n3, x1, x2: np.sqrt(3) * x1 * (x2 ** 2) + n3

    # if node == 'x1':
    #   return lambda n1: n1
    # elif node == 'x2':
    #   return lambda n2, x1: x1 + 1 + n2
    # elif node == 'x3':
    #   return lambda n3, x1, x2: 5 * (x1 + x2) + n3
    # elif node == 'x4':
    #   return lambda n4, x3: x3 + n4
    # elif node == 'x5':
    #   return lambda n5, x4: x4 + n5
    # elif node == 'x6':
    #   return lambda n6, x2: x2 + n6

  elif recourse_type in {'m1_alin', 'm1_akrr'}:

    X_train, X_test, y_train, y_test = dataset_obj.getTrainTestSplit()
    X_all = X_train.append(X_test)
    X_all = X_all[:NUM_TRAIN_SAMPLES]

    parents = causal_model_obj.getParentsForNode(node)

    if len(parents) == 0: # if root node

      return lambda noise: noise

    else:

      if recourse_type == 'm1_alin':
        model = trainRidge(dataset_obj, node, parents)
      elif recourse_type == 'm1_akrr':
        model = trainKernelRidge(dataset_obj, node, parents)

      model.fit(X_all[parents], X_all[[node]])

      return lambda noise, *parents: model.predict([[*parents]])[0][0] + noise


def measureActionSetCost(dataset_obj, factual_instance, action_set, norm_type):
  # TODO: the cost should be measured in normalized space over all features
  #       pass in dataset_obj to get..
  deltas = []
  ranges = dataset_obj.getVariableRanges()
  for key in action_set.keys():
    deltas.append((action_set[key] - factual_instance[key]) / ranges[key])
  return np.linalg.norm(deltas, norm_type)


def processDataFrame(dataset_obj, df, processing_type):
  df = df.copy() # so as not to change the underlying object
  X_train, X_test, y_train, y_test = dataset_obj.getTrainTestSplit()
  X_all = X_train.append(X_test)
  for col_idx in df.columns:
    col_min = float(min(X_all[col_idx]))
    col_max = float(max(X_all[col_idx]))
    col_mean = float(np.mean(X_all[col_idx]))
    col_std = float(np.std(X_all[col_idx]))
    if processing_type == 'normalize':
      df[col_idx] = (df[col_idx] - col_min) / (col_max - col_min)
    elif processing_type == 'standardize':
      df[col_idx] = (df[col_idx] - col_mean) / col_std
    elif processing_type == 'mean_subtract':
      df[col_idx] = (df[col_idx] - col_mean)
  return df


def deprocessDataFrame(dataset_obj, df, processing_type):
  df = df.copy() # so as not to change the underlying object
  X_train, X_test, y_train, y_test = dataset_obj.getTrainTestSplit()
  X_all = X_train.append(X_test)
  for col_idx in df.columns:
    col_min = float(min(X_all[col_idx]))
    col_max = float(max(X_all[col_idx]))
    col_mean = float(np.mean(X_all[col_idx]))
    col_std = float(np.std(X_all[col_idx]))
    if processing_type == 'normalize':
      df[col_idx] = df[col_idx] * (col_max - col_min) + col_min
    elif processing_type == 'standardize':
      df[col_idx] = df[col_idx] * col_std + col_mean
    elif processing_type == 'mean_subtract':
      df[col_idx] = df[col_idx] + col_mean
  return df


def prettyPrintDict(my_dict):
  for key, value in my_dict.items():
    my_dict[key] = np.around(value, 2)
  return my_dict


def getPredictionBatch(dataset_obj, classifier_obj, causal_model_obj, instances_df):
  sklearn_model = classifier_obj
  return sklearn_model.predict(instances_df)


def getPrediction(dataset_obj, classifier_obj, causal_model_obj, instance):
  sklearn_model = classifier_obj
  prediction = sklearn_model.predict(np.array(list(instance.values())).reshape(1,-1))[0]
  assert prediction in {0, 1}, f'Expected prediction in {0,1}; got {prediction}'
  return prediction


def didFlip(dataset_obj, classifier_obj, causal_model_obj, factual_instance, counterfactual_instance):
  return \
    getPrediction(dataset_obj, classifier_obj, causal_model_obj, factual_instance) != \
    getPrediction(dataset_obj, classifier_obj, causal_model_obj, counterfactual_instance)


# See https://stackoverflow.com/a/25670697
def lambdaWrapper(new_value):
  return lambda *args: new_value


@utils.Memoize
def trainRidge(dataset_obj, node, parents):
  assert len(parents) > 0, 'parents set cannot be empty'
  print(f'[INFO] Fitting p({node} | {", ".join(parents)}) using Ridge on {NUM_TRAIN_SAMPLES} samples; this may be very expensive, memoizing afterwards.')
  X_train, X_test, y_train, y_test = dataset_obj.getTrainTestSplit()
  X_all = X_train.append(X_test)
  X_all = X_all[:NUM_TRAIN_SAMPLES]
  param_grid = {"alpha": np.linspace(0,10,11)}
  model = GridSearchCV(Ridge(), param_grid=param_grid)
  model.fit(X_all[parents], X_all[[node]])
  return model


@utils.Memoize
def trainKernelRidge(dataset_obj, node, parents):
  assert len(parents) > 0, 'parents set cannot be empty'
  print(f'[INFO] Fitting p({node} | {", ".join(parents)}) using KernelRidge on {NUM_TRAIN_SAMPLES} samples; this may be very expensive, memoizing afterwards.')
  X_train, X_test, y_train, y_test = dataset_obj.getTrainTestSplit()
  X_all = X_train.append(X_test)
  X_all = X_all[:NUM_TRAIN_SAMPLES]
  param_grid = {
    "alpha": [1e0, 1e-1, 1e-2, 1e-3],
    "kernel": [
      ExpSineSquared(l, p)
      for l in np.logspace(-2, 2, 5)
      for p in np.logspace(0, 2, 5)
    ]
  }
  model = GridSearchCV(KernelRidge(), param_grid=param_grid)
  model.fit(X_all[parents], X_all[[node]])
  return model


@utils.Memoize
def trainCVAE(dataset_obj, node, parents):
  assert len(parents) > 0, 'parents set cannot be empty'
  print(f'[INFO] Fitting p({node} | {", ".join(parents)}) using CVAE on {NUM_TRAIN_SAMPLES} samples; this may be very expensive, memoizing afterwards.')
  X_train, X_test, y_train, y_test = dataset_obj.getTrainTestSplit()
  X_all = X_train.append(X_test)
  X_all = X_all[:NUM_TRAIN_SAMPLES]
  return train_cvae(AttrDict({
    'name': f'p({node} | {", ".join(parents)})',
    'node': processDataFrame(dataset_obj, X_all[[node]], 'standardize'),
    'parents': processDataFrame(dataset_obj, X_all[parents], 'standardize'),
    'seed': 0,
    'epochs': 100,
    'batch_size': 64,
    'learning_rate': 0.001,
    'encoder_layer_sizes': [1, 5, 5, 5], # 1 b/c the X_all[[node]] is always 1 dimensional # TODO: will change for categorical variables
    'decoder_layer_sizes': [5, 5, 5, 1], # 1 b/c the X_all[[node]] is always 1 dimensional # TODO: will change for categorical variables
    'latent_size': 2, # TODO: should this be 1 to be interpreted as noise?
    'conditional': True,
    'print_every': 1000,
    'debug_flag': DEBUG_FLAG,
    'debug_folder': experiment_folder_name,
  }))


@utils.Memoize
def trainGP(dataset_obj, node, parents, X, Y):
  assert len(parents) > 0, 'parents set cannot be empty'
  print(f'[INFO] Fitting p({node} | {", ".join(parents)}) using GP on {NUM_TRAIN_SAMPLES} samples; this may be very expensive, memoizing afterwards.')
  kernel = GPy.kern.RBF(input_dim=X.shape[1], variance=1., lengthscale=1.)
  model = GPy.models.GPRegression(X, Y, kernel)
  model.optimize_restarts(parallel=True, num_restarts = 5, verbose=False)
  return kernel, model


def sampleCVAE(dataset_obj, samples_df, node, parents, factual_instance, recourse_type):
  trained_cvae = trainCVAE(dataset_obj, node, parents)
  num_samples = samples_df.shape[0]

  x_factual = pd.DataFrame(dict(zip(
    [node],
    [num_samples * [factual_instance[node]] for node in [node]],
  )))
  pa_factual = pd.DataFrame(dict(zip(
    parents,
    [num_samples * [factual_instance[node]] for node in parents],
  )))
  pa_counter = samples_df[parents]

  if recourse_type == 'm1_cvae':
    sample_from = 'prior'
  elif recourse_type == 'm2_cvae':
    sample_from = 'posterior'
  elif recourse_type == 'm2_cvae_ps':
    sample_from = 'reweighted_prior'

  tmp = trained_cvae.reconstruct(
    x_factual=processDataFrame(dataset_obj, x_factual, 'standardize'),
    pa_factual=processDataFrame(dataset_obj, pa_factual, 'standardize'),
    pa_counter=processDataFrame(dataset_obj, pa_counter, 'standardize'),
    # sample_from='reweighted_prior',
    # sample_from='posterior' if recourse_type == 'm1_cvae' else 'prior',
    sample_from=sample_from,
  )
  tmp = tmp.rename(columns={0: node}) # bad code amir, this violates abstraction!
  samples_df[node] = deprocessDataFrame(dataset_obj, tmp, 'standardize')
  return samples_df


def sampleGP(dataset_obj, samples_df, node, parents, factual_instance, recourse_type):
  X_train, X_test, y_train, y_test = dataset_obj.getTrainTestSplit()
  X_all = X_train.append(X_test)
  X_all = X_all[:NUM_TRAIN_SAMPLES]
  # make sure factual instance is in training set (you lose indexing, but no need)
  X_all = X_all.append(factual_instance, ignore_index=True)
  X_all = processDataFrame(dataset_obj, X_all, 'mean_subtract')
  X = X_all[parents].to_numpy()
  Y = X_all[[node]].to_numpy()

  def noise_post_mean(K, sigma, Y):
    N = K.shape[0]
    S = np.linalg.inv(K + sigma * np.eye(N))
    return sigma * np.dot(S, Y)

  def noise_post_cov(K, sigma):
    N = K.shape[0]
    S = np.linalg.inv(K + sigma * np.eye(N))
    return  sigma * (np.eye(N) - sigma * S)

  def noise_post_var(K, sigma):
    N = K.shape[0]
    C = noise_post_cov(K, sigma)
    return np.array([C[i,i] for i in range(N)])

  kernel, model = trainGP(dataset_obj, node, parents, X, Y)

  K = kernel.K(X)
  sigma_noise = np.array(model.Gaussian_noise.variance)
  noise_post_means = noise_post_mean(K, sigma_noise, X)
  noise_post_vars = noise_post_var(K, sigma_noise)

  # GP posterior for node at new (intervened & conditioned) input given parents
  pred_means, pred_vars = model.predict_noiseless(samples_df[parents].to_numpy())

  if recourse_type == 'm1_gaus':
    # counterfactual distribution for node
    new_means = pred_means + noise_post_means[-1] # -1 b/c factual instance was appended as last instance
    new_vars = pred_vars + noise_post_vars[-1] # -1 b/c factual instance was appended as last instance
  elif recourse_type == 'm2_gaus':
    new_means = pred_means + 0
    new_vars = pred_vars + sigma_noise

  # sample from distribution via reparametrisation trick
  new_noise = np.random.randn(samples_df.shape[0], 1)
  new_samples = new_means + np.sqrt(new_vars) * new_noise

  samples_df[node] = new_samples
  samples_df = deprocessDataFrame(dataset_obj, samples_df, 'mean_subtract')
  return samples_df



def computeCounterfactualInstance(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, recourse_type):

  if not bool(action_set): # if action_set is empty, CFE = F
    return factual_instance

  structural_equations_new = dict(zip(factual_instance.keys(), [
    getStructuralEquation(dataset_obj, classifier_obj, causal_model_obj, node, recourse_type)
    for node in factual_instance.keys()
  ]))

  # Step 1. abduction: get value of noise variables
  # tip: pass in n* = 0 to structural_equations (lambda functions)
  noise_variables_new = dict(zip(factual_instance.keys(), [
    factual_instance[node] - structural_equations_new[node](
      0,
      *[factual_instance[node] for node in causal_model_obj.getParentsForNode(node)],
    )
    for node in factual_instance.keys()
  ]))

  # Step 2. action: update structural equations
  for key, value in action_set.items():
    node = key
    intervention_value = value
    structural_equations_new[node] = lambdaWrapper(intervention_value)
    # *args is used to allow for ignoring arguments that may be passed into this
    # function (consider, for example, an intervention on x2 which then requires
    # no inputs to call the second structural equation function, but we still pass
    # in the arugments a few lines down)

  # Step 3. prediction: compute counterfactual values starting from root node
  # CANNOT USE THE COMMENTED CODE BELOW; BECAUSE CF VALUES FOR X_i DEPENDs ON CF
  # VALUES OF PA_i, WHICH IS NOT DEFINED YET IN THE ONE-LINER
  # counterfactual_instance_new = dict(zip(factual_instance.keys(), [
  #   structural_equations_new[node](
  #     *[counterfactual_instance_new[node] for node in causal_model_obj.getParentsForNode(node)],
  #     noise_variables_new[node],
  #   )
  #   for node in factual_instance.keys()
  # ]))
  counterfactual_instance_new = {}
  for node in factual_instance.keys():
    counterfactual_instance_new[node] = structural_equations_new[node](
      noise_variables_new[node],
      *[counterfactual_instance_new[node] for node in causal_model_obj.getParentsForNode(node)],
    )

  return counterfactual_instance_new


def getRecourseDistributionSample(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, recourse_type, num_samples):

  if not bool(action_set): # if action_set is empty, CFE = F
    return pd.DataFrame(dict(zip(
      dataset_obj.getInputAttributeNames(),
      [num_samples * [factual_instance[node]] for node in dataset_obj.getInputAttributeNames()],
    )))

  counterfactual_template = dict.fromkeys(
    dataset_obj.getInputAttributeNames(),
    np.NaN,
  )

  # get intervention and conditioning sets
  intervention_set = set(action_set.keys())

  # intersection_of_non_descendents_of_intervened_upon_variables
  conditioning_set = set.intersection(*[
    causal_model_obj.getNonDescendentsForNode(node)
    for node in intervention_set
  ])

  # assert there is no intersection
  assert set.intersection(intervention_set, conditioning_set) == set()

  # set values in intervention and conditioning sets
  for node in conditioning_set:
    counterfactual_template[node] = factual_instance[node]

  for node in intervention_set:
    counterfactual_template[node] = action_set[node]

  samples_df = pd.DataFrame(dict(zip(
    dataset_obj.getInputAttributeNames(),
    [num_samples * [counterfactual_template[node]] for node in dataset_obj.getInputAttributeNames()],
  )))


  if recourse_type == 'm2_true':

    scm_do = causal_model_obj.scm
    preassigned_nodes = samples_df.columns[~samples_df.isnull().all()]
    for node in samples_df.columns[~samples_df.isnull().all()]:
      scm_do = scm_do.do(node)

    samples_df = scm_do.sample(
      n_samples = num_samples,
      set_values = dict(zip(
        preassigned_nodes,
        samples_df[preassigned_nodes].T.to_numpy()
      )),
    )

  else:

    # Simply traverse the graph in order, and populate nodes as we go!
    # IMPORTANT: DO NOT USE set(topo ordering); it sometimes changes ordering!
    for node in causal_model_obj.getTopologicalOrdering():
      # if variable value is not yet set through intervention or conditioning
      if samples_df[node].isnull().values.any():
        parents = causal_model_obj.getParentsForNode(node)
        # Confirm parents columns are present/have assigned values in samples_df
        assert not samples_df.loc[:,list(parents)].isnull().values.any()
        if DEBUG_FLAG:
          print(f'Sampling `{recourse_type}` from p({node} | {", ".join(parents)})')
        if recourse_type in {'m1_gaus', 'm2_gaus'}:
          samples_df = sampleGP(dataset_obj, samples_df, node, parents, factual_instance, recourse_type)
        elif recourse_type in {'m1_cvae', 'm2_cvae', 'm2_cvae_ps'}:
          samples_df = sampleCVAE(dataset_obj, samples_df, node, parents, factual_instance, recourse_type)

  # IMPORTANT: if for whatever reason, the columns change order (e.g., as seen in
  # scm_do.sample), reorder them as they are to be used as inputs to the fixed classifier
  samples_df = samples_df[dataset_obj.data_frame_kurz.columns[1:]]
  return samples_df


def isPointConstraintSatisfied(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, recourse_type):
  return didFlip(
    dataset_obj,
    classifier_obj,
    causal_model_obj,
    factual_instance,
    computeCounterfactualInstance(
      dataset_obj,
      classifier_obj,
      causal_model_obj,
      factual_instance,
      action_set,
      recourse_type,
    ),
  )


def getExpectationVariance(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, recourse_type):
  monte_carlo_samples_df = getRecourseDistributionSample(
    dataset_obj,
    classifier_obj,
    causal_model_obj,
    factual_instance,
    action_set,
    recourse_type,
    NUMBER_OF_MONTE_CARLO_SAMPLES,
  )
  monte_carlo_predictions = getPredictionBatch(
    dataset_obj,
    classifier_obj,
    causal_model_obj,
    monte_carlo_samples_df.to_numpy(),
  )

  expectation = np.mean(monte_carlo_predictions)
  variance = np.sum(np.power(monte_carlo_predictions - expectation, 2)) / (len(monte_carlo_predictions) - 1)

  return expectation, variance


def isDistrConstraintSatisfied(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, recourse_type):

  expectation, variance = getExpectationVariance(
    dataset_obj,
    classifier_obj,
    causal_model_obj,
    factual_instance,
    action_set,
    recourse_type,
  )

  # IMPORTANT... WE ARE CONSIDERING {0,1} LABELS AND FACTUAL SAMPLES MAY BE OF
  # EITHER CLASS. THEREFORE, THE CONSTRAINT IS SATISFIED WHEN SIGNIFICANTLY
  # > 0.5 OR < 0.5 FOR A FACTUAL SAMPLE WITH Y = 0 OR Y = 1, RESPECTIVELY.

  if getPrediction(dataset_obj, classifier_obj, causal_model_obj, factual_instance) == 0:
    return expectation - LAMBDA_LCB * np.sqrt(variance) > 0.5 # NOTE DIFFERNCE IN SIGN OF STD
  else: # factual_prediction == 1
    return expectation + LAMBDA_LCB * np.sqrt(variance) < 0.5 # NOTE DIFFERNCE IN SIGN OF STD


def getValidDiscretizedActionSets(dataset_obj):

  possible_actions_per_node = []

  for attr_name_kurz in dataset_obj.getInputAttributeNames('kurz'):

    attr_obj = dataset_obj.attributes_kurz[attr_name_kurz]

    if attr_obj.attr_type in {'numeric-real', 'numeric-int', 'binary'}:

      if attr_obj.attr_type == 'numeric-real':
        number_decimals = 4
      elif attr_obj.attr_type in {'numeric-int', 'binary'}:
        number_decimals = 0

      tmp = list(
        np.around(
          np.linspace(
            attr_obj.lower_bound,
            attr_obj.upper_bound,
            GRID_SEARCH_BINS + 1
          ),
          number_decimals,
        )
      )
      tmp.append('n/a')
      tmp = list(dict.fromkeys(tmp))
      # remove repeats from list; this may happen, say for numeric-int, where we
      # can have upper-lower < GRID_SEARCH_BINS, then rounding to 0 will result
      # in some repeated values
      possible_actions_per_node.append(tmp)

    else: # TODO

      raise NotImplementedError

  all_action_tuples = list(itertools.product(
    *possible_actions_per_node
  ))

  all_action_sets = [
    dict(zip(dataset_obj.getInputAttributeNames(), elem))
    for elem in all_action_tuples
  ]

  # Go through, and for any action_set that has a value = 'n/a', remove ONLY
  # THAT key, value pair, NOT THE ENTIRE action_set.
  valid_action_sets = []
  for action_set in all_action_sets:
    valid_action_sets.append({k:v for k,v in action_set.items() if v != 'n/a'})

  return valid_action_sets


def computeOptimalActionSet(dataset_obj, classifier_obj, causal_model_obj, factual_instance, recourse_type, optimization_approach):

  if recourse_type in ACCEPTABLE_POINT_RECOURSE:
    constraint_handle = isPointConstraintSatisfied
  elif recourse_type in ACCEPTABLE_DISTR_RECOURSE:
    constraint_handle = isDistrConstraintSatisfied
  else:
    raise Exception(f'{recourse_type} not recognized.')

  if optimization_approach == 'brute_force':

    valid_action_sets = getValidDiscretizedActionSets(dataset_obj)
    print(f'\n\t[INFO] Computing optimal `{recourse_type}`: grid searching over {len(valid_action_sets)} action sets...')

    min_cost = 1e10
    min_cost_action_set = {}
    for action_set in tqdm(valid_action_sets):
      if constraint_handle(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, recourse_type):
        cost_of_action_set = measureActionSetCost(dataset_obj, factual_instance, action_set, NORM_TYPE)
        if cost_of_action_set < min_cost:
          min_cost = cost_of_action_set
          min_cost_action_set = action_set

    print(f'\t done.')

  elif optimization_approach == 'grad_descent':

    raise NotImplementedError
    # for all possible intervention sets (without value)
    # for each child-parent that is missing
    #     get object: trained_cvae = trainCVAE(dataset_obj, node, parents)
    #     this should be a torch object
    #     then the child is a function of
    #         its factual value
    #         its parents' factual value
    #         the post-intervention value of its parents
    #     then write h() as a function of all nodes
    #     see if you can pass gradients back to the intervention value of the nodes (possibly > 1) being intervened on
    #     then add the cost function
    #     and finally minimize everything together



  else:
    raise Exception(f'{optimization_approach} not recognized.')

  return min_cost_action_set


def scatterDecisionBoundary(dataset_obj, classifier_obj, causal_model_obj, ax):
  assert len(dataset_obj.getInputAttributeNames()) == 3
  sklearn_model = classifier_obj
  fixed_model_w = sklearn_model.coef_
  fixed_model_b = sklearn_model.intercept_

  x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
  y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
  X = np.linspace(ax.get_xlim()[0] - x_range / 10, ax.get_xlim()[1] + x_range / 10, 10)
  Y = np.linspace(ax.get_ylim()[0] - y_range / 10, ax.get_ylim()[1] + y_range / 10, 10)
  X, Y = np.meshgrid(X, Y)
  Z = - (fixed_model_w[0][0] * X + fixed_model_w[0][1] * Y + fixed_model_b) / fixed_model_w[0][2]

  surf = ax.plot_wireframe(X, Y, Z, alpha=0.3)


def scatterDataset(dataset_obj, ax):
  assert len(dataset_obj.getInputAttributeNames()) == 3
  X_train, X_test, y_train, y_test = dataset_obj.getTrainTestSplit()
  X_train_numpy = X_train.to_numpy()
  X_test_numpy = X_test.to_numpy()
  y_train = y_train.to_numpy()
  y_test = y_test.to_numpy()
  number_of_samples_to_plot = 100
  for idx in range(number_of_samples_to_plot):
    color_train = 'black' if y_train[idx] == 1 else 'magenta'
    color_test = 'black' if y_test[idx] == 1 else 'magenta'
    ax.scatter(X_train_numpy[idx, 0], X_train_numpy[idx, 1], X_train_numpy[idx, 2], marker='s', color=color_train, alpha=0.2, s=10)
    ax.scatter(X_test_numpy[idx, 0], X_test_numpy[idx, 1], X_test_numpy[idx, 2], marker='o', color=color_test, alpha=0.2, s=15)


def scatterFactual(dataset_obj, factual_instance, ax):
  assert len(dataset_obj.getInputAttributeNames()) == 3
  ax.scatter(
    factual_instance['x1'],
    factual_instance['x2'],
    factual_instance['x3'],
    marker='P',
    color='black',
    s=70
  )


def scatterRecourse(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, recourse_type, marker_type, ax):

  assert len(dataset_obj.getInputAttributeNames()) == 3

  if recourse_type in ACCEPTABLE_POINT_RECOURSE:
    # point recourse

    point = computeCounterfactualInstance(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, recourse_type)
    color_string = 'green' if didFlip(dataset_obj, classifier_obj, causal_model_obj, factual_instance, point) else 'red'
    ax.scatter(point['x1'], point['x2'], point['x3'], marker = marker_type, color=color_string, s=70)

  elif recourse_type in ACCEPTABLE_DISTR_RECOURSE:
    # distr recourse

    samples_df = getRecourseDistributionSample(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, recourse_type, 100)
    for sample_idx in range(samples_df.shape[0]):
      sample = samples_df.iloc[sample_idx].to_dict()
      color_string = 'green' if didFlip(dataset_obj, classifier_obj, causal_model_obj, factual_instance, sample) else 'red'
      ax.scatter(sample['x1'], sample['x2'], sample['x3'], marker = marker_type, color=color_string, alpha=0.1, s=30)

    mean_distr_samples = {
      'x1': np.mean(samples_df['x1']),
      'x2': np.mean(samples_df['x2']),
      'x3': np.mean(samples_df['x3']),
    }
    color_string = 'green' if didFlip(dataset_obj, classifier_obj, causal_model_obj, factual_instance, mean_distr_samples) else 'red'
    ax.scatter(mean_distr_samples['x1'], mean_distr_samples['x2'], mean_distr_samples['x3'], marker = marker_type, color=color_string, alpha=0.5, s=70)

  else:

    raise Exception(f'{recourse_type} not recognized.')


def experiment1(dataset_obj, classifier_obj, causal_model_obj):
  ''' debugging: sanity check to make sure code runs '''
  X_train, X_test, y_train, y_test = dataset_obj.getTrainTestSplit()
  factual_instance = X_test.iloc[0].T.to_dict()

  # iterative over a number of action sets and compare the three counterfactuals
  # action_set = {'x1': -3}
  action_set = {'x2': +1}
  # action_set = {'x3': +1}
  # action_set = {'x1': +2, 'x3': +1, 'x5': 3}
  # action_set = {'x0': +2, 'x2': +1}
  # action_set = {'x1': +2, 'x3': +1}
  # action_set = {'x2': +1, 'x6': 2}
  # action_set = {'x3': +4}

  print(f'fc: \t\t{prettyPrintDict(factual_instance)}')
  # print(f'm0_true: \t{computeCounterfactualInstance(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, "m0_true")}')

  # print(f'm1_alin: \t{computeCounterfactualInstance(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, "m1_alin")}')
  # print(f'm1_akrr: \t{computeCounterfactualInstance(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, "m1_akrr")}')
  # print(f'm1_gaus: \n{getRecourseDistributionSample(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, "m1_gaus", 10)}')
  print(f'm1_cvae: \n{getRecourseDistributionSample(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, "m1_cvae", 10)}')

  # print(f'm2_true: \n{getRecourseDistributionSample(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, "m2_true", 10)}')
  # print(f'm2_gaus: \n{getRecourseDistributionSample(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, "m2_gaus", 10)}')
  # print(f'm2_cvae: \n{getRecourseDistributionSample(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, "m2_cvae", 10)}')
  # print(f'm2_cvae_ps: \n{getRecourseDistributionSample(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, "m2_cvae_ps", 10)}')


# DEPRECATED def experiment2


# DEPRECATED def experiment3


# DEPRECATED def experiment4


def experiment5(dataset_obj, classifier_obj, causal_model_obj, experiment_folder_name):
  ''' fixed action set: assert {m1, m2} x {gaus, cvae} working '''
  X_train, X_test, y_train, y_test = dataset_obj.getTrainTestSplit()
  factual_instance = X_test.iloc[0].T.to_dict()

  # iterative over a number of action sets and compare the three counterfactuals
  action_sets = [ \
    {'x3': +6}, \
    {'x3': +3}, \
    {'x3': +0}, \
    {'x3': -3}, \
  ]
  recourse_types = [
    'm0_true', \
    # 'm1_alin', \
    # 'm1_akrr', \
    'm1_gaus', \
    'm1_cvae', \
    'm2_true', \
    'm2_gaus', \
    'm2_cvae', \
    'm2_cvae_ps', \
  ]
  markers = ['k*', 'cD', 'mP', 'ko', 'bs', 'r+', 'gx']
  num_samples = 10

  fig, axes = pyplot.subplots(int(np.sqrt(len(action_sets))), int(np.sqrt(len(action_sets))))
  # fig.suptitle(f'Comparing conditioning on pa(I) vs pa(I) and nd(I)')

  print(f'X_train:\n{X_train}')
  print(X_train.describe())
  print(f'FC: \t\t{prettyPrintDict(factual_instance)}')

  for idx, action_set in enumerate(action_sets):

    print(f'\n\n[INFO] ACTION SET: {str(action_set)}' + ' =' * 40)

    for (recourse_type, marker) in zip(recourse_types, markers):

      if recourse_type in ACCEPTABLE_POINT_RECOURSE:
        samples = computeCounterfactualInstance(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, recourse_type)
      elif recourse_type in ACCEPTABLE_DISTR_RECOURSE:
        samples = getRecourseDistributionSample(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, recourse_type, num_samples)
      else:
        raise Exception(f'{recourse_type} not supported.')

      print(f'{recourse_type}:\t{samples}')
      axes.ravel()[idx].plot(samples['x4'], samples['x5'], marker, label=recourse_type)

    axes.ravel()[idx].set_ylabel('$x5$', fontsize='x-small')
    axes.ravel()[idx].set_xlabel('$x4$', fontsize='x-small')
    # axes.ravel()[idx].set_ylim(-10, 10)
    # axes.ravel()[idx].set_xlim(-10, 10)
    axes.ravel()[idx].tick_params(axis='both', which='major', labelsize=6)
    axes.ravel()[idx].tick_params(axis='both', which='minor', labelsize=4)
    axes.ravel()[idx].set_title(f'action_set: {str(action_set)}', fontsize='x-small')

  for ax in axes.ravel():
    ax.legend(fontsize='xx-small')
  fig.tight_layout()
  pyplot.savefig(f'{experiment_folder_name}/comparison.eps', format='eps')


def experiment6(dataset_obj, classifier_obj, causal_model_obj, experiment_folder_name):
  ''' optimal action set: figure + table '''
  X_train, X_test, y_train, y_test = dataset_obj.getTrainTestSplit()

  # Only focus on instances with h(x^f) = 0 and therfore h(x^cf) = 1
  factual_instances_dict = X_test.loc[y_test.index[y_test == 0]].iloc[:NUM_TEST_SAMPLES].T.to_dict()

  per_instance_results = {}

  start_time = time.time()
  print(f'\n' + '='*60 + '\n')
  print(f'[INFO] Hot-training ALIN, AKRR, CVAE so they do not affect runtime...')
  training_handles = [trainCVAE, trainRidge, trainKernelRidge] # TODO: add trainGP
  for training_handle in training_handles:
    print()
    for node in causal_model_obj.getTopologicalOrdering():
      parents = causal_model_obj.getParentsForNode(node)
      if len(parents): # if not a root node
        training_handle(dataset_obj, node, parents)
  end_time = time.time()
  print(f'done (total warm-up time: {end_time - start_time}.')
  print(f'\n' + '='*60 + '\n')


  recourse_types = [
    'm0_true', \
    'm1_alin', \
    # 'm1_akrr', \
    'm1_gaus', \
    'm1_cvae', \
    'm2_true', \
    'm2_gaus', \
    'm2_cvae', \
    # 'm2_cvae_ps', \
  ]
  # markers = ['k*', 'cD', 'mP', 'ko', 'bs', 'r+', 'gx']
  markers = ['*', 'D', 'P', 'o', 's', '+', 'x']

  for enumeration_idx, (key, value) in enumerate(factual_instances_dict.items()):
    factual_instance_idx = f'sample_{key}'
    factual_instance = value

    print(f'\n\n[INFO] Processing factual instance `{factual_instance_idx}` (#{enumeration_idx + 1} / {len(factual_instances_dict.keys())})...')

    per_instance_results[factual_instance_idx] = {}

    for recourse_type in recourse_types:

      tmp = {}

      start_time = time.time()
      tmp['optimal_action_set'] = computeOptimalActionSet(
        dataset_obj,
        classifier_obj,
        causal_model_obj,
        factual_instance,
        recourse_type,
        'brute_force',
      )
      end_time = time.time()

      tmp['runtime'] = np.around(end_time - start_time, 4)

      # print(f'\t[INFO] Computing SCF validity and Interventional Confidence measures for optimal action `{str(tmp["optimal_action_set"])}`...')

      tmp['scf_validity']  = isPointConstraintSatisfied(dataset_obj, classifier_obj, causal_model_obj, factual_instance, tmp['optimal_action_set'], 'm0_true')
      exp, var = getExpectationVariance(dataset_obj, classifier_obj, causal_model_obj, factual_instance, tmp['optimal_action_set'], 'm2_true') # IMPORTANT: assume h(x^f) always 0
      tmp['int_confidence_true'] = np.around(exp - LAMBDA_LCB * np.sqrt(var), 4)
      exp, var = getExpectationVariance(dataset_obj, classifier_obj, causal_model_obj, factual_instance, tmp['optimal_action_set'], 'm2_cvae') # IMPORTANT: assume h(x^f) always 0
      tmp['int_confidence_cvae'] = np.around(exp - LAMBDA_LCB * np.sqrt(var), 4)
      tmp['cost_all'] = measureActionSetCost(dataset_obj, factual_instance, tmp['optimal_action_set'], NORM_TYPE)
      tmp['cost_valid'] = tmp['cost_all'] if tmp['scf_validity'] else np.NaN

      # print(f'\t done.')

      per_instance_results[factual_instance_idx][recourse_type] = tmp

    print(f'[INFO] Saving (overwriting) results...')
    pickle.dump(per_instance_results, open(f'{experiment_folder_name}/_per_instance_results', 'wb'))
    pprint(per_instance_results, open(f'{experiment_folder_name}/_per_instance_results.txt', 'w'))
    print(f'done.')

  # Table
  metrics_summary = {}
  metrics = ['scf_validity', 'int_confidence_true', 'int_confidence_cvae', 'cost_all', 'cost_valid', 'runtime']

  for metric in metrics:
    metrics_summary[metric] = []
  # metrics_summary = dict.fromkeys(metrics, []) # BROKEN: all lists are shared causing massive headache!!!

  for recourse_type in recourse_types:
    for metric in metrics:
      metrics_summary[metric].append(
        f'{np.around(np.nanmean([v[recourse_type][metric] for k,v in per_instance_results.items()]), 4):.4f}' + \
        '+/-' + \
        f'{np.around(np.nanstd([v[recourse_type][metric] for k,v in per_instance_results.items()]), 4):.4f}'
      )
  print(pd.DataFrame(metrics_summary, recourse_types))

  # # Figure # TODO: make much cleaner, and use fig, axes = pyplot.subplots like above + add legend
  # if len(dataset_obj.getInputAttributeNames()) != 3:
  #   return

  # num_plot_rows = np.floor(np.sqrt(NUM_TEST_SAMPLES))
  # num_plot_cols = np.ceil(NUM_TEST_SAMPLES / num_plot_rows)

  # axes = []
  # for enumeration_idx, (key, value) in enumerate(factual_instances_dict.items()):

  #   factual_instance_idx = f'sample_{key}'
  #   factual_instance = value

  #   ax = pyplot.subplot(
  #     num_plot_cols,
  #     num_plot_rows,
  #     enumeration_idx + 1,
  #     projection = '3d')
  #   axes.append(ax)

  #   scatterFactual(dataset_obj, factual_instance, ax)
  #   title = f'sample_{factual_instance_idx} - {prettyPrintDict(factual_instance)}'

  #   for (recourse_type, marker) in zip(recourse_types, markers):

  #     optimal_action_set = per_instance_results[factual_instance_idx][recourse_type]['optimal_action_set']
  #     scatterRecourse(dataset_obj, classifier_obj, causal_model_obj, factual_instance, optimal_action_set, recourse_type, marker, ax)
  #     title += f'\n {recourse_type} action set: do({prettyPrintDict(optimal_action_set)}); cost: {measureActionSetCost(dataset_obj, factual_instance, optimal_action_set, NORM_TYPE):.2f}'

  #   scatterDecisionBoundary(dataset_obj, classifier_obj, causal_model_obj, ax)
  #   ax.set_xlabel('x1', fontsize=4)
  #   ax.set_ylabel('x2', fontsize=4)
  #   ax.set_zlabel('x3', fontsize=4)
  #   ax.set_title(title, fontsize=4, horizontalalignment='left')
  #   ax.view_init(elev=20, azim=-30)

  # # for ax in axes.ravel():
  # for ax in axes:
  #   ax.legend(fontsize='xx-small')
  # # fig.tight_layout()
  # pyplot.savefig(f'{experiment_folder_name}/comparison.eps', format='eps')


def visualizeDatasetAndFixedModel(dataset_obj, classifier_obj, causal_model_obj):

  fig = pyplot.figure()
  ax = pyplot.subplot(1, 1, 1, projection='3d')

  scatterDataset(dataset_obj, ax)
  scatterDecisionBoundary(dataset_obj, classifier_obj, causal_model_obj, ax)

  ax.set_xlabel('x1')
  ax.set_ylabel('x2')
  ax.set_zlabel('x3')
  ax.set_title(f'datatset')
  # ax.legend()
  ax.grid(True)

  pyplot.show()


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument(
    '-d', '--dataset_class',
    type = str,
    default = 'random',
    help = 'Name of dataset to train explanation model for: german, random, mortgage, twomoon')

  parser.add_argument(
    '-m', '--model_class',
    type = str,
    default = 'lr',
      help = 'Model class that will learn data: lr, mlp')

  parser.add_argument(
    '-p', '--process_id',
    type = str,
    default = '0',
    help = 'When running parallel tests on the cluster, process_id guarantees (in addition to time stamped experiment folder) that experiments do not conflict.')

  # parsing the args
  args = parser.parse_args()
  dataset_class = args.dataset_class
  model_class = args.model_class

  if not (dataset_class in {'random', 'mortgage', 'twomoon', 'german', 'credit', 'compass', 'adult'}):
    raise Exception(f'{dataset_class} not supported.')

  if not (model_class in {'lr', 'mlp'}):
    raise Exception(f'{model_class} not supported.')

  # create experiment folder
  setup_name = f'{dataset_class}__{model_class}'
  experiment_folder_name = f"_experiments/{datetime.now().strftime('%Y.%m.%d_%H.%M.%S')}__{setup_name}"
  os.mkdir(f'{experiment_folder_name}')

  # only load once so shuffling order is the same
  dataset_obj = loadDataset(dataset_class)
  classifier_obj = loadClassifier(dataset_class, model_class, experiment_folder_name)
  causal_model_obj = loadCausalModel(dataset_class, experiment_folder_name)
  assert set(dataset_obj.getInputAttributeNames()) == set(causal_model_obj.getTopologicalOrdering())

  # experiment1(dataset_obj, classifier_obj, causal_model_obj)
  # experiment5(dataset_obj, classifier_obj, causal_model_obj, experiment_folder_name)
  experiment6(dataset_obj, classifier_obj, causal_model_obj, experiment_folder_name)

  # sanity check
  # visualizeDatasetAndFixedModel(dataset_obj, classifier_obj, causal_model_obj)












































