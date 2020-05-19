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
from distributions import *

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

SCM_CLASS = 'sanity-add'
# SCM_CLASS = 'sanity-mult'
# SCM_CLASS = 'sanity-power'

DEBUG_FLAG = False
NORM_TYPE = 2
LAMBDA_LCB = 1
GRID_SEARCH_BINS = 5
NUM_TRAIN_SAMPLES = 500
NUM_RECOURSE_SAMPLES = 5
NUM_DISPLAY_SAMPLES = 10
NUM_MONTE_CARLO_SAMPLES = 100

ACCEPTABLE_POINT_RECOURSE = {'m0_true', 'm1_alin', 'm1_akrr'}
ACCEPTABLE_DISTR_RECOURSE = {'m1_gaus', 'm1_cvae', 'm2_true', 'm2_gaus', 'm2_cvae', 'm2_cvae_ps'}

# class Instance(object):
#   def __init__(self, endogenous_dict, exogenous_dict):

#     # assert()
#     self.endogenous_dict = endogenous_dict
#     self.exogenous_dict = exogenous_dict


@utils.Memoize
def loadDataset(dataset_class):
  return loadData.loadDataset(dataset_class, return_one_hot = True, load_from_cache = False)


@utils.Memoize
def loadClassifier(dataset_class, classifier_class, experiment_folder_name):
  return loadModel.loadModelForDataset(classifier_class, dataset_class, experiment_folder_name)


@utils.Memoize
def loadCausalModel(experiment_folder_name = None):

  if SCM_CLASS == 'sanity-add':

    structural_equations = {
      'x1': lambda n_samples,        :           n_samples,
      'x2': lambda n_samples, x1     :  2 * x1 + n_samples,
      'x3': lambda n_samples, x1, x2 : x1 + x2 + n_samples,
    }
    noises_distributions = {
      'u1': MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]),
      'u2': Normal(0, 1),
      'u3': Normal(0, 1),
    }

  elif SCM_CLASS == 'sanity-mult':

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

  elif SCM_CLASS == 'sanity-power':

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

  scm = CausalModel(structural_equations, noises_distributions)
  if experiment_folder_name is not None:
    scm.visualizeGraph(experiment_folder_name)
  return scm


def measureActionSetCost(dataset_obj, factual_instance, action_set):
  # TODO: the cost should be measured in normalized space over all features
  #       pass in dataset_obj to get..
  deltas = []
  ranges = dataset_obj.getVariableRanges()
  for key in action_set.keys():
    deltas.append((action_set[key] - factual_instance[key]) / ranges[key])
  return np.linalg.norm(deltas, NORM_TYPE)


def getIndexOfFactualInstanceInDataFrame(factual_instance, data_frame):
  # data_frame may include X and U, whereas factual_instance only includes X
  found_flag = False
  assert set(factual_instance.keys()).issubset(set(data_frame.columns))
  for enumeration_idx, (factual_instance_idx, row) in enumerate(data_frame.iterrows()):
    if np.all([
      factual_instance[key] == row[key]
      for key in factual_instance.keys()
    ]):
      found_flag = True
      break
  if not found_flag:
    raise Exception(f'Was not able to find instance in dataset.')
  return enumeration_idx


def processDataFrameOrDict(dataset_obj, obj, processing_type):
  # TODO: add support for categorical data

  X_all = getOriginalData()

  if isinstance(obj, dict):
    iterate_over = obj.keys()
  elif isinstance(obj, pd.DataFrame):
    iterate_over = obj.columns

  obj = obj.copy() # so as not to change the underlying object
  for node in iterate_over:
    if 'u' in node:
      print(f'[WARNING] Skipping over processing of noise variable {node}.')
      continue
    node_min = float(min(X_all[node]))
    node_max = float(max(X_all[node]))
    node_mean = float(np.mean(X_all[node]))
    node_std = float(np.std(X_all[node]))
    if processing_type == 'normalize':
      obj[node] = (obj[node] - node_min) / (node_max - node_min)
    elif processing_type == 'standardize':
      obj[node] = (obj[node] - node_mean) / node_std
    elif processing_type == 'mean_subtract':
      obj[node] = (obj[node] - node_mean)
  return obj


def deprocessDataFrameOrDict(dataset_obj, obj, processing_type):
  # TODO: add support for categorical data

  X_all = getOriginalData()

  if isinstance(obj, dict):
    iterate_over = obj.keys()
  elif isinstance(obj, pd.DataFrame):
    iterate_over = obj.columns

  obj = obj.copy() # so as not to change the underlying object
  for node in iterate_over:
    if 'u' in node:
      print(f'[WARNING] Skipping over processing of noise variable {node}.')
      continue
    node_min = float(min(X_all[node]))
    node_max = float(max(X_all[node]))
    node_mean = float(np.mean(X_all[node]))
    node_std = float(np.std(X_all[node]))
    if processing_type == 'normalize':
      obj[node] = obj[node] * (node_max - node_min) + node_min
    elif processing_type == 'standardize':
      obj[node] = obj[node] * node_std + node_mean
    elif processing_type == 'mean_subtract':
      obj[node] = obj[node] + node_mean
  return obj


@utils.Memoize
def getOriginalData(with_meta = False):
  if with_meta:
    X_train, X_test, U_train, U_test, y_train, y_test = dataset_obj.getTrainTestSplit(with_meta = True)
    return pd.concat(
      [
        pd.concat([X_train, U_train], axis = 1),
        pd.concat([X_test, U_test], axis = 1),
      ],
      axis = 0
    )[:NUM_TRAIN_SAMPLES]
  else:
    X_train, X_test, y_train, y_test = dataset_obj.getTrainTestSplit()
    return pd.concat([X_train, X_test], axis = 0)[:NUM_TRAIN_SAMPLES]


@utils.Memoize
def getStandardizedData():
  X_all = getOriginalData()
  X_all = processDataFrameOrDict(dataset_obj, X_all, 'standardize')
  return X_all


def getNoiseStringForNode(node):
  assert node[0] == 'x'
  return 'u' + node[1:]


def prettyPrintDict(my_dict):
  my_dict = my_dict.copy()
  for key, value in my_dict.items():
    my_dict[key] = np.around(value, 3)
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


@utils.Memoize
def trainRidge(dataset_obj, node, parents):
  assert len(parents) > 0, 'parents set cannot be empty.'
  print(f'\t[INFO] Fitting p({node} | {", ".join(parents)}) using Ridge on {NUM_TRAIN_SAMPLES} samples; this may be very expensive, memoizing afterwards.')
  X_all = getStandardizedData()
  param_grid = {'alpha': np.linspace(0,10,11)}
  model = GridSearchCV(Ridge(), param_grid=param_grid)
  model.fit(X_all[parents], X_all[[node]])
  return model


@utils.Memoize
def trainKernelRidge(dataset_obj, node, parents):
  assert len(parents) > 0, 'parents set cannot be empty.'
  print(f'\t[INFO] Fitting p({node} | {", ".join(parents)}) using KernelRidge on {NUM_TRAIN_SAMPLES} samples; this may be very expensive, memoizing afterwards.')
  X_all = getStandardizedData()
  param_grid = {
    'alpha': [1e0, 1e-1, 1e-2, 1e-3],
    'kernel': [
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
  assert len(parents) > 0, 'parents set cannot be empty.'
  print(f'\t[INFO] Fitting p({node} | {", ".join(parents)}) using CVAE on {NUM_TRAIN_SAMPLES} samples; this may be very expensive, memoizing afterwards.')
  X_all = getStandardizedData()
  return train_cvae(AttrDict({
    'name': f'p({node} | {", ".join(parents)})',
    'node': X_all[[node]],
    'parents': X_all[parents],
    'seed': 0,
    'epochs': 100,
    'batch_size': 64,
    'learning_rate': 0.001,
    'encoder_layer_sizes': [1, 10, 10], # 1 b/c the X_all[[node]] is always 1 dimensional # TODO: will change for categorical variables
    'decoder_layer_sizes': [10, 10, 1], # 1 b/c the X_all[[node]] is always 1 dimensional # TODO: will change for categorical variables
    'latent_size': 2, # TODO: should this be 1 to be interpreted as noise?
    'conditional': True,
    'print_every': 1000,
    'debug_flag': DEBUG_FLAG,
    'debug_folder': experiment_folder_name,
  }))


@utils.Memoize
def trainGP(dataset_obj, node, parents):
  assert len(parents) > 0, 'parents set cannot be empty.'
  print(f'\t[INFO] Fitting p({node} | {", ".join(parents)}) using GP on {NUM_TRAIN_SAMPLES} samples; this may be very expensive, memoizing afterwards.')
  X_all = getStandardizedData()
  kernel = GPy.kern.RBF(input_dim=len(parents), ARD=True)
  model = GPy.models.GPRegression(X_all[parents], X_all[[node]], kernel)
  model.optimize_restarts(parallel=True, num_restarts=5, verbose=False)
  X = X_all[parents].to_numpy()
  return kernel, X, model


def _getAbductionNoise(dataset_obj, classifier_obj, causal_model_obj, node, parents, factual_instance, structural_equation):
  # only applies for ANM models
  return factual_instance[node] - structural_equation(
    0,
    *[factual_instance[parent] for parent in parents],
  )


def sampleTrue(dataset_obj, classifier_obj, causal_model_obj, samples_df, factual_instance, node, parents, recourse_type):
  # Step 1. [abduction]: compute noise or load from dataset using factual_instance
  # Step 2. [action]: (skip) this step is implicitly performed in the populated samples_df columns
  # Step 3. [prediction]: run through structural equation using noise and parents from samples_df
  structural_equation = causal_model_obj.structural_equations[node]

  if recourse_type == 'm0_true':

    noise_pred = _getAbductionNoise(dataset_obj, classifier_obj, causal_model_obj, node, parents, factual_instance, structural_equation)
    XU_all = getOriginalData(with_meta = True)
    tmp_idx = getIndexOfFactualInstanceInDataFrame(factual_instance, XU_all)
    noise_true = XU_all.iloc[tmp_idx][getNoiseStringForNode(node)]
    print(f'noise_pred: {noise_pred:.8f} \t noise_true: {noise_true:.8f} \t difference: {np.abs(noise_pred - noise_true):.8f}')

    # noise_pred assume additive noise, and therefore only works with
    # models such as 'm1_alin' and 'm1_akrr' in general cases
    if recourse_type == 'm0_true':
      if SCM_CLASS != 'sanity-power':
        assert np.abs(noise_pred - noise_true) < 1e-5, 'Noise {pred, true} expected to be similar, but not.'
      noise = noise_true
    else:
      noise = noise_pred

    for row_idx, row in samples_df.iterrows():
      noise = noise
      samples_df.loc[row_idx, node] = structural_equation(
        noise,
        *samples_df.loc[row_idx, parents].to_numpy(),
      )

  elif recourse_type == 'm2_true':

    for row_idx, row in samples_df.iterrows():
      noise = causal_model_obj.noises_distributions[getNoiseStringForNode(node)].sample(),
      samples_df.loc[row_idx, node] = structural_equation(
        noise,
        *samples_df.loc[row_idx, parents].to_numpy(),
      )

  return samples_df


def _sampleRidgeKernelRidge(dataset_obj, classifier_obj, causal_model_obj, samples_df, factual_instance, node, parents, recourse_type, train_handle):
  # XU_all = getOriginalData(with_meta = True)
  # tmp_idx = getIndexOfFactualInstanceInDataFrame(factual_instance, XU_all)
  # noise_true = XU_all.iloc[tmp_idx][getNoiseStringForNode(node)]

  # All samplers EXCEPT FOR sampleTrue have been trained (and should be sampled) on standardized data
  samples_df = processDataFrameOrDict(dataset_obj, samples_df.copy(), 'standardize')
  factual_instance = processDataFrameOrDict(dataset_obj, factual_instance.copy(), 'standardize')

  # Step 1. [abduction]: compute noise or load from dataset using factual_instance
  # Step 2. [action]: (skip) this step is implicitly performed in the populated samples_df columns
  # Step 3. [prediction]: run through structural equation using noise and parents from samples_df
  trained_model = train_handle(dataset_obj, node, parents)
  structural_equation = lambda noise, *parents_values: trained_model.predict([[*parents_values]])[0][0] + noise
  noise_pred = _getAbductionNoise(dataset_obj, classifier_obj, causal_model_obj, node, parents, factual_instance, structural_equation)
  # print(f'noise_pred: {noise_pred:.8f} \t noise_true: {noise_true:.8f} \t difference: {np.abs(noise_pred - noise_true):.8f}') # TODO: check this out... why differnce?
  for row_idx, row in samples_df.iterrows():
    noise = _getAbductionNoise(dataset_obj, classifier_obj, causal_model_obj, node, parents, factual_instance, structural_equation),
    samples_df.loc[row_idx, node] = structural_equation(
      noise,
      *samples_df.loc[row_idx, parents].to_numpy(),
    )
  samples_df = deprocessDataFrameOrDict(dataset_obj, samples_df, 'standardize')
  return samples_df


def sampleRidge(dataset_obj, classifier_obj, causal_model_obj, samples_df, factual_instance, node, parents, recourse_type):
  return _sampleRidgeKernelRidge(dataset_obj, classifier_obj, causal_model_obj, samples_df, factual_instance, node, parents, recourse_type, trainRidge)


def sampleKernelRidge(dataset_obj, classifier_obj, causal_model_obj, samples_df, factual_instance, node, parents, recourse_type):
  return _sampleRidgeKernelRidge(dataset_obj, classifier_obj, causal_model_obj, samples_df, factual_instance, node, parents, recourse_type, trainKernelRidge)


def sampleCVAE(dataset_obj, classifier_obj, causal_model_obj, samples_df, factual_instance, node, parents, recourse_type):
  # All samplers EXCEPT FOR sampleTrue have been trained (and should be sampled) on standardized data
  samples_df = processDataFrameOrDict(dataset_obj, samples_df.copy(), 'standardize')
  factual_instance = processDataFrameOrDict(dataset_obj, factual_instance.copy(), 'standardize')

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

  new_samples = trained_cvae.reconstruct(
    x_factual=x_factual,
    pa_factual=pa_factual,
    pa_counter=pa_counter,
    sample_from=sample_from,
  )
  new_samples = new_samples.rename(columns={0: node}) # bad code amir, this violates abstraction!
  samples_df[node] = new_samples
  samples_df = deprocessDataFrameOrDict(dataset_obj, samples_df, 'standardize')
  return samples_df


def sampleGP(dataset_obj, classifier_obj, causal_model_obj, samples_df, factual_instance, node, parents, recourse_type):
  # All samplers EXCEPT FOR sampleTrue have been trained (and should be sampled) on standardized data
  samples_df = processDataFrameOrDict(dataset_obj, samples_df.copy(), 'standardize')
  factual_instance = processDataFrameOrDict(dataset_obj, factual_instance.copy(), 'standardize')

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

  kernel, X, model = trainGP(dataset_obj, node, parents)

  K = kernel.K(X)
  sigma_noise = np.array(model.Gaussian_noise.variance)
  noise_post_means = noise_post_mean(K, sigma_noise, X)
  noise_post_vars = noise_post_var(K, sigma_noise)

  # GP posterior for node at new (intervened & conditioned) input given parents
  # pred_means, pred_vars = model.predict_noiseless(samples_df[parents].to_numpy())
  pred_means, pred_vars = model.predict_noiseless(samples_df[parents].to_numpy())

  # IMPORTANT: Find index of factual instance in dataframe used for training GP
  #            (earlier, the factual instance was appended as the last instance)
  tmp_idx = getIndexOfFactualInstanceInDataFrame(factual_instance, getStandardizedData())

  if recourse_type == 'm1_gaus': # counterfactual distribution for node
    new_means = pred_means + noise_post_means[tmp_idx]
    new_vars = pred_vars + noise_post_vars[tmp_idx]
  elif recourse_type == 'm2_gaus': # interventional distribution for node
    new_means = pred_means + 0
    new_vars = pred_vars + sigma_noise

  # sample from distribution via reparametrisation trick
  new_noise = np.random.randn(samples_df.shape[0], 1)
  new_samples = new_means + np.sqrt(new_vars) * new_noise

  samples_df[node] = new_samples
  samples_df = deprocessDataFrameOrDict(dataset_obj, samples_df, 'standardize')
  return samples_df


def _samplingInnerLoop(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, recourse_type, num_samples):

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

  # this dataframe has populated columns set to intervention or conditioning values
  # and has NaN columns that will be set accordingly.
  samples_df = pd.DataFrame(dict(zip(
    dataset_obj.getInputAttributeNames(),
    [num_samples * [counterfactual_template[node]] for node in dataset_obj.getInputAttributeNames()],
  )))

  # Simply traverse the graph in order, and populate nodes as we go!
  # IMPORTANT: DO NOT USE set(topo ordering); it sometimes changes ordering!
  for node in causal_model_obj.getTopologicalOrdering():
    # set variable if value not yet set through intervention or conditioning
    if samples_df[node].isnull().values.any():
      parents = causal_model_obj.getParentsForNode(node)
      # Confirm parents columns are present/have assigned values in samples_df
      assert not samples_df.loc[:,list(parents)].isnull().values.any()
      if DEBUG_FLAG:
        print(f'Sampling `{recourse_type}` from p({node} | {", ".join(parents)})')
      if recourse_type in {'m0_true', 'm2_true'}:
        sampling_handle = sampleTrue
      elif recourse_type == 'm1_alin':
        sampling_handle = sampleRidge
      elif recourse_type == 'm1_akrr':
        sampling_handle = sampleKernelRidge
      elif recourse_type in {'m1_gaus', 'm2_gaus'}:
        sampling_handle = sampleGP
      elif recourse_type in {'m1_cvae', 'm2_cvae', 'm2_cvae_ps'}:
        sampling_handle = sampleCVAE
      else:
        raise Exception(f'{recourse_type} not recognized.')
      samples_df = sampling_handle(
        dataset_obj,
        classifier_obj,
        causal_model_obj,
        samples_df,
        factual_instance,
        node,
        parents,
        recourse_type,
      )
  assert \
    np.all(list(samples_df.columns) == dataset_obj.getInputAttributeNames()), \
    'Ordering of column names in samples_df has change unexpectedly'
  # samples_df = samples_df[dataset_obj.getInputAttributeNames()]
  return samples_df


def computeCounterfactualInstance(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, recourse_type):

  assert recourse_type in ACCEPTABLE_POINT_RECOURSE

  if not bool(action_set): # if action_set is empty, CFE = F
    return factual_instance

  samples_df = _samplingInnerLoop(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, recourse_type, 1)

  return samples_df.loc[0].to_dict() # return the first instance


def getRecourseDistributionSample(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, recourse_type, num_samples):

  assert recourse_type in ACCEPTABLE_DISTR_RECOURSE

  if not bool(action_set): # if action_set is empty, CFE = F
    return pd.DataFrame(dict(zip(
      dataset_obj.getInputAttributeNames(),
      [num_samples * [factual_instance[node]] for node in dataset_obj.getInputAttributeNames()],
    )))

  samples_df = _samplingInnerLoop(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, recourse_type, num_samples)

  return samples_df # return the entire data frame


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


def isDistrConstraintSatisfied(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, recourse_type):
  return computeLowerConfidenceBound() >= 0.5


def computeLowerConfidenceBound(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, recourse_type):
  monte_carlo_samples_df = getRecourseDistributionSample(
    dataset_obj,
    classifier_obj,
    causal_model_obj,
    factual_instance,
    action_set,
    recourse_type,
    NUM_MONTE_CARLO_SAMPLES,
  )
  monte_carlo_predictions = getPredictionBatch(
    dataset_obj,
    classifier_obj,
    causal_model_obj,
    monte_carlo_samples_df.to_numpy(),
  )

  expectation = np.mean(monte_carlo_predictions)
  variance = np.sum(np.power(monte_carlo_predictions - expectation, 2)) / (len(monte_carlo_predictions) - 1)

  # return expectation, variance

  # IMPORTANT... WE ARE CONSIDERING {0,1} LABELS AND FACTUAL SAMPLES MAY BE OF
  # EITHER CLASS. THEREFORE, THE CONSTRAINT IS SATISFIED WHEN SIGNIFICANTLY
  # > 0.5 OR < 0.5 FOR A FACTUAL SAMPLE WITH Y = 0 OR Y = 1, RESPECTIVELY.

  if getPrediction(dataset_obj, classifier_obj, causal_model_obj, factual_instance) == 0:
    return expectation - LAMBDA_LCB * np.sqrt(variance) # NOTE DIFFERNCE IN SIGN OF STD
  else: # factual_prediction == 1
    raise Exception(f'Should only be considering negatively predicted individuals...')
    # return expectation + LAMBDA_LCB * np.sqrt(variance) # NOTE DIFFERNCE IN SIGN OF STD


def getValidDiscretizedActionSets(dataset_obj):

  possible_actions_per_node = []

  for attr_name_kurz in dataset_obj.getInputAttributeNames('kurz'):

    attr_obj = dataset_obj.attributes_kurz[attr_name_kurz]

    if attr_obj.attr_type in {'numeric-real', 'numeric-int', 'binary'}:

      if attr_obj.attr_type == 'numeric-real':
        number_decimals = 3
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

    else:

      raise NotImplementedError # TODO

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
        cost_of_action_set = measureActionSetCost(dataset_obj, factual_instance, action_set)
        if cost_of_action_set < min_cost:
          min_cost = cost_of_action_set
          min_cost_action_set = action_set

    print(f'\t done (optimal action set: {str(min_cost_action_set)}).')

  elif optimization_approach == 'grad_descent':

    raise NotImplementedError # TODO
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


def scatterRecourse(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, recourse_type, marker_type, legend_label, ax):

  assert len(dataset_obj.getInputAttributeNames()) == 3

  if recourse_type in ACCEPTABLE_POINT_RECOURSE:
    # point recourse

    point = computeCounterfactualInstance(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, recourse_type)
    color_string = 'green' if didFlip(dataset_obj, classifier_obj, causal_model_obj, factual_instance, point) else 'red'
    ax.scatter(point['x1'], point['x2'], point['x3'], marker=marker_type, color=color_string, s=70, label=legend_label)

  elif recourse_type in ACCEPTABLE_DISTR_RECOURSE:
    # distr recourse

    samples_df = getRecourseDistributionSample(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, recourse_type, NUM_DISPLAY_SAMPLES)
    x1s = samples_df.iloc[:,0]
    x2s = samples_df.iloc[:,1]
    x3s = samples_df.iloc[:,2]
    color_strings = ['green' if didFlip(dataset_obj, classifier_obj, causal_model_obj, factual_instance, sample.to_dict()) else 'red' for _, sample in samples_df.iterrows()]
    ax.scatter(x1s, x2s, x3s, marker=marker_type, color=color_strings, alpha=0.1, s=30, label=legend_label)

    # mean_distr_samples = {
    #   'x1': np.mean(samples_df['x1']),
    #   'x2': np.mean(samples_df['x2']),
    #   'x3': np.mean(samples_df['x3']),
    # }
    # color_string = 'green' if didFlip(dataset_obj, classifier_obj, causal_model_obj, factual_instance, mean_distr_samples) else 'red'
    # ax.scatter(mean_distr_samples['x1'], mean_distr_samples['x2'], mean_distr_samples['x3'], marker=marker_type, color=color_string, alpha=0.5, s=70, label=legend_label)

  else:

    raise Exception(f'{recourse_type} not recognized.')


def getNegativelyPredictedInstances(dataset_obj, classifier_obj, causal_model_obj):
  # Only focus on instances with h(x^f) = 0 and therfore h(x^cf) = 1
  X_all = getOriginalData()
  factual_instances_dict = {}
  tmp_counter = 1
  for factual_instance_idx, row in X_all.iterrows():
    factual_instance = row.T.to_dict()
    if getPrediction(dataset_obj, classifier_obj, causal_model_obj, factual_instance) == 0:
      tmp_counter += 1
      factual_instances_dict[factual_instance_idx] = factual_instance
    if tmp_counter > NUM_RECOURSE_SAMPLES:
      break
  return factual_instances_dict


def hotTrainRecourseTypes(dataset_obj, classifier_obj, causal_model_obj, recourse_types):
  start_time = time.time()
  print(f'\n' + '='*60 + '\n')
  print(f'[INFO] Hot-training ALIN, AKRR, GAUS, CVAE so they do not affect runtime...')
  training_handles = []
  if any(['alin' in elem for elem in recourse_types]): training_handles.append(trainRidge)
  if any(['akrr' in elem for elem in recourse_types]): training_handles.append(trainKernelRidge)
  if any(['gaus' in elem for elem in recourse_types]): training_handles.append(trainGP)
  if any(['cvae' in elem for elem in recourse_types]): training_handles.append(trainCVAE)
  for training_handle in training_handles:
    print()
    for node in causal_model_obj.getTopologicalOrdering():
      parents = causal_model_obj.getParentsForNode(node)
      if len(parents): # if not a root node
        training_handle(dataset_obj, node, parents)
  end_time = time.time()
  print(f'\n[INFO] Done (total warm-up time: {end_time - start_time}).')
  print(f'\n' + '='*60 + '\n')


# DEPRECATED def experiment1


# DEPRECATED def experiment2


# DEPRECATED def experiment3


# DEPRECATED def experiment4


def experiment5(dataset_obj, classifier_obj, causal_model_obj, experiment_folder_name):
  ''' fixed action set: assert {m1, m2} x {gaus, cvae} working '''
  factual_instances_dict = getNegativelyPredictedInstances(dataset_obj, classifier_obj, causal_model_obj)

  experimental_setups = [
    ('m0_true', '*'), \
    ('m1_alin', 'v'), \
    ('m1_akrr', '^'), \
    ('m1_gaus', 'D'), \
    ('m1_cvae', 'x'), \
    ('m2_true', 'o'), \
    ('m2_gaus', 's'), \
    ('m2_cvae', '+'), \
    # ('m2_cvae_ps', 'P'), \
  ]
  recourse_types = [experimental_setup[0] for experimental_setup in experimental_setups]

  hotTrainRecourseTypes(dataset_obj, classifier_obj, causal_model_obj, recourse_types)

  print(f'Describe original data:\n{getOriginalData().describe()}')

  action_sets = [
    {'x1': causal_model_obj.noises_distributions['u1'].sample()}
    for _ in range(4)
  ]

  factual_instance = factual_instances_dict[list(factual_instances_dict.keys())[0]]

  fig, axes = pyplot.subplots(int(np.sqrt(len(action_sets))), int(np.sqrt(len(action_sets))))
  fig.suptitle(f'FC: {prettyPrintDict(factual_instance)}', fontsize='x-small')
  if len(action_sets) == 1:
    axes = np.array(axes) # weird hack we need to use so to later use flatten()

  print(f'\nFC: \t\t{prettyPrintDict(factual_instance)}')

  for idx, action_set in enumerate(action_sets):

    print(f'\n\n[INFO] ACTION SET: {str(action_set)}' + ' =' * 40)

    for experimental_setup in experimental_setups:
      recourse_type, marker = experimental_setup[0], experimental_setup[1]

      if recourse_type in ACCEPTABLE_POINT_RECOURSE:
        sample = computeCounterfactualInstance(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, recourse_type)
        print(f'{recourse_type}:\t{prettyPrintDict(sample)}')
        axes.flatten()[idx].plot(sample['x2'], sample['x3'], marker, alpha=1.0, markersize = 7, label=recourse_type)
      elif recourse_type in ACCEPTABLE_DISTR_RECOURSE:
        samples = getRecourseDistributionSample(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, recourse_type, NUM_DISPLAY_SAMPLES)
        print(f'{recourse_type}:\n{samples.head()}')
        axes.flatten()[idx].plot(samples['x2'], samples['x3'], marker, alpha=0.3, markersize = 4, label=recourse_type)
      else:
        raise Exception(f'{recourse_type} not supported.')

    axes.flatten()[idx].set_xlabel('$x2$', fontsize='x-small')
    axes.flatten()[idx].set_ylabel('$x3$', fontsize='x-small')
    # axes.flatten()[idx].set_xlim(-10, 10)
    # axes.flatten()[idx].set_ylim(-10, 10)
    axes.flatten()[idx].tick_params(axis='both', which='major', labelsize=6)
    axes.flatten()[idx].tick_params(axis='both', which='minor', labelsize=4)
    axes.flatten()[idx].set_title(f'action_set: {str(action_set)}', fontsize='x-small')

  for ax in axes.flatten():
    ax.legend(fontsize='xx-small')
  fig.tight_layout()
  # pyplot.show()
  pyplot.savefig(f'{experiment_folder_name}/comparison.pdf')


def experiment6(dataset_obj, classifier_obj, causal_model_obj, experiment_folder_name):
  ''' optimal action set: figure + table '''

  # Samples for which we seek recourse are chosen from the joint of X_train/test.
  # This is OK because the tasks of conditional density estimation and recourse
  # generation are distinct. Given the same data splicing used here and in trainGP,
  # it is guaranteed that we the factual sample for which we seek recourse is in
  # training set for GP, and hence a posterior over noise for it is computed
  # (i.e., we can cache).

  factual_instances_dict = getNegativelyPredictedInstances(dataset_obj, classifier_obj, causal_model_obj)

  experimental_setups = [
    ('m0_true', '*'), \
    ('m1_alin', 'v'), \
    ('m1_akrr', '^'), \
    ('m1_gaus', 'D'), \
    ('m1_cvae', 'x'), \
    ('m2_true', 'o'), \
    ('m2_gaus', 's'), \
    ('m2_cvae', '+'), \
    # ('m2_cvae_ps', 'P'), \
  ]
  recourse_types = [experimental_setup[0] for experimental_setup in experimental_setups]

  hotTrainRecourseTypes(dataset_obj, classifier_obj, causal_model_obj, recourse_types)

  per_instance_results = {}
  for enumeration_idx, (key, value) in enumerate(factual_instances_dict.items()):
    factual_instance_idx = f'sample_{key}'
    factual_instance = value

    print(f'\n\n\n[INFO] Processing factual instance `{factual_instance_idx}` (#{enumeration_idx + 1} / {len(factual_instances_dict.keys())})...')

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

      tmp['runtime'] = np.around(end_time - start_time, 3)

      # print(f'\t[INFO] Computing SCF validity and Interventional Confidence measures for optimal action `{str(tmp["optimal_action_set"])}`...')

      tmp['scf_validity']  = isPointConstraintSatisfied(dataset_obj, classifier_obj, causal_model_obj, factual_instance, tmp['optimal_action_set'], 'm0_true')
      tmp['int_conf_true'] = np.around(computeLowerConfidenceBound(dataset_obj, classifier_obj, causal_model_obj, factual_instance, tmp['optimal_action_set'], 'm2_true'), 3)
      tmp['int_conf_cvae'] = np.around(computeLowerConfidenceBound(dataset_obj, classifier_obj, causal_model_obj, factual_instance, tmp['optimal_action_set'], 'm2_cvae'), 3)
      tmp['cost_all'] = measureActionSetCost(dataset_obj, factual_instance, tmp['optimal_action_set'])
      tmp['cost_valid'] = tmp['cost_all'] if tmp['scf_validity'] else np.NaN

      # print(f'\t done.')

      per_instance_results[factual_instance_idx][recourse_type] = tmp

    print(f'[INFO] Saving (overwriting) results...')
    pickle.dump(per_instance_results, open(f'{experiment_folder_name}/_per_instance_results', 'wb'))
    pprint(per_instance_results, open(f'{experiment_folder_name}/_per_instance_results.txt', 'w'))
    print(f'done.')

  # Table
  metrics_summary = {}
  metrics = ['scf_validity', 'int_conf_true', 'int_conf_cvae', 'cost_all', 'cost_valid', 'runtime']

  for metric in metrics:
    metrics_summary[metric] = []
  # metrics_summary = dict.fromkeys(metrics, []) # BROKEN: all lists will be shared; causing massive headache!!!

  for recourse_type in recourse_types:
    for metric in metrics:
      metrics_summary[metric].append(
        f'{np.around(np.nanmean([v[recourse_type][metric] for k,v in per_instance_results.items()]), 3):.3f}' + \
        '+/-' + \
        f'{np.around(np.nanstd([v[recourse_type][metric] for k,v in per_instance_results.items()]), 3):.3f}'
      )
  tmp_df = pd.DataFrame(metrics_summary, recourse_types)
  print(tmp_df)
  tmp_df.to_csv(f'{experiment_folder_name}/comparison.txt', sep='\t')

  # TODO: FIX
  # # Figure
  # if len(dataset_obj.getInputAttributeNames()) != 3:
  #   print('Cannot plot in more than 3 dimensions')
  #   return

  # max_to_plot = 4
  # tmp = min(NUM_RECOURSE_SAMPLES, max_to_plot)
  # num_plot_cols = int(np.floor(np.sqrt(tmp)))
  # num_plot_rows = int(np.ceil(tmp / num_plot_cols))

  # fig, axes = pyplot.subplots(num_plot_rows, num_plot_cols, subplot_kw=dict(projection='3d'))
  # if num_plot_rows * num_plot_cols == max_to_plot:
  #   axes = np.array(axes) # weird hack we need to use so to later use flatten()

  # for idx, (key, value) in enumerate(factual_instances_dict.items()):
  #   if idx >= 1:
  #     break

  #   factual_instance_idx = f'sample_{key}'
  #   factual_instance = value

  #   ax = axes.flatten()[idx]

  #   scatterFactual(dataset_obj, factual_instance, ax)
  #   title = f'sample_{factual_instance_idx} - {prettyPrintDict(factual_instance)}'

  #   for experimental_setup in experimental_setups:
  #     recourse_type, marker = experimental_setup[0], experimental_setup[1]
  #     optimal_action_set = per_instance_results[factual_instance_idx][recourse_type]['optimal_action_set']
  #     legend_label = f'\n {recourse_type}; do({prettyPrintDict(optimal_action_set)}); cost: {measureActionSetCost(dataset_obj, factual_instance, optimal_action_set):.3f}'
  #     scatterRecourse(dataset_obj, classifier_obj, causal_model_obj, factual_instance, optimal_action_set, 'm0_true', marker, legend_label, ax)
  #     # scatterRecourse(dataset_obj, classifier_obj, causal_model_obj, factual_instance, optimal_action_set, recourse_type, marker, legend_label, ax)

  #   scatterDecisionBoundary(dataset_obj, classifier_obj, causal_model_obj, ax)
  #   ax.set_xlabel('x1', fontsize=8)
  #   ax.set_ylabel('x2', fontsize=8)
  #   ax.set_zlabel('x3', fontsize=8)
  #   ax.set_title(title, fontsize=8)
  #   ax.view_init(elev=20, azim=-30)

  # for ax in axes.flatten():
  #   ax.legend(fontsize='xx-small')
  # fig.tight_layout()
  # # pyplot.show()
  # pyplot.savefig(f'{experiment_folder_name}/comparison.pdf')


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
    '-c', '--classifier_class',
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
  classifier_class = args.classifier_class

  if not (dataset_class in {'random', 'mortgage', 'twomoon', 'german', 'credit', 'compass', 'adult'}):
    raise Exception(f'{dataset_class} not supported.')

  if not (classifier_class in {'lr', 'mlp'}):
    raise Exception(f'{classifier_class} not supported.')

  # create experiment folder
  setup_name = f'{dataset_class}__{classifier_class}'
  experiment_folder_name = f"_experiments/{datetime.now().strftime('%Y.%m.%d_%H.%M.%S')}__{setup_name}"
  os.mkdir(f'{experiment_folder_name}')

  # only load once so shuffling order is the same
  dataset_obj = loadDataset(dataset_class)
  classifier_obj = loadClassifier(dataset_class, classifier_class, experiment_folder_name)
  causal_model_obj = loadCausalModel(experiment_folder_name)
  assert set(dataset_obj.getInputAttributeNames()) == set(causal_model_obj.getTopologicalOrdering())

  # experiment1(dataset_obj, classifier_obj, causal_model_obj)
  experiment5(dataset_obj, classifier_obj, causal_model_obj, experiment_folder_name)
  # experiment6(dataset_obj, classifier_obj, causal_model_obj, experiment_folder_name)

  # sanity check
  # visualizeDatasetAndFixedModel(dataset_obj, classifier_obj, causal_model_obj)












































