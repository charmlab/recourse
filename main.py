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
import seaborn as sns

from tqdm import tqdm
from pprint import pprint
from matplotlib import pyplot
from datetime import datetime
from attrdict import AttrDict

import GPy
from scm import CausalModel
import utils
import loadSCM
import loadData
import loadModel

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import WhiteKernel, RBF

from _cvae.train import *

from debug import ipsh

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

DEBUG_FLAG = False
NORM_TYPE = 2
LAMBDA_LCB = 1
GRID_SEARCH_BINS = 5
NUM_TRAIN_SAMPLES = 500
NUM_RECOURSE_SAMPLES = 30
NUM_DISPLAY_SAMPLES = 15
NUM_MONTE_CARLO_SAMPLES = 100

ACCEPTABLE_POINT_RECOURSE = {'m0_true', 'm1_alin', 'm1_akrr'}
ACCEPTABLE_DISTR_RECOURSE = {'m1_gaus', 'm1_cvae', 'm2_true', 'm2_gaus', 'm2_cvae', 'm2_cvae_ps'}

# class Instance(object):
#   def __init__(self, endogenous_dict, exogenous_dict):

#     # assert()
#     self.endogenous_dict = endogenous_dict
#     self.exogenous_dict = exogenous_dict


@utils.Memoize
def loadCausalModel(scm_class, experiment_folder_name = None):
  return loadSCM.loadSCM(scm_class, experiment_folder_name)


@utils.Memoize
def loadDataset(scm_class, dataset_class):
  return loadData.loadDataset(dataset_class, return_one_hot = True, load_from_cache = False, meta_param = scm_class)


@utils.Memoize
def loadClassifier(dataset_class, classifier_class, experiment_folder_name):
  return loadModel.loadModelForDataset(classifier_class, dataset_class, experiment_folder_name)


def measureActionSetCost(dataset_obj, factual_instance, action_set):
  # TODO: add support for categorical data + measured in normalized space over all features
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

  if processing_type == 'raw':
    return obj

  if isinstance(obj, dict):
    iterate_over = obj.keys()
  elif isinstance(obj, pd.DataFrame):
    iterate_over = obj.columns

  obj = obj.copy() # so as not to change the underlying object
  for node in iterate_over:
    if 'u' in node:
      print(f'[WARNING] Skipping over processing of noise variable {node}.')
      continue
    # use dataset_obj stats, not X_train (in case you use more samples later,
    # e.g., validation set for cvae
    tmp = dataset_obj.data_frame_kurz.describe()[node]
    node_min = tmp['min']
    node_max = tmp['max']
    node_mean = tmp['mean']
    node_std = tmp['std']
    if processing_type == 'normalize':
      obj[node] = (obj[node] - node_min) / (node_max - node_min)
    elif processing_type == 'standardize':
      obj[node] = (obj[node] - node_mean) / node_std
    elif processing_type == 'mean_subtract':
      obj[node] = (obj[node] - node_mean)
  return obj


def deprocessDataFrameOrDict(dataset_obj, obj, processing_type):
  # TODO: add support for categorical data

  if processing_type == 'raw':
    return obj

  if isinstance(obj, dict):
    iterate_over = obj.keys()
  elif isinstance(obj, pd.DataFrame):
    iterate_over = obj.columns

  obj = obj.copy() # so as not to change the underlying object
  for node in iterate_over:
    if 'u' in node:
      print(f'[WARNING] Skipping over processing of noise variable {node}.')
      continue
    # use dataset_obj stats, not X_train (in case you use more samples later,
    # e.g., validation set for cvae
    tmp = dataset_obj.data_frame_kurz.describe()[node]
    node_min = tmp['min']
    node_max = tmp['max']
    node_mean = tmp['mean']
    node_std = tmp['std']
    if processing_type == 'normalize':
      obj[node] = obj[node] * (node_max - node_min) + node_min
    elif processing_type == 'standardize':
      obj[node] = obj[node] * node_std + node_mean
    elif processing_type == 'mean_subtract':
      obj[node] = obj[node] + node_mean
  return obj


@utils.Memoize
def getOriginalDataFrame(num_samples = NUM_TRAIN_SAMPLES, with_meta = False):
  if with_meta:
    X_train, X_test, U_train, U_test, y_train, y_test = dataset_obj.getTrainTestSplit(with_meta = True)
    return pd.concat(
      [
        pd.concat([X_train, U_train], axis = 1),
        pd.concat([X_test, U_test], axis = 1),
      ],
      axis = 0
    )[:num_samples]
  else:
    X_train, X_test, y_train, y_test = dataset_obj.getTrainTestSplit()
    return pd.concat([X_train, X_test], axis = 0)[:num_samples]


def getNoiseStringForNode(node):
  assert node[0] == 'x'
  return 'u' + node[1:]


def prettyPrintDict(my_dict):
  my_dict = my_dict.copy()
  for key, value in my_dict.items():
    my_dict[key] = np.around(value, 3)
  return my_dict


def getPredictionBatch(scm_obj, dataset_obj, classifier_obj, instances_df):
  sklearn_model = classifier_obj
  return sklearn_model.predict(instances_df)


def getPrediction(scm_obj, dataset_obj, classifier_obj, instance):
  sklearn_model = classifier_obj
  prediction = sklearn_model.predict(np.array(list(instance.values())).reshape(1,-1))[0]
  assert prediction in {0, 1}, f'Expected prediction in {0,1}; got {prediction}'
  return prediction


def didFlip(scm_obj, dataset_obj, classifier_obj, factual_instance, counterfactual_instance):
  return \
    getPrediction(scm_obj, dataset_obj, classifier_obj, factual_instance) != \
    getPrediction(scm_obj, dataset_obj, classifier_obj, counterfactual_instance)


@utils.Memoize
def trainRidge(dataset_obj, node, parents):
  assert len(parents) > 0, 'parents set cannot be empty.'
  print(f'\t[INFO] Fitting p({node} | {", ".join(parents)}) using Ridge on {NUM_TRAIN_SAMPLES} samples; this may be very expensive, memoizing afterwards.')
  X_all = processDataFrameOrDict(dataset_obj, getOriginalDataFrame(), 'standardize')
  param_grid = {'alpha': np.logspace(-2, 1, 10)}
  model = GridSearchCV(Ridge(), param_grid=param_grid)
  model.fit(X_all[parents], X_all[[node]])
  return model


@utils.Memoize
def trainKernelRidge(dataset_obj, node, parents):
  assert len(parents) > 0, 'parents set cannot be empty.'
  print(f'\t[INFO] Fitting p({node} | {", ".join(parents)}) using KernelRidge on {NUM_TRAIN_SAMPLES} samples; this may be very expensive, memoizing afterwards.')
  X_all = processDataFrameOrDict(dataset_obj, getOriginalDataFrame(), 'standardize')
  param_grid = {
    'alpha': np.logspace(-2, 1, 5),
    'kernel': [
      RBF(lengthscale)
      for lengthscale in np.logspace(-2, 1, 5)
    ]
  }
  model = GridSearchCV(KernelRidge(), param_grid=param_grid)
  model.fit(X_all[parents], X_all[[node]])
  return model


@utils.Memoize
def trainCVAE(dataset_obj, node, parents):
  assert len(parents) > 0, 'parents set cannot be empty.'
  print(f'\t[INFO] Fitting p({node} | {", ".join(parents)}) using CVAE on {NUM_TRAIN_SAMPLES} samples; this may be very expensive, memoizing afterwards.')
  X_all = processDataFrameOrDict(dataset_obj, getOriginalDataFrame(num_samples = int(NUM_TRAIN_SAMPLES * 1.2)), 'raw')

  # if scm_obj.scm_class == 'sanity-2-add':
  #   if NUM_TRAIN_SAMPLES == 5000:
  #     lambda_kld = 0.1
  #     encoder_layer_sizes = [1, 3, 3]
  #     decoder_layer_sizes = [2, 2, 1]
  #   elif NUM_TRAIN_SAMPLES == 1000:
  #     lambda_kld = 0.5
  #     encoder_layer_sizes = [1, 3, 3]
  #     decoder_layer_sizes = [2, 1]
  # elif scm_obj.scm_class == 'sanity-2-sig-add':
  #   if NUM_TRAIN_SAMPLES == 5000:
  #     lambda_kld = 0.1
  #     encoder_layer_sizes = [1, 3, 3]
  #     decoder_layer_sizes = [2, 2, 1]
  #   elif NUM_TRAIN_SAMPLES == 1000:
  #     lambda_kld = 0.5
  #     encoder_layer_sizes = [1, 3, 3]
  #     decoder_layer_sizes = [2, 1]
  # elif scm_obj.scm_class ==  'sanity-2-pow-add':
  #   if NUM_TRAIN_SAMPLES == 5000:
  #     lambda_kld = 0.5
  #     encoder_layer_sizes = [1, 3, 3]
  #     decoder_layer_sizes = [2, 2, 1]
  #   elif NUM_TRAIN_SAMPLES == 1000:
  #     lambda_kld = 0.5
  #     encoder_layer_sizes = [1, 3, 3]
  #     decoder_layer_sizes = [2, 1]
  # elif scm_obj.scm_class ==  'sanity-2-mult':
  #   # if NUM_TRAIN_SAMPLES == 5000:
  #   #   lambda_kld = 0.1
  #   #   encoder_layer_sizes = [1, 3, 3]
  #   #   decoder_layer_sizes = [2, 2, 1]
  #   if NUM_TRAIN_SAMPLES == 1000:
  #     lambda_kld = 0.5
  #     encoder_layer_sizes = [1, 3, 3]
  #     decoder_layer_sizes = [2, 1]
  # elif scm_obj.scm_class ==  'sanity-2-add-pow':
  #   # if NUM_TRAIN_SAMPLES == 5000:
  #   #   lambda_kld = 0.5
  #   #   encoder_layer_sizes = [1, 3, 3]
  #   #   decoder_layer_sizes = [2, 2, 1]
  #   if NUM_TRAIN_SAMPLES == 1000:
  #     lambda_kld = 0.5
  #     encoder_layer_sizes = [1, 3, 3]
  #     decoder_layer_sizes = [2, 1]

  lambda_kld = 0.5
  encoder_layer_sizes = [1, 3, 3]
  decoder_layer_sizes = [2, 1]

  return train_cvae(AttrDict({
    'name': f'p({node} | {", ".join(parents)})',
    'node_train': X_all[[node]].iloc[:NUM_TRAIN_SAMPLES],
    'parents_train': X_all[parents].iloc[:NUM_TRAIN_SAMPLES],
    'node_valid': X_all[[node]].iloc[NUM_TRAIN_SAMPLES:],
    'parents_valid': X_all[parents].iloc[NUM_TRAIN_SAMPLES:],
    'seed': 0,
    'epochs': 100,
    'batch_size': 128,
    'learning_rate': 0.05,
    'lambda_kld': lambda_kld,
    'encoder_layer_sizes': encoder_layer_sizes, # 1 b/c the X_all[[node]] is always 1 dimensional # TODO: add support for categorical variables
    'decoder_layer_sizes': decoder_layer_sizes, # 1 b/c the X_all[[node]] is always 1 dimensional # TODO: add support for categorical variables
    'latent_size': 1,
    'conditional': True,
    'debug_folder': experiment_folder_name,
  }))


@utils.Memoize
def trainGP(dataset_obj, node, parents):
  assert len(parents) > 0, 'parents set cannot be empty.'
  print(f'\t[INFO] Fitting p({node} | {", ".join(parents)}) using GP on {NUM_TRAIN_SAMPLES} samples; this may be very expensive, memoizing afterwards.')
  X_all = processDataFrameOrDict(dataset_obj, getOriginalDataFrame(), 'raw')
  kernel = GPy.kern.RBF(input_dim=len(parents), ARD=True)
  model = GPy.models.GPRegression(X_all[parents], X_all[[node]], kernel)
  model.optimize_restarts(parallel=True, num_restarts=5, verbose=False)
  X = X_all[parents].to_numpy()
  return kernel, X, model


def _getAbductionNoise(scm_obj, dataset_obj, classifier_obj, node, parents, factual_instance, structural_equation):
  # only applies for ANM models
  return factual_instance[node] - structural_equation(
    0,
    *[factual_instance[parent] for parent in parents],
  )


def sampleTrue(scm_obj, dataset_obj, classifier_obj, samples_df, factual_instance, node, parents, recourse_type):
  # Step 1. [abduction]: compute noise or load from dataset using factual_instance
  # Step 2. [action]: (skip) this step is implicitly performed in the populated samples_df columns
  # Step 3. [prediction]: run through structural equation using noise and parents from samples_df
  structural_equation = scm_obj.structural_equations[node]

  if recourse_type == 'm0_true':

    noise_pred = _getAbductionNoise(scm_obj, dataset_obj, classifier_obj, node, parents, factual_instance, structural_equation)
    # XU_all = getOriginalDataFrame(with_meta = True)
    # tmp_idx = getIndexOfFactualInstanceInDataFrame(factual_instance, XU_all)
    # noise_true = XU_all.iloc[tmp_idx][getNoiseStringForNode(node)]
    # # print(f'noise_pred: {noise_pred:.8f} \t noise_true: {noise_true:.8f} \t difference: {np.abs(noise_pred - noise_true):.8f}')

    # # noise_pred assume additive noise, and therefore only works with
    # # models such as 'm1_alin' and 'm1_akrr' in general cases
    # if recourse_type == 'm0_true':
    #   if scm_obj.scm_class != 'sanity-power':
    #     assert np.abs(noise_pred - noise_true) < 1e-5, 'Noise {pred, true} expected to be similar, but not.'
    #   noise = noise_true
    # else:
    #   noise = noise_pred
    noise = noise_pred

    for row_idx, row in samples_df.iterrows():
      noise = noise
      samples_df.loc[row_idx, node] = structural_equation(
        noise,
        *samples_df.loc[row_idx, parents].to_numpy(),
      )

  elif recourse_type == 'm2_true':

    for row_idx, row in samples_df.iterrows():
      noise = scm_obj.noises_distributions[getNoiseStringForNode(node)].sample(),
      samples_df.loc[row_idx, node] = structural_equation(
        noise,
        *samples_df.loc[row_idx, parents].to_numpy(),
      )

  return samples_df


def _sampleRidgeKernelRidge(scm_obj, dataset_obj, classifier_obj, samples_df, factual_instance, node, parents, recourse_type, train_handle):
  # XU_all = getOriginalDataFrame(with_meta = True)
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
  noise_pred = _getAbductionNoise(scm_obj, dataset_obj, classifier_obj, node, parents, factual_instance, structural_equation)
  # print(f'noise_pred: {noise_pred:.8f} \t noise_true: {noise_true:.8f} \t difference: {np.abs(noise_pred - noise_true):.8f}') # TODO: check this out... why differnce?
  for row_idx, row in samples_df.iterrows():
    noise = _getAbductionNoise(scm_obj, dataset_obj, classifier_obj, node, parents, factual_instance, structural_equation),
    samples_df.loc[row_idx, node] = structural_equation(
      noise,
      *samples_df.loc[row_idx, parents].to_numpy(),
    )
  samples_df = deprocessDataFrameOrDict(dataset_obj, samples_df, 'standardize')
  return samples_df


def sampleRidge(scm_obj, dataset_obj, classifier_obj, samples_df, factual_instance, node, parents, recourse_type):
  return _sampleRidgeKernelRidge(scm_obj, dataset_obj, classifier_obj, samples_df, factual_instance, node, parents, recourse_type, trainRidge)


def sampleKernelRidge(scm_obj, dataset_obj, classifier_obj, samples_df, factual_instance, node, parents, recourse_type):
  return _sampleRidgeKernelRidge(scm_obj, dataset_obj, classifier_obj, samples_df, factual_instance, node, parents, recourse_type, trainKernelRidge)


def sampleCVAE(scm_obj, dataset_obj, classifier_obj, samples_df, factual_instance, node, parents, recourse_type):
  # All samplers EXCEPT FOR sampleTrue have been trained (and should be sampled) on standardized data
  samples_df = processDataFrameOrDict(dataset_obj, samples_df.copy(), 'raw')
  factual_instance = processDataFrameOrDict(dataset_obj, factual_instance.copy(), 'raw')

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
    sample_from = 'posterior'
  elif recourse_type == 'm2_cvae':
    sample_from = 'prior'
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
  samples_df = deprocessDataFrameOrDict(dataset_obj, samples_df, 'raw')
  return samples_df


def sampleGP(scm_obj, dataset_obj, classifier_obj, samples_df, factual_instance, node, parents, recourse_type):
  # # All samplers EXCEPT FOR sampleTrue have been trained (and should be sampled) on standardized data
  samples_df = processDataFrameOrDict(dataset_obj, samples_df.copy(), 'raw')
  factual_instance = processDataFrameOrDict(dataset_obj, factual_instance.copy(), 'raw')

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

  if recourse_type == 'm1_gaus': # counterfactual distribution for node
    # IMPORTANT: Find index of factual instance in dataframe used for training GP
    #            (earlier, the factual instance was appended as the last instance)
    tmp_idx = getIndexOfFactualInstanceInDataFrame(
      factual_instance,
      processDataFrameOrDict(dataset_obj, getOriginalDataFrame(), 'raw'),
    )
    new_means = pred_means + noise_post_means[tmp_idx]
    new_vars = pred_vars + noise_post_vars[tmp_idx]
  elif recourse_type == 'm2_gaus': # interventional distribution for node
    new_means = pred_means + 0
    new_vars = pred_vars + sigma_noise

  # sample from distribution via reparametrisation trick
  new_noise = np.random.randn(samples_df.shape[0], 1)
  new_samples = new_means + np.sqrt(new_vars) * new_noise

  samples_df[node] = new_samples
  samples_df = deprocessDataFrameOrDict(dataset_obj, samples_df, 'raw')
  return samples_df


def _samplingInnerLoop(scm_obj, dataset_obj, classifier_obj, factual_instance, action_set, recourse_type, num_samples):

  counterfactual_template = dict.fromkeys(
    dataset_obj.getInputAttributeNames(),
    np.NaN,
  )

  # get intervention and conditioning sets
  intervention_set = set(action_set.keys())

  # intersection_of_non_descendents_of_intervened_upon_variables
  conditioning_set = set.intersection(*[
    scm_obj.getNonDescendentsForNode(node)
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
  for node in scm_obj.getTopologicalOrdering():
    # set variable if value not yet set through intervention or conditioning
    if samples_df[node].isnull().values.any():
      parents = scm_obj.getParentsForNode(node)
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
        scm_obj,
        dataset_obj,
        classifier_obj,
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


def computeCounterfactualInstance(scm_obj, dataset_obj, classifier_obj, factual_instance, action_set, recourse_type):

  assert recourse_type in ACCEPTABLE_POINT_RECOURSE

  if not bool(action_set): # if action_set is empty, CFE = F
    return factual_instance

  samples_df = _samplingInnerLoop(scm_obj, dataset_obj, classifier_obj, factual_instance, action_set, recourse_type, 1)

  return samples_df.loc[0].to_dict() # return the first instance


def getRecourseDistributionSample(scm_obj, dataset_obj, classifier_obj, factual_instance, action_set, recourse_type, num_samples):

  assert recourse_type in ACCEPTABLE_DISTR_RECOURSE

  if not bool(action_set): # if action_set is empty, CFE = F
    return pd.DataFrame(dict(zip(
      dataset_obj.getInputAttributeNames(),
      [num_samples * [factual_instance[node]] for node in dataset_obj.getInputAttributeNames()],
    )))

  samples_df = _samplingInnerLoop(scm_obj, dataset_obj, classifier_obj, factual_instance, action_set, recourse_type, num_samples)

  return samples_df # return the entire data frame


def isPointConstraintSatisfied(scm_obj, dataset_obj, classifier_obj, factual_instance, action_set, recourse_type):
  return didFlip(
    scm_obj,
    dataset_obj,
    classifier_obj,
    factual_instance,
    computeCounterfactualInstance(
      scm_obj,
      dataset_obj,
      classifier_obj,
      factual_instance,
      action_set,
      recourse_type,
    ),
  )


def isDistrConstraintSatisfied(scm_obj, dataset_obj, classifier_obj, factual_instance, action_set, recourse_type):
  return computeLowerConfidenceBound(scm_obj, dataset_obj, classifier_obj, factual_instance, action_set, recourse_type) >= 0.5


def computeLowerConfidenceBound(scm_obj, dataset_obj, classifier_obj, factual_instance, action_set, recourse_type):
  monte_carlo_samples_df = getRecourseDistributionSample(
    scm_obj,
    dataset_obj,
    classifier_obj,
    factual_instance,
    action_set,
    recourse_type,
    NUM_MONTE_CARLO_SAMPLES,
  )
  monte_carlo_predictions = getPredictionBatch(
    scm_obj,
    dataset_obj,
    classifier_obj,
    monte_carlo_samples_df.to_numpy(),
  )

  expectation = np.mean(monte_carlo_predictions)
  variance = np.sum(np.power(monte_carlo_predictions - expectation, 2)) / (len(monte_carlo_predictions) - 1)

  # return expectation, variance

  # IMPORTANT... WE ARE CONSIDERING {0,1} LABELS AND FACTUAL SAMPLES MAY BE OF
  # EITHER CLASS. THEREFORE, THE CONSTRAINT IS SATISFIED WHEN SIGNIFICANTLY
  # > 0.5 OR < 0.5 FOR A FACTUAL SAMPLE WITH Y = 0 OR Y = 1, RESPECTIVELY.

  if getPrediction(scm_obj, dataset_obj, classifier_obj, factual_instance) == 0:
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

      raise NotImplementedError # TODO: add support for categorical variables

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


def computeOptimalActionSet(scm_obj, dataset_obj, classifier_obj, factual_instance, recourse_type, optimization_approach):

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
      if constraint_handle(scm_obj, dataset_obj, classifier_obj, factual_instance, action_set, recourse_type):
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


def scatterDecisionBoundary(scm_obj, dataset_obj, classifier_obj, ax):
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


def scatterRecourse(scm_obj, dataset_obj, classifier_obj, factual_instance, action_set, recourse_type, marker_type, legend_label, ax):

  assert len(dataset_obj.getInputAttributeNames()) == 3

  if recourse_type in ACCEPTABLE_POINT_RECOURSE:
    # point recourse

    point = computeCounterfactualInstance(scm_obj, dataset_obj, classifier_obj, factual_instance, action_set, recourse_type)
    color_string = 'green' if didFlip(scm_obj, dataset_obj, classifier_obj, factual_instance, point) else 'red'
    ax.scatter(point['x1'], point['x2'], point['x3'], marker=marker_type, color=color_string, s=70, label=legend_label)

  elif recourse_type in ACCEPTABLE_DISTR_RECOURSE:
    # distr recourse

    samples_df = getRecourseDistributionSample(scm_obj, dataset_obj, classifier_obj, factual_instance, action_set, recourse_type, NUM_DISPLAY_SAMPLES)
    x1s = samples_df.iloc[:,0]
    x2s = samples_df.iloc[:,1]
    x3s = samples_df.iloc[:,2]
    color_strings = ['green' if didFlip(scm_obj, dataset_obj, classifier_obj, factual_instance, sample.to_dict()) else 'red' for _, sample in samples_df.iterrows()]
    ax.scatter(x1s, x2s, x3s, marker=marker_type, color=color_strings, alpha=0.1, s=30, label=legend_label)

    # mean_distr_samples = {
    #   'x1': np.mean(samples_df['x1']),
    #   'x2': np.mean(samples_df['x2']),
    #   'x3': np.mean(samples_df['x3']),
    # }
    # color_string = 'green' if didFlip(scm_obj, dataset_obj, classifier_obj, factual_instance, mean_distr_samples) else 'red'
    # ax.scatter(mean_distr_samples['x1'], mean_distr_samples['x2'], mean_distr_samples['x3'], marker=marker_type, color=color_string, alpha=0.5, s=70, label=legend_label)

  else:

    raise Exception(f'{recourse_type} not recognized.')


def getNegativelyPredictedInstances(scm_obj, dataset_obj, classifier_obj):
  # Samples for which we seek recourse are chosen from the joint of X_train/test.
  # This is OK because the tasks of conditional density estimation and recourse
  # generation are distinct. Given the same data splicing used here and in trainGP,
  # it is guaranteed that we the factual sample for which we seek recourse is in
  # training set for GP, and hence a posterior over noise for it is computed
  # (i.e., we can cache).

  # Only focus on instances with h(x^f) = 0 and therfore h(x^cf) = 1
  X_all = getOriginalDataFrame()
  factual_instances_dict = {}
  tmp_counter = 1
  for factual_instance_idx, row in X_all.iterrows():
    factual_instance = row.T.to_dict()
    if getPrediction(scm_obj, dataset_obj, classifier_obj, factual_instance) == 0:
      tmp_counter += 1
      factual_instances_dict[factual_instance_idx] = factual_instance
    if tmp_counter > NUM_RECOURSE_SAMPLES:
      break
  return factual_instances_dict


def hotTrainRecourseTypes(scm_obj, dataset_obj, classifier_obj, recourse_types):
  start_time = time.time()
  print(f'\n' + '='*80 + '\n')
  print(f'[INFO] Hot-training ALIN, AKRR, GAUS, CVAE so they do not affect runtime...')
  training_handles = []
  if any(['alin' in elem for elem in recourse_types]): training_handles.append(trainRidge)
  if any(['akrr' in elem for elem in recourse_types]): training_handles.append(trainKernelRidge)
  if any(['gaus' in elem for elem in recourse_types]): training_handles.append(trainGP)
  if any(['cvae' in elem for elem in recourse_types]): training_handles.append(trainCVAE)
  for training_handle in training_handles:
    print()
    for node in scm_obj.getTopologicalOrdering():
      parents = scm_obj.getParentsForNode(node)
      if len(parents): # if not a root node
        training_handle(dataset_obj, node, parents)
  end_time = time.time()
  print(f'\n[INFO] Done (total warm-up time: {end_time - start_time}).')
  print(f'\n' + '='*80 + '\n')


# DEPRECATED def experiment1


# DEPRECATED def experiment2


# DEPRECATED def experiment3


# DEPRECATED def experiment4


def experiment5(scm_obj, dataset_obj, classifier_obj, experiment_folder_name, factual_instances_dict, experimental_setups, recourse_types):
  ''' fixed action set: assert {m1, m2} x {gaus, cvae} working '''

  assert len(dataset_obj.getInputAttributeNames()) == 3, 'Exp 5 is only designed for 3-variable SCMs'

  print(f'Describe original data:\n{getOriginalDataFrame().describe()}')

  # action_sets = [
  #   {'x1': scm_obj.noises_distributions['u1'].sample()}
  #   for _ in range(4)
  # ]
  range_x1 = dataset_obj.data_frame_kurz.describe()['x1']
  action_sets = [
    {'x1': value_x1}
    for value_x1 in np.linspace(range_x1['min'], range_x1['max'], 9)
  ]

  factual_instance = factual_instances_dict[list(factual_instances_dict.keys())[0]]

  fig, axes = pyplot.subplots(
    int(np.sqrt(len(action_sets))),
    int(np.sqrt(len(action_sets))),
    # tight_layout=True,
    # sharex='row',
    # sharey='row',
  )
  fig.suptitle(f'FC: {prettyPrintDict(factual_instance)}', fontsize='x-small')
  if len(action_sets) == 1:
    axes = np.array(axes) # weird hack we need to use so to later use flatten()

  print(f'\nFC: \t\t{prettyPrintDict(factual_instance)}')

  for idx, action_set in enumerate(action_sets):

    print(f'\n\n[INFO] ACTION SET: {str(action_set)}' + ' =' * 40)

    for experimental_setup in experimental_setups:
      recourse_type, marker = experimental_setup[0], experimental_setup[1]

      if recourse_type in ACCEPTABLE_POINT_RECOURSE:
        sample = computeCounterfactualInstance(scm_obj, dataset_obj, classifier_obj, factual_instance, action_set, recourse_type)
        print(f'{recourse_type}:\t{prettyPrintDict(sample)}')
        axes.flatten()[idx].plot(sample['x2'], sample['x3'], marker, alpha=1.0, markersize = 7, label=recourse_type)
      elif recourse_type in ACCEPTABLE_DISTR_RECOURSE:
        samples = getRecourseDistributionSample(scm_obj, dataset_obj, classifier_obj, factual_instance, action_set, recourse_type, NUM_DISPLAY_SAMPLES)
        print(f'{recourse_type}:\n{samples.head()}')
        axes.flatten()[idx].plot(samples['x2'], samples['x3'], marker, alpha=0.3, markersize = 4, label=recourse_type)
      else:
        raise Exception(f'{recourse_type} not supported.')

    axes.flatten()[idx].set_xlabel('$x2$', fontsize='x-small')
    axes.flatten()[idx].set_ylabel('$x3$', fontsize='x-small')
    axes.flatten()[idx].tick_params(axis='both', which='major', labelsize=6)
    axes.flatten()[idx].tick_params(axis='both', which='minor', labelsize=4)
    axes.flatten()[idx].set_title(f'action_set: {str(prettyPrintDict(action_set))}', fontsize='x-small')

  # for ax in axes.flatten():
  #   ax.legend(fontsize='xx-small')

  # handles, labels = axes.flatten()[-1].get_legend_handles_labels()
  # # https://stackoverflow.com/a/43439132/2759976
  # fig.legend(handles, labels, bbox_to_anchor=(1.04, 0.5), loc='center left', fontsize='x-small')

  # https://riptutorial.com/matplotlib/example/10473/single-legend-shared-across-multiple-subplots
  handles, labels = axes.flatten()[-1].get_legend_handles_labels()
  fig.legend(
    handles=handles,
    labels=labels,        # The labels for each line
    loc="center right",   # Position of legend
    borderaxespad=0.1,    # Small spacing around legend box
    # title="Legend Title", # Title for the legend
    fontsize='xx-small',
  )
  fig.tight_layout()
  plt.subplots_adjust(right=0.85)
  # pyplot.show()
  pyplot.savefig(f'{experiment_folder_name}/comparison.pdf')


def experiment6(scm_obj, dataset_obj, classifier_obj, experiment_folder_name, factual_instances_dict, experimental_setups, recourse_types):
  ''' optimal action set: figure + table '''

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
        scm_obj,
        classifier_obj,
        factual_instance,
        recourse_type,
        'brute_force',
      )
      end_time = time.time()

      tmp['runtime'] = np.around(end_time - start_time, 3)

      # print(f'\t[INFO] Computing SCF validity and Interventional Confidence measures for optimal action `{str(tmp["optimal_action_set"])}`...')

      tmp['scf_validity']  = isPointConstraintSatisfied(scm_obj, dataset_obj, classifier_obj, factual_instance, tmp['optimal_action_set'], 'm0_true')
      tmp['int_conf_true'] = np.around(computeLowerConfidenceBound(scm_obj, dataset_obj, classifier_obj, factual_instance, tmp['optimal_action_set'], 'm2_true'), 3)
      tmp['int_conf_cvae'] = np.around(computeLowerConfidenceBound(scm_obj, dataset_obj, classifier_obj, factual_instance, tmp['optimal_action_set'], 'm2_cvae'), 3)
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
  #     scatterRecourse(scm_obj, dataset_obj, classifier_obj, factual_instance, optimal_action_set, 'm0_true', marker, legend_label, ax)
  #     # scatterRecourse(scm_obj, dataset_obj, classifier_obj, factual_instance, optimal_action_set, recourse_type, marker, legend_label, ax)

  #   scatterDecisionBoundary(scm_obj, dataset_obj, classifier_obj, ax)
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


def experiment7(scm_obj, dataset_obj, classifier_obj, experiment_folder_name, factual_instances_dict, experimental_setups, recourse_types):
  ''' optimal action set: figure + table '''

  per_instance_results = {}
  for enumeration_idx, (key, value) in enumerate(factual_instances_dict.items()):
    factual_instance_idx = f'sample_{key}'
    factual_instance = value

    print(f'\n\n\n[INFO] Processing factual instance `{factual_instance_idx}` (#{enumeration_idx + 1} / {len(factual_instances_dict.keys())})...')

    per_instance_results[factual_instance_idx] = {}

    for recourse_type in recourse_types:

      per_instance_results[factual_instance_idx][recourse_type] = {}
      per_instance_results[factual_instance_idx][recourse_type]['pred_x2_on_true_x1'] = []
      per_instance_results[factual_instance_idx][recourse_type]['pred_x3_on_true_x1_pred_x2'] = []
      per_instance_results[factual_instance_idx][recourse_type]['pred_x3_on_true_x1_true_x2'] = []

      num_samples = NUM_MONTE_CARLO_SAMPLES
      testing_template = {
        'x1': factual_instance['x1'],
        'x2': np.NaN,
        'x3': np.NaN,
      }
      # this dataframe has populated columns set to intervention or conditioning values
      # and has NaN columns that will be set accordingly.
      samples_df = pd.DataFrame(dict(zip(
        dataset_obj.getInputAttributeNames(),
        [num_samples * [testing_template[node]] for node in dataset_obj.getInputAttributeNames()],
      )))
      samples_df = tmpp(scm_obj, dataset_obj, classifier_obj, factual_instance, recourse_type, samples_df)
      per_instance_results[factual_instance_idx][recourse_type]['pred_x2_on_true_x1'] = list(samples_df['x2'])
      per_instance_results[factual_instance_idx][recourse_type]['pred_x3_on_true_x1_pred_x2'] = list(samples_df['x3'])

      num_samples = NUM_MONTE_CARLO_SAMPLES
      testing_template = {
        'x1': factual_instance['x1'],
        'x2': factual_instance['x2'],
        'x3': np.NaN,
      }
      # this dataframe has populated columns set to intervention or conditioning values
      # and has NaN columns that will be set accordingly.
      samples_df = pd.DataFrame(dict(zip(
        dataset_obj.getInputAttributeNames(),
        [num_samples * [testing_template[node]] for node in dataset_obj.getInputAttributeNames()],
      )))
      samples_df = tmpp(scm_obj, dataset_obj, classifier_obj, factual_instance, recourse_type, samples_df)
      per_instance_results[factual_instance_idx][recourse_type]['pred_x3_on_true_x1_true_x2'] = list(samples_df['x3'])

  tmp_setups = [
    ('x2','pred_x2_on_true_x1'),
    ('x3','pred_x3_on_true_x1_true_x2'),
    ('x3','pred_x3_on_true_x1_pred_x2'),
  ]

  # ============================================================================
  # SANITY 1
  # ============================================================================
  fig, axes = pyplot.subplots(len(tmp_setups), len(recourse_types), sharex='row', sharey=True, tight_layout=True)
  fig.suptitle(f'Comparison of recon error (2-rorm) for all regressor/conditionals', fontsize='x-small')

  for idx_1, (node, node_string) in enumerate(tmp_setups):
    for idx_2, recourse_type in enumerate(recourse_types):
      mse_list_for_recourse_type = [
        np.linalg.norm([sample - factual_instance[node]], 2)
        for sample in per_instance_results[factual_instance_idx][recourse_type][node_string]
        for factual_instance_idx in per_instance_results.keys()
      ]
      tmp_idx = idx_1 * len(recourse_types) + idx_2
      axes.flatten()[tmp_idx].hist(mse_list_for_recourse_type, bins=30)
      if tmp_idx % len(recourse_types) == 0: # if first subplot of row
        axes.flatten()[tmp_idx].set_ylabel(node_string, fontsize='xx-small')
      axes.flatten()[tmp_idx].tick_params(axis='both', which='major', labelsize=6)
      axes.flatten()[tmp_idx].tick_params(axis='both', which='minor', labelsize=4)
      axes.flatten()[tmp_idx].set_title(f'{recourse_type}', fontsize='xx-small')

  fig.tight_layout()
  pyplot.savefig(f'{experiment_folder_name}/_sanity_1.pdf')

  # ============================================================================
  # SANITY 2
  # ============================================================================
  tmp_recourse_types = [elem for elem in recourse_types if 'm2' in elem]

  fig, axes = pyplot.subplots(len(tmp_recourse_types), len(tmp_setups), sharex='col', sharey=True, tight_layout=True)
  fig.suptitle(f'Comparison of M2 smaples', fontsize='x-small')

  for idx_1, (node, node_string) in enumerate(tmp_setups):
    for idx_2, recourse_type in enumerate(tmp_recourse_types):
      m2_samples_for_recourse_type = [
        sample
        for sample in per_instance_results[factual_instance_idx][recourse_type][node_string]
        for factual_instance_idx in per_instance_results.keys()
      ]
      # print(f'\n{recourse_type}', m2_samples_for_recourse_type)
      tmp_idx = idx_1 + idx_2 * len(tmp_setups)
      axes.flatten()[tmp_idx].hist(m2_samples_for_recourse_type, bins=30)
      if tmp_idx % len(tmp_setups) == 0: # if first subplot of row
        axes.flatten()[tmp_idx].set_ylabel(recourse_type, fontsize='xx-small')
      if tmp_idx >= (len(tmp_recourse_types)-1) * len(tmp_setups): # if last subplot of col
        axes.flatten()[tmp_idx].set_xlabel(node_string, fontsize='xx-small')
      axes.flatten()[tmp_idx].tick_params(axis='both', which='major', labelsize=6)
      axes.flatten()[tmp_idx].tick_params(axis='both', which='minor', labelsize=4)
      # axes.flatten()[tmp_idx].set_title(f'{recourse_type}', fontsize='xx-small')

  fig.tight_layout()
  pyplot.savefig(f'{experiment_folder_name}/_sanity_2.pdf')


def tmpp(scm_obj, dataset_obj, classifier_obj, factual_instance, recourse_type, samples_df):
  # Simply traverse the graph in order, and populate nodes as we go!
  # IMPORTANT: DO NOT USE set(topo ordering); it sometimes changes ordering!
  for node in scm_obj.getTopologicalOrdering():
    # set variable if value not yet set through intervention or conditioning
    if samples_df[node].isnull().values.any():
      parents = scm_obj.getParentsForNode(node)
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
        scm_obj,
        dataset_obj,
        classifier_obj,
        samples_df,
        factual_instance,
        node,
        parents,
        recourse_type,
      )
  return samples_df


def experiment8(scm_obj, dataset_obj, classifier_obj, experiment_folder_name, factual_instances_dict, experimental_setups, recourse_types):
  ''' optimal action set: figure + table '''

  assert len(dataset_obj.getInputAttributeNames()) == 2, 'Exp 8 is only designed for 2-variable SCMs'
  assert np.all(['m2' in elem for elem in recourse_types]), 'Exp 8 is only designed for m2 recourse_types'

  per_value_x1_results = {}

  X_all = processDataFrameOrDict(dataset_obj, getOriginalDataFrame(num_samples = NUM_TRAIN_SAMPLES * 2), 'raw')

  range_x1 = dataset_obj.data_frame_kurz.describe()['x1']
  for value_x1 in np.linspace(range_x1['min'], range_x1['max'], 10):
  # for value_x1 in np.linspace(range_x1['25%'], range_x1['75%'], 5):
    value_x1 = np.around(value_x1, 2)

    per_value_x1_results[value_x1] = {}

    factual_instance = {'x1': value_x1, 'x2': -1} # TODO: this -1 should not matter

    print(f'\n\n\n[INFO] Processing factual instance `{prettyPrintDict(factual_instance)}`...')

    for recourse_type in recourse_types:

      per_value_x1_results[value_x1][recourse_type] = []

      num_samples = NUM_MONTE_CARLO_SAMPLES
      testing_template = {
        'x1': factual_instance['x1'],
        'x2': np.NaN,
      }

      # this dataframe has populated columns set to intervention or conditioning values
      # and has NaN columns that will be set accordingly.
      samples_df = pd.DataFrame(dict(zip(
        dataset_obj.getInputAttributeNames(),
        [num_samples * [testing_template[node]] for node in dataset_obj.getInputAttributeNames()],
      )))
      samples_df = tmpp(scm_obj, dataset_obj, classifier_obj, factual_instance, recourse_type, samples_df)
      per_value_x1_results[value_x1][recourse_type] = list(samples_df['x2'])

  tmp = {}
  tmp['recourse_type'] = []
  tmp['value_x1'] = []
  tmp['sample_x2'] = []
  for k1,v1 in per_value_x1_results.items():
    for k2,v2 in v1.items():
      for elem in v2:
        tmp['recourse_type'].append(k2)
        tmp['value_x1'].append(k1)
        tmp['sample_x2'].append(elem)
  tmp = pd.DataFrame.from_dict(tmp)
  # ipsh()
  ax = sns.boxplot(x="value_x1", y="sample_x2", hue="recourse_type", data=tmp, palette="Set3", showmeans=True)
  # TODO: average over high dens pdf, and show a separate plot/table for the average over things...
  # ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
  # pyplot.show()
  pyplot.savefig(f'{experiment_folder_name}/_sanity_3.pdf')


def visualizeDatasetAndFixedModel(scm_obj, dataset_obj, classifier_obj):

  fig = pyplot.figure()
  ax = pyplot.subplot(1, 1, 1, projection='3d')

  scatterDataset(dataset_obj, ax)
  scatterDecisionBoundary(scm_obj, dataset_obj, classifier_obj, ax)

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
    '-s', '--scm_class',
    type = str,
    default = 'sanity-2-add',
    help = 'Name of SCM to generate data using (see loadSCM.py)')

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
  scm_class = args.scm_class
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
  scm_obj = loadCausalModel(scm_class, experiment_folder_name)
  dataset_obj = loadDataset(scm_class, dataset_class)
  classifier_obj = loadClassifier(dataset_class, classifier_class, experiment_folder_name)
  assert set(dataset_obj.getInputAttributeNames()) == set(scm_obj.getTopologicalOrdering())

  # setup
  factual_instances_dict = getNegativelyPredictedInstances(scm_obj, dataset_obj, classifier_obj)
  experimental_setups = [
    # ('m0_true', '*'), \
    # ('m1_alin', 'v'), \
    # ('m1_akrr', '^'), \
    # ('m1_gaus', 'D'), \
    # ('m1_cvae', 'x'), \
    ('m2_true', 'o'), \
    ('m2_gaus', 's'), \
    ('m2_cvae', '+'), \
    # ('m2_cvae_ps', 'P'), \
  ]
  recourse_types = [experimental_setup[0] for experimental_setup in experimental_setups]
  hotTrainRecourseTypes(scm_obj, dataset_obj, classifier_obj, recourse_types)

  # experiment5(scm_obj, dataset_obj, classifier_obj, experiment_folder_name, factual_instances_dict, experimental_setups, recourse_types)
  # experiment6(scm_obj, dataset_obj, classifier_obj, experiment_folder_name, factual_instances_dict, experimental_setups, recourse_types)
  # experiment7(scm_obj, dataset_obj, classifier_obj, experiment_folder_name, factual_instances_dict, experimental_setups, recourse_types)
  experiment8(scm_obj, dataset_obj, classifier_obj, experiment_folder_name, factual_instances_dict, experimental_setups, recourse_types)

  # sanity check
  # visualizeDatasetAndFixedModel(scm_obj, dataset_obj, classifier_obj)






# TODO:
# merge exp5,6,8
# to confirm training correct -->for each child/parent, save m2 comparison (regression, no intervention) + report MSE/VAR between m2 methods (good choice of hyperparms)
# (=? intervention on parent node given value of sweep?)
# show 9 interventions on parent
# show table





































