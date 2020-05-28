import os
import time
import torch
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
from matplotlib import pyplot as plt
from datetime import datetime
from attrdict import AttrDict

import GPy
import gpHelper
import mmd
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

ACCEPTABLE_POINT_RECOURSE = {'m0_true', 'm1_alin', 'm1_akrr'}
ACCEPTABLE_DISTR_RECOURSE = {'m1_gaus', 'm1_cvae', 'm2_true', 'm2_gaus', 'm2_cvae', 'm2_cvae_ps'}

# class Instance(object):
#   def __init__(self, endogenous_dict, exogenous_dict):

#     # assert()
#     self.endogenous_dict = endogenous_dict
#     self.exogenous_dict = exogenous_dict

PROCESSING_SKLEARN = 'standardize'
PROCESSING_GAUS = 'raw'
PROCESSING_CVAE = 'raw'

@utils.Memoize
def loadCausalModel(args, experiment_folder_name):
  return loadSCM.loadSCM(args.scm_class, experiment_folder_name)


@utils.Memoize
def loadDataset(args, experiment_folder_name):
  # unused: experiment_folder_name
  return loadData.loadDataset(args.dataset_class, return_one_hot = True, load_from_cache = False, meta_param = args.scm_class)


@utils.Memoize
def loadClassifier(args, experiment_folder_name):
  return loadModel.loadModelForDataset(args.classifier_class, args.dataset_class, experiment_folder_name)


@utils.Memoize
def getTorchClassifier(args, objs):
  fixed_model_w = objs.classifier_obj.coef_
  fixed_model_b = objs.classifier_obj.intercept_
  fixed_model = lambda x: torch.sigmoid(
      torch.nn.functional.linear(
        x,
        torch.from_numpy(fixed_model_w).float(),
    ) + float(fixed_model_b)
  )
  return fixed_model


def measureActionSetCost(args, objs, factual_instance, action_set):
  # TODO: add support for categorical data + measured in normalized space over all features

  ranges = objs.dataset_obj.getVariableRanges()
  if \
    np.all([isinstance(elem, float) for elem in factual_instance.values()]) and \
    np.all([isinstance(elem, float) for elem in action_set.values()]):
    deltas = [
      (action_set[key] - factual_instance[key]) / ranges[key]
      for key in action_set.keys()
    ]
    return np.linalg.norm(deltas, args.norm_type)
  elif \
    np.all([isinstance(elem, torch.Tensor) for elem in factual_instance.values()]) and \
    np.all([isinstance(elem, torch.Tensor) for elem in action_set.values()]):
    deltas = torch.stack([
      (action_set[key] - factual_instance[key]) / ranges[key]
      for key in action_set.keys()
    ])
    return torch.norm(deltas, p=args.norm_type)
  else:
    raise Exception(f'Mismatching or unsupport datatypes.')


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


def processDataFrameOrDict(args, objs, obj, processing_type):
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
    # use objs.dataset_obj stats, not X_train (in case you use more samples later,
    # e.g., validation set for cvae
    tmp = objs.dataset_obj.data_frame_kurz.describe()[node]
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


def deprocessDataFrameOrDict(args, objs, obj, processing_type):
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
    # use objs.dataset_obj stats, not X_train (in case you use more samples later,
    # e.g., validation set for cvae
    tmp = objs.dataset_obj.data_frame_kurz.describe()[node]
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
def getOriginalDataFrame(objs, num_samples, with_meta = False):
  if with_meta:
    X_train, X_test, U_train, U_test, y_train, y_test = objs.dataset_obj.getTrainTestSplit(with_meta = True)
    return pd.concat(
      [
        pd.concat([X_train, U_train], axis = 1),
        pd.concat([X_test, U_test], axis = 1),
      ],
      axis = 0
    )[:num_samples]
  else:
    X_train, X_test, y_train, y_test = objs.dataset_obj.getTrainTestSplit()
    return pd.concat([X_train, X_test], axis = 0)[:num_samples]


def getNoiseStringForNode(node):
  assert node[0] == 'x'
  return 'u' + node[1:]


def prettyPrintDict(my_dict):
  # use this for grad descent logs (convert tensor accordingly)
  my_dict = my_dict.copy()
  for key, value in my_dict.items():
    my_dict[key] = np.around(value, 3)
  return my_dict


def getPredictionBatch(args, objs, instances_df):
  sklearn_model = objs.classifier_obj
  return sklearn_model.predict(instances_df)


def getPrediction(args, objs, instance):
  sklearn_model = objs.classifier_obj
  prediction = sklearn_model.predict(np.array(list(instance.values())).reshape(1,-1))[0]
  assert prediction in {0, 1}, f'Expected prediction in {0,1}; got {prediction}'
  return prediction


def didFlip(args, objs, factual_instance, counterfactual_instance):
  return \
    getPrediction(args, objs, factual_instance) != \
    getPrediction(args, objs, counterfactual_instance)


def getConditionalString(node, parents):
  return f'p({node} | {", ".join(parents)})'


@utils.Memoize
def trainRidge(args, objs, node, parents):
  assert len(parents) > 0, 'parents set cannot be empty.'
  print(f'\t[INFO] Fitting {getConditionalString(node, parents)} using Ridge on {args.num_train_samples} samples; this may be very expensive, memoizing afterwards.')
  X_all = processDataFrameOrDict(args, objs, getOriginalDataFrame(objs, args.num_train_samples), PROCESSING_SKLEARN)
  param_grid = {'alpha': np.logspace(-2, 1, 10)}
  model = GridSearchCV(Ridge(), param_grid=param_grid)
  model.fit(X_all[parents], X_all[[node]])
  return model


@utils.Memoize
def trainKernelRidge(args, objs, node, parents):
  assert len(parents) > 0, 'parents set cannot be empty.'
  print(f'\t[INFO] Fitting {getConditionalString(node, parents)} using KernelRidge on {args.num_train_samples} samples; this may be very expensive, memoizing afterwards.')
  X_all = processDataFrameOrDict(args, objs, getOriginalDataFrame(objs, args.num_train_samples), PROCESSING_SKLEARN)
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
def trainCVAE(args, objs, node, parents):
  assert len(parents) > 0, 'parents set cannot be empty.'
  print(f'\t[INFO] Fitting {getConditionalString(node, parents)} using CVAE on {args.num_train_samples} samples; this may be very expensive, memoizing afterwards.')
  X_all = processDataFrameOrDict(args, objs, getOriginalDataFrame(objs, args.num_train_samples  + args.num_validation_samples), PROCESSING_CVAE)

  # sweep_lambda_kld = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
  sweep_lambda_kld = [0.5, 0.1, 0.05]
  # 1 b/c the X_all[[node]] is always 1 dimensional # TODO: add support for categorical variables
  sweep_encoder_layer_sizes = [
    [1, 3, 3],
    [1, 5, 5],
  ]
  sweep_decoder_layer_sizes = [
    [2, 1],
    [1 + len(parents), 1],
    [2, 2, 1],
    # [3, 3, 1],
    # [5, 5, 1],
  ]

  trained_models = {}

  all_hyperparam_setups = list(itertools.product(
    sweep_lambda_kld,
    sweep_encoder_layer_sizes,
    sweep_decoder_layer_sizes,
  ))

  for idx, hyperparams in enumerate(all_hyperparam_setups):

    print(f'\n\t[INFO] Training hyperparams setup #{idx} / {len(all_hyperparam_setups)}: {str(hyperparams)}')

    trained_cvae, recon_node_train, recon_node_validation = train_cvae(AttrDict({
      'name': f'{getConditionalString(node, parents)}',
      'node_train': X_all[[node]].iloc[:args.num_train_samples],
      'parents_train': X_all[parents].iloc[:args.num_train_samples],
      'node_validation': X_all[[node]].iloc[args.num_train_samples:],
      'parents_validation': X_all[parents].iloc[args.num_train_samples:],
      'seed': 0,
      'epochs': 100,
      'batch_size': 128,
      'learning_rate': 0.05,
      'lambda_kld': hyperparams[0],
      'encoder_layer_sizes': hyperparams[1],
      'decoder_layer_sizes': hyperparams[2],
      'latent_size': 1,
      'conditional': True,
      'debug_folder': experiment_folder_name + f'/cvae_hyperparams_setup_{idx}_of_{len(all_hyperparam_setups)}',
    }))

    # # TODO: remove after models.py is corrected
    return trained_cvae

    # run mmd to verify whether training is good or not (ON VALIDATION SET)
    X_val = X_all[args.num_train_samples:].copy()
    # POTENTIAL BUG? reset index here so that we can populate the `node` column
    # with reconstructed values from trained_cvae that lack indexing
    X_val = X_val.reset_index(drop = True)

    X_true = X_val[parents + [node]]

    X_pred_posterior = X_true.copy()
    X_pred_posterior[node] = pd.DataFrame(recon_node_validation.numpy(), columns=[node])

    not_imp_factual_instance = dict.fromkeys(objs.scm_obj.getTopologicalOrdering(), -1)
    not_imp_samples_df = X_true.copy()
    X_pred_prior = sampleCVAE(args, objs, not_imp_factual_instance, not_imp_samples_df, node, parents, 'm2_cvae', trained_cvae = trained_cvae)

    X_pred = X_pred_prior

    my_statistic, statistics, sigma_median = mmd.mmd_with_median_heuristic(X_true.to_numpy(), X_pred.to_numpy())
    print(f'\t\t[INFO] test-statistic = {my_statistic} using median of {sigma_median} as bandwith')

    trained_models[f'setup_{idx}'] = {}
    trained_models[f'setup_{idx}']['hyperparams'] = hyperparams
    trained_models[f'setup_{idx}']['trained_cvae'] = trained_cvae
    trained_models[f'setup_{idx}']['test-statistic'] = my_statistic

  index_with_lowest_test_statistics = min(trained_models.keys(), key=(lambda k: trained_models[k]['test-statistic']))
  model_with_lowest_test_statistics = trained_models[index_with_lowest_test_statistics]['trained_cvae']
  return model_with_lowest_test_statistics


@utils.Memoize
def trainGP(args, objs, node, parents):
  assert len(parents) > 0, 'parents set cannot be empty.'
  print(f'\t[INFO] Fitting {getConditionalString(node, parents)} using GP on {args.num_train_samples} samples; this may be very expensive, memoizing afterwards.')
  X_all = processDataFrameOrDict(args, objs, getOriginalDataFrame(objs, args.num_train_samples), PROCESSING_GAUS)

  kernel = GPy.kern.RBF(input_dim=len(parents), ARD=True)
  # IMPORTANT: do NOT use DataFrames, use Numpy arrays; GPy doesn't like DF.
  # https://github.com/SheffieldML/GPy/issues/781#issuecomment-532738155
  model = GPy.models.GPRegression(
    X_all[parents].to_numpy(),
    X_all[[node]].to_numpy(),
    kernel,
  )
  model.optimize_restarts(parallel=True, num_restarts=5, verbose=False)

  return kernel, X_all, model


def _getAbductionNoise(args, objs, node, parents, factual_instance, structural_equation):
  # only applies for ANM models
  return factual_instance[node] - structural_equation(
    0,
    *[factual_instance[parent] for parent in parents],
  )


def sampleTrue(args, objs, factual_instance, samples_df, node, parents, recourse_type):
  # Step 1. [abduction]: compute noise or load from dataset using factual_instance
  # Step 2. [action]: (skip) this step is implicitly performed in the populated samples_df columns
  # Step 3. [prediction]: run through structural equation using noise and parents from samples_df
  structural_equation = objs.scm_obj.structural_equations[node]

  if recourse_type == 'm0_true':

    noise_pred = _getAbductionNoise(args, objs, node, parents, factual_instance, structural_equation)
    # XU_all = getOriginalDataFrame(objs, args.num_train_samples, with_meta = True)
    # tmp_idx = getIndexOfFactualInstanceInDataFrame(factual_instance, XU_all)
    # noise_true = XU_all.iloc[tmp_idx][getNoiseStringForNode(node)]
    # # print(f'noise_pred: {noise_pred:.8f} \t noise_true: {noise_true:.8f} \t difference: {np.abs(noise_pred - noise_true):.8f}')

    # # noise_pred assume additive noise, and therefore only works with
    # # models such as 'm1_alin' and 'm1_akrr' in general cases
    noise = noise_pred
    # if SCM_CLASS != 'sanity-power':
    #   assert np.abs(noise_pred - noise_true) < 1e-5, 'Noise {pred, true} expected to be similar, but not.'
    #   noise = noise_true

    samples_df[node] = structural_equation(
      noise, # may be scalar, which will be case as pd.series when being summed.
      *[samples_df[parent] for parent in parents],
    )

  elif recourse_type == 'm2_true':

    samples_df[node] = structural_equation(
      objs.scm_obj.noises_distributions[getNoiseStringForNode(node)].sample(samples_df.shape[0]),
      *[samples_df[parent] for parent in parents],
    )

  return samples_df


def _sampleRidgeKernelRidge(args, objs, factual_instance, samples_df, node, parents, recourse_type, train_handle):
  samples_df = processDataFrameOrDict(args, objs, samples_df.copy(), PROCESSING_SKLEARN)
  factual_instance = processDataFrameOrDict(args, objs, factual_instance.copy(), PROCESSING_SKLEARN)

  # Step 1. [abduction]: compute noise or load from dataset using factual_instance
  # Step 2. [action]: (skip) this step is implicitly performed in the populated samples_df columns
  # Step 3. [prediction]: run through structural equation using noise and parents from samples_df
  trained_model = train_handle(args, objs, node, parents)
  structural_equation = lambda noise, *parents_values: trained_model.predict([[*parents_values]])[0][0] + noise
  for row_idx, row in samples_df.iterrows():
    noise = _getAbductionNoise(args, objs, node, parents, factual_instance, structural_equation),
    samples_df.loc[row_idx, node] = structural_equation(
      noise,
      *samples_df.loc[row_idx, parents].to_numpy(),
    )
  samples_df = deprocessDataFrameOrDict(args, objs, samples_df, PROCESSING_SKLEARN)
  return samples_df


def sampleRidge(args, objs, factual_instance, samples_df, node, parents, recourse_type):
  return _sampleRidgeKernelRidge(args, objs, factual_instance, samples_df, node, parents, recourse_type, trainRidge)


def sampleKernelRidge(args, objs, factual_instance, samples_df, node, parents, recourse_type):
  return _sampleRidgeKernelRidge(args, objs, factual_instance, samples_df, node, parents, recourse_type, trainKernelRidge)


def sampleCVAE(args, objs, factual_instance, samples_df, node, parents, recourse_type, trained_cvae = None):
  samples_df = processDataFrameOrDict(args, objs, samples_df.copy(), PROCESSING_CVAE)
  factual_instance = processDataFrameOrDict(args, objs, factual_instance.copy(), PROCESSING_CVAE)

  if trained_cvae is None: # UGLY CODE
    trained_cvae = trainCVAE(args, objs, node, parents)
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
  samples_df = deprocessDataFrameOrDict(args, objs, samples_df, PROCESSING_CVAE)
  return samples_df


def sampleGP(args, objs, factual_instance, samples_df, node, parents, recourse_type):
  samples_df = processDataFrameOrDict(args, objs, samples_df.copy(), PROCESSING_GAUS)
  factual_instance = processDataFrameOrDict(args, objs, factual_instance.copy(), PROCESSING_GAUS)

  kernel, X_all, model = trainGP(args, objs, node, parents)
  X_parents = torch.tensor(samples_df[parents].to_numpy())

  if recourse_type == 'm1_gaus': # counterfactual distribution for node
    # IMPORTANT: Find index of factual instance in dataframe used for training GP
    #            (earlier, the factual instance was appended as the last instance)
    tmp_idx = getIndexOfFactualInstanceInDataFrame(
      factual_instance,
      processDataFrameOrDict(args, objs, getOriginalDataFrame(objs, args.num_train_samples), PROCESSING_GAUS),
    ) # TODO: can probably rewrite to just evaluate the posterior again given the same result.. (without needing to look through the dataset)
    new_samples = gpHelper.sample_from_GP_model(model, X_parents, 'cf', tmp_idx)
  elif recourse_type == 'm2_gaus': # interventional distribution for node
    new_samples = gpHelper.sample_from_GP_model(model, X_parents, 'iv')

  samples_df[node] = new_samples.numpy()
  samples_df = deprocessDataFrameOrDict(args, objs, samples_df, PROCESSING_GAUS)
  return samples_df


def _getSamplesDFTemplate(args, objs, factual_instance, action_set, recourse_type, num_samples):
  counterfactual_template = dict.fromkeys(
    objs.dataset_obj.getInputAttributeNames(),
    np.NaN,
  )

  # get intervention and conditioning sets
  intervention_set = set(action_set.keys())

  # intersection_of_non_descendents_of_intervened_upon_variables
  conditioning_set = set.intersection(*[
    objs.scm_obj.getNonDescendentsForNode(node)
    for node in intervention_set
  ])

  # assert there is no intersection
  assert set.intersection(intervention_set, conditioning_set) == set()

  # set values in intervention and conditioning sets
  for node in conditioning_set:
    counterfactual_template[node] = factual_instance[node]

  for node in intervention_set:
    counterfactual_template[node] = action_set[node]

  # return counterfactual_template

  # this dataframe has populated columns set to intervention or conditioning values
  # and has NaN columns that will be set accordingly.
  samples_df = pd.DataFrame(dict(zip(
    objs.dataset_obj.getInputAttributeNames(),
    [num_samples * [counterfactual_template[node] + 0] for node in objs.dataset_obj.getInputAttributeNames()],
  ))) # +0 important, specifically for tensor based elements, so we don't copy
      # an existing object in the computational graph, but we create a new node

  return samples_df


def _samplingInnerLoop(args, objs, factual_instance, action_set, recourse_type, num_samples):

  # counterfactual_template = _getCounterfactualTemplate(dataset_obj, classifier_obj, scm_obj, factual_instance, action_set, recourse_type, num_samples)
  samples_df = _getSamplesDFTemplate(args, objs, factual_instance, action_set, recourse_type, num_samples)

  # Simply traverse the graph in order, and populate nodes as we go!
  # IMPORTANT: DO NOT use SET(topo ordering); it sometimes changes ordering!
  for node in objs.scm_obj.getTopologicalOrdering():
    # set variable if value not yet set through intervention or conditioning
    if samples_df[node].isnull().values.any():
      parents = objs.scm_obj.getParentsForNode(node)
      # root nodes MUST always be set through intervention or conditioning
      assert len(parents) > 0
      # Confirm parents columns are present/have assigned values in samples_df
      assert not samples_df.loc[:,list(parents)].isnull().values.any()
      if args.debug_flag:
        print(f'Sampling `{recourse_type}` from {getConditionalString(node, parents)}')
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
        args,
        objs,
        factual_instance,
        samples_df,
        node,
        parents,
        recourse_type,
      )
  assert \
    np.all(list(samples_df.columns) == objs.dataset_obj.getInputAttributeNames()), \
    'Ordering of column names in samples_df has change unexpectedly'
  # samples_df = samples_df[objs.dataset_obj.getInputAttributeNames()]
  return samples_df


def computeCounterfactualInstance(args, objs, factual_instance, action_set, recourse_type):

  assert recourse_type in ACCEPTABLE_POINT_RECOURSE

  if not bool(action_set): # if action_set is empty, CFE = F
    return factual_instance

  samples_df = _samplingInnerLoop(args, objs, factual_instance, action_set, recourse_type, 1)

  return samples_df.loc[0].to_dict() # return the first instance


def getRecourseDistributionSample(args, objs, factual_instance, action_set, recourse_type, num_samples):

  assert recourse_type in ACCEPTABLE_DISTR_RECOURSE

  if not bool(action_set): # if action_set is empty, CFE = F
    return pd.DataFrame(dict(zip(
      objs.dataset_obj.getInputAttributeNames(),
      [num_samples * [factual_instance[node]] for node in objs.dataset_obj.getInputAttributeNames()],
    )))

  samples_df = _samplingInnerLoop(args, objs, factual_instance, action_set, recourse_type, num_samples)

  return samples_df # return the entire data frame


def isPointConstraintSatisfied(args, objs, factual_instance, action_set, recourse_type):
  return didFlip(
    args,
    objs,
    factual_instance,
    computeCounterfactualInstance(
      args,
      objs,
      factual_instance,
      action_set,
      recourse_type,
    ),
  )


def isDistrConstraintSatisfied(args, objs, factual_instance, action_set, recourse_type):
  return computeLowerConfidenceBound(args, objs, factual_instance, action_set, recourse_type) >= 0.5


def computeLowerConfidenceBound(args, objs, factual_instance, action_set, recourse_type):
  monte_carlo_samples_df = getRecourseDistributionSample(
    args,
    objs,
    factual_instance,
    action_set,
    recourse_type,
    args.num_mc_samples,
  )
  monte_carlo_predictions = getPredictionBatch(
    args,
    objs,
    monte_carlo_samples_df.to_numpy(),
  )

  expectation = np.mean(monte_carlo_predictions)
  variance = np.sum(np.power(monte_carlo_predictions - expectation, 2)) / (len(monte_carlo_predictions) - 1)

  # return expectation, variance

  # IMPORTANT... WE ARE CONSIDERING {0,1} LABELS AND FACTUAL SAMPLES MAY BE OF
  # EITHER CLASS. THEREFORE, THE CONSTRAINT IS SATISFIED WHEN SIGNIFICANTLY
  # > 0.5 OR < 0.5 FOR A FACTUAL SAMPLE WITH Y = 0 OR Y = 1, RESPECTIVELY.

  if getPrediction(args, objs, factual_instance) == 0:
    return expectation - args.lambda_lcb * np.sqrt(variance) # NOTE DIFFERNCE IN SIGN OF STD
  else: # factual_prediction == 1
    raise Exception(f'Should only be considering negatively predicted individuals...')
    # return expectation + args.lambda_lcb * np.sqrt(variance) # NOTE DIFFERNCE IN SIGN OF STD


def getValidDiscretizedActionSets(args, objs):

  possible_actions_per_node = []

  for attr_name_kurz in objs.dataset_obj.getInputAttributeNames('kurz'):

    attr_obj = objs.dataset_obj.attributes_kurz[attr_name_kurz]

    if attr_obj.attr_type in {'numeric-real', 'numeric-int', 'binary'}:

      if attr_obj.attr_type == 'numeric-real':
        number_decimals = 3
      elif attr_obj.attr_type in {'numeric-int', 'binary'}:
        number_decimals = 0

      tmp = list(
        np.around(
          np.linspace(
            objs.dataset_obj.data_frame_kurz.describe()[attr_name_kurz]['min'],
            # bad code amir; don't access internal object attribute
            objs.dataset_obj.data_frame_kurz.describe()[attr_name_kurz]['max'],
            args.grid_search_bins + 1
          ),
          number_decimals,
        )
      )
      tmp.append('n/a')
      tmp = list(dict.fromkeys(tmp))
      # remove repeats from list; this may happen, say for numeric-int, where we
      # can have upper-lower < args.grid_search_bins, then rounding to 0 will result
      # in some repeated values
      possible_actions_per_node.append(tmp)

    else:

      raise NotImplementedError # TODO: add support for categorical variables

  all_action_tuples = list(itertools.product(
    *possible_actions_per_node
  ))

  all_action_tuples = [
    elem1 for elem1 in all_action_tuples
    if len([
      elem2 for elem2 in elem1 if elem2 != 'n/a'
    ]) <= args.max_intervention_cardinality
  ]

  all_action_sets = [
    dict(zip(objs.dataset_obj.getInputAttributeNames(), elem))
    for elem in all_action_tuples
  ]

  # Go through, and for any action_set that has a value = 'n/a', remove ONLY
  # THAT key, value pair, NOT THE ENTIRE action_set.
  valid_action_sets = []
  for action_set in all_action_sets:
    valid_action_sets.append({k:v for k,v in action_set.items() if v != 'n/a'})

  return valid_action_sets


def getValidInterventionSets(args, objs):

  # https://stackoverflow.com/a/1482316/2759976
  def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

  all_intervention_tuples = powerset(objs.dataset_obj.getInputAttributeNames('kurz'))
  all_intervention_tuples = [
    elem for elem in all_intervention_tuples
    if len(elem) <= args.max_intervention_cardinality
    and elem is not tuple() # no interventions (i.e., empty tuple) could never result in recourse --> ignore
  ]

  return all_intervention_tuples


def getColumnIndexFromName(args, objs, column_name):
  # this is index in df, need to -1 to get index in x_counter / do_update,
  # because the first column of df is 'y'
  return objs.dataset_obj.data_frame_kurz.columns.get_loc(column_name) - 1


def tmpPlot(args, objs, factual_instance, save_path, intervention_set, recourse_type):

  assert \
    'cvae' in recourse_type or 'gaus' in recourse_type, \
    f'{args.optimization_approach} does not currently support {recourse_type}'

  # TODO: @utils.Memoize ???
  def convertDataFrameOfTensorsToSharedTensor(tmp_df, cols):
    assert isinstance(tmp_df, pd.DataFrame) # not series
    return torch.stack([
        torch.stack(
          tuple(
            tmp_df[col].to_numpy()
          )
        )
        for col in cols
    ], axis = 1)
    # return torch.stack([
    #     torch.stack([tmp_df.iloc[j,i]
    #         for j in range(tmp_df.shape[0])
    #     ], axis = 0)
    #     for i in range(tmp_df.shape[1])
    # ], axis = 1)

  range_x1 = objs.dataset_obj.data_frame_kurz.describe()['x1']
  tmp_linspace = np.linspace(range_x1['min'], range_x1['max'], 100).astype('float32')
  action_sets = [
    {'x1': torch.tensor(value_x1)}
    for value_x1 in tmp_linspace
  ]

  # IMPORTANT: watch ordering of action_set_ts and factual_instance_ts, they are
  # both tensors, but the former are trainable whereas the latter are not.
  factual_instance_ts = {k: torch.tensor(v) for k, v in factual_instance.items()}
  factual_df = pd.DataFrame({k : [v] * args.num_mc_samples for k,v in factual_instance_ts.items()})


  h = getTorchClassifier(args, objs)
  # TODO: make input args
  capped_loss = False
  num_epochs = 500
  lambda_opt = 1 # initial value
  lambda_opt_update_every = 50
  lambda_opt_learning_rate = 0.5
  action_set_learning_rate = 0.1
  print_log_every = lambda_opt_update_every
  # optimizer = torch.optim.Adam(params = list(action_set_ts.values()), lr = action_set_learning_rate)

  all_loss_totals_1 = []
  all_loss_totals_2 = []
  all_loss_totals_3 = []
  all_loss_costs = []
  all_lambda_opts = []
  all_loss_constraints = []
  # all_thetas = []

  for idx, action_set_ts in enumerate(action_sets):
    print(f'idx: {idx}')

    samples_df = _getSamplesDFTemplate(args, objs, factual_instance_ts, action_set_ts, recourse_type, args.num_mc_samples)

    # Simply traverse the graph in order, and populate nodes as we go!
    # IMPORTANT: DO NOT use SET(topo ordering); it sometimes changes ordering!
    for node in objs.scm_obj.getTopologicalOrdering():
      # set variable if value not yet set through intervention or conditioning
      if samples_df[node].isnull().values.any():
        parents = objs.scm_obj.getParentsForNode(node)
        # root nodes MUST always be set through intervention or conditioning
        assert len(parents) > 0
        # Confirm parents columns are present/have assigned values in samples_df
        assert not samples_df.loc[:,list(parents)].isnull().values.any()

        # TODO: this would change according to other recourse types
        if recourse_type == 'm1_cvae':
          sample_from = 'posterior'
        elif recourse_type == 'm2_cvae':
          sample_from = 'prior'
        elif recourse_type == 'm2_cvae_ps':
          sample_from = 'reweighted_prior'

        if 'gaus' in recourse_type:
          kernel, X_all, model = trainGP(args, objs, node, parents)
          # ipsh()
          # X_parents = torch.tensor(samples_df[parents].to_numpy())
          X_parents = convertDataFrameOfTensorsToSharedTensor(samples_df, parents)
          if recourse_type == 'm1_gaus': # counterfactual distribution for node
            # IMPORTANT: Find index of factual instance in dataframe used for training GP
            #            (earlier, the factual instance was appended as the last instance)
            tmp_idx = getIndexOfFactualInstanceInDataFrame(
              factual_instance,
              processDataFrameOrDict(args, objs, getOriginalDataFrame(objs, args.num_train_samples), PROCESSING_GAUS),
            ) # TODO: can probably rewrite to just evaluate the posterior again given the same result.. (without needing to look through the dataset)
            new_samples = gpHelper.sample_from_GP_model(model, X_parents, 'cf', tmp_idx)
          elif recourse_type == 'm2_gaus': # interventional distribution for node
            new_samples = gpHelper.sample_from_GP_model(model, X_parents, 'iv')

          samples_df[node] = [elem[0][0].float() for elem in torch.split(new_samples, 1)] # GP torch returns float.64, convert to float32

        elif 'cvae' in recourse_type:
          trained_cvae = trainCVAE(args, objs, node, parents)
          new_samples = trained_cvae.reconstruct(
            x_factual=convertDataFrameOfTensorsToSharedTensor(factual_df, [node]),
            pa_factual=convertDataFrameOfTensorsToSharedTensor(factual_df, parents),
            pa_counter=convertDataFrameOfTensorsToSharedTensor(samples_df, parents),
            sample_from=sample_from,
          )

          # split returns a tuple/list of tensors which have listed (!) values
          samples_df[node] = [elem[0][0].float() for elem in torch.split(new_samples, 1)]
          # for idx in range(samples_df.shape[0]):
          #   samples_df['x3'][idx] = new_samples[0][idx]

    # TODO: convertDataFrameOfTensorsToSharedTensor does not work if some cols
    # are nan --> so placing this after the loop above once all cols are filled
    counter_ts = convertDataFrameOfTensorsToSharedTensor(samples_df, scm_obj.getTopologicalOrdering())

    loss_cost = measureActionSetCost(args, objs, factual_instance_ts, action_set_ts)

    # compute LCB
    pred_labels = h(counter_ts)
    # When all predictions are the same (likely because all sampled points are
    # the same, likely because we are outside of the manifold OR, e.g., when we
    # intervene on all nodes and the initial epoch returns same samples), then
    # torch.std() will be 0 and therefore there is no gradinet to pass back; in
    # turn this results in torch.std() giving nan and ruining the training!
    #     tmp = torch.ones((10,1), requires_grad=True)
    #     torch.std(tmp).backward()
    #     print(tmp.grad)
    # https://github.com/pytorch/pytorch/issues/4320
    # SOLUTION: remove torch.std() when this term is small to prevent passing nans.
    if torch.std(pred_labels) < 1e-10:
      # print(f'\t\t[INFO] Removing variance term due to very small variance breaking gradient.')
      value_lcb = torch.mean(pred_labels)
    else:
      value_lcb = torch.mean(pred_labels) - args.lambda_lcb * torch.std(pred_labels)

    loss_constraint = (0.5 - value_lcb)
    if capped_loss:
      loss_constraint = torch.nn.functional.relu(loss_constraint)

    # ========================================================================
    # ========================================================================

    # for fixed lambda, optimize theta (grad descent)
    loss_total_1 = loss_cost + 1 * loss_constraint
    loss_total_2 = loss_cost + 2 * loss_constraint
    loss_total_3 = loss_cost + 3 * loss_constraint

    all_loss_totals_1.append(loss_total_1.detach().item())
    all_loss_totals_2.append(loss_total_2.detach().item())
    all_loss_totals_3.append(loss_total_3.detach().item())
    all_loss_costs.append(loss_cost.item())
    all_lambda_opts.append(lambda_opt)
    all_loss_constraints.append(loss_constraint.detach().item())
    # all_thetas.append() # show in 1d or 2d

  fig, ax = plt.subplots()
  ax.plot(tmp_linspace, all_loss_totals_1, '-', label='loss totals 1')
  ax.plot(tmp_linspace, all_loss_totals_2, '-', label='loss totals 2')
  ax.plot(tmp_linspace, all_loss_totals_3, '-', label='loss totals 3')
  ax.plot(tmp_linspace, all_loss_costs, 'g--', label='loss costs')
  ax.plot(tmp_linspace, all_loss_constraints, 'r:', label='loss constraints')
  ax.set(xlabel='theta', ylabel='loss', title='Losses vs Theta')
  ax.grid()
  ax.legend()
  plt.show()


def performGradDescentOptimization(args, objs, factual_instance, save_path, intervention_set, recourse_type):

  assert \
    'cvae' in recourse_type or 'gaus' in recourse_type, \
    f'{args.optimization_approach} does not currently support {recourse_type}'

  # TODO: @utils.Memoize ???
  def convertDataFrameOfTensorsToSharedTensor(tmp_df, cols):
    assert isinstance(tmp_df, pd.DataFrame) # not series
    return torch.stack([
        torch.stack(
          tuple(
            tmp_df[col].to_numpy()
          )
        )
        for col in cols
    ], axis = 1)
    # return torch.stack([
    #     torch.stack([tmp_df.iloc[j,i]
    #         for j in range(tmp_df.shape[0])
    #     ], axis = 0)
    #     for i in range(tmp_df.shape[1])
    # ], axis = 1)

  h = getTorchClassifier(args, objs)
  # initial_action_set, values of this dictionary are tensors which are trained
  action_set_ts = dict(zip(
    intervention_set,
    [
      torch.tensor(factual_instance[node], requires_grad=True)
      for node in intervention_set
    ]
  ))

  # IMPORTANT: watch ordering of action_set_ts and factual_instance_ts, they are
  # both tensors, but the former are trainable whereas the latter are not.
  factual_instance_ts = {k: torch.tensor(v) for k, v in factual_instance.items()}
  factual_df = pd.DataFrame({k : [v] * args.num_mc_samples for k,v in factual_instance_ts.items()})

  # TODO: make input args
  capped_loss = False
  num_epochs = 1000
  lambda_opt = 1 # initial value
  lambda_opt_update_every = 50
  lambda_opt_learning_rate = 0.5
  action_set_learning_rate = 0.1
  print_log_every = lambda_opt_update_every
  optimizer = torch.optim.Adam(params = list(action_set_ts.values()), lr = action_set_learning_rate)

  all_loss_totals = []
  all_loss_costs = []
  all_lambda_opts = []
  all_loss_constraints = []
  all_thetas = []

  start_time = time.time()
  if args.debug_flag:
    print(f'\t\t[INFO] initial action set: {str({k : np.around(v.detach().item(), 4) for k,v in action_set_ts.items()})}') # TODO: use pretty print
  for epoch in tqdm(range(1, num_epochs + 1)):

    # ========================================================================
    # ========================================================================

    samples_df = _getSamplesDFTemplate(args, objs, factual_instance_ts, action_set_ts, recourse_type, args.num_mc_samples)

    # Simply traverse the graph in order, and populate nodes as we go!
    # IMPORTANT: DO NOT use SET(topo ordering); it sometimes changes ordering!
    for node in objs.scm_obj.getTopologicalOrdering():
      # set variable if value not yet set through intervention or conditioning
      if samples_df[node].isnull().values.any():
        parents = objs.scm_obj.getParentsForNode(node)
        # root nodes MUST always be set through intervention or conditioning
        assert len(parents) > 0
        # Confirm parents columns are present/have assigned values in samples_df
        assert not samples_df.loc[:,list(parents)].isnull().values.any()

        # TODO: this would change according to other recourse types
        if recourse_type == 'm1_cvae':
          sample_from = 'posterior'
        elif recourse_type == 'm2_cvae':
          sample_from = 'prior'
        elif recourse_type == 'm2_cvae_ps':
          sample_from = 'reweighted_prior'

        if 'gaus' in recourse_type:
          kernel, X_all, model = trainGP(args, objs, node, parents)
          # ipsh()
          # X_parents = torch.tensor(samples_df[parents].to_numpy())
          X_parents = convertDataFrameOfTensorsToSharedTensor(samples_df, parents)
          if recourse_type == 'm1_gaus': # counterfactual distribution for node
            # IMPORTANT: Find index of factual instance in dataframe used for training GP
            #            (earlier, the factual instance was appended as the last instance)
            tmp_idx = getIndexOfFactualInstanceInDataFrame(
              factual_instance,
              processDataFrameOrDict(args, objs, getOriginalDataFrame(objs, args.num_train_samples), PROCESSING_GAUS),
            ) # TODO: can probably rewrite to just evaluate the posterior again given the same result.. (without needing to look through the dataset)
            new_samples = gpHelper.sample_from_GP_model(model, X_parents, 'cf', tmp_idx)
          elif recourse_type == 'm2_gaus': # interventional distribution for node
            new_samples = gpHelper.sample_from_GP_model(model, X_parents, 'iv')

          samples_df[node] = [elem[0][0].float() for elem in torch.split(new_samples, 1)] # GP torch returns float.64, convert to float32

        elif 'cvae' in recourse_type:
          trained_cvae = trainCVAE(args, objs, node, parents)
          new_samples = trained_cvae.reconstruct(
            x_factual=convertDataFrameOfTensorsToSharedTensor(factual_df, [node]),
            pa_factual=convertDataFrameOfTensorsToSharedTensor(factual_df, parents),
            pa_counter=convertDataFrameOfTensorsToSharedTensor(samples_df, parents),
            sample_from=sample_from,
          )

          # split returns a tuple/list of tensors which have listed (!) values
          samples_df[node] = [elem[0][0].float() for elem in torch.split(new_samples, 1)]
          # for idx in range(samples_df.shape[0]):
          #   samples_df['x3'][idx] = new_samples[0][idx]

    # TODO: convertDataFrameOfTensorsToSharedTensor does not work if some cols
    # are nan --> so placing this after the loop above once all cols are filled
    counter_ts = convertDataFrameOfTensorsToSharedTensor(samples_df, scm_obj.getTopologicalOrdering())

    # TODO: time analysis to speed up passing back of gradients?
    # start_time = time.time()
    # for i in range(1000):
    #   counter_ts = convertDataFrameOfTensorsToSharedTensor(samples_df, scm_obj.getTopologicalOrdering())
    # end_time = time.time()
    # print(f'\n[INFO] Done (total run-time: {end_time - start_time}).')

    # start_time = time.time()
    # for i in range(1000):
    #   samples_df['x3'] = [elem[0][0] for elem in torch.split(new_samples, 1)]
    # end_time = time.time()
    # print(f'\n[INFO] Done (total run-time: {end_time - start_time}).')

    # ========================================================================
    # ========================================================================

    # DOES NOT WORK ON GENERAL INTERVENTION_SETS!
    # counter_ts = torch.zeros((args.num_mc_samples, len(objs.dataset_obj.getInputAttributeNames())))
    # counter_ts[:,0] = factual_instance_ts['x1']
    # counter_ts[:,1] = action_set_ts[int_node] + 0 # +0 important so to have shared gradients passed back into single varable inside action_set_ts
    # trained_cvae = trainCVAE(args, objs, 'x3', ['x1', 'x2'])
    # counter_ts[:,2] = trained_cvae.reconstruct(
    #   x_factual=factual_ts[:,2].reshape(-1,1), # make dynamic
    #   pa_factual=factual_ts[:,0:2],
    #   pa_counter=counter_ts[:,0:2],
    #   sample_from='prior',
    # ).T

    # ========================================================================
    # ========================================================================

    loss_cost = measureActionSetCost(args, objs, factual_instance_ts, action_set_ts)

    # compute LCB
    pred_labels = h(counter_ts)
    # When all predictions are the same (likely because all sampled points are
    # the same, likely because we are outside of the manifold OR, e.g., when we
    # intervene on all nodes and the initial epoch returns same samples), then
    # torch.std() will be 0 and therefore there is no gradinet to pass back; in
    # turn this results in torch.std() giving nan and ruining the training!
    #     tmp = torch.ones((10,1), requires_grad=True)
    #     torch.std(tmp).backward()
    #     print(tmp.grad)
    # https://github.com/pytorch/pytorch/issues/4320
    # SOLUTION: remove torch.std() when this term is small to prevent passing nans.
    if torch.std(pred_labels) < 1e-10:
      # print(f'\t\t[INFO] Removing variance term due to very small variance breaking gradient.')
      value_lcb = torch.mean(pred_labels)
    else:
      value_lcb = torch.mean(pred_labels) - args.lambda_lcb * torch.std(pred_labels)

    loss_constraint = (0.5 - value_lcb)
    if capped_loss:
      loss_constraint = torch.nn.functional.relu(loss_constraint)

    # ========================================================================
    # ========================================================================

    # for fixed lambda, optimize theta (grad descent)
    loss_total = loss_cost + lambda_opt * loss_constraint

    # once every few epochs, optimize theta (grad ascent) manually (w/o pytorch)
    if epoch % lambda_opt_update_every == 0:
      lambda_opt = lambda_opt + lambda_opt_learning_rate * loss_constraint.detach()

    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()
    if args.debug_flag and epoch % print_log_every == 0:
      # TODO: use pretty print
      print(
        f'\t\t[INFO] epoch #{epoch:03}: ' \
        f'optimal action: {str({k : np.around(v.detach().item(), 2) for k,v in action_set_ts.items()})}    ' \
        f'loss_total: {loss_total.detach().item():02.6f}    ' \
        f'loss_cost: {loss_cost.detach().item():02.6f}    ' \
        f'lambda_opt: {lambda_opt:02.6f}    ' \
        f'loss_constraint: {loss_constraint.detach().item():02.6f}    ' \
        f'value_lcb: {value_lcb.detach().item():02.6f}    ' \
      )
    all_loss_totals.append(loss_total.detach().item())
    all_loss_costs.append(loss_cost.item())
    all_lambda_opts.append(lambda_opt)
    all_loss_constraints.append(loss_constraint.detach().item())
    # all_thetas.append() # show in 1d or 2d

  end_time = time.time()
  if args.debug_flag:
    print(f'\t\t[INFO] Done (total run-time: {end_time - start_time}).\n\n')

  # fig, ax = plt.subplots()
  # ax.plot(range(1, len(all_loss_totals) + 1), all_loss_totals, 'b-', label='loss totals')
  # ax.plot(range(1, len(all_loss_totals) + 1), all_loss_costs, 'g--', label='loss objectives')
  # ax.plot(range(1, len(all_loss_totals) + 1), all_lambda_opts, 'y-.', label='lambda_opt')
  # ax.plot(range(1, len(all_loss_totals) + 1), all_loss_constraints, 'r:', label='loss constraints')

  # fig, (ax1, ax2, ax3) = plt.subplots(1,3)
  fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
  ax1.plot(range(1, len(all_loss_totals) + 1), all_loss_totals, 'b-', label='loss totals')
  ax1.plot(range(1, len(all_loss_totals) + 1), all_loss_costs, 'g--', label='loss costs')
  ax1.plot(range(1, len(all_loss_totals) + 1), all_loss_constraints, 'r:', label='loss constraints')
  ax1.set(xlabel='epochs', ylabel='loss', title='Loss curve')
  ax1.grid()
  ax1.legend()

  ax2.plot(range(1, len(all_loss_totals) + 1), all_lambda_opts, 'y-.', label='lambda_opt')
  ax2.set(xlabel='epochs', ylabel='loss', title='Lambda curve')
  ax2.grid()
  ax2.legend()

  # ax3.plot(range(1, len(all_loss_totals) + 1), all_thetas, 'y-.', label='thetas')
  # ax3.set(xlabel='epochs', ylabel='loss', title='Loss curve')
  # ax3.grid()
  # ax3.legend()

  # plt.show()
  plt.savefig(f'{save_path}/{str(intervention_set)}.pdf')

  # Convert action_set_ts to non-tensor action_set when passing back to rest of code
  action_set = {k : v.detach().item() for k,v in action_set_ts.items()}
  return action_set


def computeOptimalActionSet(args, objs, factual_instance, save_path, recourse_type):

  if recourse_type in ACCEPTABLE_POINT_RECOURSE:
    constraint_handle = isPointConstraintSatisfied
  elif recourse_type in ACCEPTABLE_DISTR_RECOURSE:
    constraint_handle = isDistrConstraintSatisfied
  else:
    raise Exception(f'{recourse_type} not recognized.')

  if args.optimization_approach == 'brute_force':

    valid_action_sets = getValidDiscretizedActionSets(args, objs)
    print(f'\n\t[INFO] Computing optimal `{recourse_type}`: grid searching over {len(valid_action_sets)} action sets...')

    min_cost = 1e10
    min_cost_action_set = {}
    for action_set in tqdm(valid_action_sets):
      if constraint_handle(args, objs, factual_instance, action_set, recourse_type):
        cost_of_action_set = measureActionSetCost(args, objs, factual_instance, action_set)
        if cost_of_action_set < min_cost:
          min_cost = cost_of_action_set
          min_cost_action_set = action_set

    print(f'\t done (optimal action set: {str(min_cost_action_set)}).')

  elif args.optimization_approach == 'grad_descent':

    valid_intervention_sets = getValidInterventionSets(args, objs)
    print(f'\n\t[INFO] Computing optimal `{recourse_type}`: grad descent over {len(valid_intervention_sets)} intervention sets...')

    min_cost = 1e10
    min_cost_action_set = {}
    for idx, intervention_set in enumerate(valid_intervention_sets):
      print(f'\n\t[INFO] intervention set #{idx}/{len(valid_intervention_sets)}: {str(intervention_set)}')
      # tmpPlot(args, objs, factual_instance, save_path, intervention_set, recourse_type)
      action_set = performGradDescentOptimization(args, objs, factual_instance, save_path, intervention_set, recourse_type)
      if constraint_handle(args, objs, factual_instance, action_set, recourse_type):
        cost_of_action_set = measureActionSetCost(args, objs, factual_instance, action_set)
        if cost_of_action_set < min_cost:
          min_cost = cost_of_action_set
          min_cost_action_set = action_set

    print(f'\t done (optimal intervention set: {str(min_cost_action_set)}).')

    # TODOs:

    # Tuesday:
    # [x] convert sklearn to torch classifier
    # [x] compute sample mean and sample variance (of h_classifier)
    # [x] first running example of grad descent on cvae
    # [x] flush computational graph to speed up training without hogging memory
    # [x] can we apply lambda grad only when constraint >= 0
    # [x] loss_cost: use measureActionSetCost
    # [x] loss_constraint: fix BCE loss > 0.5
    # [x] learn lambda
    # [x] fix nan in theta after some epochs
    # [x] add plotting

    # Wednesday:
    # [x] cleanup models.py
    # [x] make code dynamic: use dataframes (as auxiliary store of value? think intervention on x1->x2->x3)
    # [x] investigate loss constraint 0 -> 1 -> 0 (because of not capping ()_+? yes it seems)
    # [x] see parallels in training of gp, merge torch and autograd implementations
    #     [x] get code running
    #     [x] speed up, perhaps with Memoization
    #     [x] investigate why are we seeing nan samples for 10 training samples?
    #     [x] confirm same solution for the old/new sampleGP functions on brute-force
    #     [x] build grad-descent solution on new sampleGP function
    #     [ ] find params for grad-descent solution on new sampleGP function

    # Thursday:
    # [ ] implement grad based for m0/m2_true
    # [ ] implement grad based for m1_alin
    # [ ] implement grad based for m1_akrr
    # [ ] merge all repetitive code to work for numpy and tensors gracefully (e.g., process/deprocessDf, samplingInnerLoop, etc.)
    # [ ] select hyperparms (initial values, learning rate, etc.) across settings: intervention nodes, recourse types (incl'd learned cvae model), factual instances, scms, etc.
    # [ ] select hyperparms (initial values, learning rate, etc.) in comparison with brute_force

  else:
    raise Exception(f'{args.optimization_approach} not recognized.')

  return min_cost_action_set


def scatterDecisionBoundary(args, objs, ax):
  assert len(objs.dataset_obj.getInputAttributeNames()) == 3
  sklearn_model = objs.classifier_obj
  fixed_model_w = sklearn_model.coef_
  fixed_model_b = sklearn_model.intercept_

  x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
  y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
  X = np.linspace(ax.get_xlim()[0] - x_range / 10, ax.get_xlim()[1] + x_range / 10, 10)
  Y = np.linspace(ax.get_ylim()[0] - y_range / 10, ax.get_ylim()[1] + y_range / 10, 10)
  X, Y = np.meshgrid(X, Y)
  Z = - (fixed_model_w[0][0] * X + fixed_model_w[0][1] * Y + fixed_model_b) / fixed_model_w[0][2]

  surf = ax.plot_wireframe(X, Y, Z, alpha=0.3)


def scatterDataset(args, objs, ax):
  assert len(objs.dataset_obj.getInputAttributeNames()) == 3
  X_train, X_test, y_train, y_test = objs.dataset_obj.getTrainTestSplit()
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


def scatterFactual(args, objs, factual_instance, ax):
  assert len(objs.dataset_obj.getInputAttributeNames()) == 3
  ax.scatter(
    factual_instance['x1'],
    factual_instance['x2'],
    factual_instance['x3'],
    marker='P',
    color='black',
    s=70
  )


def scatterRecourse(args, objs, factual_instance, action_set, recourse_type, marker_type, legend_label, ax):

  assert len(objs.dataset_obj.getInputAttributeNames()) == 3

  if recourse_type in ACCEPTABLE_POINT_RECOURSE:
    # point recourse

    point = computeCounterfactualInstance(args, objs, factual_instance, action_set, recourse_type)
    color_string = 'green' if didFlip(args, objs, factual_instance, point) else 'red'
    ax.scatter(point['x1'], point['x2'], point['x3'], marker=marker_type, color=color_string, s=70, label=legend_label)

  elif recourse_type in ACCEPTABLE_DISTR_RECOURSE:
    # distr recourse

    samples_df = getRecourseDistributionSample(args, objs, factual_instance, action_set, recourse_type, args.num_display_samples)
    x1s = samples_df.iloc[:,0]
    x2s = samples_df.iloc[:,1]
    x3s = samples_df.iloc[:,2]
    color_strings = ['green' if didFlip(args, objs, factual_instance, sample.to_dict()) else 'red' for _, sample in samples_df.iterrows()]
    ax.scatter(x1s, x2s, x3s, marker=marker_type, color=color_strings, alpha=0.1, s=30, label=legend_label)

    # mean_distr_samples = {
    #   'x1': np.mean(samples_df['x1']),
    #   'x2': np.mean(samples_df['x2']),
    #   'x3': np.mean(samples_df['x3']),
    # }
    # color_string = 'green' if didFlip(args, objs, factual_instance, mean_distr_samples) else 'red'
    # ax.scatter(mean_distr_samples['x1'], mean_distr_samples['x2'], mean_distr_samples['x3'], marker=marker_type, color=color_string, alpha=0.5, s=70, label=legend_label)

  else:

    raise Exception(f'{recourse_type} not recognized.')


def visualizeDatasetAndFixedModel(args, objs):

  fig = plt.figure()
  ax = plt.subplot(1, 1, 1, projection='3d')

  scatterDataset(args, objs, ax)
  scatterDecisionBoundary(args, objs, ax)

  ax.set_xlabel('x1')
  ax.set_ylabel('x2')
  ax.set_zlabel('x3')
  ax.set_title(f'datatset')
  # ax.legend()
  ax.grid(True)

  plt.show()


def getNegativelyPredictedInstances(args, objs):
  # Samples for which we seek recourse are chosen from the joint of X_train/test.
  # This is OK because the tasks of conditional density estimation and recourse
  # generation are distinct. Given the same data splicing used here and in trainGP,
  # it is guaranteed that we the factual sample for which we seek recourse is in
  # training set for GP, and hence a posterior over noise for it is computed
  # (i.e., we can cache).

  # Only focus on instances with h(x^f) = 0 and therfore h(x^cf) = 1
  X_all = getOriginalDataFrame(objs, args.num_train_samples)
  factual_instances_dict = {}
  tmp_counter = 1
  for factual_instance_idx, row in X_all.iterrows():
    factual_instance = row.T.to_dict()
    if getPrediction(args, objs, factual_instance) == 0:
      tmp_counter += 1
      factual_instances_dict[factual_instance_idx] = factual_instance
    if tmp_counter > args.num_recourse_samples:
      break
  return factual_instances_dict


def hotTrainRecourseTypes(args, objs, recourse_types):
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
    for node in objs.scm_obj.getTopologicalOrdering():
      parents = objs.scm_obj.getParentsForNode(node)
      if len(parents): # if not a root node
        training_handle(args, objs, node, parents)
  end_time = time.time()
  print(f'\n[INFO] Done (total warm-up time: {end_time - start_time}).')
  print(f'\n' + '='*80 + '\n')


# DEPRECATED def experiment1


# DEPRECATED def experiment2


# DEPRECATED def experiment3


# DEPRECATED def experiment4


def experiment5(args, objs, experiment_folder_name, factual_instances_dict, experimental_setups, recourse_types):
  ''' sub-plot sanity '''

  # action_sets = [
  #   {'x1': objs.scm_obj.noises_distributions['u1'].sample()}
  #   for _ in range(4)
  # ]
  range_x1 = objs.dataset_obj.data_frame_kurz.describe()['x1']
  action_sets = [
    {'x1': value_x1}
    for value_x1 in np.linspace(range_x1['min'], range_x1['max'], 9)
  ]

  factual_instance = factual_instances_dict[list(factual_instances_dict.keys())[0]]

  fig, axes = plt.subplots(
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

    print(f'\n\n[INFO] ACTION SET: {str(prettyPrintDict(action_set))}' + ' =' * 40)

    for experimental_setup in experimental_setups:
      recourse_type, marker = experimental_setup[0], experimental_setup[1]

      if recourse_type in ACCEPTABLE_POINT_RECOURSE:
        sample = computeCounterfactualInstance(args, objs, factual_instance, action_set, recourse_type)
        print(f'{recourse_type}:\t{prettyPrintDict(sample)}')
        axes.flatten()[idx].plot(sample['x2'], sample['x3'], marker, alpha=1.0, markersize = 7, label=recourse_type)
      elif recourse_type in ACCEPTABLE_DISTR_RECOURSE:
        samples = getRecourseDistributionSample(args, objs, factual_instance, action_set, recourse_type, args.num_display_samples)
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
  # plt.show()
  plt.savefig(f'{experiment_folder_name}/_comparison.pdf')


def experiment6(args, objs, experiment_folder_name, factual_instances_dict, experimental_setups, recourse_types):
  ''' optimal action set: figure + table '''

  os.mkdir(f'{experiment_folder_name}/optimization_results')

  per_instance_results = {}
  for enumeration_idx, (key, value) in enumerate(factual_instances_dict.items()):
    factual_instance_idx = f'sample_{key}'
    factual_instance = value

    print(f'\n\n\n[INFO] Processing factual instance `{factual_instance_idx}` (#{enumeration_idx + 1} / {len(factual_instances_dict.keys())})...')

    per_instance_results[factual_instance_idx] = {}
    per_instance_results[factual_instance_idx]['factual_instance'] = factual_instance

    for recourse_type in recourse_types:

      tmp = {}
      save_path = f'{experiment_folder_name}/optimization_results/{recourse_type}_factual_instance_{factual_instance_idx}'
      os.mkdir(save_path)

      start_time = time.time()
      tmp['optimal_action_set'] = computeOptimalActionSet(
        args,
        objs,
        factual_instance,
        save_path,
        recourse_type,
      )
      end_time = time.time()

      tmp['runtime'] = np.around(end_time - start_time, 3)

      # print(f'\t[INFO] Computing SCF validity and Interventional Confidence measures for optimal action `{str(tmp["optimal_action_set"])}`...')

      tmp['scf_validity']  = isPointConstraintSatisfied(args, objs, factual_instance, tmp['optimal_action_set'], 'm0_true')
      tmp['ic_m2_true'] = np.around(computeLowerConfidenceBound(args, objs, factual_instance, tmp['optimal_action_set'], 'm2_true'), 3)
      if recourse_type in ACCEPTABLE_DISTR_RECOURSE and recourse_type != 'm2_true':
        tmp['ic_rec_type'] = np.around(computeLowerConfidenceBound(args, objs, factual_instance, tmp['optimal_action_set'], recourse_type), 3)
      else:
        tmp['ic_rec_type'] = np.NaN
      # tmp['ic_m1_gaus'] = np.around(computeLowerConfidenceBound(args, objs, factual_instance, tmp['optimal_action_set'], 'm1_gaus'), 3)
      # tmp['ic_m1_cvae'] = np.around(computeLowerConfidenceBound(args, objs, factual_instance, tmp['optimal_action_set'], 'm1_cvae'), 3)
      # tmp['ic_m2_gaus'] = np.around(computeLowerConfidenceBound(args, objs, factual_instance, tmp['optimal_action_set'], 'm2_gaus'), 3)
      # tmp['ic_m2_cvae'] = np.around(computeLowerConfidenceBound(args, objs, factual_instance, tmp['optimal_action_set'], 'm2_cvae'), 3)
      tmp['cost_all'] = measureActionSetCost(args, objs, factual_instance, tmp['optimal_action_set'])
      tmp['cost_valid'] = tmp['cost_all'] if tmp['scf_validity'] else np.NaN

      # print(f'\t done.')

      per_instance_results[factual_instance_idx][recourse_type] = tmp

    print(f'[INFO] Saving (overwriting) results...')
    pickle.dump(per_instance_results, open(f'{experiment_folder_name}/_per_instance_results', 'wb'))
    pprint(per_instance_results, open(f'{experiment_folder_name}/_per_instance_results.txt', 'w'))
    print(f'done.')

    # Table
    metrics_summary = {}
    # metrics = ['scf_validity', 'ic_m1_gaus', 'ic_m1_cvae', 'ic_m2_true', 'ic_m2_gaus', 'ic_m2_cvae', 'cost_all', 'cost_valid', 'runtime']
    metrics = ['scf_validity', 'ic_m2_true', 'ic_rec_type', 'cost_all', 'cost_valid', 'runtime']

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
    tmp_df.to_csv(f'{experiment_folder_name}/_comparison.txt', sep='\t')
    tmp_df.to_pickle(f'{experiment_folder_name}/_comparison.pkl')

  # TODO: FIX
  # # Figure
  # if len(objs.dataset_obj.getInputAttributeNames()) != 3:
  #   print('Cannot plot in more than 3 dimensions')
  #   return

  # max_to_plot = 4
  # tmp = min(args.num_recourse_samples, max_to_plot)
  # num_plot_cols = int(np.floor(np.sqrt(tmp)))
  # num_plot_rows = int(np.ceil(tmp / num_plot_cols))

  # fig, axes = plt.subplots(num_plot_rows, num_plot_cols, subplot_kw=dict(projection='3d'))
  # if num_plot_rows * num_plot_cols == max_to_plot:
  #   axes = np.array(axes) # weird hack we need to use so to later use flatten()

  # for idx, (key, value) in enumerate(factual_instances_dict.items()):
  #   if idx >= 1:
  #     break

  #   factual_instance_idx = f'sample_{key}'
  #   factual_instance = value

  #   ax = axes.flatten()[idx]

  #   scatterFactual(args, objs, factual_instance, ax)
  #   title = f'sample_{factual_instance_idx} - {prettyPrintDict(factual_instance)}'

  #   for experimental_setup in experimental_setups:
  #     recourse_type, marker = experimental_setup[0], experimental_setup[1]
  #     optimal_action_set = per_instance_results[factual_instance_idx][recourse_type]['optimal_action_set']
  #     legend_label = f'\n {recourse_type}; do({prettyPrintDict(optimal_action_set)}); cost: {measureActionSetCost(args, objs, factual_instance, optimal_action_set):.3f}'
  #     scatterRecourse(args, objs, factual_instance, optimal_action_set, 'm0_true', marker, legend_label, ax)
  #     # scatterRecourse(args, objs, factual_instance, optimal_action_set, recourse_type, marker, legend_label, ax)

  #   scatterDecisionBoundary(args, objs, ax)
  #   ax.set_xlabel('x1', fontsize=8)
  #   ax.set_ylabel('x2', fontsize=8)
  #   ax.set_zlabel('x3', fontsize=8)
  #   ax.set_title(title, fontsize=8)
  #   ax.view_init(elev=20, azim=-30)

  # for ax in axes.flatten():
  #   ax.legend(fontsize='xx-small')
  # fig.tight_layout()
  # # plt.show()
  # plt.savefig(f'{experiment_folder_name}/comparison.pdf')


# DEPRECATED def experiment7


def experiment8(args, objs, experiment_folder_name, factual_instances_dict, experimental_setups, recourse_types):
  ''' box-plot sanity '''

  PER_DIM_GRANULARITY = 8

  for node in objs.scm_obj.getTopologicalOrdering():

    parents = objs.scm_obj.getParentsForNode(node)

    if len(parents) == 0: # if not a root node
      continue # don't want to plot marginals, because we're not learning these
    elif len(parents) > 2:
      print(f'[INFO] not able to plot sanity checks for {getConditionalString(node, parents)}')
    elif len(parents) == 1:

      all_actions_outer_product = list(itertools.product(
        *[
          np.linspace(
            objs.dataset_obj.data_frame_kurz.describe()[parent]['min'],
            objs.dataset_obj.data_frame_kurz.describe()[parent]['max'],
            PER_DIM_GRANULARITY,
          )
          for parent in parents
        ]
      ))
      action_sets = [
        dict(zip(parents, elem))
        for elem in all_actions_outer_product
      ]

      # i don't this has any affect... especially when we sweep over values of all parents and condition children
      factual_instance = factual_instances_dict[list(factual_instances_dict.keys())[0]]
      total_df = pd.DataFrame(columns=['recourse_type'] + list(objs.scm_obj.getTopologicalOrdering()))

      for idx, action_set in enumerate(action_sets):

        print(f'\n\n[INFO] ACTION SET: {str(prettyPrintDict(action_set))}' + ' =' * 40)

        for experimental_setup in experimental_setups:
          recourse_type, marker = experimental_setup[0], experimental_setup[1]

          if recourse_type in ACCEPTABLE_POINT_RECOURSE:
            sample = computeCounterfactualInstance(args, objs, factual_instance, action_set, recourse_type)
            # print(f'{recourse_type}:\t{prettyPrintDict(sample)}')
          elif recourse_type in ACCEPTABLE_DISTR_RECOURSE:
            samples = getRecourseDistributionSample(args, objs, factual_instance, action_set, recourse_type, args.num_validation_samples)
            # print(f'{recourse_type}:\n{samples.head()}')
          else:
            raise Exception(f'{recourse_type} not supported.')

          tmp_df = samples.copy()
          tmp_df['recourse_type'] = recourse_type # add column
          total_df = pd.concat([total_df, tmp_df]) # concat to overall

      # box plot
      ax = sns.boxplot(x=parents[0], y=node, hue='recourse_type', data=total_df, palette='Set3', showmeans=True)
      # TODO: average over high dens pdf, and show a separate plot/table for the average over things...
      # ax.set_xticklabels(
      #   [np.around(elem, 3) for elem in ax.get_xticks()],
      #   rotation=90,
      # )
      ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
      plt.savefig(f'{experiment_folder_name}/_sanity_{getConditionalString(node, parents)}.pdf')

    elif len(parents) == 2:
      # distribution plot

      total_df = pd.DataFrame(columns=['recourse_type'] + list(objs.scm_obj.getTopologicalOrdering()))

      X_all = processDataFrameOrDict(args, objs, getOriginalDataFrame(objs, args.num_train_samples  + args.num_validation_samples), PROCESSING_CVAE)
      X_val = X_all[args.num_train_samples:].copy()

      X_true = X_val[parents + [node]]

      not_imp_factual_instance = dict.fromkeys(objs.scm_obj.getTopologicalOrdering(), -1)
      not_imp_samples_df = X_true.copy()

      # add samples from validation set itself (the true data):
      tmp_df = X_true.copy()
      tmp_df['recourse_type'] = 'true data' # add column
      total_df = pd.concat([total_df, tmp_df]) # concat to overall

      # add samples from all m2 methods
      for recourse_type in recourse_types:

        if recourse_type == 'm2_true':
          sampling_handle = sampleTrue
        elif recourse_type == 'm2_gaus':
          sampling_handle = sampleGP
        elif recourse_type == 'm2_cvae':
          sampling_handle = sampleCVAE

        samples = sampling_handle(args, objs, not_imp_factual_instance, not_imp_samples_df, node, parents, recourse_type)
        tmp_df = samples.copy()
        tmp_df['recourse_type'] = recourse_type # add column
        total_df = pd.concat([total_df, tmp_df]) # concat to overall

      ax = sns.boxplot(x='recourse_type', y=node, data=total_df, palette='Set3', showmeans=True)
      plt.savefig(f'{experiment_folder_name}/_sanity_{getConditionalString(node, parents)}.pdf')


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument('-s', '--scm_class', type=str, default='sanity-3-lin', help='Name of SCM to generate data using (see loadSCM.py)')
  parser.add_argument('-d', '--dataset_class', type=str, default='random', help='Name of dataset to train explanation model for: german, random, mortgage, twomoon')
  parser.add_argument('-c', '--classifier_class', type=str, default='lr', help='Model class that will learn data: lr, mlp')
  parser.add_argument('-e', '--experiment', type=int, default=6, help='Which experiment to run (5,8=sanity; 6=table)')
  parser.add_argument('-p', '--process_id', type=str, default='0', help='When running parallel tests on the cluster, process_id guarantees (in addition to time stamped experiment folder) that experiments do not conflict.')

  parser.add_argument('--norm_type', type=int, default=2)
  parser.add_argument('--lambda_lcb', type=int, default=1)
  parser.add_argument('--grid_search_bins', type=int, default=5)
  parser.add_argument('--num_train_samples', type=int, default=1000)
  parser.add_argument('--num_validation_samples', type=int, default=250)
  parser.add_argument('--num_recourse_samples', type=int, default=30)
  parser.add_argument('--num_display_samples', type=int, default=15)
  parser.add_argument('--num_mc_samples', type=int, default=300)
  parser.add_argument('--debug_flag', type=bool, default=False)
  parser.add_argument('--max_intervention_cardinality', type=int, default=3)
  parser.add_argument('--optimization_approach', type=str, default='brute_force')

  args = parser.parse_args()

  if not (args.dataset_class in {'random'}):
    raise Exception(f'{args.dataset_class} not supported.')

  if not (args.classifier_class in {'lr', 'mlp'}):
    raise Exception(f'{args.classifier_class} not supported.')

  # create experiment folder
  setup_name = \
    f'{args.scm_class}__{args.dataset_class}__{args.classifier_class}' + \
    f'__ntrain_{args.num_train_samples}' +\
    f'__nmc_{args.num_mc_samples}' + \
    f'__nrecourse_{args.num_recourse_samples}' + \
    f'__lambda_lcb_{args.lambda_lcb}' + \
    f'__pid{args.process_id}'
  experiment_folder_name = f"_experiments/{datetime.now().strftime('%Y.%m.%d_%H.%M.%S')}__{setup_name}"
  os.mkdir(f'{experiment_folder_name}')

  # save all arguments to file
  args_file = open(f'{experiment_folder_name}/_args.txt','w')
  for arg in vars(args):
    print(arg, ':\t', getattr(args, arg), file = args_file)

  # only load once so shuffling order is the same
  scm_obj = loadCausalModel(args, experiment_folder_name)
  dataset_obj = loadDataset(args, experiment_folder_name)
  classifier_obj = loadClassifier(args, experiment_folder_name)
  assert set(dataset_obj.getInputAttributeNames()) == set(scm_obj.getTopologicalOrdering())
  # TODO: add more assertions for columns of dataset matching the classifer?
  objs = AttrDict({
    'scm_obj': scm_obj,
    'dataset_obj': dataset_obj,
    'classifier_obj': classifier_obj,
  })

  # TODO: describe scm_obj
  print(f'Describe original data:\n{getOriginalDataFrame(objs, args.num_train_samples).describe()}')
  # TODO: describe classifier_obj

  # if only visualizing
  if args.experiment == 0:
    visualizeDatasetAndFixedModel(args, objs)
    quit()

  # setup
  factual_instances_dict = getNegativelyPredictedInstances(args, objs)
  experimental_setups = [
    ('m0_true', '*'), \
    # ('m1_alin', 'v'), \
    # ('m1_akrr', '^'), \
    # ('m1_gaus', 'D'), \
    # ('m1_cvae', 'x'), \
    ('m2_true', 'o'), \
    # ('m2_gaus', 's'), \
    # ('m2_cvae', '+'), \
    # ('m2_cvae_ps', 'P'), \
  ]

  if args.experiment == 5:

    assert \
      len(objs.dataset_obj.getInputAttributeNames()) == 3, \
      'Exp 5 is only designed for 3-variable SCMs'

  elif args.experiment == 6:

    assert \
      len(objs.dataset_obj.getInputAttributeNames()) >= 3, \
      'Exp 6 is only designed for 3+-variable SCMs'

  elif args.experiment == 8:

    # assert \
    #   len(objs.dataset_obj.getInputAttributeNames()) == 2, \
    #   'Exp 8 is only designed for 2-variable SCMs'
    if not np.all(['m2' in elem[0] for elem in experimental_setups]):
      print('[INFO] Exp 8 is only designed for m2 recourse_types; filtering to those')
      experimental_setups = [
        elem
        for elem in experimental_setups
        if 'm2' in elem[0]
      ]

  recourse_types = [experimental_setup[0] for experimental_setup in experimental_setups]
  hotTrainRecourseTypes(args, objs, recourse_types)

  if args.experiment == 5:
    experiment5(args, objs, experiment_folder_name, factual_instances_dict, experimental_setups, recourse_types)
  elif args.experiment == 6:
    experiment6(args, objs, experiment_folder_name, factual_instances_dict, experimental_setups, recourse_types)
  elif args.experiment == 8:
    experiment8(args, objs, experiment_folder_name, factual_instances_dict, experimental_setups, recourse_types)

  # sanity check
  # visualizeDatasetAndFixedModel(args, objs)



# TODO:
# merge exp5,6,8
# to confirm training correct -->for each child/parent, save m2 comparison (regression, no intervention) + report MSE/VAR between m2 methods (good choice of hyperparams)
# (=? intervention on parent node given value of sweep?)
# show 9 interventions on parent
# show table

























