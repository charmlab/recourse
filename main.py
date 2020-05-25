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
from matplotlib import pyplot as plt
from datetime import datetime
from attrdict import AttrDict

import GPy
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


def measureActionSetCost(args, objs, factual_instance, action_set):
  # TODO: add support for categorical data + measured in normalized space over all features
  deltas = []
  ranges = objs.dataset_obj.getVariableRanges()
  for key in action_set.keys():
    deltas.append((action_set[key] - factual_instance[key]) / ranges[key])
  return np.linalg.norm(deltas, args.norm_type)


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

  sweep_lambda_kld = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
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

    # run mmd to verify whether training is good or not (ON VALIDATION SET)
    X_val = X_all[args.num_train_samples:].copy()
    # POTENTIAL BUG? reset index here so that we can populate the `node` column
    # with reconstructed values from trained_cvae that lack indexing
    X_val = X_val.reset_index(drop = True)

    X_true = X_val[parents + [node]]

    X_pred_posterior = X_true.copy()
    X_pred_posterior[node] = pd.DataFrame(recon_node_validation.numpy(), columns=[node])

    tmp_factual_instance = dict.fromkeys(objs.scm_obj.getTopologicalOrdering(), -1)
    tmp_samples_df = X_true.copy()
    X_pred_prior = sampleCVAE(args, objs, tmp_factual_instance, tmp_samples_df, node, parents, 'm2_cvae', trained_cvae = trained_cvae)

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
    # if recourse_type == 'm0_true':
    #   if objs.scm_obj.scm_class != 'sanity-power':
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
      noise = objs.scm_obj.noises_distributions[getNoiseStringForNode(node)].sample(),
      samples_df.loc[row_idx, node] = structural_equation(
        noise,
        *samples_df.loc[row_idx, parents].to_numpy(),
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

  def noise_post_mean(K, sigma_squared, Y):
    N = K.shape[0]
    S = np.linalg.inv(K + sigma_squared * np.eye(N))
    return sigma_squared * np.dot(S, Y)

  def noise_post_cov(K, sigma_squared):
    N = K.shape[0]
    S = np.linalg.inv(K + sigma_squared * np.eye(N))
    return  sigma_squared * (np.eye(N) - sigma_squared * S)

  def noise_post_var(K, sigma_squared):
    N = K.shape[0]
    C = noise_post_cov(K, sigma_squared)
    return np.array([C[i,i] for i in range(N)])

  kernel, X_all, model = trainGP(args, objs, node, parents)

  K = kernel.K(X_all[parents].to_numpy())
  Y = X_all[[node]].to_numpy()
  noise_var = np.array(model.Gaussian_noise.variance)
  noise_post_means = noise_post_mean(K, noise_var, Y)
  noise_post_vars = noise_post_var(K, noise_var)

  # GP posterior for node at new (intervened & conditioned) input given parents
  # pred_means, pred_vars = model.predict_noiseless(samples_df[parents].to_numpy())
  pred_means, pred_vars = model.predict_noiseless(samples_df[parents].to_numpy())

  if recourse_type == 'm1_gaus': # counterfactual distribution for node
    # IMPORTANT: Find index of factual instance in dataframe used for training GP
    #            (earlier, the factual instance was appended as the last instance)
    tmp_idx = getIndexOfFactualInstanceInDataFrame(
      factual_instance,
      processDataFrameOrDict(args, objs, getOriginalDataFrame(objs, args.num_train_samples), PROCESSING_GAUS),
    ) # TODO: can probably rewrite to just evaluate the posterior again given the same result.. (without needing to look through the dataset)
    new_means = pred_means + noise_post_means[tmp_idx]
    new_vars = pred_vars + noise_post_vars[tmp_idx]
  elif recourse_type == 'm2_gaus': # interventional distribution for node
    new_means = pred_means + 0
    new_vars = pred_vars + noise_var

  # sample from distribution via reparametrisation trick
  new_noise = np.random.randn(samples_df.shape[0], 1)
  new_samples = new_means + np.sqrt(new_vars) * new_noise

  samples_df[node] = new_samples
  samples_df = deprocessDataFrameOrDict(args, objs, samples_df, PROCESSING_GAUS)
  return samples_df


def _samplingInnerLoop(args, objs, factual_instance, action_set, recourse_type, num_samples):

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

  # this dataframe has populated columns set to intervention or conditioning values
  # and has NaN columns that will be set accordingly.
  samples_df = pd.DataFrame(dict(zip(
    objs.dataset_obj.getInputAttributeNames(),
    [num_samples * [counterfactual_template[node]] for node in objs.dataset_obj.getInputAttributeNames()],
  )))

  # Simply traverse the graph in order, and populate nodes as we go!
  # IMPORTANT: DO NOT USE set(topo ordering); it sometimes changes ordering!
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
            attr_obj.lower_bound,
            attr_obj.upper_bound,
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


def computeOptimalActionSet(args, objs, factual_instance, recourse_type, optimization_approach):

  if recourse_type in ACCEPTABLE_POINT_RECOURSE:
    constraint_handle = isPointConstraintSatisfied
  elif recourse_type in ACCEPTABLE_DISTR_RECOURSE:
    constraint_handle = isDistrConstraintSatisfied
  else:
    raise Exception(f'{recourse_type} not recognized.')

  if optimization_approach == 'brute_force':

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

  elif optimization_approach == 'grad_descent':

    raise NotImplementedError # TODO
    # for all possible intervention sets (without value)
    # for each child-parent that is missing
    #     get object: trained_cvae = trainCVAE(args, objs, node, parents)
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

  per_instance_results = {}
  for enumeration_idx, (key, value) in enumerate(factual_instances_dict.items()):
    factual_instance_idx = f'sample_{key}'
    factual_instance = value

    print(f'\n\n\n[INFO] Processing factual instance `{factual_instance_idx}` (#{enumeration_idx + 1} / {len(factual_instances_dict.keys())})...')

    per_instance_results[factual_instance_idx] = {}
    per_instance_results[factual_instance_idx]['factual_instance'] = factual_instance

    for recourse_type in recourse_types:

      tmp = {}

      start_time = time.time()
      tmp['optimal_action_set'] = computeOptimalActionSet(
        args,
        objs,
        factual_instance,
        recourse_type,
        'brute_force',
      )
      end_time = time.time()

      tmp['runtime'] = np.around(end_time - start_time, 3)

      # print(f'\t[INFO] Computing SCF validity and Interventional Confidence measures for optimal action `{str(tmp["optimal_action_set"])}`...')

      tmp['scf_validity']  = isPointConstraintSatisfied(args, objs, factual_instance, tmp['optimal_action_set'], 'm0_true')
      tmp['int_conf_m1_gaus'] = np.around(computeLowerConfidenceBound(args, objs, factual_instance, tmp['optimal_action_set'], 'm1_gaus'), 3)
      tmp['int_conf_m1_cvae'] = np.around(computeLowerConfidenceBound(args, objs, factual_instance, tmp['optimal_action_set'], 'm1_cvae'), 3)
      tmp['int_conf_m2_true'] = np.around(computeLowerConfidenceBound(args, objs, factual_instance, tmp['optimal_action_set'], 'm2_true'), 3)
      tmp['int_conf_m2_gaus'] = np.around(computeLowerConfidenceBound(args, objs, factual_instance, tmp['optimal_action_set'], 'm2_gaus'), 3)
      tmp['int_conf_m2_cvae'] = np.around(computeLowerConfidenceBound(args, objs, factual_instance, tmp['optimal_action_set'], 'm2_cvae'), 3)
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
    metrics = ['scf_validity', 'int_conf_m1_gaus', 'int_conf_m1_cvae', 'int_conf_m2_true', 'int_conf_m2_gaus', 'int_conf_m2_cvae', 'cost_all', 'cost_valid', 'runtime']

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
    elif len(parents) == 1 or len(parents) == 2:

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
            samples = getRecourseDistributionSample(args, objs, factual_instance, action_set, recourse_type, args.num_mc_samples)
            # print(f'{recourse_type}:\n{samples.head()}')
          else:
            raise Exception(f'{recourse_type} not supported.')

          tmp_df = samples.copy()
          tmp_df['recourse_type'] = recourse_type # add column
          total_df = pd.concat([total_df, tmp_df]) # concat to overall

      if len(parents) == 1:
        # box plot
        ax = sns.boxplot(x=parents[0], y=node, hue='recourse_type', data=total_df, palette='Set3', showmeans=True)
        # TODO: average over high dens pdf, and show a separate plot/table for the average over things...
        # ax.set_xticklabels(
        #   [np.around(elem, 3) for elem in ax.get_xticks()],
        #   rotation=90,
        # )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.savefig(f'{experiment_folder_name}/_sanity_{getConditionalString(node, parents)}.pdf')

      # TODO:
      elif len(parents) == 2:
        # contour plot
        ipsh()
        # TODO: show2x2 plot, validation set (here and abve!) X_true, m2_true, m2_gaus, m2_cvae


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
  parser.add_argument('--num_mc_samples', type=int, default=100)
  parser.add_argument('--debug_flag', type=bool, default=False)

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

  # only load once so shuffling order is the same
  scm_obj = loadCausalModel(args, experiment_folder_name)
  dataset_obj = loadDataset(args, experiment_folder_name)
  classifier_obj = loadClassifier(args, experiment_folder_name)
  assert set(dataset_obj.getInputAttributeNames()) == set(scm_obj.getTopologicalOrdering())
  # TODO: add more assertions for columns of dataset matching soething classifer?
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
    ('m1_alin', 'v'), \
    ('m1_akrr', '^'), \
    ('m1_gaus', 'D'), \
    ('m1_cvae', 'x'), \
    ('m2_true', 'o'), \
    ('m2_gaus', 's'), \
    ('m2_cvae', '+'), \
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

























