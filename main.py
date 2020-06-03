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
import mmd
import utils
import loadSCM
import loadData
import loadModel
import gpHelper
import skHelper

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import WhiteKernel, RBF

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from _cvae.train import *

from debug import ipsh

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

ACCEPTABLE_POINT_RECOURSE = {'m0_true', 'm1_alin', 'm1_akrr'}
ACCEPTABLE_DISTR_RECOURSE = {'m1_gaus', 'm1_cvae', 'm2_true', 'm2_gaus', 'm2_cvae', 'm2_cvae_ps'}

PROCESSING_SKLEARN = 'raw'
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
  classifier_obj = loadModel.loadModelForDataset(args.classifier_class, args.dataset_class, experiment_folder_name)
  dataset_obj = loadDataset(args, experiment_folder_name)
  X_train, X_test, y_train, y_test = dataset_obj.getTrainTestSplit()
  X_all = pd.concat([X_train, X_test], axis = 0)

  # samples = X_all.iloc[:5].to_numpy()
  # weights = classifier_obj.coef_
  # intercept = classifier_obj.intercept_
  # tmp = np.dot(samples, weights.T) + intercept
  # tmp = (1 + np.exp(tmp)) ** (-1)
  # # vs.
  # classifier_obj.predict_proba(X_all[:5])
  # plt.hist(classifier_obj.predict_proba(X_all)[:,1], bins=100)
  # plt.show()

  return classifier_obj


@utils.Memoize
def getTorchClassifier(args, objs):

  if isinstance(objs.classifier_obj, LogisticRegression):

    fixed_model_w = objs.classifier_obj.coef_
    fixed_model_b = objs.classifier_obj.intercept_
    fixed_model = lambda x: torch.sigmoid(
      (
        torch.nn.functional.linear(
          x,
          torch.from_numpy(fixed_model_w).float(),
        ) + float(fixed_model_b)
      )
    )

  elif isinstance(objs.classifier_obj, MLPClassifier):

    data_dim = len(objs.dataset_obj.getInputAttributeNames())
    fixed_model_width = 10 # TODO make more dynamic later and move to separate function
    assert objs.classifier_obj.hidden_layer_sizes == (fixed_model_width, fixed_model_width)
    fixed_model = torch.nn.Sequential(
      torch.nn.Linear(data_dim, fixed_model_width),
      torch.nn.ReLU(),
      torch.nn.Linear(fixed_model_width, fixed_model_width),
      torch.nn.ReLU(),
      torch.nn.Linear(fixed_model_width, 1),
      torch.nn.Sigmoid()
    )
    fixed_model[0].weight = torch.nn.Parameter(torch.tensor(objs.classifier_obj.coefs_[0].astype('float32')).t(), requires_grad=False)
    fixed_model[2].weight = torch.nn.Parameter(torch.tensor(objs.classifier_obj.coefs_[1].astype('float32')).t(), requires_grad=False)
    fixed_model[4].weight = torch.nn.Parameter(torch.tensor(objs.classifier_obj.coefs_[2].astype('float32')).t(), requires_grad=False)
    fixed_model[0].bias = torch.nn.Parameter(torch.tensor(objs.classifier_obj.intercepts_[0].astype('float32')), requires_grad=False)
    fixed_model[2].bias = torch.nn.Parameter(torch.tensor(objs.classifier_obj.intercepts_[1].astype('float32')), requires_grad=False)
    fixed_model[4].bias = torch.nn.Parameter(torch.tensor(objs.classifier_obj.intercepts_[2].astype('float32')), requires_grad=False)

  else:

    raise Exception(f'Converting {str(objs.classifier_obj.__class__)} to torch not supported.')

  X_all = getOriginalDataFrame(objs, args.num_train_samples)
  assert np.all(
    np.isclose(
      objs.classifier_obj.predict_proba(X_all[:25])[:,1],
      fixed_model(torch.tensor(X_all[:25].to_numpy(), dtype=torch.float32)).flatten(),
      atol = 1e-3,
    )
  ), 'Torch classifier is not equivalent to the sklearn model.'

  return fixed_model


def measureActionSetCost(args, objs, factual_instance, action_set, processing_type = 'raw'):
  # TODO: add support for categorical data + measured in normalized space over all features

  X_all = processDataFrameOrDict(args, objs, getOriginalDataFrame(objs, args.num_train_samples), processing_type)
  ranges = dict(zip(
    X_all.columns,
    [np.max(X_all[col]) - np.min(X_all[col]) for col in X_all.columns],
  ))
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


def getColumnIndicesFromNames(args, objs, column_names):
  # this is index in df, need to -1 to get index in x_counter / do_update,
  # because the first column of df is 'y' (amir: what if column ordering is
  # changed? this code breaks abstraction.)
  column_indices = []
  for column_name in column_names:
    tmp_1 = objs.dataset_obj.data_frame_kurz.columns.get_loc(column_name) - 1
    tmp_2 = list(objs.scm_obj.getTopologicalOrdering()).index(column_name)
    tmp_3 = list(objs.dataset_obj.getInputAttributeNames()).index(column_name)
    assert tmp_1 == tmp_2 == tmp_3
    column_indices.append(tmp_1)
  return column_indices


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


def processTensorOrDictOfTensors(args, objs, obj, processing_type, column_names):

  # To process a tensor (unlike a dict/dataframe), the getColumnIndicesFromNames
  # returns indices by always assuming that the entire tensor is present
  assert len(column_names) == len(objs.dataset_obj.getInputAttributeNames())

  if processing_type == 'raw':
    return obj

  assert \
    isinstance(obj, torch.Tensor) or \
    (
      isinstance(obj, dict) and \
      np.all([
        isinstance(value, torch.Tensor)
        for value in list(obj.values())
      ])
    ), f'Datatype `{obj.__class__}` not supported for processing.'

  iterate_over = column_names
  obj = obj.clone()

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
    node_idx = getColumnIndicesFromNames(args, objs, [node])
    if processing_type == 'normalize':
      obj[:, node_idx] = (obj[:, node_idx] - node_min) / (node_max - node_min)
    elif processing_type == 'standardize':
      obj[:, node_idx] = (obj[:, node_idx] - node_mean) / node_std
    elif processing_type == 'mean_subtract':
      obj[:, node_idx] = (obj[:, node_idx] - node_mean)
  return obj


def deprocessTensorOrDictOfTensors(args, objs, obj, processing_type, column_names):
  # To process a tensor (unlike a dict/dataframe), the getColumnIndicesFromNames
  # returns indices by always assuming that the entire tensor is present
  assert len(column_names) == len(objs.dataset_obj.getInputAttributeNames())

  if processing_type == 'raw':
    return obj

  assert \
    isinstance(obj, torch.Tensor) or \
    (
      isinstance(obj, dict) and \
      np.all([
        isinstance(value, torch.Tensor)
        for value in list(obj.values())
      ])
    ), f'Datatype `{obj.__class__}` not supported for processing.'

  iterate_over = column_names
  obj = obj.clone()

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
    node_idx = getColumnIndicesFromNames(args, objs, [node])
    if processing_type == 'normalize':
      obj[:, node_idx] = obj[:, node_idx] * (node_max - node_min) + node_min
    elif processing_type == 'standardize':
      obj[:, node_idx] = obj[:, node_idx] * node_std + node_mean
    elif processing_type == 'mean_subtract':
      obj[:, node_idx] = obj[:, node_idx] + node_mean
  return obj


def processDataFrameOrDict(args, objs, obj, processing_type):
  # TODO: add support for categorical data

  if processing_type == 'raw':
    return obj

  if isinstance(obj, dict):
    iterate_over = obj.keys()
  elif isinstance(obj, pd.DataFrame):
    iterate_over = obj.columns
  else:
    raise Exception(f'Datatype `{obj.__class__}` not supported for processing.')

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
  else:
    raise Exception(f'Datatype `{obj.__class__}` not supported for processing.')

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


def getMinimumObservableInstance(args, objs, factual_instance):
  # TODO: if we end up using this, it is important to keep in mind the processing
  # that is use when calling a specific recourse type because we would have to then
  # perhaps find the MO instance in the original data, but then initialize the action
  # set using the processed value...

  X_all = getOriginalDataFrame(objs, args.num_train_samples)
  # compute distances between factual instance and all instances in X_all
  tmp = np.array(list(factual_instance.values())).reshape(1,-1)[:,None] - X_all.to_numpy()
  tmp = tmp.squeeze()
  min_cost = 1e10
  min_observable_dict = None
  print(f'\t\t[INFO] Searching for minimum observable instance...', end = '')
  for idx in range(tmp.shape[0]):
    # CLOSEST INSTANCE ON THE OTHER SIDE!!
    distance_np = tmp[idx,:]
    distance_dict = dict(zip(
      objs.scm_obj.getTopologicalOrdering(),
      observable_np,
    ))
    if \
      getPrediction(args, objs, factual_instance) != \
      getPrediction(args, objs, distance_dict):
      if np.linalg.norm(observable_np) < min_cost:
        min_cost = np.linalg.norm(observable_np)
        min_observable_idx = idx
        min_observable_dict = dict(zip(
          objs.scm_obj.getTopologicalOrdering(),
          X_all.iloc[min_observable_idx],
        ))
  print(f'found at index #{min_observable_idx}. Initializing `action_set_ts` using these values.')
  return min_observable_dict


def getNoiseStringForNode(node):
  assert node[0] == 'x'
  return 'u' + node[1:]


def prettyPrintDict(my_dict):
  # use this for grad descent logs (convert tensor accordingly)
  my_dict = my_dict.copy()
  for key, value in my_dict.items():
    my_dict[key] = np.around(value, 3)
  return my_dict


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
   # amir: i am not proud of this, but for some reason sklearn includes the
   # X_fit_ covariates but not labels (this is needed later if we want to
   # avoid using.predict() and call from krr manually)
  model.best_estimator_.Y_fit_ = X_all[[node]].to_numpy()
  return model


@utils.Memoize
def trainCVAE(args, objs, node, parents):
  assert len(parents) > 0, 'parents set cannot be empty.'
  print(f'\t[INFO] Fitting {getConditionalString(node, parents)} using CVAE on {args.num_train_samples * 4} samples; this may be very expensive, memoizing afterwards.')
  X_all = processDataFrameOrDict(args, objs, getOriginalDataFrame(objs, args.num_train_samples * 4 + args.num_validation_samples), PROCESSING_CVAE)

  # 1 b/c the X_all[[node]] is always 1 dimensional # TODO: add support for categorical variables
  sweep_encoder_layer_sizes = [
    [1, 3, 3],
    [1, 5, 5],
    [1, 3, 3, 3],
    [1, 3, 3, 3, 3],
  ]
  sweep_decoder_layer_sizes = [
    [2, 1],
    [2, 2, 1],
    [3, 3, 1],
    [5, 5, 1],
    [3, 3, 3, 1],
    [3, 3, 3, 3, 1],
  ]
  sweep_latent_size = [1, 2, 5]
  sweep_lambda_kld = [5, 1, 0.5, 0.1, 0.05, 0.01, 0.005]

  trained_models = {}

  all_hyperparam_setups = list(itertools.product(
    sweep_lambda_kld,
    sweep_encoder_layer_sizes,
    sweep_decoder_layer_sizes,
    sweep_latent_size,
  ))

  for idx, hyperparams in enumerate(all_hyperparam_setups):

    print(f'\n\t[INFO] Training hyperparams setup #{idx} / {len(all_hyperparam_setups)}: {str(hyperparams)}')

    trained_cvae, recon_node_train, recon_node_validation = train_cvae(AttrDict({
      'name': f'{getConditionalString(node, parents)}',
      'node_train': X_all[[node]].iloc[:args.num_train_samples * 4],
      'parents_train': X_all[parents].iloc[:args.num_train_samples * 4],
      'node_validation': X_all[[node]].iloc[args.num_train_samples * 4:],
      'parents_validation': X_all[parents].iloc[args.num_train_samples * 4:],
      'seed': 0,
      'epochs': 100,
      'batch_size': 128,
      'learning_rate': 0.05,
      'lambda_kld': hyperparams[0],
      'encoder_layer_sizes': hyperparams[1],
      'decoder_layer_sizes': hyperparams[2],
      'latent_size': hyperparams[3],
      'conditional': True,
      'debug_folder': experiment_folder_name + f'/cvae_hyperparams_setup_{idx}_of_{len(all_hyperparam_setups)}',
    }))

    # ipsh()

    # # TODO: remove after models.py is corrected
    # return trained_cvae

    # run mmd to verify whether training is good or not (ON VALIDATION SET)
    X_val = X_all[args.num_train_samples * 4:].copy()
    # POTENTIAL BUG? reset index here so that we can populate the `node` column
    # with reconstructed values from trained_cvae that lack indexing
    X_val = X_val.reset_index(drop = True)

    X_true = X_val[parents + [node]]

    X_pred_posterior = X_true.copy()
    X_pred_posterior[node] = pd.DataFrame(recon_node_validation.numpy(), columns=[node])

    not_imp_factual_instance = dict.fromkeys(objs.scm_obj.getTopologicalOrdering(), -1)
    not_imp_factual_df = pd.DataFrame(dict(zip(
      objs.dataset_obj.getInputAttributeNames(),
      [X_true.shape[0] * [not_imp_factual_instance[node]] for node in objs.dataset_obj.getInputAttributeNames()],
    )))
    not_imp_samples_df = X_true.copy()
    X_pred_prior = sampleCVAE(args, objs, not_imp_factual_instance, not_imp_factual_df, not_imp_samples_df, node, parents, 'm2_cvae', trained_cvae = trained_cvae)

    X_pred = X_pred_prior

    my_statistic, statistics, sigma_median = mmd.mmd_with_median_heuristic(X_true.to_numpy(), X_pred.to_numpy())
    print(f'\t\t[INFO] test-statistic = {my_statistic} using median of {sigma_median} as bandwith')

    trained_models[f'setup_{idx}'] = {}
    trained_models[f'setup_{idx}']['hyperparams'] = hyperparams
    trained_models[f'setup_{idx}']['trained_cvae'] = trained_cvae
    trained_models[f'setup_{idx}']['test-statistic'] = my_statistic

  # index_with_lowest_test_statistics = min(trained_models.keys(), key=lambda k: abs(trained_models[k]['test-statistic'] - 0))
  index_with_lowest_test_statistics = min(trained_models.keys(), key=lambda k: trained_models[k]['test-statistic'])
  model_with_lowest_test_statistics = trained_models[index_with_lowest_test_statistics]['trained_cvae']
  # save all results
  tmp_file_name = f'{experiment_folder_name}/_cvae_params_{getConditionalString(node, parents)}.txt'
  pprint(trained_models[index_with_lowest_test_statistics]['hyperparams'], open(tmp_file_name, 'w'))
  pprint(trained_models, open(tmp_file_name, 'a'))
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


def sampleTrue(args, objs, factual_instance, factual_df, samples_df, node, parents, recourse_type):
  # Step 1. [abduction]: compute noise or load from dataset using factual_instance
  # Step 2. [action]: (skip) this step is implicitly performed in the populated samples_df columns
  # Step 3. [prediction]: run through structural equation using noise and parents from samples_df
  structural_equation = objs.scm_obj.structural_equations_np[node]

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
      np.array(noise), # may be scalar, which will be case as pd.series when being summed.
      *[samples_df[parent] for parent in parents],
    )

  elif recourse_type == 'm2_true':

    samples_df[node] = structural_equation(
      np.array(objs.scm_obj.noises_distributions[getNoiseStringForNode(node)].sample(samples_df.shape[0])),
      *[samples_df[parent] for parent in parents],
    )

  return samples_df


def sampleRidgeKernelRidge(args, objs, factual_instance, factual_df, samples_df, node, parents, recourse_type):
  samples_df = processDataFrameOrDict(args, objs, samples_df.copy(), PROCESSING_SKLEARN)
  factual_instance = processDataFrameOrDict(args, objs, factual_instance.copy(), PROCESSING_SKLEARN)

  # Step 1. [abduction]: compute noise or load from dataset using factual_instance
  # Step 2. [action]: (skip) this step is implicitly performed in the populated samples_df columns
  # Step 3. [prediction]: run through structural equation using noise and parents from samples_df
  if recourse_type == 'm1_alin':
    trained_model = trainRidge(args, objs, node, parents)
  elif recourse_type == 'm1_akrr':
    trained_model = trainKernelRidge(args, objs, node, parents)
  else:
    raise Exception(f'{recourse_type} not recognized.')
  structural_equation = lambda noise, *parents_values: trained_model.predict([[*parents_values]])[0][0] + noise
  for row_idx, row in samples_df.iterrows():
    noise = _getAbductionNoise(args, objs, node, parents, factual_instance, structural_equation)
    samples_df.loc[row_idx, node] = structural_equation(
      noise,
      *samples_df.loc[row_idx, parents].to_numpy(),
    )
  samples_df = deprocessDataFrameOrDict(args, objs, samples_df, PROCESSING_SKLEARN)
  return samples_df


def sampleCVAE(args, objs, factual_instance, factual_df, samples_df, node, parents, recourse_type, trained_cvae = None):
  samples_df = processDataFrameOrDict(args, objs, samples_df.copy(), PROCESSING_CVAE)
  factual_instance = processDataFrameOrDict(args, objs, factual_instance.copy(), PROCESSING_CVAE)

  if trained_cvae is None: # amir: UGLY CODE
    trained_cvae = trainCVAE(args, objs, node, parents)

  if recourse_type == 'm1_cvae':
    sample_from = 'posterior'
  elif recourse_type == 'm2_cvae':
    sample_from = 'prior'
  elif recourse_type == 'm2_cvae_ps':
    sample_from = 'reweighted_prior'

  new_samples = trained_cvae.reconstruct(
    x_factual=factual_df[[node]],
    pa_factual=factual_df[parents],
    pa_counter=samples_df[parents],
    sample_from=sample_from,
  )
  new_samples = new_samples.rename(columns={0: node}) # bad code amir, this violates abstraction!
  samples_df[node] = new_samples
  samples_df = deprocessDataFrameOrDict(args, objs, samples_df, PROCESSING_CVAE)
  return samples_df


def sampleGP(args, objs, factual_instance, factual_df, samples_df, node, parents, recourse_type):
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


def _getCounterfactualTemplate(args, objs, factual_instance, action_set, recourse_type):
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

  return counterfactual_template


def _samplingInnerLoop(args, objs, factual_instance, action_set, recourse_type, num_samples):

  counterfactual_template = _getCounterfactualTemplate(args, objs, factual_instance, action_set, recourse_type)

  factual_df = pd.DataFrame(dict(zip(
    objs.dataset_obj.getInputAttributeNames(),
    [num_samples * [factual_instance[node]] for node in objs.dataset_obj.getInputAttributeNames()],
  )))
  # this dataframe has populated columns set to intervention or conditioning values
  # and has NaN columns that will be set accordingly.
  samples_df = pd.DataFrame(dict(zip(
    objs.dataset_obj.getInputAttributeNames(),
    [num_samples * [counterfactual_template[node]] for node in objs.dataset_obj.getInputAttributeNames()],
  )))


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
      elif recourse_type in {'m1_alin', 'm1_akrr'}:
        sampling_handle = sampleRidgeKernelRidge
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
        factual_df,
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


def _samplingInnerLoopTensor(args, objs, factual_instance, factual_instance_ts, action_set_ts, recourse_type):

  counterfactual_template_ts = _getCounterfactualTemplate(args, objs, factual_instance_ts, action_set_ts, recourse_type)

  if recourse_type in ACCEPTABLE_POINT_RECOURSE:
    num_samples = 1
  if recourse_type in ACCEPTABLE_DISTR_RECOURSE:
    num_samples = args.num_mc_samples

  # Initialize factual_ts, samples_ts
  factual_ts = torch.zeros((num_samples, len(objs.dataset_obj.getInputAttributeNames())))
  samples_ts = torch.zeros((num_samples, len(objs.dataset_obj.getInputAttributeNames())))
  for node in objs.scm_obj.getTopologicalOrdering():
    factual_ts[:, getColumnIndicesFromNames(args, objs, [node])] = factual_instance_ts[node] + 0 # + 0 not needed because not trainable but leaving in..
    # +0 important, specifically for tensor based elements, so we don't copy
    # an existing object in the computational graph, but we create a new node
    samples_ts[:, getColumnIndicesFromNames(args, objs, [node])] = counterfactual_template_ts[node] + 0

  # Simply traverse the graph in order, and populate nodes as we go!
  # IMPORTANT: DO NOT use SET(topo ordering); it sometimes changes ordering!
  for node in objs.scm_obj.getTopologicalOrdering():
    # set variable if value not yet set through intervention or conditioning
    if torch.any(torch.isnan(samples_ts[:, getColumnIndicesFromNames(args, objs, [node])])):
      parents = objs.scm_obj.getParentsForNode(node)
      # root nodes MUST always be set through intervention or conditioning
      assert len(parents) > 0
      # Confirm parents columns are present/have assigned values in samples_ts
      assert not torch.any(torch.isnan(samples_ts[:, getColumnIndicesFromNames(args, objs, parents)]))

      if recourse_type in {'m0_true', 'm2_true'}:

        structural_equation = objs.scm_obj.structural_equations_ts[node]

        if recourse_type == 'm0_true':
          # may be scalar, which will be case as pd.series when being summed.
          noise_pred = _getAbductionNoise(args, objs, node, parents, factual_instance_ts, structural_equation)
          noises = noise_pred
        elif recourse_type == 'm2_true':
          noises = torch.tensor(
            objs.scm_obj.noises_distributions[getNoiseStringForNode(node)].sample(samples_ts.shape[0])
          ).reshape(-1,1)

        samples_ts[:, getColumnIndicesFromNames(args, objs, [node])] = structural_equation(
          noises,
          *[samples_ts[:, getColumnIndicesFromNames(args, objs, [parent])] for parent in parents],
        )

      elif recourse_type in {'m1_alin', 'm1_akrr'}:

        if recourse_type == 'm1_alin':
          training_handle = trainRidge
          sampling_handle = skHelper.sample_from_LIN_model
        elif recourse_type == 'm1_akrr':
          training_handle = trainKernelRidge
          sampling_handle = skHelper.sample_from_KRR_model

        trained_model = training_handle(args, objs, node, parents).best_estimator_
        X_parents = samples_ts[:, getColumnIndicesFromNames(args, objs, parents)]

        # Step 1. [abduction]
        # TODO: we don't need structural_equation here... get the noise posterior some other way.
        structural_equation = lambda noise, *parents_values: trained_model.predict([[*parents_values]])[0][0] + noise
        noise = _getAbductionNoise(args, objs, node, parents, factual_instance_ts, structural_equation)

        # Step 2. [action]: (skip) this step is implicitly performed in the populated samples_ts columns
        # N/A

        # Step 3. [prediction]: first get the regressed value, then get noise
        new_samples = sampling_handle(trained_model, X_parents)
        assert np.isclose( # a simple check to make sure manual sklearn is working correct
          new_samples.item(),
          trained_model.predict(X_parents.detach().numpy()).item(),
          atol = 1e-3,
        )
        new_samples = new_samples + noise

        # add back to dataframe
        samples_ts[:, getColumnIndicesFromNames(args, objs, [node])] = new_samples + 0 # TODO: not sure if +0 is needed or not

      elif recourse_type in {'m1_gaus', 'm2_gaus'}:

        kernel, X_all, model = trainGP(args, objs, node, parents)
        X_parents = samples_ts[:, getColumnIndicesFromNames(args, objs, parents)]

        if recourse_type == 'm1_gaus': # counterfactual distribution for node
          # IMPORTANT: Find index of factual instance in dataframe used for training GP
          #            (earlier, the factual instance was appended as the last instance)
          # DO NOT DO THIS: conversion from float64 to torch and back will make it impossible to find the instance idx
          # factual_instance = {k:v.item() for k,v in factual_instance_ts.items()}
          tmp_idx = getIndexOfFactualInstanceInDataFrame( # TODO: write this as ts function as well?
            factual_instance,
            processDataFrameOrDict(args, objs, getOriginalDataFrame(objs, args.num_train_samples), PROCESSING_GAUS),
          ) # TODO: can probably rewrite to just evaluate the posterior again given the same result.. (without needing to look through the dataset)
          new_samples = gpHelper.sample_from_GP_model(model, X_parents, 'cf', tmp_idx)
        elif recourse_type == 'm2_gaus': # interventional distribution for node
          new_samples = gpHelper.sample_from_GP_model(model, X_parents, 'iv')

        samples_ts[:, getColumnIndicesFromNames(args, objs, [node])] = new_samples + 0 # TODO: not sure if +0 is needed or not

      elif recourse_type in {'m1_cvae', 'm2_cvae', 'm2_cvae_ps'}:

        if recourse_type == 'm1_cvae':
          sample_from = 'posterior'
        elif recourse_type == 'm2_cvae':
          sample_from = 'prior'
        elif recourse_type == 'm2_cvae_ps':
          sample_from = 'reweighted_prior'

        trained_cvae = trainCVAE(args, objs, node, parents)
        new_samples = trained_cvae.reconstruct(
          x_factual=factual_ts[:, getColumnIndicesFromNames(args, objs, [node])],
          pa_factual=factual_ts[:, getColumnIndicesFromNames(args, objs, parents)],
          pa_counter=samples_ts[:, getColumnIndicesFromNames(args, objs, parents)],
          sample_from=sample_from,
        )
        samples_ts[:, getColumnIndicesFromNames(args, objs, [node])] = new_samples + 0 # TODO: not sure if +0 is needed or not

  return samples_ts


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
  counter_instance = computeCounterfactualInstance(
    args,
    objs,
    factual_instance,
    action_set,
    recourse_type,
  )
  return objs.classifier_obj.predict_proba(
    np.expand_dims(np.array(list(counter_instance.values())), axis=0)
  )[0][1] > 0.5


def isDistrConstraintSatisfied(args, objs, factual_instance, action_set, recourse_type):
  return computeLowerConfidenceBound(args, objs, factual_instance, action_set, recourse_type) > 0.5


def computeLowerConfidenceBound(args, objs, factual_instance, action_set, recourse_type):
  monte_carlo_samples_df = getRecourseDistributionSample(
    args,
    objs,
    factual_instance,
    action_set,
    recourse_type,
    args.num_mc_samples,
  )
  # monte_carlo_predictions = getPredictionBatch(
  #   args,
  #   objs,
  #   monte_carlo_samples_df.to_numpy(),
  # ) # gives hard classification
  monte_carlo_predictions = objs.classifier_obj.predict_proba(monte_carlo_samples_df)[:,1] # class 1 probabilities.

  expectation = np.mean(monte_carlo_predictions)
  # variance = np.sum(np.power(monte_carlo_predictions - expectation, 2)) / (len(monte_carlo_predictions) - 1)
  std = np.std(monte_carlo_predictions)

  # return expectation, variance

  # IMPORTANT... WE ARE CONSIDERING {0,1} LABELS AND FACTUAL SAMPLES MAY BE OF
  # EITHER CLASS. THEREFORE, THE CONSTRAINT IS SATISFIED WHEN SIGNIFICANTLY
  # > 0.5 OR < 0.5 FOR A FACTUAL SAMPLE WITH Y = 0 OR Y = 1, RESPECTIVELY.

  if getPrediction(args, objs, factual_instance) == 0:
    return expectation - args.lambda_lcb * std # NOTE DIFFERNCE IN SIGN OF STD
  else: # factual_prediction == 1
    raise Exception(f'Should only be considering negatively predicted individuals...')
    # return expectation + args.lambda_lcb * np.sqrt(variance) # NOTE DIFFERNCE IN SIGN OF STD


def getValidDiscretizedActionSets(args, objs):

  possible_actions_per_node = []

  # IMPORTANT: you lose ordering of columns when using setdiff! This should not
  # matter in this part of the code, but may elsewhere. For alternative, see:
  # https://stackoverflow.com/questions/46261671/use-numpy-setdiff1d-keeping-the-order
  intervenable_nodes = np.setdiff1d(
    objs.dataset_obj.getInputAttributeNames('kurz'),
    args.non_intervenable_nodes,
  )

  for attr_name_kurz in intervenable_nodes:

    attr_obj = objs.dataset_obj.attributes_kurz[attr_name_kurz]

    if attr_obj.attr_type in {'numeric-real', 'numeric-int', 'binary'}:

      if attr_obj.attr_type == 'numeric-real':
        number_decimals = 5
      elif attr_obj.attr_type in {'numeric-int', 'binary'}:
        number_decimals = 0

      tmp = list(
        np.around(
          np.linspace(
            objs.dataset_obj.data_frame_kurz.describe()[attr_name_kurz]['min'],
            # bad code amir; don't access internal object attribute
            objs.dataset_obj.data_frame_kurz.describe()[attr_name_kurz]['max'],
            args.grid_search_bins
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
    dict(zip(intervenable_nodes, elem))
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

  # IMPORTANT: you lose ordering of columns when using setdiff! This should not
  # matter in this part of the code, but may elsewhere. For alternative, see:
  # https://stackoverflow.com/questions/46261671/use-numpy-setdiff1d-keeping-the-order
  intervenable_nodes = np.setdiff1d(
    objs.dataset_obj.getInputAttributeNames('kurz'),
    args.non_intervenable_nodes,
  )

  all_intervention_tuples = powerset(intervenable_nodes)
  all_intervention_tuples = [
    elem for elem in all_intervention_tuples
    if len(elem) <= args.max_intervention_cardinality
    and elem is not tuple() # no interventions (i.e., empty tuple) could never result in recourse --> ignore
  ]

  return all_intervention_tuples


def performGradDescentOptimization(args, objs, factual_instance, save_path, intervention_set, recourse_type):

  def saveLossCurve(save_path, intervention_set, best_action_set_epoch, all_logs):
    fig, axes = plt.subplots(2 + len(intervention_set), 1, sharex=True)

    axes[0].plot(all_logs['epochs'], all_logs['loss_cost'], 'g--', label='loss costs')
    axes[0].plot(all_logs['epochs'], all_logs['loss_constraint'], 'r:', label='loss constraints')
    axes[0].plot(best_action_set_epoch, all_logs['loss_cost'][best_action_set_epoch-1], 'b*')
    axes[0].plot(best_action_set_epoch, all_logs['loss_constraint'][best_action_set_epoch-1], 'b*')
    axes[0].text(best_action_set_epoch, all_logs['loss_cost'][best_action_set_epoch-1], f"{all_logs['loss_cost'][best_action_set_epoch-1]:.3f}", fontsize='xx-small')
    axes[0].text(best_action_set_epoch, all_logs['loss_constraint'][best_action_set_epoch-1], f"{all_logs['loss_constraint'][best_action_set_epoch-1]:.3f}", fontsize='xx-small')
    axes[0].set_ylabel('loss', fontsize='xx-small')
    axes[0].set_title('Loss curve', fontsize='xx-small')
    axes[0].grid()
    axes[0].set_ylim(
      min(-1, axes[0].get_ylim()[0]),
      max(+1, axes[0].get_ylim()[1]),
    )
    axes[0].legend(fontsize='xx-small')

    axes[1].plot(range(1, len(all_logs['loss_total']) + 1), all_logs['lambda_opt'], 'y-', label='lambda_opt')
    axes[1].set_ylabel('lambda', fontsize='xx-small')
    axes[1].grid()
    axes[1].legend(fontsize='xx-small')

    for idx, node in enumerate(intervention_set):
      # print intervention values
      tmp = [elem[node] for elem in all_logs['action_set']]
      axes[idx+2].plot(all_logs['epochs'], tmp, 'b-', label='lambda_opt')
      axes[idx+2].set_ylabel(node, fontsize='xx-small')
      if idx == len(intervention_set) - 1:
        axes[idx+2].set_xlabel('epochs', fontsize='xx-small')
      axes[idx+2].grid()
      axes[idx+2].legend(fontsize='xx-small')

    plt.savefig(f'{save_path}/{str(intervention_set)}.pdf')
    plt.close()

  # IMPORTANT: if you process factual_instance here, then action_set_ts and
  #            factual_instance_ts will also be normalized down-stream. Then
  #            at the end of this method, simply deprocess action_set_ts. One
  #            thing to note is the computation of distance may not be [0,1]
  #            in the processed settings (TODO?)
  if recourse_type in {'m0_true', 'm2_true'}:
    tmp_processing_type = 'raw'
  elif recourse_type in {'m1_alin', 'm1_akrr'}:
    tmp_processing_type = PROCESSING_SKLEARN
  elif recourse_type in {'m1_gaus', 'm2_gaus'}:
    tmp_processing_type = PROCESSING_GAUS
  elif recourse_type in {'m1_cvae', 'm2_cvae', 'm2_cvae_ps'}:
    tmp_processing_type = PROCESSING_CVAE
  factual_instance = processDataFrameOrDict(args, objs, factual_instance.copy(), tmp_processing_type)

  # IMPORTANT: action_set_ts includes trainable params, but factual_instance_ts does not.
  factual_instance_ts = {k: torch.tensor(v, dtype=torch.float32) for k, v in factual_instance.items()}

  action_set_ts = dict(zip(
    intervention_set,
    [
      torch.tensor(
        # np.random.randn(),
        # objs.dataset_obj.data_frame_kurz.describe()[node]['min'].astype('float32'),
        factual_instance[node],
        # getMinimumObservableInstance(args, objs, factual_instance)[node],
        requires_grad=True,
        dtype=torch.float32,
      ) # DO NOT USE .float(), this will create a new tensor (and can't optimize over non-leaf nodes)
      for node in intervention_set
    ]
  ))

  # TODO: make input args
  min_valid_cost = 1e6  # some large number
  no_decrease_in_min_valid_cost = 0
  early_stopping_K = 10
  # DO NOT USE .copy() on the dict, the same value objects (i.e., the same trainable tensor will be used!)
  best_action_set = {k : v.item() for k,v in action_set_ts.items()}
  best_action_set_epoch = 1
  recourse_satisfied = False

  capped_loss = False
  num_epochs = args.grad_descent_epochs
  lambda_opt = 1 # initial value
  lambda_opt_update_every = 25
  lambda_opt_learning_rate = 0.5
  action_set_learning_rate = 0.1
  print_log_every = lambda_opt_update_every
  optimizer = torch.optim.Adam(params = list(action_set_ts.values()), lr = action_set_learning_rate)

  all_logs = {}
  all_logs['epochs'] = []
  all_logs['loss_total'] = []
  all_logs['loss_cost'] = []
  all_logs['lambda_opt'] = []
  all_logs['loss_constraint'] = []
  all_logs['action_set'] = []

  start_time = time.time()
  if args.debug_flag:
    print(f'\t\t[INFO] initial action set: {str({k : np.around(v.item(), 4) for k,v in action_set_ts.items()})}') # TODO: use pretty print

  # https://stackoverflow.com/a/52017595/2759976
  iterator = tqdm(range(1, num_epochs + 1))
  for epoch in iterator:

    # ========================================================================
    # CONSTRUCT COMPUTATION GRAPH
    # ========================================================================

    samples_ts = _samplingInnerLoopTensor(args, objs, factual_instance, factual_instance_ts, action_set_ts, recourse_type)

    # ========================================================================
    # COMPUTE LOSS
    # ========================================================================

    loss_cost = measureActionSetCost(args, objs, factual_instance_ts, action_set_ts, tmp_processing_type)

    # get classifier
    h = getTorchClassifier(args, objs)
    pred_labels = h(samples_ts)

    # compute LCB
    if torch.isnan(torch.std(pred_labels)) or torch.std(pred_labels) < 1e-10:
      # When all predictions are the same (likely because all sampled points are
      # the same, likely because we are outside of the manifold OR, e.g., when we
      # intervene on all nodes and the initial epoch returns same samples), then
      # torch.std() will be 0 and therefore there is no gradient to pass back; in
      # turn this results in torch.std() giving nan and ruining the training!
      #     tmp = torch.ones((10,1), requires_grad=True)
      #     torch.std(tmp).backward()
      #     print(tmp.grad)
      # https://github.com/pytorch/pytorch/issues/4320
      # SOLUTION: remove torch.std() when this term is small to prevent passing nans.
      value_lcb = torch.mean(pred_labels)
    else:
      value_lcb = torch.mean(pred_labels) - args.lambda_lcb * torch.std(pred_labels)

    loss_constraint = (0.5 - value_lcb)
    if capped_loss:
      loss_constraint = torch.nn.functional.relu(loss_constraint)

    # for fixed lambda, optimize theta (grad descent)
    loss_total = loss_cost + lambda_opt * loss_constraint

    # ========================================================================
    # EARLY STOPPING
    # ========================================================================

    # check if constraint is satisfied
    if value_lcb.detach() > 0.5:
      # check if cost decreased from previous best
      if loss_cost.detach() < min_valid_cost:
        min_valid_cost = loss_cost.item()
        # DO NOT USE .copy() on the dict, the same value objects (i.e., the same trainable tensor will be used!)
        best_action_set = {k : v.item() for k,v in action_set_ts.items()}
        best_action_set_epoch = epoch
        recourse_satisfied = True
      else:
        no_decrease_in_min_valid_cost += 1

    # stop if past K valid thetas did not improve upon best previous cost
    if no_decrease_in_min_valid_cost > early_stopping_K:
      saveLossCurve(save_path, intervention_set, best_action_set_epoch, all_logs)
      # https://stackoverflow.com/a/52017595/2759976
      iterator.close()
      break

    # ========================================================================
    # OPTIMIZE
    # ========================================================================

    # once every few epochs, optimize theta (grad ascent) manually (w/o pytorch)
    if epoch % lambda_opt_update_every == 0:
      lambda_opt = lambda_opt + lambda_opt_learning_rate * loss_constraint.detach()

    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

    # ========================================================================
    # LOGS / IMAGES
    # ========================================================================

    if args.debug_flag and epoch % print_log_every == 0:
      print(
        f'\t\t[INFO] epoch #{epoch:03}: ' \
        f'optimal action: {str({k : np.around(v.item(), 4) for k,v in action_set_ts.items()})}    ' \
        f'loss_total: {loss_total.item():02.6f}    ' \
        f'loss_cost: {loss_cost.item():02.6f}    ' \
        f'lambda_opt: {lambda_opt:02.6f}    ' \
        f'loss_constraint: {loss_constraint.item():02.6f}    ' \
        f'value_lcb: {value_lcb.item():02.6f}    ' \
      )
    all_logs['epochs'].append(epoch)
    all_logs['loss_total'].append(loss_total.item())
    all_logs['loss_cost'].append(loss_cost.item())
    all_logs['lambda_opt'].append(lambda_opt)
    all_logs['loss_constraint'].append(loss_constraint.item())
    all_logs['action_set'].append({k : v.item() for k,v in action_set_ts.items()})

    if epoch % 100 == 0:
      saveLossCurve(save_path, intervention_set, best_action_set_epoch, all_logs)

  end_time = time.time()
  if args.debug_flag:
    print(f'\t\t[INFO] Done (total run-time: {end_time - start_time}).\n\n')

  # Convert action_set_ts to non-tensor action_set when passing back to rest of code.
  # best_action_set may or may not be result of early stopping, but it will
  # either be the initial value (which was zero-cost, at the factual instance),
  # or it will be the best_action_set seen so far (smallest cost and valid const)
  # whether or not it triggered K times to initiate early stopping.
  action_set = {k : v for k,v in best_action_set.items()}
  action_set = deprocessDataFrameOrDict(args, objs, action_set, tmp_processing_type)
  return action_set, recourse_satisfied, min_valid_cost


def computeOptimalActionSet(args, objs, factual_instance, save_path, recourse_type):

  # assert factual instance has prediction = 0
  assert objs.classifier_obj.predict_proba(
    np.expand_dims(np.array(list(factual_instance.values())), axis=0)
  )[0][0] >= .50 + args.epsilon_boundary

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
    print(f'\n\t[INFO] Computing optimal `{recourse_type}`: grad descent over {len(valid_intervention_sets)} intervention sets (max card: {args.max_intervention_cardinality})...')

    min_cost = 1e10
    min_cost_action_set = {}
    for idx, intervention_set in enumerate(valid_intervention_sets):
      # print(f'\n\t[INFO] intervention set #{idx+1}/{len(valid_intervention_sets)}: {str(intervention_set)}')
      # plotOptimizationLandscape(args, objs, factual_instance, save_path, intervention_set, recourse_type)
      action_set, recourse_satisfied, cost_of_action_set = performGradDescentOptimization(args, objs, factual_instance, save_path, intervention_set, recourse_type)
      if constraint_handle(args, objs, factual_instance, action_set, recourse_type):
        assert recourse_satisfied # a bit redundant, but just in case, we check
                                  # that the MC samples from constraint_handle()
                                  # on the line above and the MC samples from
                                  # performGradDescentOptimization() both agree
                                  # that recourse has been satisfied
        assert np.isclose( # won't be exact becuase the former is float32 tensor
          cost_of_action_set,
          measureActionSetCost(args, objs, factual_instance, action_set),
          atol = 1e-2,
        )
        if cost_of_action_set < min_cost:
          min_cost = cost_of_action_set
          min_cost_action_set = action_set

    print(f'\t done (optimal intervention set: {str(min_cost_action_set)}).')

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
  number_of_samples_to_plot = args.num_display_samples
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
  # scatterDecisionBoundary(args, objs, ax)

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

  # Only focus on instances with h(x^f) = 0 and therfore h(x^cf) = 1; do not use
  # processDataFrameOrDict because classifier is trained on original data
  X_all = getOriginalDataFrame(objs, args.num_train_samples)

  # X_all = getOriginalDataFrame(objs, args.num_train_samples + args.num_validation_samples)
  # # CANNOT DO THIS:Iterate over validation set, not training set
  # # REASON: for m0_true we need the index of the factual instance to get noise
  # # variable for abduction and for m1_gaus we need the index as well.
  # X_all = X_all.iloc[args.num_train_samples:]

  predict_proba_list = objs.classifier_obj.predict_proba(X_all)[:,0]
  predict_proba_in_negative_class = predict_proba_list > 0.5 + args.epsilon_boundary
  negatively_predicted_instances = X_all[predict_proba_in_negative_class]
  factual_instances_dict = negatively_predicted_instances[
    args.batch_number * args.sample_count : (args.batch_number + 1) * args.sample_count
  ].T.to_dict()
  assert len(factual_instances_dict.keys()) == args.sample_count, 'Not enough samples.'
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

    print(f'\n\n[INFO] ACTION SET: {str(prettyPrintDict(action_set))}' + ' =' * 60)

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
  plt.close()


def experiment6(args, objs, experiment_folder_name, factual_instances_dict, experimental_setups, recourse_types):
  ''' optimal action set: figure + table '''

  os.mkdir(f'{experiment_folder_name}/_optimization_curves')

  per_instance_results = {}
  for enumeration_idx, (key, value) in enumerate(factual_instances_dict.items()):
    factual_instance_idx = f'sample_{key}'
    factual_instance = value

    os.mkdir(f'{experiment_folder_name}/_optimization_curves/factual_instance_{factual_instance_idx}')

    print(f'\n\n\n[INFO] Processing factual instance `{factual_instance_idx}` (#{enumeration_idx + 1} / {len(factual_instances_dict.keys())})...')

    per_instance_results[factual_instance_idx] = {}
    per_instance_results[factual_instance_idx]['factual_instance'] = factual_instance

    for recourse_type in recourse_types:

      tmp = {}
      save_path = f'{experiment_folder_name}/_optimization_curves/factual_instance_{factual_instance_idx}/{recourse_type}'
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
      tmp['cost_all'] = measureActionSetCost(args, objs, factual_instance, tmp['optimal_action_set'])
      tmp['cost_valid'] = tmp['cost_all'] if tmp['scf_validity'] else np.NaN

      # print(f'\t done.')

      per_instance_results[factual_instance_idx][recourse_type] = tmp

    print(f'[INFO] Saving (overwriting) results...')
    pickle.dump(per_instance_results, open(f'{experiment_folder_name}/_per_instance_results', 'wb'))
    pprint(per_instance_results, open(f'{experiment_folder_name}/_per_instance_results.txt', 'w'))
    print(f'done.')

    createAndSaveMetricsTable(per_instance_results, recourse_types, experiment_folder_name)


def createAndSaveMetricsTable(per_instance_results, recourse_types, experiment_folder_name):
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
  print(f'\nN = {len(per_instance_results.keys())}')
  tmp_df.to_csv(f'{experiment_folder_name}/_comparison.txt', sep='\t')
  with open(f'{experiment_folder_name}/_comparison.txt', 'a') as out_file:
    out_file.write(f'\nN = {len(per_instance_results.keys())}\n')
  tmp_df.to_pickle(f'{experiment_folder_name}/_comparison')

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
  plt.close()


# DEPRECATED def experiment7


def experiment8(args, objs, experiment_folder_name, factual_instances_dict, experimental_setups, recourse_types):
  ''' box-plot sanity '''

  PER_DIM_GRANULARITY = 8

  recourse_types = [elem for elem in recourse_types if elem in {'m2_true', 'm2_gaus', 'm2_cvae'}]
  if len(recourse_types) == 0:
    print(f'[INFO] Exp 8 is only designed for m2 recourse_type; skipping.')
    return

  for node in objs.scm_obj.getTopologicalOrdering():

    parents = objs.scm_obj.getParentsForNode(node)

    if len(parents) == 0: # if not a root node
      continue # don't want to plot marginals, because we're not learning these

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

        print(f'\n\n[INFO] ACTION SET: {str(prettyPrintDict(action_set))}' + ' =' * 60)

        for recourse_type in recourse_types:

          if recourse_type == 'm2_true':
            samples = getRecourseDistributionSample(args, objs, factual_instance, action_set, 'm2_true', args.num_validation_samples)
          elif recourse_type == 'm2_gaus':
            samples = getRecourseDistributionSample(args, objs, factual_instance, action_set, 'm2_gaus', args.num_validation_samples)
          elif recourse_type == 'm2_cvae':
            samples = getRecourseDistributionSample(args, objs, factual_instance, action_set, 'm2_cvae', args.num_validation_samples)

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
      plt.close()

    else:
      # distribution plot

      total_df = pd.DataFrame(columns=['recourse_type'] + list(objs.scm_obj.getTopologicalOrdering()))

      X_all = processDataFrameOrDict(args, objs, getOriginalDataFrame(objs, args.num_train_samples  + args.num_validation_samples), PROCESSING_CVAE)
      X_val = X_all[args.num_train_samples:].copy()

      X_true = X_val[parents + [node]]

      not_imp_factual_instance = dict.fromkeys(objs.scm_obj.getTopologicalOrdering(), -1)
      not_imp_factual_df = pd.DataFrame(dict(zip(
        objs.dataset_obj.getInputAttributeNames(),
        [X_true.shape[0] * [not_imp_factual_instance[node]] for node in objs.dataset_obj.getInputAttributeNames()],
      )))
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

        samples = sampling_handle(args, objs, not_imp_factual_instance, not_imp_factual_df, not_imp_samples_df, node, parents, recourse_type)
        tmp_df = samples.copy()
        tmp_df['recourse_type'] = recourse_type # add column
        total_df = pd.concat([total_df, tmp_df]) # concat to overall

      ax = sns.boxplot(x='recourse_type', y=node, data=total_df, palette='Set3', showmeans=True)
      plt.savefig(f'{experiment_folder_name}/_sanity_{getConditionalString(node, parents)}.pdf')
      plt.close()


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument('-s', '--scm_class', type=str, default='sanity-3-lin', help='Name of SCM to generate data using (see loadSCM.py)')
  parser.add_argument('-d', '--dataset_class', type=str, default='random', help='Name of dataset to train explanation model for: german, random, mortgage, twomoon')
  parser.add_argument('-c', '--classifier_class', type=str, default='lr', help='Model class that will learn data: lr, mlp')
  parser.add_argument('-e', '--experiment', type=int, default=6, help='Which experiment to run (5,8=sanity; 6=table)')
  parser.add_argument('-p', '--process_id', type=str, default='0', help='When running parallel tests on the cluster, process_id guarantees (in addition to time stamped experiment folder) that experiments do not conflict.')

  parser.add_argument('--norm_type', type=int, default=2)
  parser.add_argument('--lambda_lcb', type=float, default=1)
  parser.add_argument('--num_train_samples', type=int, default=250)
  parser.add_argument('--num_validation_samples', type=int, default=250)
  parser.add_argument('--num_display_samples', type=int, default=15)
  parser.add_argument('--num_mc_samples', type=int, default=100)
  parser.add_argument('--debug_flag', type=bool, default=False)
  parser.add_argument('--non_intervenable_nodes', nargs = '+', type=str, default='')
  parser.add_argument('--max_intervention_cardinality', type=int, default=100)
  parser.add_argument('-o', '--optimization_approach', type=str, default='brute_force')
  parser.add_argument('--grid_search_bins', type=int, default=20)
  parser.add_argument('--grad_descent_epochs', type=int, default=1000)
  parser.add_argument('--epsilon_boundary', type=int, default=0.15, help='we only consider instances that are negatively predicted and at least epsilon_boundary prob away from decision boundary.')
  parser.add_argument('--batch_number', type=int, default=0)
  parser.add_argument('--sample_count', type=int, default=10)

  args = parser.parse_args()

  if not (args.dataset_class in {'random'}):
    raise Exception(f'{args.dataset_class} not supported.')

  # create experiment folder
  setup_name = \
    f'{args.scm_class}__{args.dataset_class}__{args.classifier_class}' + \
    f'__ntrain_{args.num_train_samples}' + \
    f'__nmc_{args.num_mc_samples}' + \
    f'__lambda_lcb_{args.lambda_lcb}' + \
    f'__opt_{args.optimization_approach}' + \
    f'__batch_{args.batch_number}' + \
    f'__count_{args.sample_count}' + \
    f'__pid{args.process_id}'
  experiment_folder_name = f"_experiments/{datetime.now().strftime('%Y.%m.%d_%H.%M.%S')}__{setup_name}"
  args.experiment_folder_name = experiment_folder_name
  os.mkdir(f'{experiment_folder_name}')

  # save all arguments to file
  args_file = open(f'{experiment_folder_name}/_args.txt','w')
  for arg in vars(args):
    print(arg, ':\t', getattr(args, arg), file = args_file)

  # only load once so shuffling order is the same
  scm_obj = loadCausalModel(args, experiment_folder_name)
  dataset_obj = loadDataset(args, experiment_folder_name)
  classifier_obj = loadClassifier(args, experiment_folder_name)
  # IMPORTANT: for traversing, always ONLY use either:
  #     * objs.dataset_obj.getInputAttributeNames()
  #     * objs.scm_obj.getTopologicalOrdering()
  # DO NOT USE, e.g., for key in factual_instance.keys(), whose ordering may differ!
  # IMPORTANT: ordering may be [x3, x2, x1, x4, x5, x6] as is the case of the
  # 6-variable model.. this is OK b/c the dataset is generated as per this order
  # and consequently the model is trained as such as well (where the 1st feature
  # is x3 in the example above)
  assert \
    list(scm_obj.getTopologicalOrdering()) == \
    list(dataset_obj.getInputAttributeNames()) == \
    [elem for elem in dataset_obj.data_frame_kurz.columns if 'x' in elem] # super hack amir, but necessary right now (not much time)

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
    args.num_display_samples = 150
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
  ]

  if args.experiment == 5:

    assert \
      len(objs.dataset_obj.getInputAttributeNames()) == 3, \
      'Exp 5 is only designed for 3-variable SCMs'

  elif args.experiment == 6:

    assert \
      len(objs.dataset_obj.getInputAttributeNames()) >= 3, \
      'Exp 6 is only designed for 3+-variable SCMs'

  recourse_types = [experimental_setup[0] for experimental_setup in experimental_setups]
  hotTrainRecourseTypes(args, objs, recourse_types)

  if args.experiment == 5:
    experiment5(args, objs, experiment_folder_name, factual_instances_dict, experimental_setups, recourse_types)
  elif args.experiment == 6:
    experiment8(args, objs, experiment_folder_name, factual_instances_dict, experimental_setups, recourse_types)
    experiment6(args, objs, experiment_folder_name, factual_instances_dict, experimental_setups, recourse_types)
  elif args.experiment == 8:
    experiment8(args, objs, experiment_folder_name, factual_instances_dict, experimental_setups, recourse_types)

  # sanity check
  # visualizeDatasetAndFixedModel(args, objs)

























