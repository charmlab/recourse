# def warn(*args, **kwargs):
#     pass

# import warnings
# warnings.warn = warn # to ignore all warnings.

import sys
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import utils
import loadData
import fairRecourse
from scatter import *

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from _third_party.svm_recourse import RecourseSVM

from debug import ipsh

from random import seed
RANDOM_SEED = 54321
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

# TODO: change to be like _data_main below, and make python module
# this answer https://stackoverflow.com/a/50474562 and others
try:
  import treeUtils
except:
  print('[ENV WARNING] treeUtils not available')

SIMPLIFY_TREES = False


def trainFairClassifier(model_class, fair_kernel_type):
  if model_class != 'iw_fair_svm':

    if fair_kernel_type == 'linear':
      param_grid = [{'C': np.logspace(0, 2, 3), 'kernel': ['linear']}]
    elif fair_kernel_type == 'poly':
      param_grid = [{'C': np.logspace(0, 2, 3), 'kernel': ['poly'], 'degree':[2, 3, 5]}]
    elif fair_kernel_type == 'rbf':
      param_grid = [{'C': np.logspace(0, 2, 3), 'gamma': np.logspace(-3,0,4), 'kernel': ['rbf']}]
    elif fair_kernel_type == 'all':
      param_grid = [
        {'C': np.logspace(0, 2, 3), 'kernel': ['linear']},
        {'C': np.logspace(0, 2, 3), 'kernel': ['poly'], 'degree':[2, 3, 5]},
        {'C': np.logspace(0, 2, 3), 'gamma': np.logspace(-3,0,4), 'kernel': ['rbf']},
      ]
    else:
      raise Exception(f'unrecognized fair_kernel_type: {fair_kernel_type}')


    return GridSearchCV(estimator=SVC(probability=True), param_grid=param_grid, n_jobs=-1)

  else:

    # Note: regularisation strength C is referred to as 'ups' in RecourseSVM and is fixed to 10 by default;
    # (this correspondes to the Greek nu in the paper, see the primal form on p.3 of https://arxiv.org/pdf/1909.03166.pdf )
    lams = [0.2, 0.5, 1, 2, 10, 50, 100]
    if fair_kernel_type == 'linear':
      param_grid = [{'lam': lams, 'kernel_fn': ['linear']}]
    elif fair_kernel_type == 'rbf':
      param_grid = [{'lam': lams, 'kernel_fn': ['poly'], 'degree':[2, 3, 5]}]
    elif fair_kernel_type == 'poly':
      param_grid = [{'lam': lams, 'kernel_fn': ['rbf'], 'gamma': np.logspace(-3,0,4)}]
    elif fair_kernel_type == 'all':
      param_grid = [
        {'lam': lams, 'kernel_fn': ['linear']},
        {'lam': lams, 'kernel_fn': ['poly'], 'degree':[2, 3, 5]},
        {'lam': lams, 'kernel_fn': ['rbf'], 'gamma': np.logspace(-3,0,4)},
      ]
    else:
      raise Exception(f'unrecognized fair_kernel_type: {fair_kernel_type}')

    return GridSearchCV(estimator=RecourseSVM(), param_grid=param_grid, n_jobs=-1)



@utils.Memoize
def loadModelForDataset(model_class, dataset_string, scm_class = None, num_train_samples = 1e5, fair_nodes = None, fair_kernel_type = None, experiment_folder_name = None):

  log_file = sys.stdout if experiment_folder_name == None else open(f'{experiment_folder_name}/log_training.txt','w')

  if not (model_class in {'lr', 'mlp', 'tree', 'forest'}) and not (model_class in fairRecourse.FAIR_MODELS):
      raise Exception(f'{model_class} not supported.')

  if not (dataset_string in {'synthetic', 'mortgage', 'twomoon', 'german', 'credit', 'compass', 'adult', 'test'}):
    raise Exception(f'{dataset_string} not supported.')

  dataset_obj = loadData.loadDataset(dataset_string, return_one_hot = True, load_from_cache = False, meta_param = scm_class)

  if model_class not in fairRecourse.FAIR_MODELS:
    X_train, X_test, y_train, y_test = dataset_obj.getTrainTestSplit()
    y_all = pd.concat([y_train, y_test], axis = 0)
    assert sum(y_all) / len(y_all) == 0.5, 'Expected class balance should be 50/50%.'
  else:
    X_train, X_test, U_train, U_test, y_train, y_test = dataset_obj.getTrainTestSplit(with_meta = True, balanced = False)
    X_train = pd.concat([X_train, U_train], axis = 1)[fair_nodes]
    X_test = pd.concat([X_test, U_test], axis = 1)[fair_nodes]
    y_train = y_train * 2 - 1
    y_test = y_test * 2 - 1

  if model_class == 'tree':
    model_pretrain = DecisionTreeClassifier()
  elif model_class == 'forest':
    model_pretrain = RandomForestClassifier()
  elif model_class == 'lr':
    # IMPORTANT: The default solver changed from ‘liblinear’ to ‘lbfgs’ in 0.22;
    #            therefore, results may differ slightly from paper.
    model_pretrain = LogisticRegression() # default penalty='l2', i.e., ridge
  elif model_class == 'mlp':
    model_pretrain = MLPClassifier(hidden_layer_sizes = (10, 10))
  else:
    model_pretrain = trainFairClassifier(model_class, fair_kernel_type)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

  X_train = X_train[:num_train_samples]
  y_train = y_train[:num_train_samples]

  training_setup_string = f'[INFO] Training `{model_class}` on {X_train.shape[0]:,} samples ' + \
    f'(%{100 * X_train.shape[0] / (X_train.shape[0] + X_test.shape[0]):.2f}' + \
    f'of {X_train.shape[0] + X_test.shape[0]:,} samples)...'
  print(training_setup_string, file=log_file)
  print(training_setup_string)

  model_trained = model_pretrain.fit(X_train, y_train)

  train_accuracy_string = f'\t[INFO] Training accuracy: %{accuracy_score(y_train, model_trained.predict(X_train)) * 100:.2f}.'
  test_accuracy_string = f'\t[INFO] Testing accuracy: %{accuracy_score(y_test, model_trained.predict(X_test)) * 100:.2f}.'
  hyperparams_string = f'\t[INFO] Hyper-parameters of best classifier selected by CV:\n\t{model_trained}'

  print(train_accuracy_string, file=log_file)
  print(test_accuracy_string, file=log_file)
  print(hyperparams_string, file=log_file)
  print(train_accuracy_string)
  print(test_accuracy_string)
  print(hyperparams_string)

  # shouldn't deal with bad model; arbitrarily select offset to be 70% accuracy
  tmp = accuracy_score(y_train, model_trained.predict(X_train))

  # TODO (fair): added try except loop for use of nonlinear classifiers in fairness experiments
  try:
    assert tmp > 0.70, f'Model accuracy only {tmp}'
  except:
    print('[INFO] logistic regression accuracy may be low (<70%)')
    pass

  classifier_obj = model_trained
  visualizeDatasetAndFixedModel(dataset_obj, classifier_obj, experiment_folder_name)

  feature_names = dataset_obj.getInputAttributeNames('kurz') # easier to read (nothing to do with one-hot vs non-hit!)
  if model_class == 'tree':
    if SIMPLIFY_TREES:
      print('[INFO] Simplifying decision tree...', end = '', file=log_file)
      model_trained.tree_ = treeUtils.simplifyDecisionTree(model_trained, False)
      print('\tdone.', file=log_file)
    # treeUtils.saveTreeVisualization(model_trained, model_class, '', X_test, feature_names, experiment_folder_name)
  elif model_class == 'forest':
    for tree_idx in range(len(model_trained.estimators_)):
      if SIMPLIFY_TREES:
        print(f'[INFO] Simplifying decision tree (#{tree_idx + 1}/{len(model_trained.estimators_)})...', end = '', file=log_file)
        model_trained.estimators_[tree_idx].tree_ = treeUtils.simplifyDecisionTree(model_trained.estimators_[tree_idx], False)
        print('\tdone.', file=log_file)
      # treeUtils.saveTreeVisualization(model_trained.estimators_[tree_idx], model_class, f'tree{tree_idx}', X_test, feature_names, experiment_folder_name)

  if experiment_folder_name:
    pickle.dump(model_trained, open(f'{experiment_folder_name}/_model_trained', 'wb'))

  return model_trained

