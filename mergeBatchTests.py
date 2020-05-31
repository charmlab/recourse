import os
import glob
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from shutil import copyfile

ACCEPTABLE_POINT_RECOURSE = {'m0_true', 'm1_alin', 'm1_akrr'}
ACCEPTABLE_DISTR_RECOURSE = {'m1_gaus', 'm1_cvae', 'm2_true', 'm2_gaus', 'm2_cvae', 'm2_cvae_ps'}

from debug import ipsh

SCM_CLASS_VALUES = ['sanity-3-lin', 'sanity-3-anm', 'sanity-3-gen']
LAMBDA_LCB_VALUES = [1] # np.linspace(0,2.5,6)
OPTIMIZATION_APPROACHES = ['brute_force', 'grad_descent']

# experiments_folder_path = '/Volumes/amir/dev/recourse/_experiments/'
experiments_folder_path = '/Users/a6karimi/dev/recourse/_experiments/'
all_counter = len(SCM_CLASS_VALUES) * len(LAMBDA_LCB_VALUES) * len(OPTIMIZATION_APPROACHES)
counter = 0


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
  tmp_df.to_csv(f'{experiment_folder_name}/_comparison.txt', sep='\t')
  with open(f'{experiment_folder_name}/_comparison.txt', 'a') as out_file:
    out_file.write(f'\nN = {len(per_instance_results.keys())}')
  tmp_df.to_pickle(f'{experiment_folder_name}/_comparison')



for scm_class in SCM_CLASS_VALUES:

  for lambda_lcb in LAMBDA_LCB_VALUES:

    for optimization_approach in OPTIMIZATION_APPROACHES:

      counter = counter + 1

      specific_experiment_path = f'{scm_class}__*__lambda_lcb_{lambda_lcb}__opt_{optimization_approach}'
      # specific_experiment_path = 'adult__mlp__zero_norm__MACE_eps_1e-5'

      print(f'\n[{counter} / {all_counter}] Merging together folders for {specific_experiment_path}')

      all_batch_folders = glob.glob(f'{experiments_folder_path}*{specific_experiment_path}*')
      all_batch_folders = [elem for elem in all_batch_folders if 'batch' in elem and 'count' in elem]
      sorted_all_batch_folders = sorted(all_batch_folders, key = lambda x : x.split('__')[-3]) # sort based on batch_#
      total_per_instance_results = {}
      folders_not_found = []
      for batch_folder in tqdm(sorted_all_batch_folders):
        batch_number_string = batch_folder.split('__')[-3]
        batch_per_instance_results_path = os.path.join(batch_folder, '_per_instance_results')
        try:
          assert os.path.isfile(batch_per_instance_results_path)
          batch_per_instance_results = pickle.load(open(batch_per_instance_results_path, 'rb'))
          total_per_instance_results = {**total_per_instance_results, **batch_per_instance_results}
        except:
          folders_not_found.append(batch_number_string)

      if len(folders_not_found):
        print(f'\tCannot find minimum distance file for {folders_not_found}')

      # create new folder
      random_batch_folder = sorted_all_batch_folders[0]
      new_folder_name = '__'.join(random_batch_folder.split('/')[-1].split('__')[:-3])
      # new_folder_path = os.path.join(experiments_folder_path, '__merged_MACE_eps_1e-5', new_folder_name)
      new_folder_path = os.path.join(experiments_folder_path, '__merged', new_folder_name)

      print(f'Creating new merged folder {new_folder_path}')
      os.makedirs(new_folder_path, exist_ok = False)
      files_to_copy = {'_args.txt', 'causal_graph.pdf', 'log_training.txt'}
      for file_name in files_to_copy:
        copyfile(
          os.path.join(random_batch_folder, file_name),
          os.path.join(new_folder_path, file_name)
        )

      pickle.dump(total_per_instance_results, open(f'{new_folder_path}/_total_per_instance_results', 'wb'))
      pprint(total_per_instance_results, open(f'{new_folder_path}/total_per_instance_results.txt', 'w'))

      random_key = list(total_per_instance_results.keys())[0]
      recourse_types = [
        elem for elem in total_per_instance_results[random_key].keys()
        if elem in ACCEPTABLE_POINT_RECOURSE or elem in ACCEPTABLE_DISTR_RECOURSE
      ]
      createAndSaveMetricsTable(total_per_instance_results, recourse_types, new_folder_path)

