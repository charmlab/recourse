import os
import glob
import tqdm
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 230)


from pprint import pprint

from debug import ipsh

SCM_CLASS_VALUES = ['sanity-3-lin', 'sanity-3-anm', 'sanity-3-gen']
LAMBDA_LCB_VALUES = [0,1,2]
NUM_MC_SAMPLES_VALUES = [100, 300, 1000]


DATASET_VALUES = ['random'] # , 'twomoon'] #, 'mortgage'] #, 'twomoon'] # , 'credit']
MODEL_CLASS_VALUES = ['lr', 'mlp']
WORLD_TYPE_VALUES = ['indep', 'dep_structural']
# WORLD_TYPE_VALUES = ['indep']

# model_based_parent_path = '/Users/a6karimi/dev/counterfactual_deep_learning/_runs_bu/2020.04.03_stable_runs_post_labeling_fix/'
# verif_based_parent_path = '/Users/a6karimi/dev/counterfactual_deep_learning/_runs_bu/2020.04.03__merged_verif_vs_model_post_label_bugfix/'
model_based_parent_path = '/Users/a6karimi/dev/counterfactual_deep_learning/_runs_bu/2020.04.06_merged_model_{random,mortgage}_{lr,mlp}_{indep,dep_str}/'
verif_based_parent_path = '/Users/a6karimi/dev/counterfactual_deep_learning/_runs_bu/2020.04.07_merged_verif_{random,mortgage}_{lr,mlp}_{indep,dep_str}_eps_1e-3/'
# verif_based_parent_path = '/Users/a6karimi/dev/counterfactual_deep_learning/_runs_bu/2020.04.08_merged_verif_{random,mortgage}_{lr,mlp}_{indep,dep_str}_eps_1e-5/'

model_based_parent_path = '/Users/a6karimi/dev/counterfactual_deep_learning/_runs_bu/2020.04.11_merged_new_{random,twomoon}_{lr,mlp}_{indep,dep_str}/'
freee_based_parent_path = '/Users/a6karimi/dev/counterfactual_deep_learning/_runs_bu/2020.04.11_merged_new_{random,twomoon}_{lr,mlp}_{indep,dep_str}/'
verif_based_parent_path = '/Users/a6karimi/dev/counterfactual_deep_learning/_runs_bu/2020.04.11_merged_verif_{random,twomoon}_{lr,mlp}_{indep,dep_str}/'

parent_path = '/Volumes/amir/dev/recourse/_experiments'


def convertModelBasedResultsToDataFrame(experiment_results, run_idx):
  return pd.DataFrame.from_dict(experiment_results, orient='index') \
    .drop(columns=['fac_sample', 'int_sample', 'scf_sample', 'do_update', 'x_delta']) \
    .rename(columns={
      'scf_found': f'model_{run_idx+1}_scf_found',
      'scf_plausible': f'model_{run_idx+1}_scf_plausible',
      'scf_time': f'model_{run_idx+1}_scf_time',
      'scf_distance': f'model_{run_idx+1}_scf_distance',
      'int_distance': f'model_{run_idx+1}_int_distance',
    })


def convertFreeeBasedResultsToDataFrame(experiment_results, run_idx):
  return pd.DataFrame.from_dict(experiment_results, orient='index') \
    .drop(columns=['fac_sample', 'int_sample', 'scf_sample', 'do_update', 'x_delta']) \
    .rename(columns={
      'scf_found': f'freee_{run_idx+1}_scf_found',
      'scf_plausible': f'freee_{run_idx+1}_scf_plausible',
      'scf_time': f'freee_{run_idx+1}_scf_time',
      'scf_distance': f'freee_{run_idx+1}_scf_distance',
      'int_distance': f'freee_{run_idx+1}_int_distance',
    })


def convertVerifBasedResultsToDataFrame(experiment_results, world_type_string):
  if world_type_string == 'indep':
    tmp = pd.DataFrame.from_dict(experiment_results, orient='index') \
      .drop(columns=['fac_sample', 'cfe_sample']) \
      .rename(columns={
        'cfe_found': f'verif_scf_found', # mace/mint + scf/cfe
        'cfe_plausible': f'verif_scf_plausible', # mace/mint + scf/cfe
        'cfe_time': f'verif_scf_time', # mace/mint + scf/cfe
        'cfe_distance': f'verif_scf_distance', # mace/mint + scf/cfe
        # TOOD: copy column# 'int_distance': f'verif_int_distance', # mace/mint + scf/cfe # TODO: why doesn't this shw?? because mace? yes
      })
    tmp['verif_int_distance'] = tmp['verif_scf_distance'] # overlap in indep world (but isn't returned by mace... because not optimized that way)
    return tmp
  elif world_type_string == 'dep_structural':
    return pd.DataFrame.from_dict(experiment_results, orient='index') \
      .drop(columns=['fac_sample', 'scf_sample', 'int_sample', 'action_set']) \
      .rename(columns={
        'scf_found': f'verif_scf_found', # mace/mint + scf/cfe
        'scf_plausible': f'verif_scf_plausible', # mace/mint + scf/cfe
        'scf_time': f'verif_scf_time', # mace/mint + scf/cfe
        'scf_distance': f'verif_scf_distance', # mace/mint + scf/cfe
        'int_distance': f'verif_int_distance', # mace/mint + scf/cfe # TODO: why doesn't this shw?? because mace? yes
      })
  else:
    raise Exception(f'{DATASET} not supported.')


def gatherAndSaveResults():
  all_setups_df = pd.DataFrame()

  for scm_class in SCM_CLASS_VALUES:

    for lambda_lcb in LAMBDA_LCB_VALUES:

      for num_mc_samples in NUM_MC_SAMPLES_VALUES:

        setup_name = f'*{scm_class}__*__nmc_{num_mc_samples}__*__lambda_lcb_{lambda_lcb}*'

        matching_folders = glob.glob(
          f'{parent_path}/{setup_name}'
        )
        assert len(matching_folders) == 1, f'Expected only one folder for this setup, but found {len(matching_folders)}.'

        print(f'\n\n[INFO] Fetching results for setup `{setup_name}`...')

        experiment_results = pickle.load(open(f'{matching_verif_based_folders[0]}/_comparison', 'rb'))
        # ..
        # merge the iv. conf column and add - where we don't want information

  pickle.dump(all_setups_df, open(f'_results/all_setups_df', 'wb'))

  print(f'[INFO] done.\n')


def setToNanThoseNotFlippedOrNotPlausible(df):
  # the dataframe contains multiple sets of columns of the form:
  # {verif_, model_{i}}_scf_{found, plausible, distance, time}. The goal of this
  # function is that for any of the `i` model-based tests, if an instance does
  # not flip or does is not plausible, then set {found, plausible, distance, time}
  # all to NaN. Later, run np.nanmean and np.nanstd instead of np.mean, np.std.
  for i in range(10): # TODO: don't hardcode 10
    df.loc[
      (df[f'model_{i+1}_scf_found'] == False) | (df[f'model_{i+1}_scf_plausible'] == False),
      [ \
        f'model_{i+1}_scf_found', \
        f'model_{i+1}_scf_plausible', \
        f'model_{i+1}_scf_time', \
        f'model_{i+1}_scf_distance', \
        f'model_{i+1}_int_distance' \
      ]
    ] = np.NaN
    df.loc[
      (df[f'freee_{i+1}_scf_found'] == False) | (df[f'freee_{i+1}_scf_plausible'] == False),
      [ \
        f'freee_{i+1}_scf_found', \
        f'freee_{i+1}_scf_plausible', \
        f'freee_{i+1}_scf_time', \
        f'freee_{i+1}_scf_distance', \
        f'freee_{i+1}_int_distance' \
      ]
    ] = np.NaN
  return df

def setToNanThoseWithInfDistance(df):
  df.loc[
    (df['verif_scf_distance'] == np.Inf) | (df['verif_int_distance'] == np.Inf),
    [ \
      'verif_scf_found', \
      'verif_scf_plausible', \
      'verif_scf_time', \
      'verif_scf_distance', \
      'verif_int_distance', \
    ]
  ] = np.NaN
  return df


# def tmp(df):

#   data = []
#   for sample_idx in df.index:
#     for i in range(10):
#       data.append([ \
#         sample_idx, \
#         df.loc[sample_idx, f'model_{i+1}_scf_distance'] / df.loc[sample_idx, 'verif_scf_distance']
#       ])
#   new_df = pd.DataFrame(data, columns = ['sample_idx', 'ratio_distance']).dropna()
#   # new_df = new_df[:100]
#   unique_sample_indices = list(np.unique(new_df['sample_idx']))
#   sample_distance_stds = [ \
#     np.std(new_df.loc[new_df['sample_idx'] == sample_idx, 'ratio_distance']) \
#     for sample_idx in unique_sample_indices \
#   ]
#   ordered_sample_indices = [x for _, x in sorted(zip(sample_distance_stds,unique_sample_indices))]
#   g = sns.catplot(data=new_df, x="sample_idx", y="ratio_distance", order=ordered_sample_indices[::-1])
#   g.set_xticklabels(rotation=90)
#   # g.set_yscale("log")
#   plt.show()



def plotRelativeDistances():
  print('[INFO] Evaluating relative distances ({model, freee}-based / verif-based)')
  all_setups_df = pickle.load(open(f'_results/all_setups_df', 'rb'))
  all_setups_df = setToNanThoseNotFlippedOrNotPlausible(all_setups_df)
  all_setups_df = setToNanThoseWithInfDistance(all_setups_df)
  data = []
  print('\t[INFO] Constructing new dataframe...')
  for i in tqdm.trange(10):
    for df_idx in all_setups_df.index:
      sample_idx = all_setups_df.loc[df_idx, 'sample_idx']
      model_to_verif_dist = \
        all_setups_df.loc[df_idx, f'model_{i+1}_int_distance'] / \
        all_setups_df.loc[df_idx, 'verif_int_distance']
      if model_to_verif_dist >= 1:
        data.append([ \
          all_setups_df.loc[df_idx, 'dataset'],
          all_setups_df.loc[df_idx, 'model'],
          all_setups_df.loc[df_idx, 'world'],
          all_setups_df.loc[df_idx, 'sample_idx'],
          'model_to_verif',
          model_to_verif_dist, # TODO: flip (2020.04.11 ???)
        ])
      elif np.isnan(model_to_verif_dist):
        # do nothing
        _ = 1
      else: # dist < 1
        print(f'\t found model_to_verif_dist < 1 for sample {df_idx}! Ignoring... but good to investigage...')
      freee_to_verif_dist = \
        all_setups_df.loc[df_idx, f'freee_{i+1}_int_distance'] / \
        all_setups_df.loc[df_idx, 'verif_int_distance']
      if freee_to_verif_dist >= 1:
        data.append([ \
          all_setups_df.loc[df_idx, 'dataset'],
          all_setups_df.loc[df_idx, 'model'],
          all_setups_df.loc[df_idx, 'world'],
          all_setups_df.loc[df_idx, 'sample_idx'],
          'freee_to_verif',
          freee_to_verif_dist, # TODO: flip (2020.04.11 ???)
        ])
      elif np.isnan(freee_to_verif_dist):
        # do nothing
        _ = 1
      else: # dist < 1
        print(f'\t found freee_to_verif_dist < 1 for sample {df_idx}! Ignoring... but good to investigage...')
  print('\t[INFO] done.')
  new_df = pd.DataFrame(
    data,
    columns =['dataset', 'model', 'world', 'sample_idx', 'ratio_type', 'ratio_distance']
  ).dropna()
  tmp = len(new_df[new_df.ratio_distance < 1])
  assert tmp == 0, f'Found {tmp} cases where verif-based SCF was at further distance than model/freee-based SCF.'
  ax = sns.catplot(
    data=new_df,
    x = 'model',
    y = 'ratio_distance',
    # hue = 'dataset',
    hue = 'ratio_type',
    col = 'world',
    kind = 'boxen', # or box, with whis = np.inf
    # kind = 'violin',
    height = 2.5,
    aspect = 1,
    palette = sns.color_palette("muted", 5),
    sharey = True,
    # whis = np.inf,
  )
  # TODO: better ax management
  ax.fig.get_axes()[0].set_yscale('log')
  ax.fig.get_axes()[1].set_yscale('log')
  # ax.set_xticklabels(rotation=90)
  ax.set(ylim=(0,None))
  ax.set_ylabels(r"$\delta_{model/freee} / \delta_{verif}$")
  ax.set_xticklabels(rotation=90)
  ax.savefig(f'_results/all_setups_df.png', dpi = 400)
  # plt.show()

  # ipsh()
  # new_df = new_df.sample(500)
  # ax = sns.catplot(
  #   data=new_df,
  #   x = 'sample_idx',
  #   y = 'ratio_distance',
  #   hue = 'dataset',
  #   # hue_order = hue_order,
  #   col = 'model',
  #   # kind = 'box',
  #   # height = 2.5,
  #   # aspect = 1,
  #   # palette = sns.color_palette("muted", 5),
  #   # sharey = False,
  #   # whis = np.inf,
  # )
  # # new_df = new_df[:100]
  # unique_sample_indices = list(np.unique(new_df['sample_idx']))
  # sample_distance_stds = [ \
  #   np.std(new_df.loc[new_df['sample_idx'] == sample_idx, 'ratio_distance']) \
  #   for sample_idx in unique_sample_indices \
  # ]
  # ordered_sample_indices = [x for _, x in sorted(zip(sample_distance_stds,unique_sample_indices))]
  # g = sns.catplot(data=new_df, x="sample_idx", y="ratio_distance", order=ordered_sample_indices[::-1])
  # g.set_xticklabels(rotation=90)
  # # g.set_yscale("log")
  # plt.show()
  print('[INFO] done.')
  # ipsh()




def analyzeRelativeDistances():
  for dataset_string in DATASET_VALUES:

    for model_class_string in MODEL_CLASS_VALUES:

      setup_name = f'{dataset_string}__{model_class_string}'
      print(f'[INFO] Fetching results from setup {setup_name}...')
      df = pickle.load(open(f'_results/data_frame_all_{setup_name}', 'rb')).dropna()

      df_filtered = setToNanThoseNotFlippedOrNotPlausible(df.copy())
      tmp(df_filtered)

      # try:
      #   avg_found_verif = np.nanmean(df.filter(regex='verif.*found').to_numpy())
      #   avg_plausible_verif = np.nanmean(df.filter(regex='verif.*plausible').to_numpy())
      #   avg_distance_verif = np.nanmean(df.filter(regex='verif.*distance').to_numpy())
      #   avg_time_verif = np.nanmean(df.filter(regex='verif.*time').to_numpy())

      #   avg_found_model = np.nanmean(df.filter(regex='model.*found').to_numpy())
      #   avg_plausible_model = np.nanmean(df.filter(regex='model.*plausible').to_numpy())
      #   avg_distance_model = np.nanmean(df.filter(regex='model.*distance').to_numpy())
      #   avg_time_model = np.nanmean(df.filter(regex='model.*time').to_numpy())

      #   std_found_model = np.nanstd(df.filter(regex='model.*found').to_numpy())
      #   std_plausible_model = np.nanstd(df.filter(regex='model.*plausible').to_numpy())
      #   std_distance_model = np.nanstd(df.filter(regex='model.*distance').to_numpy())
      #   std_time_model = np.nanstd(df.filter(regex='model.*time').to_numpy())
      # except:
      #   pd.set_option('display.max_columns', None)
      #   pd.set_option('display.width', 230)
      #   ipsh()

      # # avg_found_verif_2 = np.nanmean(df_filtered.filter(regex='verif.*found').to_numpy())
      # # avg_plausible_verif_2 = np.nanmean(df_filtered.filter(regex='verif.*plausible').to_numpy())
      # # avg_distance_verif_2 = np.nanmean(df_filtered.filter(regex='verif.*distance').to_numpy())
      # # avg_time_verif_2 = np.nanmean(df_filtered.filter(regex='verif.*time').to_numpy())

      # # avg_found_model_2 = np.nanmean(df_filtered.filter(regex='model.*found').to_numpy())
      # # avg_plausible_model_2 = np.nanmean(df_filtered.filter(regex='model.*plausible').to_numpy())
      # # avg_distance_model_2 = np.nanmean(df_filtered.filter(regex='model.*distance').to_numpy())
      # # avg_time_model_2 = np.nanmean(df_filtered.filter(regex='model.*time').to_numpy())

      # # std_found_model_2 = np.nanstd(df_filtered.filter(regex='model.*found').to_numpy())
      # # std_plausible_model_2 = np.nanstd(df_filtered.filter(regex='model.*plausible').to_numpy())
      # # std_distance_model_2 = np.nanstd(df_filtered.filter(regex='model.*distance').to_numpy())
      # # std_time_model_2 = np.nanstd(df_filtered.filter(regex='model.*time').to_numpy())

      # print(f'\t[Found] \t Verif: {avg_found_verif:.4f} \t\t Model: {avg_found_model:.4f} +/- {std_found_model:.4f}')
      # print(f'\t[Plausible] \t Verif: {avg_plausible_verif:.4f} \t\t Model: {avg_plausible_model:.4f} +/- {std_plausible_model:.4f}')
      # print(f'\t[Distance] \t Verif: {avg_distance_verif:.4f} \t\t Model: {avg_distance_model:.4f} +/- {std_distance_model:.4f}')
      # print(f'\t[Time] \t\t Verif: {avg_time_verif:.4f} \t\t Model: {avg_time_model:.4f} +/- {std_time_model:.4f}')

      # # print(f'\t[Found] \t Verif: {avg_found_verif_2:.4f} \t\t Model: {avg_found_model_2:.4f} +/- {std_found_model_2:.4f}')
      # # print(f'\t[Plausible] \t Verif: {avg_plausible_verif_2:.4f} \t\t Model: {avg_plausible_model_2:.4f} +/- {std_plausible_model_2:.4f}')
      # # print(f'\t[Distance] \t Verif: {avg_distance_verif_2:.4f} \t\t Model: {avg_distance_model_2:.4f} +/- {std_distance_model_2:.4f}')
      # # print(f'\t[Time] \t\t Verif: {avg_time_verif_2:.4f} \t\t Model: {avg_time_model_2:.4f} +/- {std_time_model_2:.4f}')

      # # TODO: only average over those that are found!

      # print(f'[INFO] done.\n')




if __name__ == '__main__':
  gatherAndSaveResults()
  # analyzeRelativeDistances()
  plotRelativeDistances()
































