import os
import glob
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pprint import pprint
from shutil import copyfile
import matplotlib
import matplotlib.pyplot as plt

# sns.set(style="darkgrid")
sns.set_context("paper")

ACCEPTABLE_POINT_RECOURSE = {'m0_true', 'm1_alin', 'm1_akrr'}
ACCEPTABLE_DISTR_RECOURSE = {'m1_gaus', 'm1_cvae', 'm2_true', 'm2_gaus', 'm2_cvae'} #, 'm2_cvae_ps'}
ALL_RECOURSE_TYPES = [*ACCEPTABLE_POINT_RECOURSE, *ACCEPTABLE_DISTR_RECOURSE]

from debug import ipsh

SCM_CLASS_VALUES = ['german-credit']
LAMBDA_LCB_VALUES = np.linspace(0, 2.5, 6)
# OPTIMIZATION_APPROACHES = ['grad_descent']
CLASSIFIER_VALUES = ['lr', 'mlp', 'tree', 'forest']


# experiments_folder_path = '/Users/a6karimi/dev/recourse/_results/__merged_realworld/'
experiments_folder_path = '/Users/julisuvk/__merged_realworld_bu_2020.06.03.11.13/'
# all_counter = len(SCM_CLASS_VALUES) * len(LAMBDA_LCB_VALUES) * len(OPTIMIZATION_APPROACHES) * len(CLASSIFIER_VALUES)
all_counter = len(LAMBDA_LCB_VALUES) * len(CLASSIFIER_VALUES)
counter = 0



def latexify(fig_width=None, fig_height=None, columns=1, largeFonts=False, font_scale=1):
  """Set up matplotlib's RC params for LaTeX plotting.
  Call this before plotting a figure.

  Parameters
  ----------
  fig_width : float, optional, inches
  fig_height : float,  optional, inches
  columns : {1, 2}
  """

  # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

  # Width and max height in inches for IEEE journals taken from
  # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

  assert(columns in [1, 2])

  if fig_width is None:
      fig_width = 3.39 if columns == 1 else 6.9  # width in inches

  if fig_height is None:
      golden_mean = (np.sqrt(5) - 1.0) / 2.0    # Aesthetic ratio
      fig_height = fig_width * golden_mean  # height in inches

  MAX_HEIGHT_INCHES = 8.0
  if fig_height > MAX_HEIGHT_INCHES:
      print("WARNING: fig_height too large:" + fig_height +
            "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
      fig_height = MAX_HEIGHT_INCHES

  params = {'backend': 'ps',
            'text.latex.preamble': ['\\usepackage{gensymb}'],
            # fontsize for x and y labels (was 10)
            'axes.labelsize': font_scale * 10 if largeFonts else font_scale * 7,
            'axes.titlesize': font_scale * 10 if largeFonts else font_scale * 7,
            'font.size': font_scale * 10 if largeFonts else font_scale * 7,  # was 10
            'legend.fontsize': font_scale * 10 if largeFonts else font_scale * 7,  # was 10
            'xtick.labelsize': font_scale * 10 if largeFonts else font_scale * 7,
            'ytick.labelsize': font_scale * 10 if largeFonts else font_scale * 7,
            'text.usetex': True,
            'figure.figsize': [fig_width, fig_height],
            'font.family': 'serif',
            'xtick.minor.size': 0.5,
            'xtick.major.pad': 1.5,
            'xtick.major.size': 1,
            'ytick.minor.size': 0.5,
            'ytick.major.pad': 1.5,
            'ytick.major.size': 1,
            'lines.linewidth': 1.5,
            'lines.markersize': 0.1,
            'hatch.linewidth': 0.5
            }

  matplotlib.rcParams.update(params)
  plt.rcParams.update(params)



# iterate over and merge different lambda_lcb values

total_dict = {}
total_dict['recourse_type'] = []
total_dict['scm_class'] = []
total_dict['classifier_class'] = []
total_dict['lambda_lcb'] = []
total_dict['optimization_approach'] = []
total_dict['scf_validity_mean'] = []
total_dict['scf_validity_std'] = []
total_dict['cost_all_mean'] = []
total_dict['cost_all_std'] = []

for scm_class in SCM_CLASS_VALUES:

  for classifier_class in CLASSIFIER_VALUES:

    for lambda_lcb in LAMBDA_LCB_VALUES:

      # for optimization_approach in OPTIMIZATION_APPROACHES:
      if classifier_class in {'lr', 'mlp'}:
        optimization_approach = 'grad_descent'
      elif classifier_class in {'tree', 'forest'}:
        optimization_approach = 'brute_force'
      else:
        raise Exception(f'Classifier `{classifier_class}` not supported.')

      counter = counter + 1

      specific_experiment_path = f'{scm_class}__*__{classifier_class}__*__lambda_lcb_{lambda_lcb}__opt_{optimization_approach}'
      # specific_experiment_path = f'{scm_class}__*__{classifier_class}__*__opt_{optimization_approach}'

      print(f'\n[{counter} / {all_counter}] Merging together folders for {specific_experiment_path}')

      all_matching_folders = glob.glob(f'{experiments_folder_path}*{specific_experiment_path}*')
      assert len(all_matching_folders) == 1
      matching_folder = all_matching_folders[0]
      comparison_results_path = os.path.join(matching_folder, '_comparison')
      comparison_results_file = pickle.load(open(comparison_results_path, 'rb'))

      for recourse_type in ALL_RECOURSE_TYPES:
        total_dict['scm_class'].append(scm_class)
        total_dict['classifier_class'].append(classifier_class)
        total_dict['lambda_lcb'].append(lambda_lcb)
        total_dict['optimization_approach'].append(optimization_approach)
        total_dict['recourse_type'].append(recourse_type)
        total_dict['scf_validity_mean'].append(float(comparison_results_file.loc[recourse_type, 'scf_validity'][:6]))
        total_dict['scf_validity_std'].append(float(comparison_results_file.loc[recourse_type, 'scf_validity'][-6:]))
        total_dict['cost_all_mean'].append(float(comparison_results_file.loc[recourse_type, 'cost_all'][:6]))
        total_dict['cost_all_std'].append(float(comparison_results_file.loc[recourse_type, 'cost_all'][-6:]))

total_df = pd.DataFrame(total_dict)
total_df = total_df[total_df['classifier_class'] == 'lr']

# ipsh()

# latexify(1.5 * 6, 6, font_scale = 1.2)
# sns.set_style("whitegrid")
# ax = sns.catplot(
#   x = 'dataset',
#   y = 'counterfactual distance',
#   hue = 'approach',
#   hue_order = hue_order,
#   col = 'norm',
#   data = model_specific_df,
#   kind = 'box',
#   # kind = 'violin',
#   # kind = 'swarm',
#   height = 3.5,
#   aspect = .9,
#   palette = sns.color_palette("muted", 5),
#   sharey = False,
#   whis = np.inf,
#   legend_out = False,
# )

m0_true_array = total_df[total_df['recourse_type'] == 'm0_true']
m1_alin_array = total_df[total_df['recourse_type'] == 'm1_alin']
m1_akrr_array = total_df[total_df['recourse_type'] == 'm1_akrr']
m1_gaus_array = total_df[total_df['recourse_type'] == 'm1_gaus']
m1_cvae_array = total_df[total_df['recourse_type'] == 'm1_cvae']
m2_gaus_array = total_df[total_df['recourse_type'] == 'm2_gaus']
m2_cvae_array = total_df[total_df['recourse_type'] == 'm2_cvae']
m2_true_array = total_df[total_df['recourse_type'] == 'm2_true']
lambda_lcb_array = m0_true_array['lambda_lcb']
# print(total_df)
# plot version 1
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
ax1.plot(lambda_lcb_array, 100 * np.ones_like(lambda_lcb_array) * np.mean(m0_true_array['scf_validity_mean']), '--', label=r'$\mathcal{M}_{\bigstar}$', linewidth=2)
ax1.plot(lambda_lcb_array, 100 * np.ones_like(lambda_lcb_array) * np.mean(m1_alin_array['scf_validity_mean']), '--', label=r'$\mathcal{M}_{\mathrm{LIN}}$', linewidth=2)
ax1.plot(lambda_lcb_array, 100 * np.ones_like(lambda_lcb_array) * np.mean(m1_akrr_array['scf_validity_mean']), '--', label=r'$\mathcal{M}_{\mathrm{KRR}}$', linewidth=2)
ax1.plot(lambda_lcb_array, 100 * m1_gaus_array['scf_validity_mean'], '-o', label=r'$\mathcal{M}_{\mathrm{GP}}$', linewidth=2)
ax1.plot(lambda_lcb_array, 100 * m1_cvae_array['scf_validity_mean'], '-s', label=r'$\mathcal{M}_{\mathrm{CVAE}}$', linewidth=2)
ax1.plot(lambda_lcb_array, 100 * m2_true_array['scf_validity_mean'], '-^', label=r'$\mathrm{CATE}_{\bigstar}$', linewidth=2)
ax1.plot(lambda_lcb_array, 100 * m2_gaus_array['scf_validity_mean'], '-v', label=r'$\mathrm{CATE}_{\mathrm{GP}}$', linewidth=2)
ax1.plot(lambda_lcb_array, 100 * m2_cvae_array['scf_validity_mean'], '-d', label=r'$\mathrm{CATE}_{\mathrm{CVAE}}$', linewidth=2)
# ax1.legend(ncol=2)
ax1.set_xlabel(r'$\gamma_{\mathrm{LCB}}$')
ax1.set_ylabel('Validity (%)')
ax1.set_xlim(left = -0.15, right = 2.65)
ax1.set_xticks(lambda_lcb_array)
ax1.set_xticklabels(['0', '0.5', '1', '1.5', '2', '2.5'])
# ax1.set_yticklabels(np.linspace(0,100,11, dtype=int))
ax1.set_ylim(bottom=10, top=105)

ax2.plot(lambda_lcb_array, 100 * np.ones_like(lambda_lcb_array) * np.mean(m0_true_array['cost_all_mean']), '--', label=r'$\mathcal{M}_{\bigstar}$', linewidth=2)
ax2.plot(lambda_lcb_array, 100 * np.ones_like(lambda_lcb_array) * np.mean(m1_alin_array['cost_all_mean']), '--', label=r'$\mathcal{M}_{\mathrm{LIN}}$', linewidth=2)
ax2.plot(lambda_lcb_array, 100 * np.ones_like(lambda_lcb_array) * np.mean(m1_akrr_array['cost_all_mean']), '--', label=r'$\mathcal{M}_{\mathrm{KRR}}$', linewidth=2)
ax2.plot(lambda_lcb_array, 100 * m1_gaus_array['cost_all_mean'], '-o', label=r'$\mathcal{M}_{\mathrm{GP}}$', linewidth=2)
ax2.plot(lambda_lcb_array, 100 * m1_cvae_array['cost_all_mean'], '-s', label=r'$\mathcal{M}_{\mathrm{CVAE}}$', linewidth=2)
ax2.plot(lambda_lcb_array, 100 * m2_true_array['cost_all_mean'], '-^', label=r'$\mathrm{CATE}_{\bigstar}$', linewidth=2)
ax2.plot(lambda_lcb_array, 100 * m2_gaus_array['cost_all_mean'], '-v', label=r'$\mathrm{CATE}_{\mathrm{GP}}$', linewidth=2)
ax2.plot(lambda_lcb_array, 100 * m2_cvae_array['cost_all_mean'], '-d', label=r'$\mathrm{CATE}_{\mathrm{CVAE}}$', linewidth=2)
ax2.legend(ncol=2)
ax2.set_xlabel(r'$\gamma_{\mathrm{LCB}}$')
ax2.set_ylabel('Cost (%)')
ax2.set_xlim(left = -0.15, right = 2.65)
ax2.set_xticks(lambda_lcb_array)
ax2.set_xticklabels(['0', '0.5', '1', '1.5', '2', '2.5'])
# ax1.set_yticklabels(np.linspace(0,100,11, dtype=int))
# ax2.set_ylim(bottom=10, top=105)
fig.savefig('gamma_lcb_wider.pdf', bboxinches='tight')
plt.show()

# fig, axes = plt.subplots(1, 2)
# sns.lineplot(
#   x='lambda_lcb',
#   y='scf_validity_mean',
#   hue='recourse_type',
#   data=total_df,
#   ax = axes[0],
# markers=True,
# )
# sns.lineplot(
#   x='lambda_lcb',
#   y='cost_all_mean',
#   hue='recourse_type',
#   data=total_df,
#   ax = axes[1],
# markers=True,
# )
# plt.show()
# axes[0].set_xlabel('lambda_LCB')

# plot version 2
# total_df = pd.DataFrame(total_dict)
# # tmp_df = total_df[total_df['classifier_class'] == 'mlp']
# import seaborn as sns; sns.set()
# import matplotlib.pyplot as plt

# fig, axes = plt.subplots(2, 2) #, sharex=True, sharey=True)
# acceptable_classifier_class = ['lr', 'mlp', 'tree', 'forest']
# for idx, ax in enumerate(axes.flatten()):
#   print(ax)
#   classifier_class = acceptable_classifier_class[idx]
#   tmp_df = total_df[total_df['classifier_class'] == classifier_class]
#   sns.lineplot(
#     x='cost_all_mean',
#     y='scf_validity_mean',
#     hue='recourse_type',
#     data=tmp_df,
#     ax = ax,
#   )
#   ax.set_title(classifier_class)
# plt.show()



## validity
# ax = sns.catplot(
#   x = 'lambda_lcb',
#   y = 'scf_validity_mean',
#   col = 'classifier_class',
#   data = total_df,
#   kind = 'box',
# )

## cost
# ax = sns.catplot(
#   x = 'lambda_lcb',
#   y = 'cost_all_mean',
#   col = 'classifier_class',
#   data = total_df,
#   kind = 'box',
# )



# order = ['m0_true'
#  'm1_alin',
#  'm1_akrr',
#  'm1_gaus',
#  'm1_cvae',
#  'm2_true',
#  'm2_gaus',
#  'm2_cvae']
# ax = sns.catplot(
#   x = 'recourse_type',
#   order = order,
#   y = 'scf_validity_mean',
#   col = 'lambda_lcb',
#   data = total_df,
#   kind = 'box',
# )

plt.show()
ax.fig.get_axes()[0].legend().remove()
ax.fig.get_axes()[2].legend(loc='upper left', fancybox = True, shadow = True, fontsize = 'small')
ax.set(ylim=(0,None))
ax.set_axis_labels("", r"Distance $\delta$ to" + "\nNearest Counterfactual")
ax.set_titles('{col_name}')
ax.set_xlabels() # remove "dataset" on the x-axis
# ax.savefig(f'_results/{tmp_constrained}__all_distances_appendix__{model_string}.png', dpi = 400)

































