import numpy as np

# SCM_CLASS_VALUES = ['sanity-3-lin', 'sanity-3-anm', 'sanity-3-gen']
# LAMBDA_LCB_VALUES = [2.]
# OPTIMIZATION_APPROACHES = ['brute_force', 'grad_descent']
# CLASSIFIER_VALUES = ['lr']

# ==============================================================================
# # ==============================================================================

# SCM_CLASS_VALUES = ['german-credit']
# LAMBDA_LCB_VALUES = [2.5] # np.linspace(0, 2.5, 6)
# OPTIMIZATION_APPROACHES = ['grad_descent']
# CLASSIFIER_VALUES = ['lr', 'mlp']

# # ==============================================================================
# # ==============================================================================

# SCM_CLASS_VALUES = ['german-credit']
# LAMBDA_LCB_VALUES = [2.5] # np.linspace(0, 2.5, 6)
# OPTIMIZATION_APPROACHES = ['brute_force']
# CLASSIFIER_VALUES = ['tree', 'forest']

# ==============================================================================
# ==============================================================================

SCM_CLASS_VALUES = ['fair-IMF-LIN', 'fair-CAU-LIN', 'fair-CAU-ANM',
                    'fair-IMF-LIN-radial', 'fair-CAU-LIN-radial', 'fair-CAU-ANM-radial']
LAMBDA_LCB_VALUES = [1]
OPTIMIZATION_APPROACHES = ['brute_force']
CLASSIFIER_VALUES = ['vanilla_svm',
                     'vanilla_lr',
                     'vanilla_mlp',
                     'nonsens_svm',
                     'nonsens_lr',
                     'nonsens_mlp',
                     'unaware_svm',
                     'unaware_lr',
                     'unaware_mlp',
                     'cw_fair_svm',
                     'cw_fair_lr',
                     'cw_fair_mlp',
                     'iw_fair_svm']

# if set to 'all', will select best kernel type based on CV;
# else uses linear kernel for 'linear' datasets and 'poly' for nonlinear ones
FAIR_KERNEL_TYPE = 'NOT all'
# FAIR_KERNEL_TYPE = 'all'

NUM_BATCHES = 1
NUM_NEG_SAMPLES_PER_BATCH = 200
request_memory = 8192*8


if FAIR_KERNEL_TYPE == 'all':
  sub_file = open('fair_recourse_all_kernels.sub','w')
else:
  sub_file = open('fair_recourse.sub','w')
# print('executable = /home/julisuvk/recourse/_venv/bin/python', file=sub_file)
print('executable = /home/amir/dev/recourse/_venv/bin/python', file=sub_file)
print('error = _cluster_logs/test.$(Process).err', file=sub_file)
print('output = _cluster_logs/test.$(Process).out', file=sub_file)
print('log = _cluster_logs/test.$(Process).log', file=sub_file)
print(f'request_memory = {request_memory}', file=sub_file)
print('request_cpus = 4', file=sub_file)
print('\n' * 2, file=sub_file)

for scm_class in SCM_CLASS_VALUES:
  for classifier_class in CLASSIFIER_VALUES:
    for lambda_lcb in LAMBDA_LCB_VALUES:
      for optimization_approach in OPTIMIZATION_APPROACHES:
        for batch_number in range(NUM_BATCHES):
          command = \
            f'arguments = main.py' \
            f' --scm_class {scm_class}' \
            f' --classifier_class {classifier_class}' \
            f' --experimental_setups m0_true m1_alin m1_akrr' \
            f' --lambda_lcb {lambda_lcb}' \
            f' --optimization_approach {optimization_approach}'


          # run-specific options
          if optimization_approach == 'grad_descent':
            command += f' --grad_descent_epochs 1000'
          elif optimization_approach == 'brute_force':
            if scm_class == 'german-credit':
              command += f' --grid_search_bins 10'
            else:
              command += f' --grid_search_bins 15'

          # if scm_class == 'sanity-3-gen':
          #   command += f' --non_intervenable_nodes x3'

          if scm_class == 'german-credit':
            command += f' --non_intervenable_nodes x1 x2 x5'

          # for fair experiments
          if 'fair' in scm_class:
            command += f' -e 9'
            command += f' --sensitive_attribute_nodes x1'
            command += f' --num_train_samples 500'
            command += f' --num_fair_samples 50'
            if 'svm' in classifier_class:
              if FAIR_KERNEL_TYPE == 'all':
                command += f' --fair_kernel_type all'
              else:
                if 'radial' in scm_class:
                  command += f' --fair_kernel_type poly'
                else:
                  command += f' --fair_kernel_type linear'

          # finally add batch, samples, and process id params
          command += f' --batch_number {batch_number}'
          command += f' --sample_count {NUM_NEG_SAMPLES_PER_BATCH}'
          command += f' -p $(Process)'
          print(command, file=sub_file)
          print('queue', file=sub_file)
          print('\n', file=sub_file)

