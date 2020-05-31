import numpy as np

SCM_CLASS_VALUES = ['sanity-3-lin', 'sanity-3-anm', 'sanity-3-gen']
LAMBDA_LCB_VALUES = [1] # np.linspace(0,2.5,6)
OPTIMIZATION_APPROACHES = ['brute_force', 'grad_descent']

request_memory = 8192*4

NUM_BATCHES = 10
NUM_NEG_SAMPLES_PER_BATCH = 5

sub_file = open('test.sub','w')
print('executable = /home/amir/dev/recourse/_venv/bin/python', file=sub_file)
print('error = _cluster_logs/test.$(Process).err', file=sub_file)
print('output = _cluster_logs/test.$(Process).out', file=sub_file)
print('log = _cluster_logs/test.$(Process).log', file=sub_file)
print(f'request_memory = {request_memory}', file=sub_file)
print('request_cpus = 4', file=sub_file)
print('\n' * 2, file=sub_file)

for scm_class in SCM_CLASS_VALUES:
  for lambda_lcb in LAMBDA_LCB_VALUES:
    for optimization_approach in OPTIMIZATION_APPROACHES:
      for batch_number in range(NUM_BATCHES):
        print(f'arguments = main.py' + \
           f' --scm_class {scm_class}' \
           f' --lambda_lcb {lambda_lcb}' \
           f' --optimization_approach {optimization_approach}' \
           # f' --max_intervention_cardinality 3'
           f' --batch_number {batch_number}' \
           f' --sample_count {NUM_NEG_SAMPLES_PER_BATCH}', \
           f' -p $(Process)', \
        file=sub_file)
        print('\n', file=sub_file)

