import numpy as np

SCM_CLASS_VALUES = ['sanity-3-lin'] #, 'sanity-3-anm', 'sanity-3-gen']
LAMBDA_LCB_VALUES = np.linspace(0,2.5,6)
OPTIMIZATION_APPROACHES = ['brute_force', 'grad_descent']
REPEAT_COUNT = 1

request_memory = 8192

sub_file = open('test.sub','w')
print('executable = /home/amir/dev/recourse/_venv/bin/python', file=sub_file)
print('error = _cluster_logs/test.$(Process).err', file=sub_file)
print('output = _cluster_logs/test.$(Process).out', file=sub_file)
print('log = _cluster_logs/test.$(Process).log', file=sub_file)
print(f'request_memory = {request_memory}', file=sub_file)
print('request_cpus = 2', file=sub_file)
print('\n' * 2, file=sub_file)

for (
  scm_class,
  lambda_lcb,
  optimization_approach,
) in tuple(
  SCM_CLASS_VALUES,
  LAMBDA_LCB_VALUES,
  OPTIMIZATION_APPROACHES,
):
  print(f'arguments = main.py' + \
     f' --scm_class {scm_class}' \
     f' --lambda_lcb {lambda_lcb}' \
     f' --optimization_approach {optimization_approach}' \
     f' -p $(Process)', \
  file=sub_file)
  print(f'queue {REPEAT_COUNT}', file=sub_file)
  print('\n', file=sub_file)

