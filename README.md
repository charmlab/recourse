# General

This repository provides code and examples for generating sub-population based algorithmic recourse: https://arxiv.org/abs/2006.06831

# Code Pre-requisites

First,
```console
$ git clone https://github.com/amirhk/recourse.git
$ pip install virtualenv
$ cd recourse
$ virtualenv -p python3 _venv
$ source _venv/bin/activate
$ pip install -r pip_requirements.txt
```


# Repro:

### Table 1:
```python
python main.py --scm_class sanity-3-lin --classifier_class lr --lambda_lcb 2. --optimization_approach grad_descent --grad_descent_epochs 1000 --batch_number 0 --sample_count 100
python main.py --scm_class sanity-3-anm --classifier_class lr --lambda_lcb 2. --optimization_approach grad_descent --grad_descent_epochs 1000 --batch_number 0 --sample_count 100
python main.py --scm_class sanity-3-gen --classifier_class lr --lambda_lcb 2. --optimization_approach grad_descent --grad_descent_epochs 1000 --batch_number 0 --sample_count 100
```

### Table 2:
```python
python main.py --scm_class german-credit --classifier_class lr     --lambda_lcb 2.5 --optimization_approach grad_descent --grad_descent_epochs 1000 --non_intervenable_nodes x1 x2 x5 --batch_number 0 --sample_count 100
python main.py --scm_class german-credit --classifier_class mlp    --lambda_lcb 2.5 --optimization_approach grad_descent --grad_descent_epochs 1000 --non_intervenable_nodes x1 x2 x5 --batch_number 0 --sample_count 100
python main.py --scm_class german-credit --classifier_class forest --lambda_lcb 2.5 --optimization_approach brute_force  --grid_search_bins 10      --non_intervenable_nodes x1 x2 x5 --batch_number 0 --sample_count 100
```


### Fair recourse:
```python
python main.py -e 9 -s fair-3-lin --batch_number 0 --sample_count 50 --sensitive_attribute_nodes x1 --optimization_approach brute_force --grid_search_bins 5
```