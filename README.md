# General

This repository provides code and examples for generating sub-population based algorithmic recourse.

# Code Pre-requisites

First,
```console
$ git clone https://github.com/pseudonym/recourse.git
$ pip install virtualenv
$ cd recourse
$ virtualenv -p python3 _venv
$ source _venv/bin/activate
$ pip install -r pip_requirements.txt
```


# Repro:

### Table 1:
```python
python main.py --scm_class sanity-3-lin --classifier_class lr --lambda_lcb 2. --optimization_approach grad_descent --grad_descent_epochs 1000 --batch_number 0 --sample_count 50
python main.py --scm_class sanity-3-anm --classifier_class lr --lambda_lcb 2. --optimization_approach grad_descent --grad_descent_epochs 1000 --batch_number 0 --sample_count 50
python main.py --scm_class sanity-3-gen --classifier_class lr --lambda_lcb 2. --optimization_approach grad_descent --grad_descent_epochs 1000 --batch_number 0 --sample_count 50
```

### Table 2:
```python
python main.py --scm_class german-credit --classifier_class lr     --lambda_lcb 2.5 --optimization_approach grad_descent --grad_descent_epochs 1000 --non_intervenable_nodes x1 x2 x5 --batch_number 0 --sample_count 50
python main.py --scm_class german-credit --classifier_class mlp    --lambda_lcb 2.5 --optimization_approach grad_descent --grad_descent_epochs 1000 --non_intervenable_nodes x1 x2 x5 --batch_number 0 --sample_count 50
python main.py --scm_class german-credit --classifier_class forest --lambda_lcb 2.5 --optimization_approach brute_force  --grid_search_bins 10      --non_intervenable_nodes x1 x2 x5 --batch_number 0 --sample_count 50
```


# Sanity checks:

EXPERIMENT 5 (sub-plots):
If:
* LINEAR SCM
* sufficient training data
Expect:
* m1_alin to match m0_true perfectly within of data manifold
* m1_alin to match m0_true perfectly outside of data manifold

<> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <>

EXPERIMENT 5 (sub-plots):
If:
* ANM SCM (not linear)
* sufficient training data
Expect:
* m1_akrr to match m0_true perfectly within data manifold
* m1_gaus to have smaller variance than m2_gaus
* m1_gaus to be (ideally) close and centered around to m0_true
* m2_gaus to match m2_true within data manifold

<> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <>

EXPERIMENT 5 (sub-plots):
If:
* GENERAL SCM
* sufficient training data
* reasonable hyperparams for cvae
Expect:
* m1_cvae to have smaller variance than m2_cvae
* m2_cvae to match m2_true within data manifold
* m2_true to be near-ish to m0_true assuming the abducted value of noise variables is likely-ish under the priors

<> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <> <>

EXPERIMENT 8 (box-plots):
If:
* assumptions for gp (additive noise) and cvae (sigmoid, sin/cos, exp nonlin only) are satisfied
* sufficient training data
* reasonable hyperparams for cvae
Expect: box-plots should match in mean and variance within data manifold (p(x1) > 0.05)

