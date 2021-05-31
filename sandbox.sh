


[x] go through reproducability checklist
[x] add more assertions to code
[x] cost calculated correctly?
[x] git commit -m "[MAJOR] Added support for heterogenous data (particularly categorical data) to run m1_cvae on adult dataset. Completed reproducability check successfully."
[x] indiv nan???
[x] show some statistic on groups?
[x] bug in max_indiv_delta_cost_valid computation
[x] compute avg max indiv delta_cost_valid
[x] fixed regressed bug in eisting fair setups
[x] use _per_instance_results to keep everything
[x] why some groups/indivs fail?? resolved --> predict() vs predict_proba() ... those instances were not neatively predicted
[?] why does adult run on m* and m1_alin/akrr?? --> ignore for now
[?] why does cvae training fail? --> ignore for now
[x] baby sit the cvae fit for x7 x8 (hopefully this results in more diverse action sets, and different values of x7/8 for twins)
[x] not only compute the max indiv cost, but also save the factual_idx so to report in the table.
[x] let run overnight using script below.
[x] describe in paper

[ ] why recourse action suggesting fixing a factual value (w/o changing parents)
[?] why local/cluster results different? --> diff training seed in torch?




[x] try increased grid search
[x] try more instances in each group
[?] try grad descent? --> doesn\'t work for other fair models
[?] try diff combinations of sensitive attributes x1, x2, x3; if we only consider 1 sens attr, compare to FairSVM

LOW PRI:
[ ] change nonsese ==> unaware
[x] add args to select approx SCM

[x] set marital status as non-intervnable (not sensitive) attribute
[x] does prior from loadSCM affect the CVAE training? NO
[x] swap table 1 for \hat{M}_KR from table 2 and add log reg.
[x] "we fit a model to the data under an additive noise assumption (see suppl.)"



We generate recourse actions for \#\# negatively predicted individuals uniformly sampled from
\#\# different sensitive groups ($2^{|\text{bin. sens. attr.}|} = 2^{|\Acal|}$) on a \#\# classifier.
The results are then averaged per sensitive group and the maximum pair-wise dist/cost between any
pair of groups is evaluated per \cref{def:IW-fair}~\cite{gupta2019equalizing} ($\Delta_\textbf{dist}$),
and our causal group-level criteria \cref{def:MINT-fair} ($\Delta_\textbf{cost}$). This demonstration
clearly highlights the presence of group-level discrimination on this classifier. Furthermore, while
not directly comparable, the order-of-magnitude difference in $\Delta_\textbf{dist}$ and $\Delta_\textbf{cost}$
under assumption of expert opinion on causal assumptions (see \citet{chiappa2019path,nabi2018fair}) further
warrants the need for causal notions of discrimination whose severity is otherwise undetectable using non-causal counterparts.

  shows  then evaluate the group-level performance ($\Delta_\textbf{dist}$ and $\Delta_\textbf{cost}$) of the generated
In aggregate, we evalute  the same metrics as the synthetic experiments above.





$\Delta_\textbf{ind}$

# on cluster
cd ~/dev/recourse/_experiments/
# find 2021.05.20_11.13* -name _optimization_curves | xargs rm -rf # (optional) delete the _optimization_curves directory and contents
zip -r _fair_neurips 2021.05.20_11.13*

# on local machine
scp -r amir@login.cluster.is.localnet:~/dev/recourse/_experiments/_fair_neurips.zip _fair_neurips.
scp -r amir@login.cluster.is.localnet:~/dev/recourse/_experiments/2021.* .


# REPRODUCABILITY CHECkLIST (to be ran before/after a commit)

python main.py \
	--scm_class sanity-3-lin \
	--classifier_class lr \
	--lambda_lcb 1 \
	--optimization_approach brute_force \
	--grid_search_bins 4 -e 6 \
	--num_train_samples 500 \
	--num_fair_samples 2 \
	--batch_number 0 \
	--sample_count 3

python main.py \
	--scm_class sanity-3-anm \
	--classifier_class lr \
	--lambda_lcb 1 \
	--optimization_approach grad_descent \
	--grid_search_bins 4 -e 6 \
	--num_train_samples 500 \
	--num_fair_samples 2 \
	--batch_number 0 \
	--sample_count 3

python main.py \
	--scm_class sanity-3-gen \
	--classifier_class mlp \
	--lambda_lcb 1 \
	--optimization_approach grad_descent \
	--grid_search_bins 4 -e 6 \
	--num_train_samples 500 \
	--num_fair_samples 2 \
	--batch_number 0 \
	--sample_count 3

python main.py \
	--scm_class german-credit \
	--classifier_class mlp \
	--lambda_lcb 1 \
	--optimization_approach brute_force \
	--grid_search_bins 2 -e 6 \
	--num_train_samples 500 \
	--num_fair_samples 2 \
	--batch_number 0 \
	--sample_count 3

python main.py \
	--scm_class german-credit \
	--classifier_class tree \
	--lambda_lcb 1 \
	--optimization_approach brute_force \
	--grid_search_bins 2 -e 6 \
	--num_train_samples 500 \
	--num_fair_samples 2 \
	--batch_number 0 \
	--sample_count 3


################################################################################
################################################################################


python main.py \
	--scm_class fair-IMF-LIN \
	--classifier_class vanilla_mlp \
	--lambda_lcb 1 \
	--optimization_approach brute_force \
	--grid_search_bins 3 -e 9 \
	--sensitive_attribute_nodes x1 \
	--num_train_samples 500 \
	--num_fair_samples 2 \
	--batch_number 0 \
	--sample_count 200

python main.py \
	--scm_class fair-IMF-LIN \
	--classifier_class nonsens_lr \
	--lambda_lcb 1 \
	--optimization_approach brute_force \
	--grid_search_bins 3 -e 9 \
	--sensitive_attribute_nodes x1 \
	--num_train_samples 500 \
	--num_fair_samples 2 \
	--batch_number 0 \
	--sample_count 200

python main.py \
	--scm_class fair-IMF-LIN-radial \
	--classifier_class iw_fair_svm \
	--lambda_lcb 1 \
	--optimization_approach brute_force \
	--grid_search_bins 3 -e 9 \
	--sensitive_attribute_nodes x1 \
	--num_train_samples 500 \
	--num_fair_samples 2 \
	--batch_number 0 \
	--sample_count 200

python main.py \
	--scm_class fair-IMF-LIN-radial \
	--classifier_class unaware_svm \
	--lambda_lcb 1 \
	--optimization_approach brute_force \
	--grid_search_bins 3 -e 9 \
	--sensitive_attribute_nodes x1 \
	--num_train_samples 500 \
	--num_fair_samples 2 \
	--batch_number 0 \
	--sample_count 200

python main.py \
	--scm_class fair-IMF-LIN-radial \
	--classifier_class cw_fair_svm \
	--lambda_lcb 1 \
	--optimization_approach brute_force \
	--grid_search_bins 3 -e 9 \
	--sensitive_attribute_nodes x1 \
	--num_train_samples 500 \
	--num_fair_samples 2 \
	--batch_number 0 \
	--sample_count 200


################################################################################
################################################################################


python main.py \
	--scm_class adult \
	--dataset_class adult \
	--classifier_class nonsens_mlp \
	--lambda_lcb 1 \
	--optimization_approach brute_force \
	--grid_search_bins 4 -e 9 \
	--sensitive_attribute_nodes x1 x2 x3 \
	--non_intervenable_nodes x4 \
	--num_train_samples 1500 \
	--num_fair_samples 10 \
	--batch_number 0 \
	--sample_count 1000



python main.py \
	--scm_class adult \
	--dataset_class adult \
	--classifier_class nonsens_mlp \
	--lambda_lcb 1 \
	--optimization_approach brute_force \
	--grid_search_bins 3 -e 9 \
	--sensitive_attribute_nodes x1 x2 x3 \
	--non_intervenable_nodes x4 \
	--num_train_samples 1500 \
	--num_fair_samples 2 \
	--batch_number 0 \
	--sample_count 1000



# python main.py --scm_class fair-IMF-LIN --classifier_class cw_fair_svm --lambda_lcb 1 --optimization_approach brute_force --grid_search_bins 5 -e 9 --sensitive_attribute_nodes x1 --num_train_samples 500 --num_fair_samples 2 --fair_kernel_type linear --batch_number 0 --sample_count 200
# python main.py --scm_class fair-IMF-LIN --classifier_class cw_fair_lr --lambda_lcb 1 --optimization_approach brute_force --grid_search_bins 5 -e 9 --sensitive_attribute_nodes x1 --num_train_samples 500 --num_fair_samples 2 --fair_kernel_type linear --batch_number 0 --sample_count 200
# python main.py --scm_class fair-IMF-LIN --classifier_class cw_fair_mlp --lambda_lcb 1 --optimization_approach brute_force --grid_search_bins 5 -e 9 --sensitive_attribute_nodes x1 --num_train_samples 500 --num_fair_samples 2 --fair_kernel_type linear --batch_number 0 --sample_count 200

# * add 4 rows for LR (appendix)
# * recreate half of table 1 for adult or semi-synth german
# * find clever story to tell of the plights of 1 individual (candidates: high indiv unfairness in IMF world; compare recourse action and cost for orig-vs-twin)


# race (indep of intelligence)
# intelligence
# education (how mcuh they've been taught)
# effort

# together go into state/private school


# match marginals, or joint, but not everything
# to allow for admissions dataaset to have intervenable nodes that are non-descendants




