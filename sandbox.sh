# on cluster
cd ~/dev/recourse/_experiments/
# find 2021.05.20_11.13* -name _optimization_curves | xargs rm -rf # (optional) delete the _optimization_curves directory and contents
zip -r _fair_neurips 2021.05.20_11.13*

# on local machine
scp -r amir@login.cluster.is.localnet:~/dev/recourse/_experiments/_fair_neurips.zip _fair_neurips.
scp -r amir@login.cluster.is.localnet:~/dev/recourse/_experiments/2021.* .


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


python main.py \
	--scm_class adult \
	--dataset_class adult \
	--classifier_class nonsens_mlp \
	--lambda_lcb 1 \
	--optimization_approach brute_force \
	--grid_search_bins 2 -e 9 \
	--sensitive_attribute_nodes x1 x2 x3 \
	--num_train_samples 500 \
	--num_fair_samples 2 \
	--batch_number 0 \
	--sample_count 400


--non_intervenable_nodes  \



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



