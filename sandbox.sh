# on cluster
cd ~/dev/recourse/_experiments/
# find 2021.05.20_11.13* -name _optimization_curves | xargs rm -rf # (optional) delete the _optimization_curves directory and contents
zip -r _fair_neurips 2021.05.20_11.13*

# on local machine
scp -r amir@login.cluster.is.localnet:~/dev/recourse/_experiments/_fair_neurips.zip _fair_neurips