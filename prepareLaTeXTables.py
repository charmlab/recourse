import os
import glob
import pickle
import numpy as np
import pandas as pd

from debug import ipsh

for approx_scm in ['m0_true', 'm1_alin', 'm1_akrr']:
    print(r'\midrule')
    if approx_scm == 'm0_true':
        print(r'\multirow{5}{*}{$\Mcal^\star$}')
    elif approx_scm == 'm1_alin':
        print(r'\multirow{5}{*}{$\Hat{\Mcal}_{\text{LIN}}$}')
    elif approx_scm == 'm1_akrr':
        print(r'\multirow{5}{*}{$\Hat{\Mcal}_{\text{KR}}$}')
    for fair_model in [
        'vanilla_svm',
        'nonsens_svm',
        'iw_fair_svm',
        'unaware_svm',
        'cw_fair_svm',
        # 'vanilla_*', # lr/mlp, depending on scm_setup
        # 'nonsens_*', # lr/mlp, depending on scm_setup
        # 'unaware_*', # lr/mlp, depending on scm_setup
        # 'cw_fair_*', # lr/mlp, depending on scm_setup
        ]:
        if fair_model == 'vanilla_svm':
            print('\t' + r'& SVM$(\Xb, A)$')
        elif fair_model == 'nonsens_svm':
            print('\t' + r'& SVM$(\Xb)$')
        elif fair_model == 'iw_fair_svm':
            print('\t' + r'& FairSVM$(\Xb,A)$')
        elif fair_model == 'unaware_svm':
            print('\t' + r'& SVM$(\Xb_{\nd(A)})$')
        elif fair_model == 'cw_fair_svm':
            print('\t' + r'& SVM$(\Xb_{\nd(A)}, \Ub_{\d(A)})$')
        elif fair_model == 'vanilla_*':
            print('\t' + r'& LR$(\Xb, A)$')
        elif fair_model == 'nonsens_*':
            print('\t' + r'& LR$(\Xb)$')
        elif fair_model == 'unaware_*':
            print('\t' + r'& LR$(\Xb_{\nd(A)})$')
        elif fair_model == 'cw_fair_*':
            print('\t' + r'& LR$(\Xb_{\nd(A)}, \Ub_{\d(A)})$')
        for scm_setup in [
            'fair-IMF-LIN__',
            'fair-CAU-LIN__',
            'fair-CAU-ANM__',
            'fair-IMF-LIN-radial__',
            'fair-CAU-LIN-radial__',
            'fair-CAU-ANM-radial__',
        ]:
            if '*' in fair_model:
                if 'radial' not in scm_setup:
                    fair_model = fair_model[:-1] + 'lr'
                else:
                    fair_model = fair_model[:-1] + 'mlp'

            # find folder
            folders = glob.glob(f'./_experiments/_fair_neurips/_bu_table1,2/*{scm_setup}*{fair_model}*')
            # folders = glob.glob(f'./_experiments/_fair_neurips/_table1,2/*{scm_setup}*{fair_model}*')
            assert len(folders) == 1
            folder_path = folders[0]

            # load log_training and search for test accuracy.
            log_training_file = open(os.path.join(folder_path, 'log_training.txt'), 'r')
            all_lines = log_training_file.readlines()
            test_accuracy_line = [line for line in all_lines if 'Testing accuracy' in line][0]
            tmp_index = test_accuracy_line.find('%')
            test_acc = np.around(float(test_accuracy_line[tmp_index+1:tmp_index+6]), 1)

            try:
                # load data frame with remainder of results
                comparison_df = pickle.load(open(os.path.join(folder_path, f'_comparison_{fair_model}'), 'rb'))
            except Exception as e:
                print(e)
                continue

            # print f'{approx_scm} {fair_model} {scm_setup}'
            print( '\t\t' + \
                f'& {np.around(test_acc, 2)} ' + \
                f'& {np.around(comparison_df.loc[approx_scm, "delta_dist_to_db"], 2):.2f} ' + \
                f'& {np.around(comparison_df.loc[approx_scm, "delta_cost_valid"], 2):.2f} ' + \
                f'& {np.around(comparison_df.loc[approx_scm, "max_delta_indiv_cost"], 2):.2f}' \
            )
        print('\t\t\\\\')
