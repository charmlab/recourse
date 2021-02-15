# def warn(*args, **kwargs):
#     pass

# import warnings
# warnings.warn = warn # to ignore all warnings.

from cvxopt import matrix, solvers
import numpy as np
import pandas as pd

# import plotly.plotly as ply
# import plotly.tools as tls
import matplotlib.pyplot as plt
from functools import partial
from sklearn.utils import check_random_state

import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split

from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel, chi2_kernel, laplacian_kernel, sigmoid_kernel

from sklearn.datasets.samples_generator import make_blobs, make_moons, make_circles, make_classification
# import scipy

from sklearn import svm
from itertools import compress

from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
import os, sys, time

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

CREDIT = "../datasets/credit_processed.csv"
PROPUBLICA = "../datasets/propublica/propublica-recidivism_numerical-binsensitive.csv"
GERMAN = "../datasets/german_numerical-binsensitive.csv"

cvfold = 10
noiterations = 10
labelizer = lambda x: -1 if x < 0 else 1
vlabel = np.vectorize(labelizer)
rectol = 1e-20

def read_CSV(filepath, k, pct, rstate=None):
    # first column is target variable
    # second column is sensitive attr
    rstate = check_random_state(rstate)

    dataset = pd.read_csv(filepath)
    n = dataset.shape[0]
    d = dataset.shape[1] - 2
    # print('Original Data\n',len(dataset.values[dataset.values[:,0] == 1]), len(dataset.values[dataset.values[:,0] == -1]))

    if k > n:
        k = n

    data = dataset.values[rstate.choice(np.arange(n), k, replace=False), :]

    # if normalization is required
    scaler = StandardScaler()
    data[:, 2:] = scaler.fit_transform(data[:, 2:])

    df = {}

    df['trainx'], df['testx'], df['traingrp'], df['testgrp'], df['trainy'], df['testy'] = train_test_split(data[:,2:], data[:,1], data[:,0], test_size=pct, random_state=rstate)
    df['both'] = np.concatenate((df['trainx'], df['testx']), axis=0)

    df['train'] = np.hstack((df['traingrp'].reshape(-1, 1), df['trainx']))
    df['test'] = np.hstack((df['testgrp'].reshape(-1, 1), df['testx']))

    return df, d

class RecourseSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel_fn='linear', lam=1, ups=10, noiter=10, C=1, gamma=0.5, degree=1):
        self.gamma = gamma
        self.degree = degree
        self.lam = lam
        self.ups = ups
        self.noiter = noiter

        self.kernel_params = {'kernel': kernel_fn, 'gamma':self.gamma, 'degree':self.degree}
        self.kernel_fn = kernel_fn

        self.Q1 = None
        self.Q2 = None
        self.Q3 = None

        self.train_samples = None
        self.vanilla = {'wnorm':None, 'coeff1':None, 'coeff2':None, 'bias':None}
        self.converged = {'wnorm':None, 'coeff1':None, 'coeff2':None, 'bias':None}

        self.fitted = False

    def kernel(self, *params):
        if (self.kernel_fn =='linear'):
            return linear_kernel(*params)
        elif (self.kernel_fn =='rbf'):
            return partial(rbf_kernel, gamma=self.gamma)(*params)
        elif (self.kernel_fn =='poly'):
            return partial(polynomial_kernel, degree=self.degree)(*params)

    def fit(self, X, y=None):
        xtr = X[:,1:]
        gtr = X[:,0]

        n = xtr.shape[0]
        self.C = np.eye(xtr.shape[1])
        self.train_samples = xtr

        cx = np.matmul(xtr, self.C)
        Y_matrix = np.tile(y, (len(y), 1))
        YiYj = Y_matrix * Y_matrix.T

        self.Q1 = self.kernel(cx, xtr)
        self.Q2 = self.kernel(xtr)
        self.Q3 = self.kernel(cx, cx)

        YiYjQ2 = YiYj * self.Q2

        e = np.hstack((-1 * np.ones(n), 0.0)).T
        b = np.array([0])
        G = np.vstack((-1 * np.eye(n + 1), np.eye(n + 1)))

        e = matrix(e, tc='d')
        G = matrix(G, tc='d')
        b = matrix(b, tc='d')

        solvers.options['show_progress'] = False

        results = {}

        itr = 0
        lastmu = None

        clf = svm.SVC(**self.kernel_params)
        clf.fit(xtr, y)
        ppred = clf.predict(xtr)

        dissimilar = [0]

        results = []
        while (itr < 5 or dissimilar):
            itr = itr + 1

            if itr > self.noiter:
                break

            l = self.lam
            if itr < 3:
                l = 0

            neg_gtr = gtr[ppred == -1]
            cnt = {1: neg_gtr[neg_gtr == 1].shape[0], -1: neg_gtr[neg_gtr == -1].shape[0]}

            s1 = np.array(gtr)
            s2 = np.array(([float(1 - ppred[i]) / (2 * cnt[gtr[i]]) if cnt[gtr[i]] > 0 else 0 for (i, _) in enumerate(gtr)]))
            s = s1 * s2

            cons_ytr = np.hstack((y.T, sum(s)))
            A = cons_ytr[np.newaxis]
            A = matrix(A, tc='d')

            S_matrix = np.tile(np.diag(s).diagonal(), (len(s), 1))
            YiSj = Y_matrix.T * S_matrix
            SiSj = S_matrix * S_matrix.T
            del(S_matrix)

            SiYjQ1 = np.array([sum(YiSj.T * self.Q1)])
            SiSjQ3 = np.array([sum(sum(SiSj * self.Q3))])
            del(YiSj, SiSj)

            M = np.block([[YiYjQ2, SiYjQ1.T], [SiYjQ1, SiSjQ3]])
            M = matrix(M, tc='d')
            del (SiYjQ1, SiSjQ3)

            h = np.hstack((np.zeros(n), l, self.ups * np.ones(n)/n, l))
            h = matrix(h, tc='d')


            sol = solvers.qp(M, e, G, h, A, b, initvals=lastmu)
            mu = sol['x']

            mu_arr = np.array(mu).flatten()

            if sol['status'] == 'optimal':
                self.fitted = True

                lastmu = mu

                self.converged['coeff2'] = mu_arr[0:n] * y[0:n]
                self.converged['coeff1'] = mu_arr[n] * s

                tol = 1e-6
                mu_n =  mu_arr[0:n]
                sup_indices = np.where(np.logical_and(mu_n > tol, mu_n < self.ups/n))[0]

                bias = 0
                for sv_ind in sup_indices:
                    bias += y[sv_ind] - np.dot(self.converged['coeff2'], self.Q2[sv_ind, :]) - np.dot(self.converged['coeff1'], self.Q1[:, sv_ind])
                self.converged['bias'] = bias / len(sup_indices)

                pred = np.matmul(self.converged['coeff2'], self.Q2) + np.matmul(self.converged['coeff1'], self.Q1) + self.converged['bias']
                pred = vlabel(pred)

                train_acc = float(np.sum(pred * y > 0))/ len(y)

                self.converged['wnorm'] = np.sqrt(np.matmul(np.matmul(mu_arr.T, M), mu_arr))

                neg_gtr = gtr[pred == -1]
                cnt = {1: neg_gtr[neg_gtr == 1].shape[0], -1: neg_gtr[neg_gtr == -1].shape[0]}

                s1 = np.array(gtr)
                s2 = np.array(([float(1 - pred[i]) / (2 * cnt[gtr[i]]) if cnt[gtr[i]] > 0 else 0 for (i, _) in enumerate(gtr)]))
                s = s1 * s2

                rec_diff_train = np.sum(s * (np.matmul(self.converged['coeff2'], self.Q2) + np.matmul(self.converged['coeff1'], self.Q3) + bias))/self.converged['wnorm']

                dissimilar = [i for i, j in zip(pred, ppred) if i != j]
                ppred = pred.copy()

                results.append([train_acc, abs(rec_diff_train)])
            else:
                break

            if itr == 2:
                self.vanilla = self.converged.copy()

        l = len(results)
        if l < self.noiter:
            results.extend([results[-1] for i in range(self.noiter-l)])

        return results

    def predict_core(self, X):
        trained_model = self.converged
        if not self.fitted:
            print('Call fit first!!')
            return

        xtst = X[:,1:]
        gtst = X[:,0]

        cx = np.matmul(self.train_samples, self.C)
        cxtst = np.matmul(xtst, self.C)

        k1 = self.kernel(self.train_samples, xtst)
        k2 = self.kernel(cx, xtst)
        k3 = self.kernel(self.train_samples, cxtst)
        k4 = self.kernel(cx, cxtst)

        tpreds = np.matmul(trained_model['coeff2'], k1) + np.matmul(trained_model['coeff1'], k2) + trained_model['bias']
        tpreds = vlabel(tpreds)

        neg_gtest = gtst[tpreds == -1]
        cntest = {1:neg_gtest[neg_gtest==1].shape[0], -1:neg_gtest[neg_gtest==-1].shape[0]}

        s1tst = np.array(gtst)
        s2tst = np.array(([float(1 - tpreds[i]) / (2 * cntest[gtst[i]]) if cntest[gtst[i]] > 0 else 0 for (i, _) in enumerate(gtst)]))
        stst = s1tst * s2tst

        # # Amir's additions
        # stst_group_1 = stst.copy()
        # stst_group_2 = stst.copy()

        # stst_group_1[stst_group_1 < 0] = 0
        # stst_group_2[stst_group_2 > 0] = 0

        # rec_dist_group_1 = np.sum(stst_group_1 * (np.matmul(trained_model['coeff2'], k3) + np.matmul(trained_model['coeff1'], k4) + trained_model['bias']))/trained_model['wnorm']
        # rec_dist_group_2 = np.sum(stst_group_2 * (np.matmul(trained_model['coeff2'], k3) + np.matmul(trained_model['coeff1'], k4) + trained_model['bias']))/trained_model['wnorm']

        # rec_diff_test = rec_dist_group_1 + rec_dist_group_2 # one is neg, other is pos. take abs() for true distances.

        rec_diff_test = np.sum(stst * (np.matmul(trained_model['coeff2'], k3) + np.matmul(trained_model['coeff1'], k4) + trained_model['bias']))/trained_model['wnorm']
        return tpreds, rec_diff_test

    def predict(self, X):
        tpreds, rec_diff_test = self.predict_core(X)
        return tpreds

    def decision_function(self, X):
        tpreds, rec_diff_test = self.predict_core(X)
        return np.abs(rec_diff_test)

    def main_eval(self, trained_model, X, y=None):
        if not self.fitted:
            print('Call fit first!!')
            return

        xtst = X[:,1:]
        gtst = X[:,0]

        cx = np.matmul(self.train_samples, self.C)
        cxtst = np.matmul(xtst, self.C)

        k1 = self.kernel(self.train_samples, xtst)
        k2 = self.kernel(cx, xtst)
        k3 = self.kernel(self.train_samples, cxtst)
        k4 = self.kernel(cx, cxtst)

        tpreds = np.matmul(trained_model['coeff2'], k1) + np.matmul(trained_model['coeff1'], k2) + trained_model['bias']
        tpreds = vlabel(tpreds)

        neg_gtest = gtst[tpreds == -1]
        cntest = {1:neg_gtest[neg_gtest==1].shape[0], -1:neg_gtest[neg_gtest==-1].shape[0]}

        s1tst = np.array(gtst)
        s2tst = np.array(([float(1 - tpreds[i]) / (2 * cntest[gtst[i]]) if cntest[gtst[i]] > 0 else 0 for (i, _) in enumerate(gtst)]))
        stst = s1tst * s2tst

        rec_diff_test = np.sum(stst * (np.matmul(trained_model['coeff2'], k3) + np.matmul(trained_model['coeff1'], k4) + trained_model['bias']))/trained_model['wnorm']

        test_acc = float(np.sum(tpreds * y > 0))/len(y)

        return ([tpreds, test_acc, abs(rec_diff_test)])

    def vanilla_eval(self, X, y=None):
        return self.main_eval(self.vanilla, X, y)

    def fairrec_eval(self, X, y=None):
        return self.main_eval(self.converged, X, y)

    def score(self, X, y=None):
        _, _, rec1 = self.vanilla_eval(X, y)
        _, _, rec2 = self.fairrec_eval(X, y)

        if rec2 < rectol and rec1 < rectol:
            return 0
        else:
            return (rec2 - rec1)/rec2 if (rec2 > rec1) else (rec1 - rec2)/rec1

def create_plots(bef, aft, name, title):
    box_fig = plt.figure()
    xlabels = ['Before', 'After']

    median_col = '#fc8d59'
    colors_rec = ['#ffffff', '#999999']
    colors_acc = ['#ffffff', '#91bfdb']

    whiskstyle = {'linestyle': '--'}
    meanstyle = {'markeredgecolor':'black', 'marker':'o', 'markerfacecolor':'#ffffbf', 'markersize':8}
    medianstyle = {'color':median_col, 'linewidth': 2.5}

    # train recourse plot
    ax = box_fig.add_subplot(221)
    bp1 = ax.boxplot([before['train_rec'], after['train_rec']], patch_artist=True, labels=xlabels,showfliers=False, showmeans=True, whiskerprops=whiskstyle, meanprops=meanstyle, medianprops=medianstyle)
    ax.yaxis.grid(True)
    ax.set_ylabel('Recourse Difference')
    ax.set_title('Train')

    # test recourse plot
    ax = box_fig.add_subplot(222)
    bp2 = ax.boxplot([before['test_rec'], after['test_rec']], patch_artist=True, labels=xlabels, showfliers=False, showmeans=True, whiskerprops=whiskstyle, meanprops=meanstyle, medianprops=medianstyle)
    ax.yaxis.grid(True)
    ax.set_title('Test')

    # train accuracy plot
    ax = box_fig.add_subplot(223)
    bp3 = ax.boxplot([before['train_acc'], after['train_acc']], patch_artist=True, labels=xlabels, showfliers=False, showmeans=True, whiskerprops=whiskstyle, meanprops=meanstyle, medianprops=medianstyle)
    ax.yaxis.grid(True)
    ax.set_ylabel('Accuracy')
    # ax.set_title('Train accuracies')

    # test accuracy plot
    ax = box_fig.add_subplot(224)
    bp4 = ax.boxplot([before['test_acc'], after['test_acc']], patch_artist=True, labels=xlabels, showfliers=False, showmeans=True, whiskerprops=whiskstyle, meanprops=meanstyle, medianprops=medianstyle)
    ax.yaxis.grid(True)
    # ax.set_title('Test accuracies')

    for bplot in (bp1, bp2):
        for patch, color in zip(bplot['boxes'], colors_rec):
            patch.set_facecolor(color)

    for bplot in (bp3, bp4):
        for patch, color in zip(bplot['boxes'], colors_acc):
            patch.set_facecolor(color)

    box_fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(name)

get_data = read_CSV

def vanillasvm(xtr, ytr, xtst, params):
    clf = svm.SVC(**params)
    clf.fit(xtr, ytr)
    ptest = clf.predict(xtst)
    ptrn = clf.predict(xtr)

    return (ptrn, ptest)

if __name__ == '__main__':

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    dsname = 'GERMAN'
    filepath = GERMAN

    noruns = 10

    t = time.time()

    lams = [0.2, 0.5, 1, 2, 10, 50, 100]
    param_grids = [[
        {'lam': lams, 'kernel_fn': ['linear']}
    ],
    [
        {'lam': lams, 'kernel_fn': ['poly'], 'degree':[2, 3, 5]}
    ]]

    for i in range(2):
        data, d = get_data(filepath, 1000, 0.8)

        clf = GridSearchCV(cv=cvfold, estimator=RecourseSVM(noiter=noiterations), param_grid=param_grids[i], n_jobs=-1)
        clf.fit(data['train'], data['trainy'])
        # print(clf.cv_results_)
        print(clf.best_params_)

        # clf.refit(data['train'], data['trainy'])

        train_aggr = np.zeros((noiterations, 2))
        van_test_aggr = np.zeros((2,))
        test_aggr = np.zeros((2,))

        before = {'train_acc':[], 'test_acc':[], 'train_rec':[], 'test_rec':[]}
        after = {'train_acc':[], 'test_acc':[], 'train_rec':[], 'test_rec':[]}

        for j in range(noruns):
            print('Run {}'.format(j+1))
            data, d = get_data(filepath, 1000, 0.8)

            C = np.eye(d) #/ d
            # ups = 10

            rsvm = RecourseSVM(**clf.best_params_)
            res1 = rsvm.fit(data['train'], data['trainy'])
            for r in res1:
                print(r)
            print('\n')

            res0 = rsvm.vanilla_eval(data['test'], data['testy'])[1:]
            res2 = rsvm.fairrec_eval(data['test'], data['testy'])[1:]

            print(res0)
            print(res2)
            print('\n')

            train_aggr += np.array(res1)
            van_test_aggr += np.array(res0)
            test_aggr += np.array(res2)

            before['train_acc'].append(res1[1][0])
            before['train_rec'].append(res1[1][1])
            before['test_acc'].append(res0[0])
            before['test_rec'].append(res0[1])

            after['train_acc'].append(res1[-1][0])
            after['train_rec'].append(res1[-1][1])
            after['test_acc'].append(res2[0])
            after['test_rec'].append(res2[1])

        print('Train: \n{}'.format(train_aggr/noruns))
        print('Test(Vanilla): \n{}'.format(van_test_aggr/noruns))
        print('Test: \n{}'.format(test_aggr/noruns))

        create_plots(before, after, '{}-{}.pdf'.format(dsname, clf.best_params_['kernel_fn']), dsname)

    elapsed_time = time.time() - t
    print('Time elapsed: {}\n'.format(elapsed_time))
