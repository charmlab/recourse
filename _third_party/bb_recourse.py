import numpy as np 
import pandas as pd 
import sklearn
from sklearn.utils import check_random_state
from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import warnings
from sklearn.exceptions import DataConversionWarning

import os, sys, time

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from collections import Counter
from sklearn.metrics import accuracy_score
# from scipy import stats

# import plotly.plotly as ply
# import plotly.tools as tls
import matplotlib.pyplot as plt
# import seaborn as sns

# from multiprocessing import Pool
import ray
import psutil

CREDIT = "../datasets/credit_processed.csv"
PROPUBLICA = "../datasets/propublica-recidivism_numerical-binsensitive.csv"
GERMAN = "../datasets/german_numerical-binsensitive.csv"

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

# rstate = check_random_state(100)
TOL = 1e-10
ratio = .3

classifiers = {
    'RF':partial(RandomForestClassifier, max_depth=4, random_state=None),
    'SVC':partial(SVC, C=1, kernel='rbf', degree=3, gamma='scale', probability=True),
    'ADA':AdaBoostClassifier, #partial(AdaBoostClassifier, DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
    'LR':LogisticRegression,
    'GB':partial(GradientBoostingClassifier, max_depth  = 4, random_state = None)
}

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
    ntr = df['trainx'].shape[0]
    ntst = k - ntr

    return df, d, ntr, ntst

# This class heavily uses code snippets from the repo
# https://github.com/marcotcr/lime/tree/master/lime , especially
# two files `lime_base.py` and `lime_tabular.py`. Although we have
# modified them for our purposes.

class LimeTabularClassification(object):
    def __init__(self, train_data, kernel_width=None, kernel=None, sample_around_instance=False, random_state=None, feature_selection='highest_weights'):
        self.random_state = check_random_state(random_state)
        self.sample_around_instance = sample_around_instance
        self.feature_selection = feature_selection

        if kernel_width is None:
            kernel_width = np.sqrt(train_data.shape[1]) * 0.75
        kernel_width = float(kernel_width)
        
        if kernel is None:
            def krnl(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        self.kernel_fn = partial(krnl, kernel_width=kernel_width)
        
        self.scaler = None
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(train_data)

    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)
    
    def select_features(self, data, labels, weights, num_features, method):
        if method == 'none':
            return np.array(range(data.shape[1]))
        
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)

        elif method == 'highest_weights':
            clf = Ridge(alpha=0, fit_intercept=True,random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)
            feature_weights = sorted(zip(range(data.shape[0]), clf.coef_ * data[0]), key=lambda x: np.abs(x[1]), reverse=True)
            return np.array([x[0] for x in feature_weights[:num_features]])

    def cal_distance(self, data_rows, predict_fn, num_samples=5000, num_features=10, distance_metric='euclidean', neighbors=(None, [])):
        n = data_rows.shape[0]
        distsnpreds = np.repeat(np.zeros(3), n, 0).reshape(n, 3)
        sample, ws = neighbors
        weights = []
        # ss = []
        # print(weights)

        for i in range(n):
            if np.any(sample):
                data = sample
            else:
                smaller = 0
                thr = 0.01
                data = []
                yss = []

                while smaller < thr:
                    data = self.get_neighborhood(data_rows[0,:], num_samples)
                    yss = predict_fn(data)
                    # smaller = 0.2
                    smaller = np.min([len(yss[:,0])/len(yss[:,1]), len(yss[:,1])/len(yss[:,0])])
                    # smaller /= len(yss)
                    # print(smaller)
                
                # ss.append(smaller)
                sample = data
            
            data[0] = data_rows[i, :].copy()
            yss = predict_fn(data)
            # print(yss[:10])

            scaled_data = (data - self.scaler.mean_) / self.scaler.scale_

            if ws:
                weights.append(ws[i])
            else:
                distances = sklearn.metrics.pairwise_distances(
                        scaled_data,
                        scaled_data[0].reshape(1, -1),
                        metric=distance_metric
                ).ravel()
                
                # print(yss)

                weights.append(self.kernel_fn(distances))

            # print(i, len(weights))

            labels_column = yss[:,1]
            used_features = self.select_features(scaled_data, labels_column, weights[i], num_features, self.feature_selection)

            simple_model = Ridge(alpha=1, fit_intercept=True,random_state=self.random_state)
            simple_model.fit(scaled_data[:, used_features], labels_column, sample_weight=weights[i])
            
            # print(simple_model.coef_, simple_model.intercept_)

            # print(np.linalg.norm(simple_model.coef_))
            # if np.linalg.norm(simple_model.coef_) < TOL:
            #     distsnpreds[i] = [0, 0, 0]
            # else:
            d = (np.dot(data_rows[i, used_features], simple_model.coef_) + simple_model.intercept_)/np.linalg.norm(simple_model.coef_)

            s = simple_model.score(scaled_data[:, used_features],labels_column, sample_weight=weights[i])

            p = simple_model.predict(scaled_data[0, used_features].reshape(1, -1))
            distsnpreds[i] = [d, s, p]
        # print(ss)
        return (sample, weights, distsnpreds)

    def get_neighborhood(self, data_row, num_samples=1000):
        data = np.zeros((num_samples, data_row.shape[0]))
        data = self.random_state.normal(0, 1, num_samples * data_row.shape[0]).reshape(num_samples, data_row.shape[0])
        if self.sample_around_instance:
            data = data * self.scaler.scale_ + data_row
        else:
            data = data * self.scaler.scale_ + self.scaler.mean_
        # data[0] = data_row.copy()
        # inverse = data.copy()

        return data

@ray.remote
def run_agnostic(filepath, clfname, k, pct, sed=1):
    best_acc = 0
    chosen_samples = []
    chosen_weights = []
    test_samples = []
    test_weights = []

    np.random.seed(sed) 
    data, d, ntr, ntst = read_CSV(filepath, k, pct, None)

    results = []

    ntrial = 0
    while ntrial < 5:
        ntrial += 1

        samples = []
        weights = []

        rst = check_random_state(np.random.randint(100))

        clf = classifiers[clfname]()
        clf.fit(data['trainx'], data['trainy'], sample_weight=np.ones(ntr))
        ypred = clf.predict(data['trainx'])
 
        lf = LimeTabularClassification(data['trainx'], sample_around_instance=False, random_state=rst)

        runs = 2
        probs = np.zeros((runs, ntr))

        for i in range(runs):
            samp, ws, distsnpreds = lf.cal_distance(data['trainx'], clf.predict_proba)
            samples.append(samp)
            weights.append(ws)
            probs[i] += distsnpreds[:,2]
        
        avgProbs = np.average(probs, axis=0)
        avgProbs[avgProbs < 0.5] = -1
        avgProbs[avgProbs >= 0.5] = 1

        acc = float(np.sum(avgProbs * ypred > 0))/len(avgProbs)
        if acc > best_acc:
            best_acc = acc
            chosen_samples = samples.copy()
            chosen_weights = weights.copy()
        print(ntrial)

    print(best_acc)

    while True:
        rst = check_random_state(np.random.randint(100))

        # BEFORE 
        clf = classifiers[clfname]()
        clf.fit(data['trainx'], data['trainy'], sample_weight=np.ones(ntr))
        
        yall = clf.predict(data['both'])
        ypred = yall[:ntr]
        ytest = yall[ntr:]

        ltrain = LimeTabularClassification(data['trainx'], sample_around_instance=False, random_state=rst)
        ltest = LimeTabularClassification(data['testx'], sample_around_instance=False, random_state=rst)

        runs = 2
        tdists = np.zeros((runs, ntst))
        dists = np.zeros((runs, ntr))

        for i in range(runs):
            _, _, distsnpreds = ltrain.cal_distance(data['trainx'], clf.predict_proba, neighbors=(chosen_samples[i], chosen_weights[i]))
            # s, w, distsnpreds = ltrain.cal_distance(data['trainx'], clf.predict_proba)
            # chosen_samples.append(s)
            # chosen_weights.append(w)

            ts, tw, tdistsnpreds = ltest.cal_distance(data['testx'], clf.predict_proba)
            test_samples.append(ts)
            test_weights.append(tw)

            dists[i] = dists[i] + distsnpreds[:,0]
            tdists[i] += tdistsnpreds[:,0]
        
        avgTraindist = np.average(dists, 0)
        avgTstdist = np.average(tdists, 0)
        
        minTrainD, maxTrainD = np.min(avgTraindist[ypred==-1]), np.max(avgTraindist[ypred==-1])

        negavgtraindist = avgTraindist[ypred==-1]

        negtrainpredgrps = data['traingrp'][ypred==-1]
        traingrpcnt = Counter(negtrainpredgrps)
        gposTrainAvg = (np.sum(negavgtraindist[negtrainpredgrps==1])/traingrpcnt[1]) if traingrpcnt[1] != 0 else 0
        gnegTrainAvg = (np.sum(negavgtraindist[negtrainpredgrps==-1])/traingrpcnt[-1]) if traingrpcnt[-1] != 0 else 0

        recourse_diff_train = (gposTrainAvg - gnegTrainAvg)/(maxTrainD - minTrainD)


        minTestD, maxTestD = np.min(avgTstdist[ytest==-1]), np.max(avgTstdist[ytest==-1])
        negavgtestdist = avgTstdist[ytest==-1]

        negtestpredgrps = data['testgrp'][ytest==-1]
        testgrpcnt = Counter(negtestpredgrps)
        gposTestAvg = (np.sum(negavgtestdist[negtestpredgrps==1])/testgrpcnt[1]) if testgrpcnt[1] != 0 else 0
        gnegTestAvg = (np.sum(negavgtestdist[negtestpredgrps==-1])/testgrpcnt[-1]) if testgrpcnt[-1] != 0 else 0

        recourse_diff_test = (gposTestAvg - gnegTestAvg)/(maxTestD - minTestD)

        acc_train = float(np.sum(ypred * data['trainy'] > 0))/len(ypred)     
        acc_test = float(np.sum(ytest * data['testy'] > 0))/len(ytest)

        results.extend([abs(recourse_diff_train), abs(recourse_diff_test), acc_train, acc_test])

        # AFTER
        # training with weights inversely proportional to approx distance
        countneg = len(negavgtraindist)
        countneg_posdist = len(negavgtraindist[negavgtraindist > 0])
        m = np.min(negavgtraindist[negavgtraindist > 0]) if (2*countneg_posdist > countneg) else np.min(negavgtraindist)

        new_weights = np.ones(ntr)
        new_weights[ypred==-1] = m/negavgtraindist
        new_weights[(new_weights < 0)] = 1

        clf.fit(data['trainx'], data['trainy'], sample_weight=new_weights)
        yall = clf.predict(data['both'])
        ypred = yall[:ntr]
        ytest = yall[ntr:]

        tdists = np.zeros((runs, ntst))
        dists = np.zeros((runs, ntr))
        probs = np.zeros((runs, ntr))

        for i in range(runs):
            distsnpreds = ltrain.cal_distance(data['trainx'], clf.predict_proba, neighbors=(chosen_samples[i], chosen_weights[i]))[2]
            tdistsnpreds = ltest.cal_distance(data['testx'], clf.predict_proba, neighbors=(test_samples[i], test_weights[i]))[2]

            dists[i] = dists[i] + distsnpreds[:,0]
            tdists[i] += tdistsnpreds[:,0]

        avgTraindist = np.average(dists, 0)
        avgTstdist = np.average(tdists, 0)

        minTrainD, maxTrainD = np.min(avgTraindist), np.max(avgTraindist)
        negavgtraindist = avgTraindist[ypred==-1]

        negtrainpredgrps = data['traingrp'][ypred==-1]
        traingrpcnt = Counter(negtrainpredgrps)
        gposTrainAvg = (np.sum(negavgtraindist[negtrainpredgrps==1])/traingrpcnt[1]) if traingrpcnt[1] != 0 else 0
        gnegTrainAvg = (np.sum(negavgtraindist[negtrainpredgrps==-1])/traingrpcnt[-1]) if traingrpcnt[-1] != 0 else 0

        recourse_diff_train = (gposTrainAvg - gnegTrainAvg)/(maxTrainD - minTrainD)

        minTestD, maxTestD = np.min(avgTstdist), np.max(avgTstdist)
        negavgtestdist = avgTstdist[ytest==-1]

        negtestpredgrps = data['testgrp'][ytest==-1]
        testgrpcnt = Counter(negtestpredgrps)
        gposTestAvg = (np.sum(negavgtestdist[negtestpredgrps==1])/testgrpcnt[1]) if testgrpcnt[1] != 0 else 0
        gnegTestAvg = (np.sum(negavgtestdist[negtestpredgrps==-1])/testgrpcnt[-1]) if testgrpcnt[-1] != 0 else 0

        recourse_diff_test = (gposTestAvg - gnegTestAvg)/(maxTestD - minTestD)

        acc_train = float(np.sum(ypred * data['trainy'] > 0))/len(ypred)
        acc_test = float(np.sum(ytest * data['testy'] > 0))/len(ytest)

        results.extend([abs(recourse_diff_train), abs(recourse_diff_test), acc_train, acc_test])

        return results

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

if __name__ == "__main__":

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    DATA = CREDIT
    dsname = 'CREDIT'

    # if len(sys.argv) != 2:
    #     print('minimum 2 arguments')
    #     exit(0)

    # num_processes = int(sys.argv[1])
    num_processes = psutil.cpu_count(logical=False)
    ray.init(num_cpus=num_processes)

    t = time.time()

    howmany = 5
    stats = []

    for name in ['RF', 'LR', 'ADA']:
        before = {'train_acc':[], 'test_acc':[], 'train_rec':[], 'test_rec':[]}
        after = {'train_acc':[], 'test_acc':[], 'train_rec':[], 'test_rec':[]}

        # with Pool(processes=num_processes) as p:
        #     stats = np.array(p.map(run_agnostic1, [np.random.randint(1, 1000) for _ in range(howmany)]))
        
        stats = np.array(ray.get([run_agnostic.remote(DATA, name, 100, 0.8, np.random.randint(1000)) for _ in range(howmany)]))
        # print(stats)

        # remove all zero rows
        stats = stats[~np.all(stats==0, axis=1)]

        for row in stats:
            before['train_acc'].append(row[2])
            before['train_rec'].append(row[0])
            before['test_acc'].append(row[3])
            before['test_rec'].append(row[1])

            after['train_acc'].append(row[6])
            after['train_rec'].append(row[4])
            after['test_acc'].append(row[7])
            after['test_rec'].append(row[5])

        print(stats)

        # if not all are bad
        if stats.shape[0] > 0:
            print('Min')
            print(np.min(stats, axis=0))

            print('Max')
            print(np.max(stats, axis=0))

            print('Mean')
            print(np.mean(stats, axis=0))

            print('SD')
            print(np.std(stats, axis=0))

        create_plots(before, after, '{}-{}.pdf'.format(dsname, name), dsname)

    elapsed_time = time.time() - t
    print('Time elapsed: {}'.format(elapsed_time))

