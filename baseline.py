
import numpy as np
import pandas as pd
import hickle as hkl
import random
import yaml
import cv2
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from train_funcs import unpack_configs

config = None
trainA_filenames = None
valA_filenames = None
trainB_filenames = None
valB_filenames = None
train_labels = None
val_labels = None
img_mean = None

def load_config():
    global config, trainA_filenames, valA_filenames, \
            trainB_filenames, valB_filenames, \
            train_labels, val_labels, img_mean
    with open('config.yaml', 'r') as f:                                                                         
        config = yaml.load(f)
    with open('spec_1gpu.yaml', 'r') as f:
        config = dict(config.items() + yaml.load(f).items())
    
    (flag_para_load,
            trainA_filenames, valA_filenames,
            trainB_filenames, valB_filenames,
            train_labels, val_labels, img_mean) = unpack_configs(config)

def load_training_batch(minibatch_index=0):
    x1 = hkl.load(str(trainA_filenames[minibatch_index])) - img_mean
    x2 = hkl.load(str(trainB_filenames[minibatch_index])) - img_mean
    
    #x1 = x1.reshape([-1, x1.shape[3]]).transpose()
    #x2 = x2.reshape([-1, x2.shape[3]]).transpose()

    x1 = x1.transpose().reshape([x1.shape[3], -1])
    x2 = x2.transpose().reshape([x2.shape[3], -1])

    X = np.hstack((x1, x2))

    batch_size = len(x1)
    y = train_labels[minibatch_index * batch_size:
                     (minibatch_index + 1) * batch_size]
    
    return X, y

def load_batches(start=0, end=10):
    print "Loading data..."

    X, y = None, None
    for i in xrange(start, end+1):
        Xi, yi = load_training_batch(0)
        if X is None:
            X = Xi
            y = yi
        else:
            X = np.vstack((X, Xi))
            y = np.concatenate((y, yi))

    count = X.shape[0]
    train_size = np.int(count*0.8)

    # Scale appropriately
    scaler = StandardScaler()
    scaler.fit(X[:train_size])
    X = scaler.transform(X)
    
    Xtrain = X[:train_size]
    ytrain = y[:train_size]
    Xtest = X[train_size:]
    ytest = y[train_size:]

    return Xtrain, ytrain, Xtest, ytest

def make_model(X, y, choice='log'):
    clf = None
    
    log_params = {
         "loss": ["log"],
         "penalty": ["l1", "l2"], # , "elasticnet"],
         "alpha": 10.0**-np.arange(1,7),
         "epsilon": 10.0**np.arange(2,7),
         "n_iter": [5],
         "shuffle": [True]
        }
    huber_params = {
         "loss": ["modified_huber"],
         "penalty": ["l1", "l2", "elasticnet"],
         "alpha": 10.0**-np.arange(1,7),
         "epsilon": 10.0**np.arange(0,7),
         "n_iter": [5],
         "shuffle": [True]
        }
    SVM_params = {
        'C': 10.0**-np.arange(0,7),
        'gamma': 10.0**-np.arange(0,7),
        'kernel': ['linear'], # 'poly', 'rbf'
        'probability': [True]
        }
    RF_params = {
        'n_estimators' : [1000],
        'bootstrap' : [False]
        }
    NN_params = {}
    
    if choice=='log':
        params = log_params
        model = SGDClassifier()
    elif choice=='huber':
        params = huber_params
        model = SGDClassifier()
    elif choice=='svm':
        params = SVM_params
        model = svm.SVC(C=1)
    elif choice=='rf':
        params = RF_params
        model = RandomForestClassifier(n_estimators=1000, bootstrap=False)
        #clf = RandomForestClassifier(n_estimators=1000, bootstrap=False)
    
    # Set up Grid Search
    print "Grid search..."
    clf = GridSearchCV(model, params, n_jobs=2, scoring='f1')
    clf.fit(X[:255], y[:255]) # Grid search only on part of the data
    clf = clf.best_estimator_
    print clf

    clf.fit(X, y)
    
    return clf

def test(clf, Xtest, ytest):
    print "Testing..."
    ztest = clf.predict(Xtest)
    
    metrics =  "Cross-tabulation of test results:\n"
    metrics +=  pd.crosstab(ytest, ztest, rownames=['actual'], colnames=['preds']).to_string()
    metrics += "\n\n"
    
    metrics +=  "Classification Report:\n"
    metrics +=  classification_report(ytest, ztest)
    metrics += "\n"
    
    # Display metrics
    print metrics

    print("Total "+str(sum(ytest!=ztest))+" incorrect, out of "+str(len(ytest)))


def run():
    load_config()
    Xtrain, ytrain, Xtest, ytest = load_batches(0, 20)
    clf = make_model(Xtrain, ytrain)
    test(clf, Xtest, ytest)

run()

