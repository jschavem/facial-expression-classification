from sklearn.metrics import confusion_matrix
from threading import Thread
import time
from Queue import Queue, Empty
import multiprocessing

from lib.in_subject_cross_validation import *
import lib.in_subject_cross_validation as libcv
import sys
import argparse


q = Queue()

# Performs recursive feature elimination until 'attribute count' has been reached
def _eliminate_features(X_test, X_train, attribute_count, y_train):
    clf = LinearSVC(class_weight='auto')
    rfe = RFE(clf, n_features_to_select=attribute_count, step=1)
    fit = rfe.fit(X_train, y_train)

    # Reduce the feature matrices to contain just the selected features
    X_train = [fit.transform(X) for X in X_train]
    X_test = [fit.transform(X) for X in X_test]
    return X_test, X_train


def _cv_instances(Xs, ys, test_index, train_index, result_pairs, attribute_count):
    # print "Cross validating with %d left out" % test_index
    Xs_train, Xs_test = flatten(Xs[train_index]), flatten(Xs[test_index])
    ys_train, ys_test = flatten(ys[train_index]), flatten(ys[test_index])

    transformer = preprocessing.MinMaxScaler().fit(to_float(flatten(Xs)))
    Xs_train = transformer.transform(to_float(Xs_train))
    Xs_test = transformer.transform(to_float(Xs_test))

    if attribute_count is not None:
        Xs_test, Xs_train = _eliminate_features(Xs_test, Xs_train, attribute_count, ys_train)
        Xs_test = flatten(Xs_test)
        Xs_train = flatten(Xs_train)

    clf = SVC(**SVC_parameters)
    # clf = LinearSVC(class_weight='auto')
    clf.fit(to_float(Xs_train), ys_train)

    ys_pred = clf.predict(to_float(Xs_test))
    predicted_class = list(ys_pred)
    actual_class = ys_test

    print "%d, %.3f" % (test_index[0], accuracy_score(actual_class, predicted_class))

    # print "Finished cross validation for %d" % test_index

    result_pairs.append((actual_class, predicted_class))


def threaded_worker():
    while True:
        try:
            arguments = q.get(False)
            _cv_instances(*arguments)
            q.task_done()
        except Empty:
            break


def cross_validate_combined_dataset(Xs, ys, num_attributes=None, threaded=False):
    leave_one_out = cross_validation.LeaveOneOut(len(ys))

    result_pairs = []
    threads = []

    for train_index, test_index in leave_one_out:
        if threaded:
            q.put((Xs, ys, test_index, train_index, result_pairs, num_attributes))
        else:
            _cv_instances(Xs, ys, test_index, train_index, result_pairs, num_attributes)

    if threaded:
        for num in range(1, multiprocessing.cpu_count()):
            print "Starting thread %d" % num
            thread = Thread(target=threaded_worker)
            threads.append(thread)
            thread.start()

        [thread.join() for thread in threads]

    actual_classes = [actual for (actual, _) in result_pairs]
    predicted_classes = [predicted for (_, predicted) in result_pairs]

    return flatten(actual_classes), flatten(predicted_classes)


def flatten(list):
    return [item for sublist in list for item in sublist]


def to_float(list):
    return [[float(item) for item in sublist] for sublist in list]


def print_report(actual, attr_count, class_id, dataset, predicted):
    # Print the performance to the console
    conf_matrix = confusion_matrix(actual, predicted, ['low', 'high'])
    print ""
    print conf_matrix
    scores = f1_score(actual, predicted, ['low', 'high'], 'low', average=None)
    average_f1 = np.average(scores)
    accuracy = accuracy_score(actual, predicted)
    print "\nAverage F1 score: %.3f" % average_f1
    print "Average accuracy: %.3f" % accuracy
    low_ratings = [p for (idx, p) in enumerate(predicted) if actual[idx] == 'low']
    high_ratings = [p for (idx, p) in enumerate(predicted) if actual[idx] == 'high']
    print "Low accuracy: %.3f" % (float(low_ratings.count('low')) / len(low_ratings))
    print "High accuracy: %.3f" % (float(high_ratings.count('high')) / len(high_ratings))
    attr_names = ["valence", "arousal", "control"]
    print "%s,leave-one-subject-out%s,%s,%s,%.3f,%.3f" % (
        dataset, '' if (attr_count is None) else '-rfe', attr_names[class_id], time.strftime('%Y-%m-%d'), average_f1,
        accuracy)




import numpy as np
import pylab as pl

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV


parser = argparse.ArgumentParser(description='Perform cross-validation on the dataset, cross-validating the behavior of one specific subject.')
parser.add_argument('dataset', help='name of the dataset folder')
parser.add_argument('class_id', type=int, help='target class id, 0-2')
parser.add_argument('ground_truth_count', type=int, help='number of ground truth values, 1-3')

args = parser.parse_args()
print args

##############################################################################
# Load and prepare data set
#
# dataset for grid search
Xs, ys = libcv._load_full_dataset(args.dataset, args.class_id, args.ground_truth_count)
X = to_float(flatten(Xs))
y = flatten(ys)

# It is usually a good idea to scale the data for SVM training.
# We are cheating a bit in this example in scaling all of the data,
# instead of fitting the transformation on the training set and
# just applying it on the test set.

scaler = StandardScaler()
X = scaler.fit_transform(X)

##############################################################################
# Train classifier
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

C_range = 10.0 ** np.arange(-1, 5)
param_grid = dict(C=C_range)
cv = StratifiedKFold(y=y, n_folds=3)
grid = GridSearchCV(SVC(kernel='linear'), param_grid=param_grid, cv=cv, n_jobs=-1)
grid.fit(X, y)

print("The best classifier is: ", grid.best_estimator_)