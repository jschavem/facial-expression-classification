from sklearn.metrics import confusion_matrix
from threading import Thread
import time
from Queue import Queue, Empty
import multiprocessing

from lib.in_subject_cross_validation import *
import lib.in_subject_cross_validation as libcv
import sys
import argparse


attribute_counts = {
    'koelstra-approach': {
        # 0: [3, 4, 28, 32, 33, 41, 62, 70],
        0: 38,
        1: 25,
        2: 18
    },
    'koelstra-normalized': {
        0: 8,
        1: 7,
        2: 36
    },
    'au-counts': {
        0: 35,
        1: 10,
        2: 24
    },
    'au-counts-valence': {
        0: 32,
        1: 27,
        2: 24
    },
    'au-counts-weighted': {
        0: 12,
        1: 13,
        2: 10
    },
    'au-counts-avg': {
        0: 18
    },
    'mime-koelstra': {
        0: 7
    },
    'bined': {
        0: 27
    },
    'bined-koelstra': {
        0: 26
    },
    'mahnob-hr': {
        0: 12,
        1: 5,
        2: 11
    },
    'mahnob-hr-au-count': {
        0: 16,
        1: 36,
        2: 41
    },
    'mime-au-counts': {
        0: 21
    },
    'mime-orientation': {
        0: 3
    },
    'mime-orientation-au-counts': {
        0: 8
    },
    'bined-orientation' : {
        0: 4
    },
    'bined-orientation-au-counts': {
        0: 19
    },
    'mahnob-orientation': {
        0: 5
    },
    'mahnob-orientation-au-counts': {
        0: 37
    }
}

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


def main():
    parser = argparse.ArgumentParser(description='Perform cross-validation on the dataset, cross-validating the behavior of one specific subject.')
    parser.add_argument('dataset', help='name of the dataset folder')
    parser.add_argument('class_id', type=int, help='target class id, 0-2')
    parser.add_argument('ground_truth_count', type=int, help='number of ground truth values, 1-3')
    parser.add_argument('rfe', type=int, default=1, help='perform RFE?')

    args = parser.parse_args()
    print args

    attr_count = None if args.rfe == 0 else attribute_counts[args.dataset][args.class_id]

    # Load our dataset into memory
    Xs, ys = libcv._load_full_dataset(args.dataset, args.class_id, args.ground_truth_count)

    # Perform cross-validation on the dataset, using RFE to achieve the target attr_count
    actual, predicted = cross_validate_combined_dataset(Xs, ys, attr_count, threaded=False)

    print_report(actual, attr_count, args.class_id, args.dataset, predicted)


if __name__ == '__main__':
    main()
