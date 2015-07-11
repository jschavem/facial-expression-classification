from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from threading import Thread
import time
from Queue import Queue, Empty

from lib.in_subject_cross_validation import *
import lib.in_subject_cross_validation as libcv
import multiprocessing
from sklearn.decomposition import RandomizedPCA

attribute_counts = {
    'koelstra-approach': {
        # 0: [3, 4, 28, 32, 33, 41, 62, 70],
        0: 22,
        1: 25,
        2: 18
    },
    'koelstra-normalized': {
        0: 8,
        1: 7,
        2: 36
    },
    'au-counts': {
        0: 32,
        1: 27,
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
    }
}

q = Queue()

# Performs recursive feature elimination until 'attribute count' has been reached
def _eliminate_features(X_test, X_train, attribute_count, y_train):
    print "Eliminating features until %d has been reached" % attribute_count

    pca = RandomizedPCA(n_components=attribute_count+10).fit(X_train)
    X_train = pca.transform(to_float(X_train))
    print "Finished pca"

    clf = SVC(**SVC_parameters)
    rfe = RFE(clf, n_features_to_select=attribute_count, step=0.1)
    fit = rfe.fit(X_train, y_train)
    print "Finished rfe"

    # Reduce the feature matrices to contain just the selected features
    X_train = [fit.transform(X) for X in X_train]
    X_test = [fit.transform(X) for X in pca.transform(to_float(X_test))]
    return X_test, X_train


def _cv_instances(Xs, ys, test_index, train_index, result_pairs, attribute_count):
    # print "Cross validating with %d left out" % test_index
    Xs_train, Xs_test = flatten(Xs[train_index]), flatten(Xs[test_index])
    ys_train, ys_test = flatten(ys[train_index]), flatten(ys[test_index])

    if attribute_count is not None:
        Xs_test, Xs_train = _eliminate_features(Xs_test, Xs_train, attribute_count, ys_train)
        Xs_test = flatten(Xs_test)
        Xs_train = flatten(Xs_train)

    # clf = SVC(**SVC_parameters)
    clf = GaussianNB()
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
    class_counts = [sum(row) for row in conf_matrix]
    average_f1 = np.average(scores, weights=class_counts)
    accuracy = accuracy_score(actual, predicted)
    print "\nAverage F1 score: %.3f" % average_f1
    print "Average accuracy: %.3f" % accuracy
    low_ratings = [p for (idx, p) in enumerate(predicted) if actual[idx] == 'low']
    high_ratings = [p for (idx, p) in enumerate(predicted) if actual[idx] == 'high']
    #print low_ratings
    #print high_ratings
    print "Low accuracy: %.3f" % (float(low_ratings.count('low')) / len(low_ratings))
    print "High accuracy: %.3f" % (float(high_ratings.count('high')) / len(high_ratings))
    attr_names = ["valence", "arousal", "control"]
    print "%s\tLeave-one-subject-out%s\t%s\t%s\t%.3f\t%.3f" % (
        dataset, '' if (attr_count is None) else '-rfe', attr_names[class_id], time.strftime('%Y-%m-%d'), average_f1,
        accuracy)


def main():
    dataset = 'koelstra-approach'
    class_id = 0
    ground_truth_variable_count = 3
    attr_count = attribute_counts[dataset][class_id]
    # attr_count = None

    # Load our dataset into memory
    Xs, ys = libcv._load_full_dataset(dataset, class_id, ground_truth_variable_count)

    # Perform cross-validation on the dataset, using RFE to achieve the target attr_count
    actual, predicted = cross_validate_combined_dataset(Xs, ys, attr_count, threaded=False)

    print_report(actual, attr_count, class_id, dataset, predicted)


if __name__ == '__main__':
    main()
