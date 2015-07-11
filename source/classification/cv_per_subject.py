from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as np
import time
import argparse

from lib.in_subject_cross_validation import cross_validate_dataset


##
# cross_validate_per_subject_rfe: Performs cross validation on a per-subject basis where 19 sessions are used for
# training and the 20th is used to test.
##

# The number of attributes that should be passed on to the classifier for each class.
attribute_counts = {
    'koelstra-approach': {
        0: 25,
        1: 21,
        2: 10
    },
    'koelstra-normalized': {
        0: 8,
        1: 7,
        2: 36
    },
    'au-counts': {
        0: 18,
        1: 16,
        2: 4
    },
    'mahnob-au+face': {
        0: 30
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
    }
}


def print_report(actual, class_id, dataset, predicted):
    conf_matrix = confusion_matrix(actual, predicted, ['low', 'high'])
    print ""
    print conf_matrix
    scores = f1_score(actual, predicted, ['low', 'high'], 'low', average=None)
    average_f1 = np.average(scores)
    accuracy = accuracy_score(actual, predicted)
    #print [1 if actual[idx] == predicted[idx] else 0 for (idx, _) in enumerate(actual)]
    print "\nAverage F1 score: %.3f" % average_f1
    print "Average accuracy: %.3f" % accuracy
    print "Low F1 score: %.3f" % scores[0]
    print "High F1 score: %.3f" % scores[1]
    low_ratings = [p for (idx, p) in enumerate(predicted) if actual[idx] == 'low']
    high_ratings = [p for (idx, p) in enumerate(predicted) if actual[idx] == 'high']
    print "Low accuracy: %.3f" % (float(low_ratings.count('low')) / len(low_ratings))
    print "High accuracy: %.3f" % (float(high_ratings.count('high')) / len(high_ratings))
    attr_names = ["valence", "arousal", "control"]
    print "%s,leave-one-session-out-rfe,%s,%s,%.3f,%.3f" % (
        dataset, attr_names[class_id], time.strftime('%Y-%m-%d'), average_f1, accuracy)


def main():
    parser = argparse.ArgumentParser(
        description='Perform cross-validation on the dataset, cross-validating the behavior of all but one subjects '
                    'using an SVM classifier.'
    )
    parser.add_argument('dataset', help='name of the dataset folder')
    parser.add_argument('class_id', type=int, help='target class id, 0-2')
    parser.add_argument('ground_truth_count', type=int, help='number of ground truth values, 1-3')
    parser.add_argument('rfe', type=int, default=1, help='perform RFE?')

    args = parser.parse_args()
    print args

    attr_count = None if args.rfe == 0 else attribute_counts[args.dataset][args.class_id]
    actual, predicted = cross_validate_dataset(args.dataset, args.class_id, attribute_count=attr_count,
                                               ground_truth_count=args.ground_truth_count)

    print_report(actual, args.class_id, args.dataset, predicted)


if __name__ == '__main__':
    main()
