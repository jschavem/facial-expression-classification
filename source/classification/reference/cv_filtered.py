from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as np
import time

from lib.in_subject_cross_validation import cross_validate_dataset


selected_attributes = {
    'koelstra-approach': {
        # 0: [3, 4, 28, 32, 33, 41, 62, 70],
        0: [3, 12, 32, 43, 44],
        1: [13, 17, 46, 57, 72, 73, 75],
        2: [3, 4, 5, 13, 14, 17, 18, 24, 30, 32, 34, 46, 53, 59, 61, 71, 75, 88]
    },
    'koelstra-normalized': {
        0: [3, 4, 28, 32, 33, 41, 62, 70],
        1: [3, 17, 30, 31, 32, 34, 41, 42, 49, 59, 60, 61, 62, 72, 73, 75, 78, 86, 88, 89],
        2: [3, 4, 5, 12, 14, 15, 17, 24, 28, 30, 31, 32, 33, 34, 41, 43, 44, 46, 47, 53, 57, 59, 60, 61, 62, 63, 70, 71,
            72, 73, 75, 76, 82, 86, 88, 89],
    }
}


def main():
    dataset = 'koelstra-approach'
    attribute_index = 0
    attributes = selected_attributes[dataset][attribute_index]

    actual, predicted = cross_validate_dataset(dataset, attribute_index, ground_truth_count=attributes)

    conf_matrix = confusion_matrix(actual, predicted, ['low', 'high'])

    print conf_matrix
    print ""

    scores = f1_score(actual, predicted, ['low', 'high'], 'low', average=None)
    class_counts = [sum(row) for row in conf_matrix]

    average_f1 = np.average(scores, weights=class_counts)
    accuracy = accuracy_score(actual, predicted)
    print "Average F1 score: %.3f" % average_f1
    print "Average accuracy: %.3f" % accuracy

    attr_names = ["valence", "arousal", "control"]
    print "py-%s-rfe,%s,%s,%.3f,%.3f" % (
        dataset, attr_names[attribute_index], time.strftime('%Y-%m-%d'), average_f1, accuracy)


if __name__ == '__main__':
    main()
