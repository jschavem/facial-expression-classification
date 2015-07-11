from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import numpy as np
import csv
import glob

from sklearn import datasets
iris = datasets.load_iris()


def c_validate(filename):
    data = []

    with open(filename, 'rb') as csvfile:
        for idx, row in enumerate(csv.reader(csvfile, delimiter=',', quotechar='"')):
            if idx == 0 and len(data) > 0:
                continue

            data.append(row)

    print "Attribute %s" % data[0][0]
    X = np.array([row[3:] for row in data[1:]])
    y = np.array([row[0] for row in data[1:]])

    print("Loaded %d sessions and %d truthy values for %s" % (len(X), len(y), filename))

    leave_one_out = cross_validation.LeaveOneOut(len(y))

    act = []
    pred = []
    for train_index, test_index in leave_one_out:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = SVC()
        clf.fit(X_train, y_train)
        p = clf.predict(X_test)

        act.append(y_test[0])
        pred.append(p[0])
        #print "%s, predicted %s" % (y_test[0], y_pred[0])

    return act, pred


num_test_files = 35
dataset = 'koelstra-approach'

actual = []
predicted = []
for idx, filename in enumerate(glob.glob('../output/%s/*.csv' % dataset)):
    if idx == num_test_files:
        break

    a, p = c_validate(filename)
    actual += a
    predicted += p


conf_matrix = confusion_matrix(actual, predicted, ['low', 'high'])
print conf_matrix

print ""
scores = f1_score(actual, predicted, ['low', 'high'], 'low', average=None)
class_counts = [sum(row) for row in conf_matrix]
average_f1 = np.average(scores, weights=class_counts)

print "Average F1 score: %.3f" % average_f1
print "Average accuracy: %.3f" % accuracy_score(actual, predicted)