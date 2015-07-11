from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC
import csv
import random
import glob
from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing
import sys

def load_dataset(dataset, num_files):
    data = []
    list = glob.glob('../output/%s/*.csv' % dataset)
    #random.shuffle(list)
    for filename in list:
        num_files -= 1
        if num_files < 0:
            break

        print filename

        try:
            with open(filename, 'rb') as csvfile:
                for idx, row in enumerate(csv.reader(csvfile, delimiter=',', quotechar='"')):
                    if idx == 0 and len(data) > 0:
                        continue

                    data.append(row)
        except IOError:
            next
    return data


def select_features(data, class_attribute):
    X = [[float(v) for v in row[3:]] for row in data[1:]]
    X = preprocessing.MinMaxScaler().fit_transform(X)
    y = [0 if row[class_attribute] == 'low' else 1 for row in data[1:]]
    print("Loaded %d sessions" % len(y))
    svc = LinearSVC(class_weight='auto')
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(y, 10), scoring='f1')
    print("RFE in progress")
    rfecv = rfecv.fit(X, y)
    return rfecv


# Plot number of features VS. cross-validation scores
def plot_grid_scores(grid_scores):
    import pylab as pl
    pl.figure()
    pl.xlabel("Number of features selected")
    pl.ylabel("Cross validation F1 score")
    pl.plot(range(1, len(grid_scores) + 1), grid_scores)
    pl.show()


def main():
    training_set_size = int(sys.argv[2])
    class_attribute_index = int(sys.argv[3])

    data = load_dataset(sys.argv[1], training_set_size)
    rfecv = select_features(data, class_attribute_index)

    print("Selected features:")
    # for idx, value in enumerate(rfecv.support_):
    #     if value:
    #         print(data[0][idx+3])

    print("\nOptimal number of features : %d" % rfecv.n_features_)

    selected_attributes = [idx + 3 for idx, value in enumerate(rfecv.support_) if value]
    print selected_attributes

    plot_grid_scores(rfecv.grid_scores_)


if __name__ == '__main__':
    main()
