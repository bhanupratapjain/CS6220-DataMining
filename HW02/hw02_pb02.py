from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from  sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pprint
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score


class KNN:
    def __init__(self, train_features, train_labels):
        self.train_features = train_features
        self.train_labels = train_labels

    def get_distances(self, query_features):
        diff = query_features - self.train_features
        distances = np.sqrt(np.sum(diff ** 2, axis=1))
        return distances

    def get_k_nearest_neighbours(self, k, query_features):
        distances = self.get_distances(query_features)
        neighbours = np.argsort(distances)[0:k]
        return neighbours

    def predict(self, k, query_features):
        neighbours = self.get_k_nearest_neighbours(k, query_features)
        y_predictions = self.train_labels[neighbours]
        counter = Counter(y_predictions)
        y_predict = counter.most_common(1)[0][0]
        return y_predict


def z_normalize(features):
    return (features - np.mean(features, axis=0)) / np.std(features, axis=0)


def get_accuracy(test_labels, predictions):
    correct_results = np.sum(np.isclose(test_labels, predictions))
    return correct_results / float(len(test_labels))


def get_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    print "Reading spambase from [{}] with Shape:{}".format(filename, data.shape)
    return data


def k_fold_generator(X, y, k_fold):
    subset_size = (X.shape[0]) / k_fold
    for k in range(1, k_fold + 1):
        start_valid = (k - 1) * subset_size
        end_valid = start_valid + subset_size
        valid_rows = np.arange(start_valid, end_valid)
        train_rows = [x for x in range(X.shape[0]) if x not in valid_rows]
        X_train = X[train_rows, :]
        X_valid = X[valid_rows, :]
        y_train = y[train_rows]
        y_valid = y[valid_rows]
        yield X_train, y_train, X_valid, y_valid


def part_a():
    train_features = get_data("spambase/spambase_train.txt")
    train_labels = get_data("spambase/spambase_train_label.txt")
    test_features = get_data("spambase/spambase_test.txt")
    test_labels = get_data("spambase/spambase_test_label.txt")

    knn = KNN(train_features=train_features, train_labels=train_labels)

    k_values = [1, 5, 11, 21, 41, 61, 81, 101, 201, 401]
    accuracy = []
    accuracy_skl = []
    for k in k_values:
        predictions = []
        for i in range(len(test_features)):
            predictions.append(knn.predict(k, test_features[i]))
        accuracy.append(get_accuracy(test_labels, predictions))
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(train_features, train_labels)
        predictions_skl = neigh.predict(test_features)
        accuracy_skl.append(accuracy_score(test_labels, predictions_skl))
    print accuracy
    print accuracy_skl

    fig, ax = plt.subplots(2)
    fig.set_size_inches(6, 8)
    ax[0].plot(k_values, accuracy, 'bo-')
    ax[0].set_ylabel("accuracy")
    ax[0].set_xlabel("k")
    ax[0].set_title("KNN: k vs. accuracy")

    ax[1].plot(k_values, accuracy_skl, 'ro-')
    ax[1].set_ylabel("accuracy")
    ax[1].set_xlabel("k")
    ax[1].set_title("KNN with sklearn: k vs. accuracy")


def part_b():
    train_features = get_data("spambase/spambase_train.txt")
    train_labels = get_data("spambase/spambase_train_label.txt")
    test_features = get_data("spambase/spambase_test.txt")
    test_labels = get_data("spambase/spambase_test_label.txt")

    # print train_features[0]
    # print test_features[0]

    # Normalize
    train_features = z_normalize(train_features)
    test_features = z_normalize(test_features)

    # print train_features[0]
    # print test_features[0]

    knn = KNN(train_features=train_features, train_labels=train_labels)

    k_values = [1, 5, 21, 41, 61, 81, 101, 201, 401]
    accuracy = []
    for k in k_values:
        predictions = []
        for i in range(len(test_features)):
            predictions.append(knn.predict(k, test_features[i]))
        accuracy.append(get_accuracy(test_labels, predictions))
    print accuracy

    fig, ax = plt.subplots(1)
    # fig.set_size_inches(6, 8)
    ax.plot(k_values, accuracy, 'bo-')
    ax.set_ylabel("accuracy")
    ax.set_xlabel("k")
    ax.set_title("KNN: k vs. accuracy")


def part_c():
    train_features = get_data("spambase/spambase_train.txt")
    train_labels = get_data("spambase/spambase_train_label.txt")
    test_features = get_data("spambase/spambase_test.txt")
    test_labels = get_data("spambase/spambase_test_label.txt")

    # Normalize
    train_features = z_normalize(train_features)
    test_features = z_normalize(test_features)

    knn = KNN(train_features=train_features, train_labels=train_labels)
    k_values = [1, 5, 21, 41, 61, 81, 101, 201, 401]
    result = {}
    for i in range(50):
        predictions = []
        for k in k_values:
            predictions.append(knn.predict(k, test_features[i]))
        result[i + 1] = map(lambda x: "spam" if x == 1.0 else "no", predictions)
    pprint.pprint(result, width=150)


def part_d():
    train_features = get_data("spambase/spambase_train.txt")
    train_labels = get_data("spambase/spambase_train_label.txt")
    train_features = z_normalize(train_features)
    k_values = [1, 5, 21, 41, 61, 81, 101, 201, 401]
    k_fold = 5
    cv = []

    for k in k_values:
        accuracy = []
        kf = cross_validation.KFold(len(train_labels), n_folds=k_fold, shuffle=True)
        for train_index, test_index in kf:
            X_train, y_train, X_valid, y_valid = \
                train_features[train_index], train_labels[train_index], \
                train_features[test_index], train_labels[test_index]
            knn = KNN(train_features=X_train, train_labels=y_train)
            predictions = map(lambda x: knn.predict(k, x), X_valid)
            accuracy.append(get_accuracy(y_valid, predictions))
        cv.append(accuracy)
    pprint.pprint(cv)
    accuracy_on_k = np.mean(np.array(cv), axis=1)
    pprint.pprint(accuracy_on_k)
    optimal_k_index = np.argmax(accuracy_on_k)
    print "Optimal Value for K is [{}] with Accuracy [{}]".format(k_values[optimal_k_index],
                                                                  accuracy_on_k[optimal_k_index])

    score = []
    for k in k_values:
        neigh = KNeighborsClassifier(n_neighbors=k)
        score.append(cross_val_score(neigh, train_features, train_labels, cv=5))
    pprint.pprint(score)
    accuracy_on_k_with_skl = np.mean(np.array(score), axis=1)
    pprint.pprint(accuracy_on_k_with_skl )
    optimal_k_index_with_skl = np.argmax(accuracy_on_k_with_skl)
    print "Optimal Value with sklearn for K is [{}] with Accuracy [{}]".format(k_values[optimal_k_index_with_skl],
                                                                               accuracy_on_k_with_skl[
                                                                                   optimal_k_index_with_skl])


if __name__ == "__main__":
    # part_a()
    # part_b()
    # part_c()
    part_d()
    # plt.show()
