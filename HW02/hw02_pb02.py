from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from  sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pprint

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
    print "Reading data from [{}] with Shape:{}".format(filename, data.shape)
    return data


def part_a():
    train_features = get_data("data/spambase_train.txt")
    train_labels = get_data("data/spambase_train_label.txt")
    test_features = get_data("data/spambase_test.txt")
    test_labels = get_data("data/spambase_test_label.txt")

    knn = KNN(train_features=train_features, train_labels=train_labels)

    k_values = [1, 5, 21, 41, 61, 81, 101, 201, 401]
    accuracy = []
    accuracy_skl = []
    for k in k_values:
        predictions = []
        predictions_skl = []
        for i in range(len(test_features)):
            predictions.append(knn.predict(k, test_features[i]))
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(train_features, train_labels)
            predictions_skl.append(neigh.predict(test_features[i].reshape(1, -1)))
        # print predictions
        # print test_labels
        accuracy.append(get_accuracy(test_labels, predictions))
        accuracy_skl.append(accuracy_score(test_labels, predictions_skl))
        # accuracy.append(accuracy_score(test_labels, predictions))
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
    train_features = get_data("data/spambase_train.txt")
    train_labels = get_data("data/spambase_train_label.txt")
    test_features = get_data("data/spambase_test.txt")
    test_labels = get_data("data/spambase_test_label.txt")

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
    train_features = get_data("data/spambase_train.txt")
    train_labels = get_data("data/spambase_train_label.txt")
    test_features = get_data("data/spambase_test.txt")
    test_labels = get_data("data/spambase_test_label.txt")

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
        result[i+1] = map(lambda x: "spam" if x == 1.0 else "no", predictions)
    pprint.pprint(result, width=150)


if __name__ == "__main__":
    # part_a()
    # part_b()
    part_c()
    plt.show()
