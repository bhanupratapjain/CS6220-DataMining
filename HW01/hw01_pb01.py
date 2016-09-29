import numpy as np
import matplotlib.pyplot as plt
import glob


class RigeRegression:
    def __init__(self, X, y, lmda):
        I = np.eye(X.shape[1])
        I[0, 0] = 0
        self.X = X
        self.XT = self.X.T
        self.y = y
        self.I = I
        self.lmda = lmda
        self.weights = None

    def fit(self):
        self.weights = np.dot(np.linalg.inv(np.dot(self.XT, self.X) + self.lmda * self.I), np.dot(self.XT, self.y))

    def predict(self, X):
        return np.dot(X, self.weights)


def load_data(filename):
    print"Running Regression on {}".format(filename)
    data = np.loadtxt(filename, skiprows=1, delimiter=',')
    print "Input Data Shape:{}".format(data.shape)
    cols = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    X = data[:, cols]
    y = data[:, 10:11]
    return X, y


def split_data_file():
    data = np.loadtxt('data/train-1000-100.csv', skiprows=1, delimiter=',')
    data_50 = np.split(data, [50])
    data_100 = np.split(data, [100])
    data_150 = np.split(data, [150])
    np.savetxt("data/train-50(1000)-100.csv", data_50[0], delimiter=',')
    np.savetxt("data/train-100(1000)-100.csv", data_100[0], delimiter=',')
    np.savetxt("data/train-150(1000)-100.csv", data_150[0], delimiter=',')


def read_data_files(filename):
    data = np.loadtxt(filename, skiprows=1, delimiter=',')


def divide_files(files):
    file_map = {}
    for file in files:
        if 'train' in file:
            if '(1000)' in file:
                file_map[file] = 'data/test-1000-100.csv'
            else:
                file_map[file] = file.replace('train', 'test')
    return file_map


def get_mse(X, y):
    mse = []
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    for i in range(151):
        print "lambda =>{}".format(i)
        rReg = RigeRegression(X, y, i)
        rReg.fit()
        mse.append(np.mean((rReg.predict(X) - y) ** 2))
    return mse


def part_a():
    split_data_file()
    files = glob.glob("data/*.csv")
    file_map = divide_files(files)
    fig, ax = plt.subplots(4, 2)
    fig.tight_layout()
    row, col = 0, 0
    for train in file_map:
        mse_train = get_mse(load_data(train))
        mse_test = get_mse(load_data(file_map[train]))
        ax[row][col].plot(np.arange(151), mse_train, label="Train")
        ax[row][col].plot(np.arange(151), mse_test, label="Test")
        ax[row][col].set_xlabel("lambda")
        ax[row][col].set_ylabel("mse")
        ax[row][col].set_title(train + ";" + file_map[train])
        ax[row][col].legend()
        row += 1
        if row == 4:
            row = 0
            col = 1
    fig.delaxes(ax[3][1])
    plt.show()


def part_b():
    lambdas = [1, 46, 150]
    X_train, y_train = load_data('data/train-1000-100.csv')
    X_test, y_test = load_data('data/test-1000-100.csv')
    mse_lambda = {}
    for l in lambdas:
        mse_test = []
        for m in range(50, X_train.shape[0]):
            mse = 0
            for i in range(10):
                X_train_random = X_train[np.random.randint(X_train.shape[0], size=m), :]
                X_train_random = np.hstack((np.ones((X_train_random.shape[0], 1)), X_train_random))
                y_train_random = y_train[np.random.randint(y_train.shape[0], size=m), :]

                rReg = RigeRegression(X_train_random, y_train_random, l)
                rReg.fit()

                X_test_random = X_test[np.random.randint(X_test.shape[0], size=m), :]
                X_test_random = np.hstack((np.ones((X_test_random.shape[0], 1)), X_test_random))
                y_test_random = y_test[np.random.randint(y_test.shape[0], size=m), :]
                mse += np.mean((rReg.predict(X_test_random) - y_test_random) ** 2)
            mse_test.append(mse / 10)
        mse_lambda[l] = mse_test
    fig, ax = plt.subplots(3)
    fig.tight_layout()
    row = 0
    for l in lambdas:
        ax[row].plot(np.arange(50, X_train.shape[0]), mse_lambda[l])
        ax[row].set_xlabel("m")
        ax[row].set_ylabel("mse")
        ax[row].set_title("lamda = {}".format(l))
        row += 1
    plt.show()


def part_c():
    pass


if __name__ == '__main__':
    # part_a()
    # part_b()
    part_c()
