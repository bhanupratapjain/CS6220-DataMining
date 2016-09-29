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
    data_50[0] = np.vstack((np.zeros((1,data_50[0].shape[1])), data_50[0]))
    data_100 = np.split(data, [100])
    data_100[0] = np.vstack((np.zeros((1,data_100[0].shape[1])), data_100[0]))
    data_150 = np.split(data, [150])
    data_150[0] = np.vstack((np.zeros((1,data_150[0].shape[1])), data_150[0]))
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
    fig.canvas.set_window_title('HW01_PB01_A')
    fig.tight_layout()
    row, col = 0, 0
    for train in file_map:
        X_train,y_train=load_data(train)
        X_test,y_test=load_data(file_map[train])
        mse_train = get_mse(X_train,y_train)
        mse_test = get_mse(X_test,y_test)
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
    fig.canvas.set_window_title('HW01_PB01_B')
    fig.tight_layout()
    row = 0
    for l in lambdas:
        ax[row].plot(np.arange(50, X_train.shape[0]), mse_lambda[l])
        ax[row].set_xlabel("m")
        ax[row].set_ylabel("mse")
        ax[row].set_title("lamda = {}".format(l))
        row += 1


def k_fold_generator(X, y, k_fold):
    subset_size = (X.shape[0]) / k_fold
    for k in range(1, k_fold+1):
        start_valid = (k - 1) * subset_size
        end_valid = start_valid + subset_size
        valid_rows = np.arange(start_valid,end_valid)
        train_rows = [x for x in range(X.shape[0]) if x not in valid_rows]

        X_train = X[train_rows, :]
        X_valid = X[valid_rows, :]
        y_train = y[train_rows, :]
        y_valid = y[valid_rows, :]
        # print X_train.shape,X_valid.shape,y_train.shape,y_valid.shape
        yield X_train, y_train, X_valid, y_valid


def get_cv(cv_array):
    arr = np.array(cv_array)
    return np.mean(arr, axis=0)


def part_c():
    k_fold = 5
    split_data_file()
    files = glob.glob("data/*.csv")
    file_map = divide_files(files)
    row, col = 0, 0
    cv = {}
    for train_file in file_map:
        X, y = load_data(train_file)
        cv_array = []
        for X_train, y_train, X_valid, y_valid in k_fold_generator(X, y, k_fold):
            mse_array = []
            X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
            X_valid = np.hstack((np.ones((X_valid.shape[0], 1)), X_valid))
            for l in range(151):
                rReg = RigeRegression(X_train, y_train, l)
                rReg.fit()
                mse_array.append(np.mean((rReg.predict(X_valid) - y_valid) ** 2))
            cv_array.append(mse_array)
        cv_for_all_l = get_cv(cv_array)
        min_vc_index = np.argmin(cv_for_all_l)
        cv[train_file] = (min_vc_index, cv_for_all_l[min_vc_index])
    print cv


if __name__ == '__main__':
    part_a()
    part_b()
    part_c()
    plt.show()
