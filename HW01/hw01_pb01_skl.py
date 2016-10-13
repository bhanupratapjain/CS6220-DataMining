# import matplotlib
# matplotlib.use('Agg')
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

train_data = np.loadtxt('spambase/train-100-10.csv', skiprows=1, delimiter=',')
test_data = np.loadtxt('spambase/test-100-10.csv', skiprows=1, delimiter=',')

# print train_data
# print len(train_data)
# print len(test_data)

X = np.delete(train_data, 10, axis=1)
y = train_data[:, 10]
error = []

for i in range(0, 150):
    clf = linear_model.Ridge(alpha=i)
    clf.fit(X, y)
    y_predictions = clf.predict(X)
    error.append(mean_squared_error(y, y_predictions))
print error
fig = plt.figure(figsize=(20, 6))

ax = fig.add_subplot(111)
# ax = plt.gca()
ax.plot(error,np.arange(150))
# ax.set_xscale('log')
# plt.xlabel('alpha')
# plt.ylabel('mse')
# plt.title('Ridge coefficients as a function of the regularization')
# plt.axis('tight')
# plt.savefig('fig')
plt.show()