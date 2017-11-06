import numpy as np
from sklearn.linear_model import SGDClassifier

from sklearn import datasets
from sklearn.linear_model import SGDClassifier


# import some data to play with


def load():
    file = '/Users/junix/dataset/iris-species/Iris.csv'
    rs = np.recfromcsv(file)
    cols = ('sepallengthcm', 'sepalwidthcm')  # , 'petallengthcm', 'petalwidthcm')
    cols = [rs[c].reshape(-1, 1) for c in cols]
    X = np.concatenate(cols, axis=1)
    Y = rs['species']  # .reshape(-1, 1)
    Y[Y == b'Iris-setosa'] = b'1'
    Y[Y == b'Iris-versicolor'] = b'2'
    Y[Y == b'Iris-virginica'] = b'3'
    return X, Y.astype(np.int)


if __name__ == '__main__':
    # iris = datasets.load_iris()
    # X = iris.data[:, :2]
    # y = iris.target
    # colors = "bry"
    #
    # # shuffle
    # idx = np.arange(X.shape[0])
    # np.random.seed(13)
    # np.random.shuffle(idx)
    # X = X[idx]
    # y = y[idx]
    #
    # # standardize
    # mean = X.mean(axis=0)
    # std = X.std(axis=0)
    # X = (X - mean) / std
    #
    # h = .02  # step size in the mesh
    #
    # clf = SGDClassifier(alpha=0.001, max_iter=100).fit(X, y)
    #
    # print(X)



    X, Y = load()
    c = SGDClassifier(alpha=0.0001, max_iter=10000)
    c.fit(X, Y)
    Y1 = c.predict(X)
    print((Y1 == Y).sum())
    print(np.concatenate((Y.reshape(-1, 1), Y1.reshape(-1, 1)), axis=1))
