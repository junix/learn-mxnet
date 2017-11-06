import numpy as np


def load():
    csv = '/Users/junix/dataset/iris-species/Iris.csv'
    np.recfromcsv(csv)


if __name__ == '__main__':
    load()
