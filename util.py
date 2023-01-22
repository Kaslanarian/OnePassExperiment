from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    load_svmlight_file,
)
import numpy as np
import os
from typing import Literal


def load_mnist(type: Literal['mnist', 'fashion', 'kmnist']):

    path = r'./data/{}MNIST/raw'.format({
        'mnist': '',
        'fashion': 'Fashion',
        'kmnist': 'K'
    }.get(type))

    with open(os.path.join(path, 'train-images-idx3-ubyte')) as fd:
        loaded = np.fromfile(fd, dtype=np.uint8)
    X = loaded[16:].reshape(60000, 28 * 28).astype(float)

    with open(os.path.join(path, 'train-labels-idx1-ubyte')) as fd:
        loaded = np.fromfile(fd, dtype=np.uint8)
    y = loaded[8:].reshape(60000).astype(int)

    return X, y


def load_sklearn(dataset: str):
    return {
        "breast_cancer": load_breast_cancer,
        "digits": load_digits,
        "iris": load_iris,
        "wine": load_wine,
    }[dataset](return_X_y=True)


def load_tabular(dataset: str):
    data = load_svmlight_file('data/tabular/{}'.format(dataset))
    X, y = np.asarray(data[0].todense()), data[1]
    y[y != -1] = 0
    return X, y


def load_dataset(dataset: str):
    if dataset in ['mnist', 'fashion', 'kmnist']:
        return load_mnist(dataset)
    elif dataset == 'fashion':
        return load_mnist(True)
    elif dataset in {'breast_cancer', 'digits', 'iris', 'wine'}:
        return load_sklearn(dataset)
    elif dataset in {'australian', 'diabetes', 'splice', 'svmguide3'}:
        return load_tabular(dataset)
    else:
        raise ValueError("Dataset \"{}\" is not supported".format(dataset))