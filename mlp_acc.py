import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import argparse
from joblib.parallel import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from util import load_dataset

parser = argparse.ArgumentParser(description='One-Pass training')
parser.add_argument('dataset', help='dataset for experiment')
parser.add_argument('--lr', type=float, default=1., help='learning rate')
parser.add_argument('--decay',
                    action='store_true',
                    help='learning rate decay: 1 / sqrt(t)')
parser.add_argument('--std', action='store_true', help='standardize the data')
parser.add_argument('--minmax', action='store_true', help='minmax the data')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--n_jobs', type=int, default=1, help='using parallel')

args = parser.parse_args()
assert not (args.std and args.minmax)
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)

# Shuffle dataset
X, y = load_dataset(args.dataset)
shuffle = np.random.permutation(X.shape[0])
X, y = X[shuffle], y[shuffle]

# Onehot encoding
y = OneHotEncoder(sparse=False).fit_transform(np.c_[y])

# Cross Validation
kfold = KFold(n_splits=5)


def train_and_valid(train_index, test_index):
    train_X, test_X, train_y, test_y = (
        X[train_index],
        X[test_index],
        y[train_index],
        y[test_index],
    )

    if args.std:
        scaler = StandardScaler().fit(train_X)
        train_X, test_X = scaler.transform(train_X), scaler.transform(test_X)
    if args.minmax:
        scaler = MinMaxScaler().fit(train_X)
        train_X, test_X = scaler.transform(train_X), scaler.transform(test_X)

    model = nn.Sequential(
        nn.Linear(train_X.shape[1], 128),
        nn.Tanh(),
        nn.Linear(128, train_y.shape[1]),
        nn.LogSoftmax(dim=0),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        (lambda t: 1 / np.sqrt(t + 1)) if args.decay else lambda t: 1.,
    )
    data = torch.tensor(train_X, dtype=torch.float)
    target = torch.tensor(train_y, dtype=torch.float)

    for t in tqdm(range(train_X.shape[0])):
        xt, yt = data[t], target[t]
        pt = model(xt)
        lt = -(pt * yt).sum()

        optimizer.zero_grad()
        lt.backward()
        optimizer.step()
        scheduler.step()

    test = torch.tensor(test_X, dtype=torch.float)
    with torch.no_grad():
        score = model(test)
    pred = score.numpy()
    return accuracy_score(test_y.argmax(-1), pred.argmax(-1))


acc_list = Parallel(n_jobs=args.n_jobs)(delayed(train_and_valid)(*index)
                                        for index in kfold.split(X))
print(args.dataset, np.mean(acc_list), np.std(acc_list))

with open('mlp_perf', 'a') as f:
    f.write("{}    {}    {}    {}    {:.3f}   {:.3f}\n".format(
        args.dataset.rjust(12),
        str(args.lr).ljust(5),
        str(args.decay).ljust(5),
        str(args.std).ljust(5),
        np.mean(acc_list),
        np.std(acc_list),
    ))