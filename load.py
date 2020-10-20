import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def next_batch(train_data, train_target, batch_size, step):
    index = [i for i in range(batch_size * step, batch_size * (step + 1))]
    np.random.shuffle(index)
    batch_data = []
    batch_target = []
    for i in range(0, batch_size):
        batch_data.append(train_data[index[i]])
        batch_target.append(train_target[index[i]])
    return np.asarray(batch_data), np.asarray(batch_target)  # 返回


def loaddataset(filename, line):
    data = pd.read_csv(filename, nrows=line)
    data['Label'] = pd.Categorical(data['Label']).codes
    cols = list(data)
    for item in cols:
        data = data[~data[item].isin([np.nan, np.inf])]
    Labelset = data.iloc[:, 79:80].values.tolist()
    Dataset = data.iloc[:, 3:79].values.tolist()
    for item in Labelset:
        if item[0] is 0:
            item.append(1)
        else:
            item.append(0)
    X_train, X_test, Y_train, Y_test = train_test_split(Dataset, Labelset, test_size=0.20, random_state=50)
    return X_train, X_test, Y_train, Y_test
