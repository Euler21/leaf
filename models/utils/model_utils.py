import json
import numpy as np
import os
from collections import defaultdict


def batch_data(data, batch_size, seed):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(seed)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


def ternarize(gradients):
    '''
    Quantize gradient to {-1, 0, 1}.
    Introduced in
    [TernGrad: Ternary Gradients to Reduce Communication in Distributed Deep
     Learning](https://arxiv.org/pdf/1705.07878.pdf)
    Args:
        gradients: list of numpy.ndarray
    Return:
        ternary_grad: sign of each elements in input gradients rescaled by the
        largest element, with small values clipped by some probability.
    '''
    ternary_grad = []

    for grad in gradients:
        std = np.std(grad)

        s = np.amax(abs(grad))

        sign = np.sign(grad)
        mask = (np.random.rand(*grad.shape) < (abs(grad)/s))
        # according to the TernGrad paper, clip small values with probability
        # 1 - abs(g_tk)/s_t
        s = min([s, 2.5*std])
        ternary_grad.append(s*np.multiply(sign, mask))

    return ternary_grad