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


def read_file(file_path):
    with open(file_path, 'r') as inf:
        cdata = json.load(inf)
    
    users = cdata['users']
    hierarchies = cdata['hierarchies'] if 'hierarchies' in cdata else []
    user_data = cdata['user_data']
    return users, hierarchies, user_data

def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    file_paths = list_json_paths(data_dir)
    for file_path in file_paths:
        users, hierarchies, user_data = read_file(file_path)
        clients.extend(users)
        groups.extend(hierarchies)
        data.update(user_data)

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


def generate_data_shard(train_data_dir, test_data_dir, num_client_servers=1):
    train_paths = list_json_paths(train_data_dir)
    test_paths = list_json_paths(test_data_dir)
    k = len(train_paths)
    n = num_client_servers
    # distribute data among all client servers as evenly as possible
    # i.e. client servers created by each data file should not differ by more than 1
    client_servers_per_file = [n//k + 1]*(n % k) + [n//k]*(k - n % k)
    # Assume the corresponding train and test files named in the same pattern
    # i.e. 01_train.json -- 01_test.json, 02_train.json -- 02_test.json etc
    for train_path, test_path, num_cs in zip(sorted(train_paths), sorted(test_paths), client_servers_per_file):
        yield train_path, test_path, num_cs


def list_json_paths(data_dir):
    files = os.listdir(data_dir)
    return [os.path.join(data_dir, f) for f in files if f.endswith('.json')]
