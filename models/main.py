"""Script to run the baselines."""
import argparse
import importlib
import numpy as np
import os
import sys
import random
import ray
import tensorflow as tf

import metrics.writer as metrics_writer

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from client_server import ClientServer
from client import Client
from server import Server
from model import ServerModel

from utils.args import parse_args
from utils.model_utils import generate_data_shard, list_json_paths, read_data, read_file
from utils.compression_utils import *

STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'
DATA_PATH=['/global', 'cfs', 'cdirs', 'mp156', 'rayleaf_dataset']

def main():

    args = parse_args()

    # Set the random seed if provided (affects client sampling, and batching)
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    tf.set_random_seed(123 + args.seed)

    model_path = '%s/%s.py' % (args.dataset, args.model)
    if not os.path.exists(model_path):
        print('Please specify a valid dataset and a valid model.')
    model_path = '%s.%s' % (args.dataset, args.model)
    
    print('############################## %s ##############################' % model_path)
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')

    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]
    num_client_servers = args.num_client_servers
    sketcher = eval(args.sketcher + '()')

    # Suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)

    # Create 2 models
    model_params = MODEL_PARAMS[model_path]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)

    # Create client model, and share params with server model
    tf.reset_default_graph()
    server_model = ClientModel(args.seed, *model_params)
    
    if args.multi_node:
        ray.init(address='auto', redis_password='5241590000000000')
    else:
        ray.init(local_mode=args.no_parallel)

    # Create clients
    client_servers = setup_client_servers(args.dataset, 
                                          args.seed,
                                          model_params,
                                          sketcher,
                                          ClientModel,
                                          args.use_val_set, 
                                          num_client_servers,
                                          deferred_loading=args.defer_data_loading)

    # Create server
    server = Server(server_model, client_servers, sketcher)

    # clients = setup_clients(args.dataset, client_model, args.use_val_set)
    client_ids, client_groups, client_num_samples = server.get_clients_info(all_clients=True)
    print('Clients in Total: %d' % len(client_ids))

    # Initial status
    print('--- Random Initialization ---')
    stat_writer_fn = get_stat_writer_function(client_ids, client_groups, client_num_samples, args)
    sys_writer_fn = get_sys_writer_function(args)
    print_stats(0, server, client_num_samples, args, stat_writer_fn, args.use_val_set)

    # Simulate training
    for i in range(num_rounds):
        print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))

        # Select clients to train this round
        server.select_clients(i, num_clients=clients_per_round)
        c_ids, c_groups, c_num_samples = server.get_clients_info()

        # Simulate server model training on selected clients' data
        sys_metrics = server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size, minibatch=args.minibatch)
        sys_writer_fn(i + 1, c_ids, sys_metrics, c_groups, c_num_samples)

        # Update server model
        server.update_model()

        # Test model
        if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
            print_stats(i + 1, server, c_num_samples, args, stat_writer_fn, args.use_val_set)
    
    # Save server model
    ckpt_path = os.path.join('checkpoints', args.dataset)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_path = server.save_model(os.path.join(ckpt_path, '{}.ckpt'.format(args.model)))
    print('Model saved in path: %s' % save_path)

    # Close models
    server.close_model()
    # stop Ray driver after job finish
    ray.shutdown()

def online(clients):
    """We assume all users are always online."""
    return clients

def partition_data(n, users, groups, train_data, test_data):
    # divide the list into n pieces more evenly =]
    index_part = np.array_split(np.arange(len(users)), n)

    users_part = [[users[i] for i in indices] for indices in index_part]
    groups_part = [[groups[i] for i in indices] for indices in index_part]

    train_data_part = [{u: train_data[u] for u in us} 
                       for us in users_part]
    test_data_part = [{u: test_data[u] for u in us} 
                       for us in users_part]

    return zip(users_part, groups_part, train_data_part, test_data_part)

def create_client_servers(seed, 
                          params, 
                          users, 
                          groups, 
                          train_data, 
                          test_data, 
                          sketcher,
                          model_cls, 
                          num_client_servers):
    if len(groups) == 0:
        groups = [[] for _ in users]
    assert(len(groups) == len(users))

    partition = partition_data(num_client_servers,
                               users,
                               groups,
                               train_data,
                               test_data)

    # this will return list of ObjectIDs for ClientServer actors
    return [ClientServer.remote(seed, 
                         params, 
                         cs_users, 
                         cs_groups, 
                         cs_train_data, 
                         cs_test_data, 
                         sketcher,
                         model_cls) 
            for cs_users, cs_groups, cs_train_data, cs_test_data in partition]
    #clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
    #return clients


@ray.remote(resources={"Node": 1})
def load_data_and_create_client_servers(seed, params, train_path, test_path, sketcher, model_cls, num_client_servers):
    """
    
    """
    # refactor this with read_dir
    train_users, train_groups, train_data = read_file(train_path)
    test_users, test_groups, test_data = read_file(test_path)

    assert train_users == test_users
    assert train_groups == test_groups

    return create_client_servers(seed, params, train_users, train_groups, train_data, test_data, sketcher, model_cls, num_client_servers)


def setup_client_servers(dataset, seed, params, sketcher, model_cls, use_val_set=False, num_client_servers=1, deferred_loading=False):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    eval_set = 'test' if not use_val_set else 'val'
    # train_data_dir = os.path.join('..', 'data', dataset, 'data', 'train')
    # test_data_dir = os.path.join('..', 'data', dataset, 'data', eval_set)
    train_dir = DATA_PATH + ['data', dataset, 'train']
    test_dir = DATA_PATH + ['data', dataset, eval_set]
    train_data_dir = os.path.join(*train_dir)
    test_data_dir = os.path.join(*test_dir)

    # 
    if deferred_loading:
        client_servers = []
        for train_path, test_path, num_cs in generate_data_shard(train_data_dir, test_data_dir):
            client_servers += ray.get(
                load_data_and_create_client_servers.remote(seed, params, train_path, test_path, sketcher, model_cls, num_cs))
    else:
        users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

        client_servers = create_client_servers(seed, params, users, groups, train_data, test_data, sketcher, model_cls, num_client_servers)

    return client_servers


def get_stat_writer_function(ids, groups, num_samples, args):

    def writer_fn(num_round, metrics, partition):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, partition, args.metrics_dir, '{}_{}'.format(args.metrics_name, 'stat'))

    return writer_fn


def get_sys_writer_function(args):

    def writer_fn(num_round, ids, metrics, groups, num_samples):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, 'train', args.metrics_dir, '{}_{}'.format(args.metrics_name, 'sys'))

    return writer_fn


def print_stats(
    num_round, server, num_samples, args, writer, use_val_set):
    
    train_stat_metrics = server.test_model(clients_to_test=None, set_to_use='train')
    print_metrics(train_stat_metrics, num_samples, prefix='train_')
    writer(num_round, train_stat_metrics, 'train')

    eval_set = 'test' if not use_val_set else 'val'
    test_stat_metrics = server.test_model(clients_to_test=None, set_to_use=eval_set)
    print_metrics(test_stat_metrics, num_samples, prefix='{}_'.format(eval_set))
    writer(num_round, test_stat_metrics, eval_set)


def print_metrics(metrics, weights, prefix=''):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    to_ret = None
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 90)))


if __name__ == '__main__':
    main()
