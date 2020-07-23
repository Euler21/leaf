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
from utils.model_utils import read_data, split_shard_path

STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'

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
    
    # please swap the two lines below if deploying on cluster.
    #ray.init(address='auto', redis_password='5241590000000000')
    ray.init(lru_evict=True)

    # Create clients
    client_servers = setup_client_servers(args.dataset, 
                                          args.seed,
                                          model_params,
                                          ClientModel,
                                          args.use_val_set, 
                                          num_client_servers,
                                          defer_data_loading=args.defer_data_loading)

    # Create server
    server = Server(server_model, client_servers)

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
                         model_cls) 
            for cs_users, cs_groups, cs_train_data, cs_test_data in partition]
    #clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
    #return clients


def create_shard_client_servers(
        seed, params, train_data_dir, test_data_dir, model_cls):
    sharding = split_shard_path(train_data_dir, test_data_dir)

    client_server_ids = []
    for train_data_path, test_data_path in sharding:
        cs = ClientServer.options(num_cpus=10).remote(
                seed, params, [], [], [], [], model_cls,
                train_data_path, test_data_path
            )
        client_server_ids.append(cs)

    return client_server_ids



def setup_client_servers(dataset, seed, params, model_cls, use_val_set=False, num_client_servers=1, defer_data_loading=False):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    eval_set = 'test' if not use_val_set else 'val'
    train_data_dir = os.path.join('..', 'data', dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', dataset, 'data', eval_set)
    if not defer_data_loading:
        users, groups, train_data, test_data = read_data(
            train_data_dir, test_data_dir)

        client_servers = create_client_servers(
            seed, params, users, groups, train_data, test_data, model_cls,
            num_client_servers)
    else:
        # for now we are creating same number of client_servers as number of data shards
        client_servers = create_shard_client_servers(
            seed, params, train_data_dir, test_data_dir, model_cls)

    return client_servers


def get_stat_writer_function(ids, groups, num_samples, args):

    def writer_fn(num_round, metrics, partition):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, partition,
            args.metrics_dir, '{}_{}'.format(args.metrics_name, 'stat'))

    return writer_fn


def get_sys_writer_function(args):

    def writer_fn(num_round, ids, metrics, groups, num_samples):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, 'train',
            args.metrics_dir, '{}_{}'.format(args.metrics_name, 'sys'))

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
