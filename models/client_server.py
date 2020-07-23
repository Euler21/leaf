from collections import defaultdict

import numpy as np
import ray

from client import Client
from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY
from utils.model_utils import read_data_file

@ray.remote
class ClientServer:
    
    def __init__(
            self, seed, params, users, groups, train_data, test_data, model_cls, train_data_path=None, test_data_path=None):
        self.client_model = model_cls(seed, *params)

        if train_data_path:
            train_data = defaultdict(lambda : None)
            test_data = defaultdict(lambda : None)
            train_groups, test_groups = [], []
            train_data, train_groups = read_data_file(train_data_path, [], train_groups, train_data)
            test_datai, test_groups = read_data_file(test_data_path, [], test_groups, test_data)
            users = list(sorted(train_data.keys()))
            groups = train_groups if len(train_groups) != 0 else [[] for _ in users]

        self.clients = [
            Client(u, g, train_data[u], test_data[u], self.client_model) 
            for u, g in zip(users, groups)]
        self.model = self.client_model.get_params()
        self.selected_clients = []
        self.updates = []

    def select_clients(self, my_round, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.
        
        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        possible_clients = self.clients
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None):
        """Trains self.model on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            metrics: a dict of metrics on communication and compute per client
            updates: a list of the updates for each client on this server
        """
        if clients is not None:
            raise NotImplementedError("Client selection not yet implemented")
        clients = self.selected_clients
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0} for c in clients}
        for c in clients:
            c.model.set_params(self.model)
            comp, num_samples, update = c.train(num_epochs, batch_size, minibatch)

            sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
            sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
            sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp

            self.updates.append((num_samples, update))

        return sys_metrics, self.updates

    def update_model(self, new_model):
        self.model = new_model
        self.updates = []

    def test_model(self, clients_to_test, set_to_use='test'):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """
        metrics = {}

        if clients_to_test is not None:
            raise NotImplementedError("Client selection not yet implemented")
        clients_to_test = self.selected_clients

        for client in clients_to_test:
            client.model.set_params(self.model)
            c_metrics = client.test(set_to_use)
            metrics[client.id] = c_metrics
        
        return metrics

    def get_clients_info(self, all_clients=False, clients=None):
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if all_clients:
            clients = self.clients
        elif clients is not None:
            raise NotImplementedError("Client selection not yet implemented")
        else:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, groups, num_samples

    def save_model(self, path):
        """Saves the server model on checkpoints/dataset/model.ckpt."""
        # Save server model
        self.client_model.set_params(self.model)
        model_sess =  self.client_model.sess
        return self.client_model.saver.save(model_sess, path)

    def close_model(self):
        self.client_model.close()
