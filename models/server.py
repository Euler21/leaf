import numpy as np
import ray

from client_server import ClientServer

class Server:
    
    def __init__(self, server_model, client_servers):
        self.server_model = server_model
        self.client_servers = client_servers
        self.model = server_model.get_params()
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
        samples_futures = []
        for cs in self.client_servers:
            samples_future = cs.select_clients.remote(my_round, num_clients)
            samples_futures.append(samples_future)

        return [p for future in samples_futures for p in ray.get(future)]

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
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """
        if clients is not None:
            raise NotImplementedError("Client selection not yet implemented")
        sys_metrics = {}
        metrics_updates_futures = []
        for cs in self.client_servers:
            metrics_updates_future = cs.train_model.remote(
                num_epochs, batch_size, minibatch)
            metrics_updates_futures.append(metrics_updates_future)

        for future in metrics_updates_futures:
            # if we don't want this to block sequentially, can switch to ray.wait
            metrics, updates = ray.get(future)
            sys_metrics.update(metrics)
            self.updates += updates

        return sys_metrics

    def update_model(self):
        total_weight = 0.
        base = [0] * len(self.updates[0][1])
        for (client_samples, client_model) in self.updates:
            total_weight += client_samples
            for i, v in enumerate(client_model):
                base[i] += (client_samples * v.astype(np.float64))
        averaged_soln = [v / total_weight for v in base]

        self.model = averaged_soln
        self.updates = []
        for cs in self.client_servers:
            cs.update_model.remote(self.model)

    def test_model(self, clients_to_test=None, set_to_use='test'):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """
        metrics = {}

        if clients_to_test is not None:
            raise NotImplementedError("Client selection not yet implemented")

        metrics_futures = []
        for cs in self.client_servers:
            metrics_future = cs.test_model.remote(
                clients_to_test=None, set_to_use=set_to_use)
            metrics_futures.append(metrics_future)
        for future in metrics_futures:
            metrics.update(ray.get(future))
        
        return metrics

    def get_clients_info(self, all_clients=False, clients=None):
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is not None:
            raise NotImplementedError("Client selection not yet implemented")

        ids = []
        groups = {}
        num_samples = {}
        client_info_futures = []
        for cs in self.client_servers:
            client_info_future = cs.get_clients_info.remote(
                all_clients=all_clients, clients=clients)
            client_info_futures.append(client_info_future)

        for future in client_info_futures:
            # if we don't want this to block sequentially, can switch to ray.wait
            cs_ids, cs_groups, cs_num_samples = ray.get(future)
            ids += cs_ids
            groups.update(cs_groups)
            num_samples.update(cs_num_samples)
        return ids, groups, num_samples

    def save_model(self, path):
        """Saves the server model on checkpoints/dataset/model.ckpt."""
        # Save server model
        self.server_model.set_params(self.model)
        model_sess =  self.server_model.sess
        return self.server_model.saver.save(model_sess, path)

    def close_model(self):
        for cs in self.client_servers:
            cs.close_model.remote()
        self.server_model.close()
