from abc import ABC, abstractmethod
import tensorflow.compat.v1 as tf
import numpy as np
from sklearn.decomposition import TruncatedSVD


class Sketcher(ABC):
    @abstractmethod
    def compress(self, updates, params=None):
        pass
    @abstractmethod
    def uncompress(self, compressed_updates):
        pass

class VoidSketcher(Sketcher):
    def compress(self, updates, params=None):
        return updates
    def uncompress(self, compressed_updates):
        return compressed_updates

class HalkoSVD(Sketcher):
    def compress(self, updates, params=None):
        pass
    def uncompress(self, compressed_updates):
        pass
class SVD(Sketcher):
    def compress(self, updates, params=None):
        compressed_updates = []
        rank = params["rank"]
        if rank == -1:
            rank = 2048
        for i in range(len(updates)):
            update = updates[i]
            if update.shape == (3136, 2048):
                u, s, v = np.linalg.svd(updates[i], full_matrices=False)
                u = u[:, :rank]
                s = s[:rank]
                v = v[:rank, :]
                c = [u,s,v]
                compressed_updates += [c]
            else:
                compressed_updates += [update]

        return compressed_updates
    
    
    def uncompress(self, compressed_updates):
        uncompressed_list = []
        for i in range(len(compressed_updates)):
            compressed = compressed_updates[i]
            if type(compressed) == list:
                u, s, v = compressed[0], compressed[1], compressed[2]
                uncompressed = np.dot(u * s, v)
                uncompressed_list += [uncompressed]
            else:
                uncompressed_list += [compressed]
        return uncompressed_list
