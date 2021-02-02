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
        flops = 0
        m = 3136
        n = 2048
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
            flops += 14 * m * (n**2) + (8 * (m ** 3))

        return (compressed_updates, flops)
    
    
    def uncompress(self, compressed_updates):
        uncompressed_list = []
        flops = 0
        m = 3136
        n = 2048
        for i in range(len(compressed_updates)):
            compressed = compressed_updates[i]
            if type(compressed) == list:
                
                u, s, v = compressed[0], compressed[1], compressed[2]
                r = u.shape[1]

                uncompressed = np.dot(u * s, v)
                uncompressed_list += [uncompressed]
                flops += ((2 * r  - 1) * m * r) + ((2 * r - 1) * m * n)

            else:
                uncompressed_list += [compressed]
        return (uncompressed_list, flops)
