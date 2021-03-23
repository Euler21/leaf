from abc import ABC, abstractmethod
import tensorflow.compat.v1 as tf
from sklearn.utils.extmath import randomized_svd
import numpy as np
from sklearn.decomposition import TruncatedSVD
# from flops_utils import randomized_svd_flops, svd_flops
# import flops_utils
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

class RandomizedSVD(Sketcher):
    def compress(self, updates, params=None):
        compressed_updates = []
        rank = params["rank"]
        if rank == -1:
            rank = 2048
        flops = 0
        m = 3136
        n = 2048
        num_iterations= 0 # change this to change number of iterations
        for i in range(len(updates)):
            update = updates[i]
            if update.shape == (3136, 2048):
                u, s, v = randomized_svd(updates[i], n_components=rank, n_iter=num_iterations, random_state=None)
                u = u[:, :rank]
                s = s[:rank]
                v = v[:rank, :]
                c = [u,s,v]
                compressed_updates += [c]
            else:
                compressed_updates += [update]
            flops += randomized_svd_flops(m, n, rank)#14 * m * (n**2) + (8 * (m ** 3))
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
#                 sum_squared = sum([si**2 for si in s])
#                 percent_variance = [(si**2.0) / sum_squared  for si in s]
#                 print("PERCENTAGE OF VARIANCES: " + str(percent_variance))
#                 print("HIGHEST PERCENTAGE OF VARIANCES: " + str(percent_variance[0]))
#                 normalized_s = s / s[0]
#                 for i in range(len(normalized_s)):
#                     print("singular value " + str(i) + ": " + str(normalized_s[i]))
                u = u[:, :rank]
                s = s[:rank]
                v = v[:rank, :]
                c = [u,s,v]
                compressed_updates += [c]
            else:
                compressed_updates += [update]
            flops += svd_flops(m, n, rank)#14 * m * (n**2) + (8 * (m ** 3))
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
    
def randomized_svd_flops(m, n, r):
    r2 = np.power(r, 2)
    r3 = np.power(r, 3)
    list_of_terms = [2 * m * n*r, 
                     2 * m * (r**2) - ((2.0/3.0) * (r**3)),
                    4*(r**2)*m + 2*(r**3) - 2*(r**2),
                    14*n*(r**2) + 8*(r**3),
                    2*m*(r**2)]
    return int(sum(list_of_terms))

def svd_flops(m, n, r):
    n2 = n**2
    n3 = n**3
    k = 5
    list_of_terms = [4*m*n2 - ((4.0/3.0) * n3),
                     k*r*n,
                     4*r*m*n - 2*r*n2 + 2*r*n,
                     2*r*n*(n+1)
                    ]
    return int(sum(list_of_terms))