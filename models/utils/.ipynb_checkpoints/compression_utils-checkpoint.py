from abc import ABC, abstractmethod
import tensorflow.compat.v1 as tf
import numpy as np

class Sketcher(ABC):
    @abstractmethod
    def compress(self, updates):
        pass
    @abstractmethod
    def uncompress(self, compressed_updates):
        pass

class VoidSketcher(Sketcher):
    def compress(self, updates):
        return updates
    def uncompress(self, compressed_updates):
        return compressed_updates

class SVD(Sketcher):
    def compress(self, updates):
        compressed_updates = []
        for i in range(len(updates)):
#             print("++++++++++++++++++++++ SVD COMPRESS WAS CALLED: " + str(updates[i].shape))
            update = updates[i]
            if len(update.shape) > 1:
                u, s, v = np.linalg.svd(updates[i], full_matrices=False)
#                 print("++++++++++++++++++++++ SVD COMPRESS WAS CALLED. u:" + str(u.shape) + " s:" + str(s.shape) + " v:" + str(v.shape))
                c = [u,s,v]
                compressed_updates += [c]
            else:
                compressed_updates += [update]
#             print("++++++++++++++++++++++ SVD COMPRESS WAS CALLED. u:" + str(u.shape) + " s:" + str(s.shape) + " v:" + str(v.shape))
        return compressed_updates
    
    
    def uncompress(self, compressed_updates):
#         print("++++++++++++++++++++++ SVD UNCOMPRESS WAS CALLED: " + str(type(compressed_updates[0])) + " len:" + str(len(compressed_updates[1])))
        compressed_update_list = compressed_updates[1]
        uncompressed_list = []
        for i in range(len(compressed_update_list)):
            compressed = compressed_update_list[i]
#             print("++++++++++++++++++++++ SVD UNCOMPRESS WAS CALLED: " + str(type(compressed)))
            if type(compressed) == list:
                u, s, v = compressed[0], compressed[1], compressed[2]
#                 print("++++++++++++++++++++++ SVD UNCOMPRESS WAS CALLED. u:" + str(u.shape) + " s:" + str(s.shape) + " v:" + str(v.shape))
                uncompressed = np.matmul(u * s[..., None, :], v)
                uncompressed_list += [uncompressed]
            else:
                uncompressed_list += [compressed]
        return [compressed_updates[0], uncompressed_list]
