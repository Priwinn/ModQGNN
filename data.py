import os
import tensorflow as tf
import sparse
import numpy as np


def read(name):
    Gs = sparse.load_npz(name + '.npz')
    chong = np.load(name + '_chong.npy')
    return Gs, chong


def make_windows(name, size, stride):
    Gs, chong = read(name)
    T = Gs.shape[0]
    Gs = np.concatenate([Gs, np.stack([Gs[-1]] * size)])
    edges = [Gs[stride * i:stride * i + size].coords.T for i in range(int(np.ceil(T / stride)))]

    node_input = chong[:-1:stride]
    target = chong[1::stride]
    # if target.shape[0]==0:
    #     print(node_input.shape,target,len(edges))
    #     print(chong.shape,T)
    #     print([i for i in range(1,301-(T%stride),stride)])
    return edges, node_input, target


def load_dir(path, size, stride, file_range=(0,10000)):
    edges = []
    node_input = []
    target = []
    for file in os.listdir(path)[file_range[0]:file_range[1]]:
        if '.npz' in file:
            load = make_windows(os.path.join(path, file.strip('.npz')), size, stride)
            empty_indices=[]
            for i, edge in enumerate(load[0]):
                if edge.shape[0] == 0:
                    empty_indices.append(i)
            for index in sorted(empty_indices, reverse=True):
                del load[0][index]
            edges += load[0]
            node_input.append(np.delete(load[1],empty_indices,axis=0))
            target.append(np.delete(load[2],empty_indices,axis=0))
    return tf.ragged.stack(edges), np.concatenate(node_input), np.concatenate(target)


def get_ds(path, size, stride, file_range=(0,1000)):
    edges, node_input, target = load_dir(path, size, stride, file_range=file_range)
    ds = tf.data.Dataset.from_tensor_slices(((edges, node_input), target)).shuffle(65536).prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == '__main__':
    a = get_ds('random_circuits', 5, 5, 10)
