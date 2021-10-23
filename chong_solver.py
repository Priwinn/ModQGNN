import os.path
import numpy as np
import networkx as nx
import random
import sparse
import warnings
from multiprocessing import Pool
from multiprocessing import cpu_count
random.seed(1)
warnings.filterwarnings("ignore", message="invalid value encountered in multiply")


def validate_partition(G, P):
    if isinstance(P, list):
        P = np.array(P)
    if isinstance(G, nx.Graph):
        for (u, v) in G.edges:
            if P[u] != P[v]:
                print(f'Nodes {u},{v} are not in the same partition')
                return np.inf
        return 0
    else:
        edges = np.where(G != 0)
        if any(P[edges[0]] != P[edges[1]]):
            return np.inf
        return 0


def roee(A, G, N, part=None):
    '''
    :param A: Weight matrix
    :param G: Interaction graph
    :param N: Number of partitions
    :param part: Initial partition (generates random if None)
    :return: rOEE solution
    '''
    n_nodes = A.shape[0]
    n_per_part = int(n_nodes / N)
    if part is None:
        part = [i for i in range(N) for _ in range(n_per_part)]
        random.shuffle(part)
    g_max = 1
    # Step 7
    while g_max > 0:
        if validate_partition(G, part) == 0:
            break
        # Step 1
        C = [i for i in range(n_nodes)]
        index = 0
        W = np.zeros([n_nodes, N])
        D = np.empty([n_nodes, N])
        # Precompute partitions
        P = np.stack([A[np.where(np.array(part) == i)[0]] for i in range(N)])
        for i in range(n_nodes):
            for l in range(N):
                W[i, l] = np.sum(P[l, :, i])
        for i in range(n_nodes):
            for l in range(N):
                D[i, l] = W[i, l] - W[i, part[i]]
        g = []
        # Step 4
        while len(C) > 1:
            # Step 2
            g.append([-np.inf, None, None])
            for i in C:
                for j in C:
                    g_aux = D[i, part[j]] + D[j, part[i]] - 2 * A[i, j]
                    if g_aux > g[index][0]:
                        g[index][1] = i
                        g[index][2] = j
                        g[index][0] = g_aux
            C.remove(g[index][1])
            if g[index][1] != g[index][2]:
                C.remove(g[index][2])
            a = g[index][1]
            b = g[index][2]

            # Step 3
            for i in C:
                for l in range(N):
                    if l == part[a]:
                        if part[i] != part[a] and part[i] != part[b]:
                            D[i, l] = D[i, l] + A[i, b] - A[i, a]
                            # print(D[i, l],i,l,1)
                        if part[i] == part[b]:
                            D[i, l] = D[i, l] + 2 * A[i, b] - 2 * A[i, a]
                            # print(D[i, l],i,l,2)
                    elif l == part[b]:
                        if part[i] != part[a] and part[i] != part[b]:
                            D[i, l] = D[i, l] + A[i, a] - A[i, b]
                            # print(D[i, l],i,l,3)
                        if part[i] == part[a]:
                            D[i, l] = D[i, l] + 2 * A[i, a] - 2 * A[i, b]
                            # print(D[i, l],i,l,4)
                    else:
                        if part[i] == part[a]:
                            D[i, l] = D[i, l] + A[i, a] - A[i, b]
                            # print(D[i, l],i,l,5)
                        elif part[i] == part[b]:
                            D[i, l] = D[i, l] + A[i, b] - A[i, a]
                            # print(D[i, l],i,l,6)
            index += 1
        g_max = np.cumsum([i[0] for i in g])
        m = np.argmax(g_max)
        g_max = g_max[m]
        for i in g[:m + 1]:
            part[i[1]], part[i[2]] = part[i[2]], part[i[1]]
    if validate_partition(G, part) != 0:
        print('Valid partition not found')
    return part


def roee_vect(A, G, N, part=None):
    '''
    :param A: Weight matrix
    :param G: Interaction graph
    :param N: Number of partitions
    :param part: Initial partition (generates random if None)
    :return: rOEE solution
    '''
    if isinstance(A,sparse.COO):
        A=A.todense()
    n_nodes = A.shape[0]
    n_per_part = int(n_nodes / N)
    if part is None:
        part = [i for i in range(N) for _ in range(n_per_part)]
        random.shuffle(part)
    g_max = 1
    # Step 7
    while g_max > 0:
        if validate_partition(G, part) == 0:
            break
        # Step 1
        C = np.arange(n_nodes)
        index = 0
        # Precompute partitions
        P = np.stack([A[np.where(np.array(part) == i)[0]] for i in range(N)], axis=-1)
        W = np.sum(P, axis=0)
        D = W - np.expand_dims(W[range(n_nodes), part].T,-1)
        g = []
        # Step 4
        while C.shape[0] > 1:
            # Step 2
            g.append([-np.inf, None, None])

            g_aux = D[np.meshgrid(C, part[C])] + D[np.meshgrid(C,part[C])].T - 2 * A[np.meshgrid(C, C)]
            g[index][1], g[index][2] = np.unravel_index(np.argmax(g_aux, axis=None), g_aux.shape)
            g[index][0] = g_aux[g[index][1], g[index][2]]
            a = C[g[index][1]]
            b = C[g[index][2]]
            if g[index][1] != g[index][2]:
                C=np.delete(C,g[index][1:])
            else:
                C=np.delete(C,g[index][1])

            g[index][1]=a
            g[index][2]=b
            # Step 3
            indices = C[np.where((part[C] != part[a]) & (part[C] != part[b]))[0]]
            non_ab = np.where((np.arange(N) != part[a]) & (np.arange(N) != part[b]))
            D[indices, part[a]] = D[indices, part[a]] + A[indices, b] - A[indices, a]
            D[indices, part[b]] = D[indices, part[b]] + A[indices, a] - A[indices, b]
            indices = C[np.where(part[C] == part[b])[0]]
            D[indices, part[a]] = D[indices, part[a]] + 2 * A[indices, b] - 2 * A[indices, a]
            D[np.meshgrid(indices, non_ab)] = D[np.meshgrid(indices, non_ab)] + A[indices, b] - A[indices, a]
            indices = C[np.where(part[C] == part[a])[0]]
            D[indices, part[b]] = D[indices, part[b]] + 2 * A[indices, a] - 2 * A[indices, b]
            D[np.meshgrid(indices, non_ab)] = D[np.meshgrid(indices, non_ab)] + A[indices, a] - A[indices, b]
            index += 1
        g_max = np.cumsum([i[0] for i in g])
        m = np.argmax(g_max)
        g_max = g_max[m]
        for i in g[:m + 1]:
            part[i[1]], part[i[2]] = part[i[2]], part[i[1]]
    if validate_partition(G, part) != 0:
        print('Valid partition not found')
    return part


def lookahead(Gs, func='exp', sigma=1, inf=2 ** 16):
    W = inf * Gs[0]
    W.fill_value = np.float64(0.0)
    n = 1
    if func == 'exp':
        pass
    else:
        raise NotImplementedError('For the time being only exp is implemented')
    for G in Gs:
        aux = G * 2 ** (-n / sigma)
        aux.fill_value = np.float64(0.0)
        W = W + aux
        n+=1
    return W


def sequence_solver(file, N=10, path='random_circuits', out_path='random_circuits'):
    if '.npz' in file:
        Gs = sparse.load_npz(os.path.join(path, file))
    else:
        return
    save_path = os.path.join(out_path, file.strip('.npz') + '_chong.npy')
    if os.path.exists(save_path):
        print(f'Solution for {file} exists, skipping')
        return
    print(f'Solving for {file}')
    n_nodes = Gs.shape[1]
    n_per_part = int(n_nodes / N)
    # Init Partitions, add extra 1 for initial random assignment
    Ps = np.zeros((Gs.shape[0] + 1, Gs.shape[1]), dtype=int)
    part = [i for i in range(N) for _ in range(n_per_part)]
    random.shuffle(part)
    Ps[0] = part
    T = Gs.shape[0]
    for i, G in enumerate(Gs):
        L = lookahead(Gs[i:])
        Ps[i + 1] = roee_vect(L, G, N, Ps[i].copy())

    np.save(save_path, Ps)
    print(f'Solution written to {save_path}')


if __name__ == '__main__':
    path = 'random_circuits_remove_empty'
    N = 10
    out_path = 'random_circuits_remove_empty'
    for file in os.listdir(path):
        sequence_solver(file,path=path,out_path=out_path)

