import networkx as nx
from networkx.generators.random_graphs import fast_gnp_random_graph, barabasi_albert_graph, watts_strogatz_graph
from networkx.generators.degree_seq import configuration_model
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, product
from random import shuffle


def shortest_cycle(G):
    V = len(G.nodes) + 1
    dist = np.full([V, V], np.inf)
    next = np.full([V, V], np.nan)
    for (u, v) in G.edges:
        dist[u, v] = 1
        next[u, v] = v
    for k in G.nodes:
        for i in G.nodes:
            for j in G.nodes:
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next[i][j] = next[i][k]
    s = np.argmin(np.diagonal(dist))
    if np.isnan(next[s, s]):
        return []
    u = int(next[s, s])
    path = [(s, u)]
    while u != s:
        path.append((u, next[u][s]))
        u = int(next[u][s])
    return path


def cost_step(p1, p2):
    cost = 0
    # Number of partitions
    n_part = np.max(p1) + 1
    # Find indices where the partition has changed
    changes = np.where(np.array(p1) != np.array(p2))[0]
    # Initialize cost graph
    cost_M = nx.MultiDiGraph()
    # Add nodes for each partition
    cost_M.add_nodes_from(range(n_part))
    # Add edges for each node that needs to be moved
    edges = [(p1[i], p2[i]) for i in changes]
    cost_M.add_edges_from(edges)
    # Transform to weighted graph
    cost_G = nx.DiGraph()
    cost_G.add_nodes_from(range(n_part))
    for u, v, data in cost_M.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cost_G.has_edge(u, v):
            cost_G[u][v]['weight'] += w
        else:
            cost_G.add_edge(u, v, weight=w)
    # First shortest cycle
    shortest = shortest_cycle(cost_G)
    while shortest:
        # Remove cycle
        cost += len(shortest) - 1
        for (u, v) in shortest:
            cost_G[u][v]['weight'] -= 1
            if cost_G[u][v]['weight'] == 0:
                cost_G.remove_edge(u, v)

        shortest = shortest_cycle(cost_G)
    for (u, v) in cost_G.edges:
        cost += cost_G[u][v]['weight']
    return cost


# Determine whether all interacting q-bits are in the same partition. Return inf if not, 0 else
def validate_partition(G, P):
    if isinstance(P,list):
        P=np.array(P)
    if isinstance(G,nx.Graph):
        for (u, v) in G.edges:
            if P[u] != P[v]:
                print(f'Nodes {u},{v} are not in the same partition')
                return np.inf
                break
        return 0
    else:
        edges=np.where(G!=0)
        if any(P[edges[0]]!=P[edges[1]]):
            return np.inf
        return 0


def total_cost(Gs, partitions, validate=True):
    C = 0
    for i, p in enumerate(partitions):
        if validate:
            C += validate_partition(Gs[i], p)
        if i == 0:
            continue
        C += cost_step(partitions[i - 1], p)
    return C


def grid_search_partitions(Gs, N, n):
    T = len(Gs)
    grid = set(permutations([i for i in range(N) for _ in range(n)], N * n))
    valid = [[] for _ in range(T)]
    min_cost = np.inf
    min_perm = [i for i in range(N) for _ in range(n)]
    for t, G in enumerate(Gs):
        print(f'Looking for valid partitions for time step {t}')
        for i, part in enumerate(grid):
            if i % 300 == 0:
                print(i)
            if validate_partition(G, part) == 0:
                valid[t].append(part)
    valid_size = 1
    for t in valid:
        valid_size = valid_size * len(t)
    print(f'Size of validated grid:{valid_size}')
    for i, parts in enumerate(product(*valid)):
        if i % 10000 == 0:
            print(f'Progress:{100 * i / valid_size}%')
        cost = total_cost(Gs, parts, validate=False)
        if cost < min_cost:
            min_cost = cost
            min_parts = parts
            print(f'new min_cost={min_cost:.2f}')
        if cost == 0:
            break
    return min_cost, min_parts


def roee(A,G, N, part=None):
    n_nodes = A.shape[0]
    n_per_part = int(n_nodes / N)
    if not part:
        part = [i for i in range(N) for _ in range(n_per_part)]
        shuffle(part)
    g_max=1
    #Step 7
    while g_max>0 and validate_partition(G,part)!=0:
        #Step 1
        C=[i for i in range(n_nodes)]
        index=0
        W = np.zeros([n_nodes, N])
        D = np.empty([n_nodes, N])
        P= np.stack([A[np.where(np.array(part) == i)[0]] for i in range(N)])
        for i in range(n_nodes):
            for l in range(N):
                W[i, l] = np.sum(P[l,:,i])
        for i in range(n_nodes):
            for l in range(N):
                D[i,l] = W[i,l]-W[i,part[i]]
        g=[]
        #Step 4
        while len(C)>1:
            #Step 2
            g.append([-np.inf,None,None])
            for i in C:
                for j in C:
                    g_aux=D[i,part[j]]+D[j,part[i]]-2*A[i,j]
                    if g_aux>g[index][0]:
                        g[index][1]=i
                        g[index][2]=j
                        g[index][0]=g_aux
            C.remove(g[index][1])
            if g[index][1]!=g[index][2]:
                C.remove(g[index][2])
            a=g[index][1]
            b=g[index][2]

            #Step 3
            for i in C:
                for l in range(n_part):
                    if l==part[a]:
                        if part[i]!=part[a] and part[i]!=part[b]:
                            D[i,l]=D[i,l]+A[i,b]-A[i,a]
                        if part[i]==part[b]:
                            D[i,l]=D[i,l]+2*A[i,b]-2*A[i,a]
                    elif l==part[b]:
                        if part[i]!=part[a] and part[i]!=part[b]:
                            D[i,l]=D[i,l]+A[i,a]-A[i,b]
                        if part[i]==part[a]:
                            D[i,l]=D[i,l]+2*A[i,a]-2*A[i,b]
                    else:
                        if part[i]==part[a]:
                            D[i,l]=D[i,l]+A[i,a]-A[i,b]
                        elif part[i]==part[b]:
                            D[i,l]=D[i,l]+A[i,b]-A[i,a]
            index+=1
        g_max=[]
        for i in range(len(g)):
            g_max.append(np.sum(g[:i+1][0]))
        m=np.argmax(g_max)
        g_max=g_max[m]
        print(g_max)


        for i in g[:m+1]:
            part[i[1]],part[i[2]]=part[i[2]],part[i[1]]
    return part





if __name__ == "__main__":
    n_part = 10
    n = 10
    N = 100
    k = 1/2
    Gs = [fast_gnp_random_graph(N, k / (N - 1)), fast_gnp_random_graph(N, k / (N - 1)),
          fast_gnp_random_graph(N, k / (N - 1))]
    for G in Gs:
        if len(max(nx.connected_components(G), key=len)) > n:
            raise ValueError(
                'Generated graph has too many interactions. Try reducing k or rerun for another random seed.')
    #Brute force
    #a = grid_search_partitions(Gs, n_part, n)
    #roee
    a=roee(100*nx.convert_matrix.to_numpy_array(Gs[0]),nx.convert_matrix.to_numpy_array(Gs[0]),n_part)
