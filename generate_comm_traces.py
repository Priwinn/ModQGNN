import math
import os
import numpy
import networkx as nx
import matplotlib.pyplot as plt

def sequence_tracer(file, N=10, time_windows=10, slices_per_window=None, path='random_circuits', out_path='random_circuits'):
    if '11-28-2021_170231rollback_True_relax_2_cost.npy' in file and 'qft' in file:
        Ps = numpy.load(os.path.join(path, file))
    else:
        return
    save_path = os.path.join(out_path, file.strip('.npy') + '_comm_trace.npy')
    if os.path.exists(save_path):
        print(f'Solution for {file} exists, skipping')
        return
    print(f'Solving for {file}')
    n_nodes = Ps.shape[1] # Total number of qubits
    n_per_part = int(n_nodes / N) # Number of qubits per core
    time_slices = Ps.shape[0] # Number of ime slices in the compiled algorithm
    if slices_per_window is None:
        slices_per_window = math.ceil(time_slices/time_windows)
    else:
        time_windows = math.ceil(time_slices/slices_per_window)
    graphs = []
    for i in range(time_windows):
        G = nx.DiGraph()
        G.add_nodes_from(range(N))
        graphs.append(G)
    full_graph = nx. DiGraph()
    full_graph.add_nodes_from(range(N))

    max_weight = 0
    for i, P in enumerate(Ps):
        if i > 0:
            G = graphs[i//slices_per_window]
            part = Ps[i]
            prev_part = Ps[i-1]
            for q,_ in enumerate(part):
                if part[q] != prev_part[q]:
                    print(f'{i},{q},{prev_part[q]},{part[q]}') # añadir traza comunicación
                    if G.has_edge(prev_part[q], part[q]):
                        # we added this one before, just increase the weight by one
                        G[prev_part[q]][part[q]]['weight'] += 0.1
                        if G[prev_part[q]][part[q]]['weight'] > max_weight:
                            max_weight = G[prev_part[q]][part[q]]['weight']
                    else:
                        # new edge. add with weight=1
                        G.add_edge(prev_part[q], part[q], weight = 0.1)
                    # Now the full graph
                    if full_graph.has_edge(prev_part[q], part[q]):
                        # we added this one before, just increase the weight by one
                        full_graph[prev_part[q]][part[q]]['weight'] += 0.01
                    else:
                        # new edge. add with weight=1
                        full_graph.add_edge(prev_part[q], part[q], weight = 0.01)

    pos = nx.spring_layout(full_graph)
    edge_widths = [w for (*edge, w) in full_graph.edges.data('weight')]
    nx.draw(full_graph, pos, width=edge_widths, with_labels=True, connectionstyle='arc3, rad=.15')
    plt.show()
    for G in graphs:
        # pos = nx.spring_layout(G) # Uncomment this line if you want to change the graph layout dynamically
        edge_widths = [w for (*edge, w) in G.edges.data('weight')]
        nx.draw(G, pos, width=edge_widths, with_labels=True, connectionstyle='arc3, rad=.15')
        plt.show()
        print("Done")
        print(nx.pagerank(G))
        for n in G.nodes():
            print(f'in_degree ({n}) = {G.in_degree(n)}')
        for n in G.nodes():
            print(f'out_degree ({n}) = {G.out_degree(n)}')
        for e in G.edges(data = True):
            w = e[2].get('weight', 1)
            print(f'weight ({e[0:2]}) = {w}')
        print(f'max_weight = {max_weight}')
    
    numpy.save(save_path, graphs)
    print(f'Solution written to {save_path}')


if __name__=='__main__':
    # path = 'random_circuits_remove_empty'
    path = 'real_circuits/qft_circuits'
    N = 10 # Number of cores = number of partitions
    tw = 10 # Number of time windows for the graph plotting
    slices_per_window = None # Set a value different from None to contrl directly time duration of the time window
    # out_path = 'random_circuits_remove_empty'
    out_path = 'real_circuits/qft_circuits'
    file='qft_q100_111-28-2021_170231rollback_True_relax_2_cost.npy'
    sequence_tracer(file,N=N,time_windows=tw,slices_per_window=slices_per_window,path=path,out_path=out_path)
    # for file in os.listdir(path):
    #     sequence_tracer(file,N=N,time_windows=tw,slices_per_window=slices_per_window,path=path,out_path=out_path)


