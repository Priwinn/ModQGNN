import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment as sp_lsa
from numpy import vectorize
import os
import json
from data import make_windows, read
from chong_solver import roee_vect, lookahead
import networkx as nx
sp_lsa = vectorize(sp_lsa, signature='(n,n)->(n),(n)')
import time

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


def load(load_path):
    with open(os.path.join(load_path, 'config.json')) as file:
        config = json.load(file)
    model = tf.keras.models.load_model(load_path, compile=False)
    decayed_lr = tf.keras.optimizers.schedules.ExponentialDecay(config['initial_lr'],
                                                                config['decay_steps'],
                                                                config['decay_rate'])
    model.compile(loss=tf.keras.metrics.sparse_categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(decayed_lr),
                  metrics=['accuracy', lsa_acc],
                  # run_eagerly=True #This is to debug, disable if training
                  )
    return model, config




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


def total_cost(partitions, Gs=None):
    C = 0
    for i, p in enumerate(partitions):
        if Gs:
            C += validate_partition(Gs[i], p)
        if i == 0:
            continue
        C += cost_step(partitions[i - 1], p)
    return C


def evaluate_sequences(model, config, path,file_range=(600,800),target_suffix='_chong.npy',relax=0):
    files = [file for file in os.listdir(path) if '.npz' in file][file_range[0]:file_range[1]]


    save_path_target=os.path.join(path,f'cost{target_suffix}')
    save_path_model=os.path.join(path,model.name+f'relax_{relax}_cost.npy')
    if os.path.exists(save_path_target):
        result_chong=np.load(save_path_target)
    else:
        cost_chong=[]
        time_chong=[]
        it_chong=[]
    if os.path.exists(save_path_model):
        result_gnn=np.load(save_path_model)
    else:
        cost_model=[]
        time_model=[]
        it_model=[]
        total_time_model=[]
    for n,file in enumerate(files):
        if not os.path.exists(save_path_model):
            time_model_aux=[]
            it_model_aux=[]
            total_time_model_aux=[]
            print(f'File {n}/{len(files)}')
            Gs, chong = read(os.path.join(path, file.strip('.npz')))
            L=lookahead(Gs)
            node_input, edges, _ = make_windows(os.path.join(path, file.strip('.npz')), config['window_size'], 1)
            gnn_parts=np.empty([node_input.shape[0],config['n_nodes']],dtype=int)
            start_gnn=time.time()
            logits = model((np.expand_dims(node_input[0], axis=0), tf.ragged.constant(np.expand_dims(edges[0], axis=0))))
            assignments = tf.squeeze(tf.cast(lsa(logits, n_per_part=config['n_per_part']),tf.int32)).numpy()
            start_gnn_roee=time.time()
            assignments,it = roee_vect(L, Gs[0], config['n_parts'], assignments,return_n_it=True)
            end_gnn_roee=time.time()
            time_model_aux.append(end_gnn_roee-start_gnn_roee)
            total_time_model_aux.append(end_gnn_roee-start_gnn)
            it_model_aux.append(it)
            gnn_parts[0] = assignments.copy()
        if not os.path.exists(save_path_target):
            time_chong_aux=[]
            it_chong_aux=[]
            L=lookahead(Gs)
            start_roee=time.time()
            fgp,it = roee_vect(L, Gs[0], config['n_parts'], node_input[0],return_n_it=True)
            time_chong_aux.append(time.time()-start_roee)
            it_chong_aux.append(it)
        for i in range(1,node_input.shape[0]):
            if not os.path.exists(save_path_model):
                L=lookahead(Gs[i:])
                start_gnn=time.time()
                logits = model((np.expand_dims(gnn_parts[i-1], axis=0), tf.ragged.constant(np.expand_dims(edges[i], axis=0))))
                assignments = tf.squeeze(tf.cast(lsa(logits, n_per_part=config['n_per_part']),tf.int32)).numpy()
                start_gnn_roee=time.time()
                assignments,it = roee_vect(L, Gs[i], config['n_parts'], assignments,return_n_it=True)
                end_gnn_roee=time.time()
                time_model_aux.append(end_gnn_roee-start_gnn_roee)
                total_time_model_aux.append(end_gnn_roee-start_gnn)
                it_model_aux.append(it)
                gnn_parts[i] = assignments.copy()
            if not os.path.exists(save_path_target):
                L=lookahead(Gs[i:])
                start_roee=time.time()
                fgp,it=roee_vect(L, Gs[i], config['n_parts'], fgp,return_n_it=True)
                time_chong_aux.append(time.time()-start_roee)
                it_chong_aux.append(it)
        if not os.path.exists(save_path_model):
            cost_model.append(total_cost(np.concatenate([np.expand_dims(node_input[0],axis=0),gnn_parts],axis=0)))
            time_model.append(np.mean(time_model_aux))
            it_model.append(np.mean(it_model_aux))
            total_time_model.append(np.mean(total_time_model_aux))
        if not os.path.exists(save_path_target):
            cost_chong.append(total_cost(chong))
            time_chong.append(np.mean(time_chong_aux))
            it_chong.append(np.mean(it_chong_aux))
    if not os.path.exists(save_path_model):
        result_gnn=np.array([cost_model,it_model,time_model,total_time_model])
        np.save(save_path_model,result_gnn)
    if not os.path.exists(save_path_target):
        result_chong=np.array([cost_chong,it_chong,time_chong])
        np.save(save_path_target,result_chong)
    return result_chong,result_gnn

def message_ffn(hidden, activation, dropout, node_state_dim):
    model = tf.keras.Sequential([])
    for h, d in zip(hidden, dropout):
        model.add(tf.keras.layers.Dense(h, activation=activation))
        model.add(tf.keras.layers.Dropout(d))
    model.add(tf.keras.layers.Dense(node_state_dim, activation=activation))
    return model


def readout_ffn(hidden, activation, dropout, n_parts):
    model = tf.keras.Sequential([])
    for h, d in zip(hidden, dropout):
        model.add(tf.keras.layers.Dense(h, activation=activation))
        model.add(tf.keras.layers.Dropout(d))
    model.add(tf.keras.layers.Dense(n_parts, activation=tf.nn.softmax))
    return model


def _lsa(y_pred, n_per_part=10):
    """
    Solve linear sum assignment minimization problem
    :y_pred: logits for each partition
    :n_per_part: number or slots in each partition
    """
    # Negative log_likelihood
    logs = -tf.math.log(y_pred)
    # Repeats for each slot in each partition
    logs = tf.repeat(logs, axis=-1, repeats=n_per_part)
    # Solve linear sum assignment minimization problem
    assignments = tf.cast(tf.floor(sp_lsa(logs.numpy())[1] / n_per_part),
                          tf.float32)
    return assignments


def lsa(y_pred, n_per_part=10):
    """
    tf wrapper for _lsa
    """

    return tf.py_function(lambda x: _lsa(x, n_per_part=n_per_part), [y_pred], tf.float32)


def lsa_acc(y_true, y_pred):
    """
    Accuracy metric for lsa assignments
    """
    assignments = lsa(y_pred)
    return tf.reduce_mean(tf.cast(y_true == assignments, tf.float32))


def _invalid_int(inputs):
    src, dst = inputs
    return tf.reduce_sum(tf.cast(src.to_tensor() != dst.to_tensor(), tf.float32))


def batched_invalid_int(src, dst):
    return tf.map_fn(_invalid_int, (src, dst), fn_output_signature=tf.float32)


def graph_batch(inputs, loader='window_chong', temp_mode='center', n_nodes=100, window_size=1):
    """
    Input preprocessing and batching
    :input: (node_input,edges)
    """
    n_nodes_total = n_nodes * window_size
    if loader == 'lookahead_chong':
        node_input = inputs[0]
        batch_edges = tf.cast(inputs[1][..., 1:], tf.int32)
        batch_size = tf.shape(node_input)[0]
        offsets = tf.reshape(tf.range(batch_size, dtype=tf.int32), [batch_size, 1, 1])
        node_indices = batch_edges[..., 0:1] + offsets * n_nodes
        neighbour_indices = batch_edges[..., 1:] + offsets * n_nodes
        node_indices = node_indices.merge_dims(0, 1)
        neighbour_indices = neighbour_indices.merge_dims(0, 1)
        node_indices = tf.squeeze(node_indices, axis=-1)
        neighbour_indices = tf.squeeze(neighbour_indices, axis=-1)
        lookahead_weights = inputs[1][..., 0:1].merge_dims(0, 1).to_tensor()
        return node_input, node_indices, neighbour_indices, lookahead_weights
    else:
        node_input = inputs[0]
        batch_edges = tf.cast(inputs[1], tf.int32)
        batch_size = tf.shape(node_input)[0]
        offsets = tf.reshape(tf.range(batch_size, dtype=tf.int32), [batch_size, 1, 1])
        node_indices = batch_edges[..., 0:1] * n_nodes + \
                       batch_edges[..., 1:2] + offsets * n_nodes_total  # Temporal + Batch offsets
    neighbour_indices = batch_edges[..., 0:1] * n_nodes + \
                        batch_edges[..., 2:] + offsets * n_nodes_total
    node_indices = node_indices.merge_dims(0, 1)
    neighbour_indices = neighbour_indices.merge_dims(0, 1)
    node_indices = tf.squeeze(node_indices, axis=-1)
    neighbour_indices = tf.squeeze(neighbour_indices, axis=-1)

    if temp_mode == 'double':
        temp_edges = tf.constant([[i, i + n_nodes] for i in range(n_nodes_total - n_nodes)],
                                 dtype=tf.int32)

    if temp_mode == 'cycle':
        temp_edges_first_to_last = tf.constant(
            [[i, i + n_nodes_total - n_nodes] for i in range(n_nodes)], dtype=tf.int32)
        temp_edges_backward = tf.constant(
            [[i, i + n_nodes] for i in range(n_nodes_total - n_nodes)], dtype=tf.int32)
        temp_edges = tf.concat([temp_edges_backward, temp_edges_first_to_last],
                               axis=0)

    if temp_mode == 'center':
        temp_edges = tf.constant(
            [[i, i + j * n_nodes] for i in range(n_nodes) for j in range(window_size)],
            dtype=tf.int32)

    if temp_mode:
        temp_edges = tf.repeat(tf.expand_dims(temp_edges, axis=0), batch_size,
                               axis=0) + offsets * n_nodes_total
        temp_edges = tf.reshape(temp_edges, [-1, 2])
        temp_node_indices = tf.squeeze(temp_edges[..., 0:1])
        temp_neighbour_indices = tf.squeeze(temp_edges[..., 1:2])
        return node_input, node_indices, neighbour_indices, \
               temp_node_indices, temp_neighbour_indices
    else:
        return node_input, node_indices, neighbour_indices, \
               None, None
