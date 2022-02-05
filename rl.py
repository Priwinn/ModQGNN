import argparse
from pathlib import Path

import networkx as nx

import torch
import torch.nn as nn

from torch_geometric.data import Data, DataLoader, Batch, Dataset
from torch_geometric.nn import GATConv, GlobalAttention,GATv2Conv
from torch_geometric.utils import to_networkx, degree

import numpy as np

import scipy
from scipy.sparse import coo_matrix
from scipy.io import mmread
from scipy.spatial import Delaunay

import math
import copy
import timeit
import os
from itertools import combinations
import sparse


# Baker lookahead function
def lookahead(Gs, func='exp', sigma=1, inf=1):
    W = inf * Gs[0]
    W.fill_value = np.float64(0.0)
    if func == 'exp':
        pass
    else:
        raise NotImplementedError('For the time being only exp is implemented')
    L = np.arange(1, Gs.shape[0])[:, np.newaxis, np.newaxis]
    L = Gs[1:] * 2 ** (-L / sigma)
    L.fill_value = np.single(0.0)
    return np.sum(L, axis=0) + W


# Calc lookaheads
def torch_lookaheads(Gs, inf=1, normalize=True):
    edges = []
    edge_attr = []
    t = []
    for i in range(Gs.shape[0]):
        L = lookahead(Gs[i:], inf=inf)
        # Normalize to (0,1) (almost)
        if normalize:
            L = L / inf

        edges_aux = L.coords
        edges.append(edges_aux[:, np.where(edges_aux[0, :] < edges_aux[1, :])[0]])
        weights = np.expand_dims(L.data, axis=-1)[np.where(edges_aux[0, :] < edges_aux[1, :])[0]]
        edge_attr.append(weights)
        t.append(np.ones_like(weights) * i)
    return torch.tensor(np.concatenate(edge_attr), dtype=torch.float), \
           torch.tensor(np.concatenate(edges, axis=1), dtype=torch.long), \
           torch.tensor(np.concatenate(t), dtype=torch.long)


# Check if all interactions are satisfied

def validate_partition(edges, P):
    if any(P[edges[0]] != P[edges[1]]):
        return np.inf
    return 0


# Change the partition of the selected qbits

def change_pair(state, pair):
    q1, q2 = pair
    state.x[q1], state.x[q2] = state.x[q2], state.x[q1]
    return state


# Build a pytorch geometric graph with features [1,0] form a networkx
# graph. Then it turns the feature of one of the vertices with minimum
# degree into [0,1]


def torch_from_graph(Gs):
    edge_attr, edges, t = torch_lookaheads(Gs)
    graph_torch = Data(edge_index=edges, edge_attr=edge_attr, t=t)
    return graph_torch


# Training dataset from npz sparse files
def training_dataset_from_files(path, out=None, num_files=100):
    dataset = []
    files = [file for file in os.listdir(path) if '.npz' in file][:num_files]
    for i, file in enumerate(files):

        if out:
            pt_path = os.path.join(out, f"{file.strip('.npz')}.pt")
            if os.path.exists(pt_path):
                Gs = torch.load(pt_path)
            else:
                Gs = sparse.load_npz(os.path.join(path, file))
                Gs = torch_from_graph(Gs)
                torch.save(Gs, os.path.join(out, f"{file.strip('.npz')}.pt"))
        else:
            Gs = sparse.load_npz(os.path.join(path, file))
            Gs = torch_from_graph(Gs)
        dataset.append(Gs)
        print(f"Loaded {i}/{len(files)}")

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return loader


def training_dataset_random(n, n_min=50, n_max=100):
    dataset = []
    for i in range(n):
        num_nodes = np.random.choice(np.arange(n_min, n_max + 1, 2))
        points = np.random.random_sample((num_nodes, 2))
        g = graph_delaunay_from_points(points)
        g_t = torch_from_graph(g)
        dataset.append(g_t)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return loader


def one_hot(n, N):
    h = torch.zeros(N)
    h[n] = 1
    return h


def one_hot_to_sparse(h):
    s = h.cpu().numpy()
    s = np.where(s == 1)[1]
    return s


# DRL training loop

def training_loop(
        model,
        training_dataset,
        gamma,
        time_to_sample,
        coeff,
        optimizer,
        print_loss,
        device
):
    i = 0
    N = 10
    # Here start the main loop for training
    for graph in training_dataset:
        num_nodes = 100
        start_part = torch.stack([one_hot(j, N) for i in range(10) for j in range(N)])
        if i % print_loss == 0 and i > 0:
            print('Graph:', i, '  reward:', rew_partial, 'actor_loss:',torch.mean(actor_loss),'critic_loss:',torch.mean(critic_loss))
        Gs = graph
        Gs.x = start_part
        len_episode = 1  # length of an episode

        rew_partial = 0

        rews, vals, logprobs = [], [], []
        t_edge_idx = Gs.t == 0
        slice = Data(x=Gs.x, edge_index=Gs.edge_index[:, torch.squeeze(t_edge_idx)],
                     edge_attr=Gs.edge_attr[t_edge_idx].view([-1,1]))
        slice = Batch.from_data_list([slice]).to(device)

        # Here starts the episode related to the graph "start" TODO: Add temporal dimension of the interaction graphs
        for time_step in range(torch.max(Gs.t)):
            it = 0
            immediate_interactions = slice.edge_index[:, torch.squeeze(slice.edge_attr >= 1)].cpu().numpy()
            interactions=np.sum(one_hot_to_sparse(slice.x)[immediate_interactions[0]] !=
                                one_hot_to_sparse(slice.x)[immediate_interactions[1]])
            while it < len_episode and validate_partition(immediate_interactions, one_hot_to_sparse(slice.x)) != 0:
                # we evaluate the A2C agent on the graph
                policy, values = model(slice)
                probs = policy.view(-1).clone().detach().cpu().numpy()

                action = np.random.choice(np.arange(slice.num_nodes), p=probs, size=(2,), replace=False)
                new_state = slice.clone()
                # We swap the partition of the selected vertices
                old0 = new_state.x[action[0]].clone()
                new_state.x[action[0]] = new_state.x[action[1]]
                new_state.x[action[1]] = old0
                # Update the state
                slice = new_state.to(device)
                invalid_ints = np.sum(one_hot_to_sparse(slice.x)[immediate_interactions[0]] !=
                                      one_hot_to_sparse(slice.x)[immediate_interactions[1]])

                # Collect all the rewards in this episode
                # rew_partial += -np.sum(
                #     one_hot_to_sparse(slice.x)[immediate_interactions[0]] != one_hot_to_sparse(slice.x)[
                #         immediate_interactions[1]]) / num_nodes ** 2
                rew_partial += interactions-invalid_ints
                # Collect the log-probability of the chosen action TODO:Reevaluate this calculation
                logprobs.append(torch.log(policy.view(-1)[action[0]] *
                                          policy.view(-1)[action[1]] / (1 - policy.view(-1)[action[0]]) +
                                          policy.view(-1)[action[1]] *
                                          policy.view(-1)[action[0]] / (1 - policy.view(-1)[action[1]])))
                # Collect the value of the chosen action
                vals.append(values)
                # Collect the reward
                # rews.append(-np.sum(one_hot_to_sparse(slice.x)[immediate_interactions[0]] != one_hot_to_sparse(slice.x)[
                #     immediate_interactions[1]]) / num_nodes ** 2)
                rews.append(interactions-invalid_ints)
                interactions=invalid_ints

                it += 1
            t_edge_idx = Gs.t == (time_step + 1)
            slice = Data(x=slice.x, edge_index=Gs.edge_index[:, torch.squeeze(t_edge_idx)],
                         edge_attr=Gs.edge_attr[t_edge_idx].view([-1,1]))
            slice = Batch.from_data_list([slice]).to(device)


        # After time_to_sample episodes we update the loss
            if (time_step % time_to_sample == 0 or time_step == torch.max(Gs.t)) and logprobs:  # and i > 0:

                logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
                vals = torch.stack(vals).flip(dims=(0,)).view(-1)
                rews = torch.tensor(rews).flip(dims=(0,)).view(-1)

                # Compute the advantage
                R = []
                R_partial = torch.tensor([0.])
                for j in range(rews.shape[0]):
                    R_partial = rews[j] + gamma * R_partial
                    R.append(R_partial)

                R = torch.stack(R).view(-1).to(device)
                advantage = R - vals.detach()

                # Actor loss
                actor_loss = (-1 * logprobs * advantage)

                # Critic loss
                critic_loss = torch.pow(R - vals, 2)

                # Finally we update the loss
                optimizer.zero_grad()

                loss = torch.mean(actor_loss) + \
                       torch.tensor(coeff) * torch.mean(critic_loss)

                rews, vals, logprobs = [], [], []

                loss.backward()

                optimizer.step()


        i += 1

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--out', default='./temp_edge/', type=str)
    parser.add_argument(
        "--nmin",
        default=50,
        help="Minimum graph size",
        type=int)
    parser.add_argument(
        "--nmax",
        default=100,
        help="Maximum graph size",
        type=int)
    parser.add_argument(
        "--ntrain",
        default=1000,
        help="Number of training graphs",
        type=int)
    parser.add_argument(
        "--print_rew",
        default=1,
        help="Steps to take before printing the reward",
        type=int)
    parser.add_argument("--batch", default=32, help="Batch size", type=int)
    parser.add_argument(
        "--lr",
        default=0.002,
        help="Learning rate",
        type=float)
    parser.add_argument(
        "--gamma",
        default=0.99,
        help="Gamma, discount factor",
        type=float)
    parser.add_argument(
        "--coeff",
        default=0.1,
        help="Critic loss coefficient",
        type=float)
    parser.add_argument(
        "--units_conv",
        default=[
            10,
            10,
            10,
            10],
        help="Number of units in conv layers",
        nargs='+',
        type=int)
    parser.add_argument(
        "--units_dense",
        default=[
            5,
            5,
            5],
        help="Number of units in linear layers",
        nargs='+',
        type=int)

    torch.manual_seed(1)
    np.random.seed(2)

    args = parser.parse_args()
    outdir = args.out + '/'
    Path(outdir).mkdir(parents=True, exist_ok=True)

    n_min = args.nmin
    n_max = args.nmax
    n_train = args.ntrain
    coeff = args.coeff
    print_loss = args.print_rew

    time_to_sample = args.batch
    lr = args.lr
    gamma = args.gamma
    hid_conv = args.units_conv
    hid_lin = args.units_dense


    # Deep neural network that models the DRL agent

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = GATv2Conv(10, hid_conv[0],edge_dim=1)
            self.conv2 = GATv2Conv(hid_conv[0], hid_conv[1],edge_dim=1)
            self.conv3 = GATv2Conv(hid_conv[1], hid_conv[2],edge_dim=1)
            self.conv4 = GATv2Conv(hid_conv[2], hid_conv[3],edge_dim=1)

            self.l1 = nn.Linear(hid_conv[3], hid_lin[0])
            self.l2 = nn.Linear(hid_lin[0], hid_lin[1])
            self.actor1 = nn.Linear(hid_lin[1], hid_lin[2])
            self.actor2 = nn.Linear(hid_lin[2], 1)

            self.GlobAtt = GlobalAttention(
                nn.Sequential(
                    nn.Linear(
                        hid_lin[1], hid_lin[1]), nn.Tanh(), nn.Linear(
                        hid_lin[1], 1)))
            self.critic1 = nn.Linear(hid_lin[1], hid_lin[2])
            self.critic2 = nn.Linear(hid_lin[2], 1)

        def forward(self, graph):
            x_start, edge_index, batch = graph.x, graph.edge_index, graph.batch

            x = self.conv1(graph.x, edge_index,graph.edge_attr)
            x = torch.tanh(x)
            x = self.conv2(x, edge_index,graph.edge_attr)
            x = torch.tanh(x)
            x = self.conv3(x, edge_index,graph.edge_attr)
            x = torch.tanh(x)
            x = self.conv4(x, edge_index,graph.edge_attr)
            x = torch.tanh(x)

            x = self.l1(x)
            x = torch.tanh(x)
            x = self.l2(x)
            x = torch.tanh(x)

            x_actor = self.actor1(x)
            x_actor = torch.tanh(x_actor)
            x_actor = self.actor2(x_actor)
            # flipped = torch.where(
            #     (x_start == torch.tensor([0., 1.])).all(axis=-1))[0]
            # x_actor.data[flipped] = torch.tensor(-np.Inf)
            x_actor = torch.softmax(x_actor, dim=0)

            x_critic = self.GlobAtt(x.detach(), batch)
            x_critic = self.critic1(x_critic)
            x_critic = torch.tanh(x_critic)
            x_critic = self.critic2(x_critic)

            return x_actor, x_critic


    # dataset_path=''
    # if os.path.exists(dataset_path):
    #     dataset=Dataset.processed_file_names=[dataset_path]
    dataset = training_dataset_from_files('random_circuits_remove_empty', 'torch_rl_random', num_files=1000)
    device = torch.device('cuda')
    model = Model().to(device)
    print(model)
    print('Model parameters:',
          sum([w.nelement() for w in model.parameters()]))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training
    print('Start training')
    t0 = timeit.default_timer()
    model = training_loop(
        model,
        dataset,
        gamma,
        time_to_sample,
        coeff,
        optimizer,
        print_loss,
        device)
    ttrain = timeit.default_timer() - t0
    print('Training took:', ttrain, 'seconds')

    # Saving the model
    torch.save(model.state_dict(), outdir + 'model_coarsest')
