import json
import tensorflow as tf
from scipy.optimize import linear_sum_assignment as sp_lsa
from numpy import vectorize
import mpnn_utils

sp_lsa = vectorize(sp_lsa, signature='(n,n)->(n),(n)')


class MPNN(tf.keras.Model):
    def __init__(self,
                 name='temp',
                 n_nodes=100,
                 n_parts=10,
                 window_size=4,
                 node_state_dim=128,
                 message_hidden=[64, 64, 64, 64],
                 message_activation='gelu',
                 message_dropout=[0, 0, 0, 0],
                 readout_hidden=[64, 64, 64, 64],
                 readout_activation='gelu',
                 readout_dropout=[0, 0, 0, 0],
                 loader='window_chong',
                 t=4,
                 temp_mode='center',
                 use_metrics=False,  # Adds informative metrics regarding invalid interactions,
                 # number of valid partitions and number of changes, but adds
                 # significant overhead due to the use of scipy lsa
                 t_temp=1,
                 t_int=1, **kwargs):
        super(MPNN, self).__init__(name=name)
        self.n_parts = n_parts
        self.window_size = window_size
        self.n_nodes = n_nodes
        self.n_per_part = int(n_nodes / n_parts)
        self.n_nodes_total = window_size * n_nodes
        self.node_state_dim = node_state_dim
        self.loader = loader
        self.t = t
        self.t_int = t_int
        self.t_temp = t_temp
        self.use_metrics = use_metrics
        self.temp_mode = temp_mode
        # Update function
        self.int_update = tf.keras.layers.GRUCell(node_state_dim, name='int_update')
        self.temp_update = tf.keras.layers.GRUCell(node_state_dim, name='temp_update')

        # Message function
        self.message_func = mpnn_utils.message_ffn(message_hidden,
                                                   message_activation,
                                                   message_dropout,
                                                   node_state_dim)
        # Readout function
        self.readout = mpnn_utils.readout_ffn(readout_hidden,
                                              readout_activation,
                                              readout_dropout,
                                              n_parts)
        self.embed = tf.keras.layers.Embedding(10, node_state_dim)

    def call(self, inputs, training=True):
        # edges,initial partition
        if self.loader == 'lookahead_chong':
            node_inputs, node_indices, neighbour_indices, lookahead_weights = mpnn_utils.graph_batch(inputs,
                                                                                                     self.loader,
                                                                                                     self.temp_mode,
                                                                                                     self.n_nodes,
                                                                                                     self.window_size)
        else:
            node_inputs, node_indices, neighbour_indices, temp_node_indices, temp_neighbour_indices = mpnn_utils.graph_batch(
                inputs, self.loader, self.temp_mode, self.n_nodes, self.window_size)
        # Initialize node states. We use an Embedding layer to obtain
        # a dense state from each partition number
        node_states = self.embed(node_inputs)
        batch_size = tf.shape(node_inputs)[0]
        node_states = tf.repeat(node_states, repeats=self.window_size, axis=0)  # Repeat states for every time slice
        node_states = tf.reshape(node_states,
                                 [-1, self.node_state_dim])  # Flatten batch dimension
        # Message passing
        for i in range(self.t):
            # Interaction messages
            for j in range(self.t_int):
                node_gather = tf.gather(node_states, node_indices, name='node_gather')
                neighbour_gather = tf.gather(node_states, neighbour_indices, name='neighbour_gather')
                # node_gather = tf.squeeze(node_gather, name='squeeze_node_gather')
                # neighbour_gather = tf.squeeze(neighbour_gather, name='squeeze_neighbour_gather')
                # neighbour_gather = tf.ensure_shape(neighbour_gather,
                #                                    [None, self.node_state_dim])
                if self.loader == 'lookahead_chong':
                    int_msg_input0 = tf.concat([node_gather, neighbour_gather, lookahead_weights], axis=1)
                    int_msg_input1 = tf.concat([neighbour_gather, node_gather, lookahead_weights], axis=1)
                    int_msg_input = tf.concat([int_msg_input0, int_msg_input1], axis=0)
                    # int_msg_input = tf.ensure_shape(int_msg_input,
                    #                                 [None,
                    #                                  self.node_state_dim * 2 + 1])
                else:
                    int_msg_input0 = tf.concat([node_gather, neighbour_gather], axis=1)
                    int_msg_input1 = tf.concat([neighbour_gather, node_gather], axis=1)
                    int_msg_input = tf.concat([int_msg_input0, int_msg_input1], axis=0)
                    # int_msg_input = tf.ensure_shape(int_msg_input,
                    #                                 [None, self.node_state_dim * 2])
                int_message = self.message_func(int_msg_input)
                # Interaction Aggregation
                int_mean = tf.math.unsorted_segment_mean(int_message,
                                                         tf.concat([neighbour_indices, node_indices], axis=0),
                                                         tf.shape(node_states)[0])
                # int_mean = tf.ensure_shape(int_mean, [None, self.node_state_dim])
                # Update
                node_states, _ = self.int_update(int_mean, [node_states])
            # Temporal messages
            if self.temp_mode:
                for k in range(self.t_temp):
                    temp_node_gather = tf.gather(node_states, temp_node_indices, name='temp_node_gather')
                    temp_neighbour_gather = tf.gather(node_states, temp_neighbour_indices, name='temp_neighbour_gather')
                    temp_node_gather = tf.squeeze(temp_node_gather, name='temp_squeeze_node_gather')
                    # temp_neighbour_gather = tf.squeeze(temp_neighbour_gather, name='temp_squeeze_neighbour_gather')
                    # temp_neighbour_gather = tf.ensure_shape(temp_neighbour_gather,
                    #                                         [None,
                    #                                          self.node_state_dim])
                    temp_msg_input0 = tf.concat([temp_node_gather, temp_neighbour_gather], axis=1)
                    temp_msg_input1 = tf.concat([temp_neighbour_gather, temp_node_gather], axis=1)
                    temp_msg_input = tf.concat([temp_msg_input0, temp_msg_input1], axis=0)
                    # temp_msg_input = tf.ensure_shape(temp_msg_input,
                    #                                  [None, self.node_state_dim * 2])
                    temp_message = self.message_func(temp_msg_input)
                    # Temporal aggregation
                    temp_mean = tf.math.unsorted_segment_mean(temp_message,
                                                              tf.concat([temp_neighbour_indices, temp_node_indices],
                                                                        axis=0),
                                                              tf.shape(node_states)[0])
                    # temp_mean = tf.ensure_shape(temp_mean,
                    #                             [None, self.node_state_dim])

                    node_states, _ = self.temp_update(temp_mean, [node_states])

        # Readout
        readout_indices = tf.repeat(tf.expand_dims(tf.range(self.n_nodes), axis=0), batch_size, axis=0)
        readout_indices = readout_indices + tf.reshape(tf.range(batch_size), [batch_size, 1]) * self.n_nodes_total
        nn_output = self.readout(tf.gather(node_states, readout_indices))
        # Check valid partitions
        if self.use_metrics:
            assignments = tf.cast(mpnn_utils.lsa(nn_output), tf.int64)
            self.add_metric(tf.reduce_sum(tf.cast(assignments != inputs[0], tf.float32), axis=1), aggregation='mean',
                            name='num_changes')
            if self.loader != 'lookahead_chong':
                target_nodes = tf.ragged.boolean_mask(inputs[1][..., 1:2],
                                                      inputs[1][..., 0:1] == 0)
                target_neighbours = tf.ragged.boolean_mask(inputs[1][..., 2:],
                                                           inputs[1][..., 0:1] == 0)
                src_gather = tf.gather(assignments, target_nodes, batch_dims=1)
                dst_gather = tf.gather(assignments, target_neighbours, batch_dims=1)

                invalid_ints = mpnn_utils.batched_invalid_int(src_gather, dst_gather)
                self.add_metric(tf.cast(invalid_ints, tf.float32), aggregation='mean',
                                name='invalid_ints')
                self.add_metric(tf.cast(invalid_ints == 0, tf.float32), aggregation='mean',
                                name='valid_parts')
        return nn_output
