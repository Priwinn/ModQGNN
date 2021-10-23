import tensorflow as tf
import configparser


class MPNN(tf.keras.Model):
    def __init__(self, config):
        super(MPNN, self).__init__()
        self.config = config
        # Update function
        self.int_update = tf.keras.layers.GRUCell(int(self.config['HYPERPARAMETERS']['node_state_dim']), name='int_update')
        self.temp_update = tf.keras.layers.GRUCell(int(self.config['HYPERPARAMETERS']['node_state_dim']), name='temp_update')

        # Message function
        self.message_func = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(1024,
                                      activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['node_state_dim']),
                                      activation=tf.nn.relu)
            ]
        )
        # Readout function
        self.readout = tf.keras.Sequential([
            tf.keras.layers.Dense(1024,
                                  activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512,
                                  activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        self.embed = tf.keras.layers.Embedding(10, self.config['HYPERPARAMETERS']['node_state_dim'])
        self.window_size = self.config['DATA_PARAMS']['window_size']
        self.n_nodes = self.config['DATA_PARAMS']['n_nodes']
        self.n_nodes_total = self.window_size * self.n_nodes
        self.temp_edges = self.get_temp_edges()

    def get_temp_edges(self):
        temp_edges0 = tf.constant([[i, i + self.n_nodes] for i in range(self.n_nodes_total - self.n_nodes)])
        temp_edges1 = tf.constant([[i, i - self.n_nodes] for i in range(self.n_nodes, self.n_nodes_total)])
        temp_edges = tf.concat([temp_edges0, temp_edges1], axis=0)
        return temp_edges[:,0],temp_edges[:,1]

    @tf.function
    def call(self, inputs):
        # edges,initial partition
        int_edges, nodes = inputs
        int_edges = int_edges.to_tensor()

        node_indices = tf.squeeze(int_edges[..., 0:1] * self.n_nodes + int_edges[..., 1:2],name='squeeze_node_indices')
        neighbour_indices = tf.squeeze(int_edges[..., 0:1] * self.n_nodes + int_edges[..., 2:],name='squeeze_neihgbour_indices')
        # Initialize node states. We use an Embedding layer to obtain
        # a dense state from each partition number (Another option is to use one-hot and pad)
        node_states = tf.squeeze(self.embed(nodes))
        node_states = tf.repeat(node_states, repeats=self.window_size, axis=0)

        # Message passing
        for _ in range(int(self.config['HYPERPARAMETERS']['t'])):
            # Interaction messages
            node_gather = tf.gather(node_states, node_indices,name='node_gather')
            neighbour_gather = tf.gather(node_states, neighbour_indices,name='neighbour_gather')
            node_gather = tf.squeeze(node_gather,name='squeeze_node_gather')
            neighbour_gather = tf.squeeze(neighbour_gather,name='squeeze_neighbour_gather')
            neighbour_gather = tf.ensure_shape(neighbour_gather,[None,int(self.config['HYPERPARAMETERS']['node_state_dim'])])
            int_msg_input = tf.concat([node_gather, neighbour_gather], axis=1)
            int_msg_input = tf.ensure_shape(int_msg_input,
                                            [None, int(self.config['HYPERPARAMETERS']['node_state_dim']) * 2])
            int_message = self.message_func(int_msg_input)
            # Interaction Agregation
            int_mean = tf.math.unsorted_segment_mean(int_message,
                                                     neighbour_indices,
                                                     self.window_size *
                                                     self.n_nodes)
            int_mean = tf.ensure_shape(int_mean, [None, int(self.config['HYPERPARAMETERS']['node_state_dim'])])
            node_states, _ = self.int_update(int_mean, [node_states])
            # Temporal messages
            temp_node_gather = tf.gather(node_states, self.temp_edges[0],name='temp_node_gather')
            temp_neighbour_gather = tf.gather(node_states, self.temp_edges[1],name='temp_neighbour_gather')
            temp_node_gather = tf.squeeze(temp_node_gather,name='temp_squeeze_node_gather')
            temp_neighbour_gather = tf.squeeze(temp_neighbour_gather,name='temp_squeeze_neighbour_gather')
            temp_neighbour_gather = tf.ensure_shape(temp_neighbour_gather,[None,int(self.config['HYPERPARAMETERS']['node_state_dim'])])
            temp_msg_input = tf.concat([temp_node_gather, temp_neighbour_gather], axis=1)
            temp_msg_input = tf.ensure_shape(temp_msg_input,
                                            [None, int(self.config['HYPERPARAMETERS']['node_state_dim']) * 2])
            temp_message = self.message_func(temp_msg_input)
            #Temporal aggregation
            temp_mean = tf.math.unsorted_segment_mean(temp_message,
                                                      self.temp_edges[1],
                                                      self.window_size *
                                                      self.n_nodes)
            temp_mean = tf.ensure_shape(temp_mean, [None, int(self.config['HYPERPARAMETERS']['node_state_dim'])])


            # Update


            node_states, _ = self.temp_update(temp_mean, [node_states])

        # Readout
        nn_output = self.readout(tf.gather(node_states, tf.range(self.n_nodes), axis=0,name='readout_gather'))
        return nn_output
