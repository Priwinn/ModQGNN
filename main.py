import mpnn
import data
import configparser
import tensorflow as tf


if __name__=='__main__':
    path='random_circuits'

    config_dict={
        'HYPERPARAMETERS':{'t':16,
                           'node_state_dim':128},
        'DATA_PARAMS':{'window_size': 5,
                       'stride':100,
                       'n_nodes':100,
                       'N':10}
    }
    ds=data.get_ds('random_circuits',
                                     config_dict['DATA_PARAMS']['window_size'],
                                     config_dict['DATA_PARAMS']['stride'],
                                     file_range=[0,2])
    val_ds=data.get_ds('random_circuits',
                   config_dict['DATA_PARAMS']['window_size'],
                   config_dict['DATA_PARAMS']['stride'],
                   file_range=[900,1000])
    model=mpnn.MPNN(config_dict)
    decayed_lr = tf.keras.optimizers.schedules.ExponentialDecay(0.001,
                                                                10000,
                                                                0.1,
                                                                staircase=False)
    model.compile(loss=tf.keras.metrics.sparse_categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(decayed_lr),metrics=['accuracy'])
    history=model.fit(ds,
              validation_data=val_ds,
              epochs=10000,
              verbose=True,
              use_multiprocessing=True)