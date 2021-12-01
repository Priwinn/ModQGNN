import json
import os.path

import mpnn
import data
import configparser
import tensorflow as tf
import logging
import tensorflow_addons as tfa
from datetime import datetime

if __name__ == '__main__':
    config = {
        'name': datetime.now().strftime('%m-%d-%Y_%H%M%S'),
        't': 4,
        'node_state_dim': 128,
        't_int': 1,
        't_temp': 1,
        'batch_size': 64,
        'shuffle': 4096,
        'temp_mode': None,
        'message_hidden': [64, 64, 64, 64],
        'message_activation': 'gelu',
        'message_dropout': [0, 0, 0, 0],
        'readout_hidden': [64, 64, 64, 64],
        'readout_activation': 'gelu',
        'readout_dropout': [0, 0, 0, 0],
        'window_size': 1,
        'stride': 1,
        'n_nodes': 100,
        'n_parts': 10,
        'n_per_part':10,
        'loader': 'lookahead_chong',
        'use_metrics': False,
        'initial_lr': 0.0017,
        'decay_steps': 1000,
        'decay_rate': 0.99
    }
    # Dataset from files
    # train_ds=data.get_ds('random_circuits_remove_empty',
    #                                  config_dict['DATA_PARAMS']['window_size'],
    #                                  config_dict['DATA_PARAMS']['stride'],
    #                file_range=[0,60],
    #                batch_size=config_dict['HYPERPARAMETERS']['batch_size'])
    # if val:
    #     val_ds=data.get_ds('random_circuits_remove_empty',
    #                    config_dict['DATA_PARAMS']['window_size'],
    #                    config_dict['DATA_PARAMS']['stride'],
    #                    file_range=[600,800],
    #                    batch_size=config_dict['HYPERPARAMETERS']['batch_size'])
    # else:
    #     val_ds=None

    # Dataset from TFRecords
    train_ds, val_ds, _ = data.TFRecord_write_load('random_circuits_remove_empty',
                                                   'data_lookahead_relax2',
                                                   config['window_size'],
                                                   config['stride'],
                                                   batch_size=config['batch_size'],
                                                   shuffle=config['shuffle'],
                                                   loader=config['loader'],
                                                   target_suffix='_chong_relax2.py')

    mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = mpnn.MPNN(**config)
    decayed_lr = tf.keras.optimizers.schedules.ExponentialDecay(config['initial_lr'],
                                                                config['decay_steps'],
                                                                config['decay_rate'])
    model.compile(loss=tf.keras.metrics.sparse_categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(decayed_lr),
                  metrics=['accuracy'],
                  # run_eagerly=True #This is to debug, disable if training
                  )
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=100,
                        use_multiprocessing=True,
                        verbose=True,
                        callbacks=[tf.keras.callbacks.TensorBoard(log_dir=os.path.join('logs', config['name']),
                                                                  profile_batch=0)]
                        )
    model.save(os.path.join('models', config['name']))
    with open(os.path.join('models', config['name'], 'config.json'), 'w') as f:
        json.dump(config, f)
