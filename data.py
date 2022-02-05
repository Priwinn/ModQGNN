import os
import tensorflow as tf
import sparse
import numpy as np
from chong_solver import lookahead


def read(name, target_suffix='_chongv2.npy'):
    Gs = sparse.load_npz(name + '.npz')
    chong = np.load(name + target_suffix)
    return Gs, chong


def make_windows(name, size, stride, target_suffix='_chongv2.npy'):
    Gs, chong = read(name, target_suffix=target_suffix)
    T = Gs.shape[0]
    Gs = np.concatenate([Gs, np.stack([Gs[-1]] * size)])
    edges = [Gs[stride * i:stride * i + size].coords.T for i in range(int(np.ceil(T / stride)))]
    edges = [e[np.where(e[:, 1] < e[:, 2])].tolist() for e in edges]

    node_input = chong[:-1:stride]
    target = chong[1::stride]
    return node_input, edges, target


def make_windows_naive(name, stride):
    Gs = sparse.load_npz(name + '.npz')
    naive = np.load(name + '_naive.npy')
    T = Gs.shape[0]
    edges = [Gs[stride * i:stride * i + 1].coords.T.tolist() for i in range(int(np.ceil(T / stride)))]
    node_input = np.repeat(np.expand_dims(np.arange(Gs.shape[1]), axis=0), int(np.ceil(T / stride)), axis=0)
    target = naive[::stride]
    return node_input, edges, target


def load_dir(path, size, stride, file_range=(0, 10000)):
    edges = []
    node_input = []
    target = []
    files = [file for file in os.listdir(path) if '.npz' in file]
    for file in files[file_range[0]:file_range[1]]:
        load = make_windows(os.path.join(path, file.strip('.npz')), size, stride)
        edges += load[1]
        node_input.append(load[0])
        target.append(load[2])
    if len(edges) > 1:
        return np.concatenate(node_input), tf.ragged.stack(edges), np.concatenate(target)
    else:
        return np.concatenate(node_input), tf.expand_dims(edges, axis=0), np.concatenate(target)


def get_ds(path, window_size, stride, file_range=(0, 1000), batch_size=32):
    node_input, edges, target = load_dir(path, window_size, stride, file_range=file_range)
    ds = tf.data.Dataset.from_tensor_slices(((node_input, edges), target)) \
        .shuffle(1024).batch(batch_size, drop_remainder=True) \
        .prefetch(tf.data.AUTOTUNE)

    return ds


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(node_inputs, edges, target):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        'node_inputs': _bytes_feature(tf.io.serialize_tensor(node_inputs)),
        'edges': _bytes_feature(tf.io.serialize_tensor(edges)),
        'target': _bytes_feature(tf.io.serialize_tensor(target)),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def TFRecord_writer(in_path, out_path, size=4, stride=1, file_range=(0, 1000), loader='window_chong',target_suffix='_chongv2.npy'):
    with tf.io.TFRecordWriter(out_path) as file_writer:
        files = [file for file in os.listdir(in_path) if '.npz' in file]
        for file in files[file_range[0]:file_range[1]]:
            if loader == 'window_chong':
                load = make_windows(os.path.join(in_path, file.strip('.npz')), size, stride,target_suffix=target_suffix)
            elif loader == 'naive':
                load = make_windows_naive(os.path.join(in_path, file.strip('.npz')), stride)
            for x, y, z in zip(*load):
                file_writer.write(serialize_example(x, y, z))


def TFRecord_write_load(in_path, out_path, size, stride, splits=(6, 2, 2), batch_size=64, shuffle=4096,
                        loader='window_chong', target_suffix='_chongv2.npy',inf=1):
    splits_total=np.sum(splits)
    n_files = len([file for file in os.listdir(in_path) if '.npz' in file])
    for i in range(splits[0]):
        start = int(i * n_files/splits_total)
        end = int((i + 1) * n_files/splits_total)
        save_path = os.path.join(out_path, f'train_{size}_{stride}_{start}-{end}.tfrecord')
        if not os.path.exists(save_path):
            if loader == 'lookahead_chong':
                write_lookaheads(in_path, save_path, stride=stride, file_range=(start, end),inf=inf,target_suffix=target_suffix)
            else:
                TFRecord_writer(in_path, save_path, size, stride, (start, end),target_suffix=target_suffix)
    for i in range(splits[1]):
        start = int(splits[0]*n_files/splits_total + i * n_files/splits_total)
        end = int(splits[0]*n_files/splits_total + (i + 1) * n_files/splits_total)
        save_path = os.path.join(out_path, f'val_{size}_{stride}_{start}-{end}.tfrecord')
        if not os.path.exists(save_path):
            if loader == 'lookahead_chong':
                write_lookaheads(in_path, save_path, stride=stride, file_range=(start, end),inf=inf,target_suffix=target_suffix)
            else:
                TFRecord_writer(in_path, save_path, size, stride, (start, end),target_suffix=target_suffix)
    for i in range(splits[2]):
        start = int((splits[0]+splits[1])*n_files/splits_total + i * n_files/splits_total)
        end = int((splits[0]+splits[1])*n_files/splits_total + (i + 1) * n_files/splits_total)
        save_path = os.path.join(out_path, f'test_{size}_{stride}_{start}-{end}.tfrecord')
        if not os.path.exists(save_path):
            if loader == 'lookahead_chong':
                write_lookaheads(in_path, save_path, stride=stride, file_range=(start, end),inf=inf,target_suffix=target_suffix)
            else:
                TFRecord_writer(in_path, save_path, size, stride, (start, end),target_suffix=target_suffix)
    return TFRecord_ds(out_path, size, stride, batch_size=batch_size, shuffle=shuffle, loader=loader)


def deserialize_example(example_proto):
    feature_description = {
        'node_inputs': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'edges': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'target': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }
    features = tf.io.parse_single_example(example_proto, feature_description)
    return (tf.ensure_shape(tf.io.parse_tensor(features['node_inputs'], out_type=tf.int64), (100,)),
            tf.RaggedTensor.from_tensor(
                tf.ensure_shape(tf.io.parse_tensor(features['edges'], out_type=tf.int32), (None, 3)))), \
           tf.ensure_shape(tf.io.parse_tensor(features['target'], out_type=tf.int64), (100,))


def process_TFRecord(ds, batch_size, shuffle):
    return ds.shuffle(shuffle) \
        .batch(batch_size, drop_remainder=True). \
        prefetch(tf.data.AUTOTUNE)


def TFRecord_ds(path, size, stride, batch_size=64, shuffle=1024, loader='window_chong'):
    if loader == 'window_chong' or loader == 'naive':
        deserialize = deserialize_example
    elif loader == 'lookahead_chong':
        deserialize = deserialize_example_lookahead
    files = os.listdir(path)
    train_files = [os.path.join(path, file) for file in files if f'train_{size}_{stride}' in file]
    val_files = [os.path.join(path, file) for file in files if f'val_{size}_{stride}' in file]
    test_files = [os.path.join(path, file) for file in files if f'test_{size}_{stride}' in file]
    train_ds = tf.data.TFRecordDataset(train_files).map(deserialize, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = tf.data.TFRecordDataset(val_files).map(deserialize, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = tf.data.TFRecordDataset(test_files).map(deserialize, num_parallel_calls=tf.data.AUTOTUNE)
    return process_TFRecord(train_ds, batch_size, shuffle), \
           process_TFRecord(val_ds, batch_size, shuffle), \
           process_TFRecord(test_ds, batch_size, shuffle)


def write_lookaheads(in_path, out_path, stride=1, inf=1, normalize=True, file_range=(0, 1000),target_suffix='chongv2.npy'):
    files = [file for file in os.listdir(in_path) if '.npz' in file]
    with tf.io.TFRecordWriter(out_path) as file_writer:
        for file in files[file_range[0]:file_range[1]]:
            print(file)
            Gs, chong = read(os.path.join(in_path, file.strip('.npz')),target_suffix=target_suffix)
            node_inputs = chong[:-1:stride]
            target = chong[1::stride]
            for i in range(Gs.shape[0]):
                L = lookahead(Gs[i:], inf=inf)
                # Normalize to (0,1) (almost)
                if normalize:
                    L = L / inf
                edges = np.concatenate([np.expand_dims(L.data, axis=0), L.coords], axis=0).T
                edges = edges[np.where(edges[:, 1] < edges[:, 2])[0]]
                file_writer.write(serialize_example(node_inputs[i], edges.astype(np.single), target[i]))


def deserialize_example_lookahead(example_proto):
    feature_description = {
        'node_inputs': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'edges': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'target': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }
    features = tf.io.parse_single_example(example_proto, feature_description)
    return (tf.ensure_shape(tf.io.parse_tensor(features['node_inputs'], out_type=tf.int64), (100,)),
            tf.RaggedTensor.from_tensor(
                tf.ensure_shape(tf.io.parse_tensor(features['edges'], out_type=tf.float32), (None, 3)))), \
           tf.ensure_shape(tf.io.parse_tensor(features['target'], out_type=tf.int64), (100,))


if __name__ == '__main__':
    TFRecord_write_load('random_circuits_remove_empty', 'data_lookahead', 1, 1, loader='lookahead_chong')
