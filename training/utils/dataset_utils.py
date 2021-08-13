import glob
import tensorflow as tf
import numpy as np


def get_dataset(records_path, shuffle_size, batch_size):
    tf_records = glob.glob(records_path)
    dataset = tf.data.TFRecordDataset(tf_records)
    dataset = dataset.map(parse_dataset)
    dataset = dataset.shuffle(shuffle_size).repeat().batch(batch_size)
    return dataset


def parse_dataset(record):
    name_to_features = {
        'image': tf.io.FixedLenFeature([], tf.string),
    }
    features = tf.io.parse_example([record], features=name_to_features)
    image = tf.io.decode_raw(features['image'], np.uint8)
    image = tf.reshape(image, (256, 256, 3))
    return image