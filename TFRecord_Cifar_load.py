
import os

from six.moves import cPickle as pickle
from six.moves import range
import tensorflow as tf

image_size = 32
num_labels = 10
num_channels = 3  # RGB
examples_per_mode = {'train': 45000, 'validation': 5000, 'test': 10000}


def get_files(basepath='./', mode='train'):
    return os.path.join(basepath, mode+'.tfrecords')


def serialize(input_file):

    filename_queue = tf.train.string_input_producer([input_file])

    reader = tf.TFRecordReader()
    _, serialised_example = reader.read(filename_queue)

    return serialised_example


def TFR_parse(example):
    features_desc = {
        'data': tf.FixedLenFeature([], tf.string),
        'labels': tf.FixedLenFeature([], tf.int64)
    }

    features = tf.parse_single_example(example, features=features_desc)

    image = tf.decode_raw(features['data'], tf.uint8)
    image.set_shape([num_channels * image_size * image_size])
    image = tf.cast(
        tf.transpose(
            tf.reshape(image, (num_channels, image_size, image_size)
                       ), (1, 2, 0)
        ), tf.float32)
    #label = tf.cast(features['labels'],tf.int32)
    label = tf.cast(features['labels'], tf.int64)
        
    return image, label


def make_batch(batch_size=100, mode='train', basepath='./'):

    filename = get_files(basepath, mode)
    print(filename)
    image, label = TFR_parse(serialize(filename))
    image = tf.image.per_image_standardization(image)

    if mode == 'train':
        # so that the shuffeling is good enough
        min_examples = int(examples_per_mode['train'] * 0.4)

        data_batch, label_batch = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size,
            capacity=examples_per_mode['train'],
            min_after_dequeue=min_examples, num_threads=8)

    else:

        data_batch, label_batch = tf.train.batch(
            [image, label], batch_size=examples_per_mode[mode],
            capacity=examples_per_mode[mode], num_threads=8)
    tf.summary.image('images', data_batch)
    #tf.add_to_collection('summaries', image_summary)
    #print(label_batch)
    return data_batch, label_batch
