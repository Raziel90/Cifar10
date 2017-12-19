import numpy as np
import tensorflow as tf
#import pickle
from six.moves import cPickle as pickle
from six.moves import range
import tarfile
import os


CIFAR_FILENAME = 'cifar-10-python.tar.gz'
CIFAR_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME

tr_data_files = ['data_batch_1', 'data_batch_2',
                 'data_batch_3', 'data_batch_4', 'data_batch_5']
test_data_files = ['test_batch']


def maybe_download(destination_path='./'):
    tf.contrib.learn.datasets.base.maybe_download(
        CIFAR_FILENAME, destination_path, CIFAR_DOWNLOAD_URL)
    tarfile.open(os.path.join(destination_path, CIFAR_FILENAME),
                 'r:gz').extractall(destination_path)


def _get_batch_names():
    files = dict()
    files['train'] = tr_data_files[0:3]
    files['validation'] = [tr_data_files[4]]
    files['test'] = [test_data_files[0]]
    return files


# extract data from pickle files
def unpickle(file):
    with open(file, 'rb') as fo:
        #file_dict = pickle.load(fo, encoding='bytes') #use if python 3 but seems not to work with TF-slim
        file_dict = pickle.load(fo)
    return file_dict

# define the features for the protocol


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))

# write TFRecods


def write_TF_records(input_files, output_file):
    print("Generating TF Records in ", output_file, "....")
    with tf.python_io.TFRecordWriter(output_file) as TFR_writer:
        for batch_file in input_files:

            data_dict = unpickle(batch_file)
            print(data_dict['batch_label'].decode("utf-8"))
            # for every datapoint
            for data_idx in range(len(data_dict['labels'])):
                #print('tipo',type(data_dict[b'data'][data_idx,:].tobytes()))
                record = tf.train.Example(features=tf.train.Features(
                    feature={
                        'data': _bytes_feature(data_dict['data'][data_idx,:].tobytes()) ,
                        'labels': _int64_feature(data_dict['labels'][data_idx])
                    }
                ))
                TFR_writer.write(record.SerializeToString())


# loading dataset directly from pickle alternative
def import_data_set(basepath='./'):
    """Loads and converts the dataset to a tf friendly format """
    batch_data = []
    batch_labels = []
    for batch_file in tr_data_files:
        batch_dict = unpickle(basepath + batch_file)
        print(batch_dict['batch_label'].decode("utf-8"))
        batch_data += [batch_dict['data'].reshape((-1, num_channels, image_size, image_size)).astype(
            np.float32).transpose((0, 2, 3, 1))/255]  # normalised (n_samples,im_size,im_size,n_channels)
        batch_labels += [(np.arange(num_labels) == np.array(batch_dict[b'labels'])
                          [:, None]).astype(np.float32)]  # 1-Hot encoded
    train_data = np.concatenate(batch_data)
    train_labels = np.concatenate(batch_labels)
    del batch_dict, batch_data, batch_labels
    test_dict = unpickle(basepath + test_data_files[0])  # only one test file
    print(test_dict['batch_label'].decode("utf-8"))
    test_data = test_dict['data'].reshape(
        (-1, num_channels, image_size, image_size)).astype(np.float32).transpose((0, 2, 3, 1))/255
    test_labels = (np.arange(num_labels) == np.array(
        test_dict[b'labels'])[:, None]).astype(np.float32)

    return train_data, train_labels, test_data, test_labels


def main():
    print("Download data...")
    current_fol = './'
    maybe_download(current_fol)
    print('Downloaded!')
    dataset_files = _get_batch_names()
    #print('files',dataset_files)
    for mode, filenames in dataset_files.items():
        input_file = [os.path.join(current_fol, 'cifar-10-batches-py', filename) for filename in filenames]
        output_file = os.path.join(current_fol, 'cifar-10-batches-py', mode + '.tfrecords')
        try:
            os.remove(output_file)
        except OSError:
            pass
        write_TF_records(input_file, output_file)


if __name__ == '__main__':
    main()
    print('Extraction Completed!')
