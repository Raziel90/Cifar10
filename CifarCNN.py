"""Convolutionary Neural Network on CIFAR 10 Dataset."""
from __future__ import print_function
import numpy as np
import tensorflow as tf
#import pickle
from six.moves import cPickle as pickle
from six.moves import range


image_size = 32
num_labels = 10
num_channels = 3  # RGB


tr_data_files = ['data_batch_1', 'data_batch_2',
                 'data_batch_3', 'data_batch_4', 'data_batch_5']
test_data_files = ['test_batch']

"""
pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

"""


# Reformat into a TensorFlow-friendly shape:
# convolutions need the image data formatted as a cube (width by height by #channels)
# labels as float 1-hot encodings.
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

"""
def reformat(dataset, labels):
    dataset = dataset.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.uint8)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.uint8)
    return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
"""

def load_data_set(basepath='./'):
    """Loads and converts the dataset to a tf friendly format """
    batch_data = [] 
    batch_labels = []
    for batch_file in tr_data_files:
        batch_dict = unpickle(basepath + batch_file)
        print(batch_dict[b'batch_label'].decode("utf-8") )
        batch_data += [batch_dict[b'data'].reshape((-1, num_channels, image_size, image_size)).astype(
            np.float32).transpose((0, 2, 3, 1))/255]  # normalised (n_samples,im_size,im_size,n_channels)
        batch_labels += [(np.arange(num_labels) == np.array(batch_dict[b'labels'])
                          [:, None]).astype(np.float32)]  # 1-Hot encoded
    train_data = np.concatenate(batch_data)
    train_labels = np.concatenate(batch_labels)
    del batch_dict, batch_data, batch_labels
    test_dict = unpickle(basepath + test_data_files[0])  # only one test file
    print(test_dict[b'batch_label'].decode("utf-8") )
    test_data = test_dict[b'data'].reshape(
        (-1, num_channels, image_size, image_size)).astype(np.float32).transpose((0, 2, 3, 1))/255
    test_labels = (np.arange(num_labels) == np.array(
        test_dict[b'labels'])[:, None]).astype(np.float32)

    return train_data, train_labels, test_data, test_labels


train_dataset, train_labels, test_dataset, test_labels = load_data_set()


"""

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])



image_size = 28
num_labels = 10
num_channels = 1 # grayscale

batch_size = 200
patch_size = [5,3,2]
depth = [6,18,50]
pool_strides = [2, 2, 2]
num_hidden = 128

graph = tf.Graph()
def define_conv(x,W,b,stride=1,keep_prob=1):
    conv1 = tf.nn.conv2d(input=x,filter=W,strides=[1, stride, stride, 1],padding="SAME")
    conv1 = tf.nn.bias_add(conv1,b)
    conv1 = tf.nn.relu(conv1)
    if keep_prob < 1:
        conv1 = tf.nn.dropout(conv1,keep_prob=keep_prob)
    return conv1
    
def define_max_pool(x,k=2):
    
    return tf.nn.max_pool(x,strides=[1, k, k, 1],ksize=[1, k, k, 1],padding="SAME")

with graph.as_default() as g:
    
    global_step=tf.Variable(0)
    init_learning_rate=tf.placeholder(tf.float32)
    tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size,image_size,image_size,num_channels))
    tf_train_labels = tf.placeholder(tf.float32,shape=(batch_size,num_labels))
    
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    
    hidden_1_weights = tf.Variable(tf.truncated_normal(
        [patch_size[0], patch_size[0], num_channels, depth[0]],
        stddev = 0.01 
    ))
    hidden_1_bias = tf.Variable(tf.zeros([depth[0]]))
    hidden_2_weights = tf.Variable(tf.truncated_normal(
        [patch_size[1], patch_size[1], depth[0], depth[1]], stddev = 0.1
    ))
    hidden_2_bias = tf.Variable(tf.ones([depth[1]]))
    hidden_3_weights = tf.Variable(tf.truncated_normal(
        [patch_size[2], patch_size[2], depth[1], depth[2] ], stddev = 0.1
    ))
    hidden_3_bias = tf.Variable(tf.ones(depth[2]))
    hidden_4_weights = tf.Variable(tf.truncated_normal(
        [-(-image_size // np.prod(pool_strides)) * -(-image_size // np.prod(pool_strides)) * depth[2], num_hidden ], stddev = 0.1
    ))
    hidden_4_bias = tf.Variable(tf.ones([num_hidden]))
    out_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_labels], stddev = 0.1
    ))
    out_bias = tf.Variable(tf.ones([num_labels]))
    
    def define_model(data,training_flg=False):
        keep1=(training_flg * 0.75 + (not training_flg) )#.astype(float)
        #print(keep1)
        conv1=define_conv(data,hidden_1_weights,hidden_1_bias,stride=1,keep_prob=keep1)
        pool1=define_max_pool(conv1,k=pool_strides[0])
#         pool1=conv1
        conv2=define_conv(pool1,hidden_2_weights,hidden_2_bias,stride=1,keep_prob=keep1)
#         pool2=conv2
        pool2=define_max_pool(conv2,k=pool_strides[1])
        conv3=define_conv(pool2,hidden_3_weights,hidden_3_bias,stride=1,keep_prob=keep1)
#         pool2=conv2
        pool3=define_max_pool(conv3,k=pool_strides[2])
        
        shape = pool3.get_shape().as_list()
        
        reshaped = tf.reshape(pool3, [shape[0], shape[1] * shape[2] * shape[3]])
        #print(data.get_shape().as_list())
        #print(shape)
        #print(reshaped.get_shape().as_list())
        full_layer = tf.nn.relu(tf.nn.bias_add(tf.matmul(reshaped, hidden_4_weights), hidden_4_bias))
        
        return tf.nn.bias_add(tf.matmul(full_layer, out_weights), out_bias)
    
    train_model=define_model(tf_train_dataset,training_flg=True)
    
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=train_model, 
            labels=tf_train_labels
        )
    )
    
    reg1 = 0.02*(tf.nn.l2_loss(hidden_1_bias)+tf.nn.l2_loss(hidden_1_weights))
    reg2 = 0.02*(tf.nn.l2_loss(hidden_2_bias)+tf.nn.l2_loss(hidden_2_weights))
    reg3 = 0.02*(tf.nn.l2_loss(hidden_3_bias)+tf.nn.l2_loss(hidden_3_weights))
    reg4 = 0.05*(tf.nn.l2_loss(hidden_4_bias)+tf.nn.l2_loss(hidden_4_weights))
    
    reg_loss = loss + reg1 + reg2 + reg3 + reg4
    
    learning_rate=tf.train.exponential_decay(learning_rate=init_learning_rate,global_step=global_step, decay_steps=200,decay_rate=0.86, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(reg_loss,global_step=global_step)
    
    
    train_prediction = tf.nn.softmax(define_model(tf_train_dataset))
    valid_prediction = tf.nn.softmax(define_model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(define_model(tf_test_dataset)) 
    


num_steps = 10001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, init_learning_rate : 5e-2}
    _, l, predictions= session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 1000 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
      #print(r)
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))



  """
