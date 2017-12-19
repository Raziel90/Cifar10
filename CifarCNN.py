"""Convolutionary Neural Network on CIFAR 10 Dataset."""
from __future__ import print_function
import numpy as np
import tensorflow as tf
#import pickle
from six.moves import cPickle as pickle
from six.moves import range
import matplotlib.pyplot as plt
import os
import cv2
image_size = 32
num_labels = 10
num_channels = 3  # RGB
batch_len = 200
examples_per_mode = {'train' : 45000, 'validation' : 5000, 'test' : 10000}

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
def get_files(basepath='./',mode='train'):
    return os.path.join(basepath,mode+'.tfrecords')

def serialize(input_file):

    filename_queue = tf.train.string_input_producer([input_file],num_epochs=1)

    reader = tf.TFRecordReader()
    _, serialised_example = reader.read(filename_queue)

    return serialised_example

def TFR_parse(example):
    features_desc = { 
    'data': tf.FixedLenFeature([], tf.string) ,
    'labels': tf.FixedLenFeature([],tf.int64)
    }
   
    features = tf.parse_single_example(example,features=features_desc)
    
    image = tf.decode_raw(features['data'],tf.uint8)
    image.set_shape([num_channels * image_size * image_size])
    image = tf.cast(
        tf.transpose(
            tf.reshape(image,(num_channels, image_size, image_size)
                ),( 1, 2, 0)
            ),tf.float32)
    #label = tf.cast(features['labels'],tf.int32)
    label = tf.cast(
        tf.one_hot(
            tf.cast(features['labels'],tf.int32),num_labels),
        tf.float32)
    return image,label

def make_batch(batch_size=100,mode='train',basepath='./'):

    filename = get_files(basepath,mode)
    print(filename)
    image,label = TFR_parse(serialize(filename))
    #dataset = tf.contrib.data.TFRecordDataset(filename).repeat()

    #dataset = dataset.map(TFR_parse, num_threads=batch_size, output_buffer_size = 2 * batch_size)

    if mode == 'train':
        min_examples = int(examples_per_mode['train'] * 0.4) # so that the shuffeling is good enough

        data_batch, label_batch = tf.train.shuffle_batch([image,label],batch_size=batch_len,capacity=examples_per_mode['train'],min_after_dequeue=min_examples)

    else:
        min_examples = int(examples_per_mode[mode] * 0.4) # so that the shuffeling is good enough

        data_batch, label_batch = tf.train.batch([image,label],batch_size=examples_per_mode[mode],capacity=examples_per_mode[mode])
        #dataset = dataset.shuffe(buffer_size=min_queue_examples + 3 * batch_size)

    #finally create the batches
    #dataset = dataset.batch(batch_size)
    #iterator = dataset.make_one_shot_iterator()
    #data_batch, label_batch = iterator.get_next()

    return data_batch,label_batch


#Definition of the Architecture

#batch_size = 200
patch_size = [5,3,2]
depth = [6,18,50]
pool_strides = [2, 2, 2]
num_hidden = 128

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

def define_conv(x,W,b,stride=1,keep_prob=1):
    conv1 = tf.nn.conv2d(input=x,filter=W,strides=[1, stride, stride, 1],padding="SAME")
    conv1 = tf.nn.bias_add(conv1,b)
    conv1 = tf.nn.relu(conv1)
    if keep_prob < 1:
        conv1 = tf.nn.dropout(conv1,keep_prob=keep_prob)
    return conv1
    
def define_max_pool(x,k=2):
    
    return tf.nn.max_pool(x,strides=[1, k, k, 1],ksize=[1, k, k, 1],padding="SAME")



graph = tf.Graph()
with graph.as_default() as g:
    tf_train_dataset,tf_train_labels = make_batch(batch_len,'train',basepath='./cifar-10-batches-py')
    tf_valid_dataset,tf_valid_dataset_labels = make_batch(batch_len,'validation',basepath='./cifar-10-batches-py')
    tf_test_dataset,tf_test_dataset_labels = make_batch(batch_len,'test',basepath='./cifar-10-batches-py')
    # The op for initializing the variables.
    



    global_step=tf.Variable(0)
    init_learning_rate=tf.placeholder(tf.float32)
    #tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size,image_size,image_size,num_channels))
    #tf_train_labels = tf.placeholder(tf.float32,shape=(batch_size,num_labels))
    
    #tf_valid_dataset = tf.constant(valid_dataset)
    #tf_test_dataset = tf.constant(test_dataset)
    
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
    
    init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())


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

with tf.Session(graph=graph) as sess:
    
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    valid_data, valid_labels, test_data, test_labels = sess.run([tf_valid_dataset,tf_valid_dataset_labels,tf_test_dataset,tf_test_dataset_labels])
    print('Initialized')
    for step in range(num_steps):
        
        batch_data, batch_labels = sess.run([tf_train_dataset,tf_train_labels])
        
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, init_learning_rate : 5e-2}
        _, l, predictions= sess.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 1000 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), valid_labels))
                #print(r)
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

"""
with tf.Session(graph=graph) as sess:
    sess.run(init_op)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    final_image,final_lab = sess.run([tf_train_dataset,tf_train_labels])
    print(len(final_image),final_image.shape,final_lab.shape)
    print(final_lab)
    
    plt.imshow(final_image[10,:,:,::-1])
    plt.show()


"""



