"""Convolutionary Neural Network on CIFAR 10 Dataset."""
from __future__ import print_function
import numpy as np
import tensorflow as tf

# import pickle
from TFRecord_Cifar_load import make_batch
import matplotlib.pyplot as plt

image_size = 32
num_labels = 10
num_channels = 3  # RGB
batch_len = 100
examples_per_mode = {'train': 45000, 'validation': 5000, 'test': 10000}
INIT_L_RATE = 0.1  # 5e-2
LEARNING_RATE_DECAY_FACTOR = 0.1
NUM_EPOCHS_PER_DECAY = 350.0

# Definition of the Architecture


def activation_info(layer):
    tensor_name = layer.op.name
    tf.summary.histogram(tensor_name + '/activations', layer)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(layer))


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def conv_params(patch_sz, prev_depth, curr_depth, stddev=5e-2):
    weights = tf.Variable(tf.truncated_normal(
        [patch_sz, patch_sz, prev_depth, curr_depth],
        stddev=0.01
    ))
    bias = tf.Variable(0.1 * tf.ones([curr_depth]))
    return weights, bias


def define_conv(x, W, b, stride=1, keep_prob=1):

    # print(x.get_shape().as_list(), W.get_shape().as_list())
    conv1 = tf.nn.conv2d(input=x, filter=W, strides=[1, stride, stride, 1],
                         padding="SAME")
    conv1 = tf.nn.bias_add(conv1, b)
    conv1 = tf.nn.relu(conv1)
    if keep_prob < 1:
        conv1 = tf.nn.dropout(conv1, keep_prob=keep_prob)
    return conv1


def define_max_pool(x, k=2):
    return tf.nn.max_pool(x, strides=[1, k, k, 1],
                          ksize=[1, k, k, 1], padding="SAME")


def define_conv_layer(input_tensor, weights, bias, conv_stride, pool_stride,
                      want_pooling=True, keep_prob=1.0):

    conv_layer = define_conv(input_tensor, weights,
                             bias, conv_stride, keep_prob)
    if want_pooling:
        return define_max_pool(conv_layer, pool_stride)
    else:
        return conv_layer


def l2_regularize(weights, bias, decay):
    return decay * tf.nn.l2_loss(weights), decay * tf.nn.l2_loss(bias)

patch_size = [3, 3, 3, 3, 3]
depth = [32, 32, 64, 64, 128, 128]
poolings = [True, False, True, False, True]
pool_strides = [2, 2, 2, 2, 2]

# num_hidden = 500

graph = tf.Graph()
with graph.as_default() as g:

    summary_writer = tf.summary.FileWriter('./log', g)
    tf_train_dataset, tf_train_labels = make_batch(
        batch_len, 'train', basepath='./cifar-10-batches-py')
    tf_valid_dataset, tf_valid_dataset_labels = make_batch(
        batch_len, 'validation', basepath='./cifar-10-batches-py')
    tf_test_dataset, tf_test_dataset_labels = make_batch(
        batch_len, 'test', basepath='./cifar-10-batches-py')
    # The op for initializing the variables.

    global_step = tf.Variable(0)
    init_learning_rate = tf.placeholder(tf.float32)
    # tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size,image_size,image_size,num_channels))
    # tf_train_labels = tf.placeholder(tf.float32,shape=(batch_size,num_labels))

    # tf_valid_dataset = tf.constant(valid_dataset)
    # tf_test_dataset = tf.constant(test_dataset)
    hidden_1_weights, hidden_1_bias = conv_params(
        patch_size[0], num_channels, depth[0])
    hidden_2_weights, hidden_2_bias = conv_params(
        patch_size[1], depth[0], depth[1])
    hidden_3_weights, hidden_3_bias = conv_params(
        patch_size[2], depth[1], depth[2])
    hidden_4_weights, hidden_4_bias = conv_params(
        patch_size[3], depth[2], depth[3])
    hidden_5_weights, hidden_5_bias = conv_params(
        patch_size[4], depth[3], depth[4])
  
    def define_model(data, training_flg=False):
        keep1 = (training_flg * 0.8 + (not training_flg))

        conv_l1 = define_conv_layer(data, hidden_1_weights, hidden_1_bias,
                                    1, pool_strides[0], want_pooling=True, keep_prob=keep1)
        # conv1=define_conv(data,hidden_1_weights,hidden_1_bias,stride=patch_size[0],keep_prob=keep1)
        # pool1=define_max_pool(conv1,k=pool_strides[0])
        conv_l2 = define_conv_layer(conv_l1, hidden_2_weights, hidden_2_bias,
                                    1, pool_strides[1], want_pooling=False, keep_prob=keep1)
        # conv2=define_conv(conv_l1,hidden_2_weights,hidden_2_bias,stride=patch_size[1],keep_prob=keep1)
        # pool2=define_max_pool(conv2,k=pool_strides[1])
        conv_l3 = define_conv_layer(conv_l2, hidden_3_weights, hidden_3_bias,
                                    1, pool_strides[2], want_pooling=True, keep_prob=keep1)
        # conv3=define_conv(pool2,hidden_3_weights,hidden_3_bias,stride=patch_size[2],keep_prob=keep1)
        # pool3=define_max_pool(conv3,k=pool_strides[2])
        conv_l4 = define_conv_layer(conv_l3, hidden_4_weights, hidden_4_bias,
                                    1, pool_strides[3], want_pooling=False, keep_prob=keep1)
        # conv4=define_conv(pool3,hidden_4_weights,hidden_4_bias,stride=patch_size[3],keep_prob=keep1)
        # pool4=define_max_pool(conv4,k=pool_strides[3])
        conv_l5 = define_conv_layer(conv_l4, hidden_5_weights, hidden_5_bias,
                                    1, pool_strides[4], want_pooling=True, keep_prob=keep1)
        # conv5=define_conv(pool4,hidden_5_weights,hidden_5_bias,stride=patch_size[4],keep_prob=keep1)
        # pool5=define_max_pool(conv5,k=pool_strides[3])

        activation_info(conv_l1)
        activation_info(conv_l2)
        activation_info(conv_l3)
        activation_info(conv_l4)
        activation_info(conv_l5)

        # flattening_value = np.prod(pool_strides)
        flattening_value = 8

        shape = conv_l5.get_shape().as_list()
        reshaped = tf.reshape(
            conv_l5, [shape[0], shape[1] * shape[2] * shape[3]])

        out_weights = tf.Variable(tf.truncated_normal(
            [-(-image_size // flattening_value) *
             -(-image_size // flattening_value) * depth[4], num_labels], stddev=0.1))
        out_bias = tf.Variable(tf.ones([num_labels]))

        # print(data.get_shape().as_list())
        # print(shape)
        # print(reshaped.get_shape().as_list())
        # full_layer = tf.nn.relu(tf.nn.bias_add(tf.matmul(reshaped, hidden_5_weights), hidden_5_bias))
        # print(reshaped.get_shape().as_list(), out_weights.get_shape().as_list())
        return tf.nn.bias_add(tf.matmul(reshaped, out_weights), out_bias)

    train_model = define_model(tf_train_dataset, training_flg=True)

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=train_model,
            labels=tf_train_labels
        )
    )

    reg1 = l2_regularize(hidden_1_weights, hidden_1_bias, 1e-4)[0]
    reg2 = l2_regularize(hidden_2_weights, hidden_2_bias, 1e-4)[0]
    reg3 = l2_regularize(hidden_3_weights, hidden_3_bias, 1e-4)[0]
    reg4 = l2_regularize(hidden_4_weights, hidden_4_bias, 1e-4)[0]
    reg5 = l2_regularize(hidden_5_weights, hidden_5_bias, 1e-4)[0]

    reg_loss = loss + reg1 + reg2 + reg3 + reg4 + reg5
    num_batches_per_epoch = examples_per_mode['train'] / batch_len
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    learning_rate = tf.train.exponential_decay(
        learning_rate=init_learning_rate, global_step=global_step,
        decay_steps=decay_steps, decay_rate=LEARNING_RATE_DECAY_FACTOR, staircase=True)
    optimizer = tf.train.AdamOptimizer(
        learning_rate).minimize(reg_loss, global_step=global_step)

    train_prediction = tf.nn.softmax(define_model(tf_train_dataset))
    valid_prediction = tf.nn.softmax(define_model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(define_model(tf_test_dataset))

    # tr_accuracy=tf.metrics.accuracy(predictions=tf.argmax(train_prediction, 1), labels=tf.argmax(tf_train_labels,1))
    # valid_accuracy=tf.metrics.accuracy(predictions=tf.argmax(valid_prediction, 1),labels=tf.argmax(tf_valid_dataset_labels,1))
    # test_accuracy=tf.metrics.accuracy(predictions=tf.argmax(test_prediction, 1),labels=tf.argmax(tf_test_dataset_labels,1))

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    merged = tf.summary.merge_all()


num_steps = 10001

with tf.Session(graph=graph) as sess:

    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    valid_data, valid_labels, test_data, test_labels = sess.run(
        [tf_valid_dataset, tf_valid_dataset_labels,
         tf_test_dataset, tf_test_dataset_labels])
    print('Initialized')
    tr_acc = []
    valid_acc = []
    test_acc = []
    loss_in_time = []
    for step in range(num_steps):

        batch_data, batch_labels = sess.run(
            [tf_train_dataset, tf_train_labels])

        feed_dict = {tf_train_dataset: batch_data,
                     tf_train_labels: batch_labels, init_learning_rate: INIT_L_RATE}
        _, l, predictions, summary = sess.run(
            [optimizer, loss, train_prediction,
             # tr_accuracy,valid_accuracy,test_accuracy
             merged], feed_dict=feed_dict)
        summary_writer.add_summary(summary, step)

        if (step % 300 == 0):
            tr_a = accuracy(predictions, batch_labels)
            val_a = accuracy(test_prediction.eval(), test_labels)
            tr_acc += tr_a
            valid_acc += val_a
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % tr_a)
            print('Validation accuracy: %.1f%%' % val_a)
    test_acc = accuracy(test_prediction.eval(), test_labels)
    print('Test accuracy: %.1f%%' % test_acc)
    coord.request_stop()
    coord.join(threads)

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
