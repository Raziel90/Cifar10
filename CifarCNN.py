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
batch_len = 150
examples_per_mode = {'train': 45000, 'validation': 5000, 'test': 10000}
INIT_L_RATE = 5e-1
LEARNING_RATE_DECAY_FACTOR = 0.1
NUM_EPOCHS_PER_DECAY = 350.0
num_steps = 10000

# Definition of the Architecture
patch_size = [5, 3, 5, 3, 3]
depth = [32, 32, 64, 64, 128, 128]
want_pooling = [True, False, True, False, True]
want_norm = [False, True, False, True, False]
pool_strides = [2, 2, 2, 2, 2]
weight_decay = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4]
keep_prob = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
flattening_value = np.prod(
    np.array(pool_strides)[np.array(want_pooling)]
).tolist()  # to flatten the convolution output


def activation_info(layer):
    tensor_name = layer.op.name
    tf.summary.histogram(tensor_name + '/activations', layer)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(layer))


def accuracy(predictions, labels):
    """
    return (100.0 * np.sum(np.argmax(predictions, 1) == labels)
        / labels.shape[0])
    """

    return (100.0 * tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.argmax(predictions, 1),
                labels),
            tf.float32)))

    # return tf.metrics.accuracy(labels, tf.argmax(predictions, 1))


def conv_params(patch_sz, prev_depth, curr_depth, stddev=5e-2):
    weights = tf.get_variable("weights",
                              [patch_sz, patch_sz, prev_depth, curr_depth],
                              initializer=tf.random_normal_initializer(
                                  stddev=0.1)
                              # collections=[tf.GraphKeys.WEIGHTS]
                              )

    bias = tf.get_variable("biases", [curr_depth],
                           initializer=tf.constant_initializer(0.1)
                           # collections=[tf.GraphKeys.BIASES]
                           )
    return weights, bias


def define_conv(x, W, b, stride=1, keep_prob=1):

    # print(x.get_shape().as_list(), W.get_shape().as_list())
    conv1 = tf.nn.conv2d(input=x, filter=W, strides=[1, stride, stride, 1],
                         padding="SAME",
                         name='conv_op')
    conv1 = tf.nn.bias_add(conv1, b)
    conv1 = tf.nn.relu(conv1)
    if keep_prob < 1:
        conv1 = tf.nn.dropout(conv1, keep_prob=keep_prob)
    return conv1


def define_max_pool(x, k=2):
    return tf.nn.max_pool(x, strides=[1, k, k, 1],
                          ksize=[1, k, k, 1], padding="SAME",
                          name='max_pooling')


def define_conv_layer(input_tensor, weights, bias, conv_stride, pool_stride,
                      want_pooling=True, keep_prob=1.0):

    conv_layer = define_conv(input_tensor, weights,
                             bias, conv_stride, keep_prob)
    if want_pooling:
        return define_max_pool(conv_layer, pool_stride)
    else:
        return conv_layer


def local_normalization(input_tensor,
                        depth_radius=1.0, bias=1, beta=0.75, alpha=0.001 / 9.0,
                        want_norm=True):
    if want_norm:
        return tf.nn.lrn(input_tensor,
                         depth_radius=depth_radius,
                         bias=bias, beta=beta, alpha=alpha)
    else:
        return input_tensor


def l2_regularize(weights, bias, decay):
    loss_weight = decay * tf.nn.l2_loss(weights, name='bias_weight')
    loss_bias = decay * tf.nn.l2_loss(bias, name='bias_loss')
    tf.add_to_collection('losses_w', loss_weight)
    tf.add_to_collection('losses_b', loss_bias)


def get_decay_loss():
    return (
        tf.add_n(tf.get_collection('losses_w'), name='total_loss_w'),
        tf.add_n(tf.get_collection('losses_b'), name='total_loss_b')
    )


def define_training(logits, labels, start_lrate, global_step):

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=labels,
            name='main_loss'
        )
    )
    reg_loss = loss + get_decay_loss()[0]
    num_batches_per_epoch = examples_per_mode['train'] / batch_len
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    learning_rate = tf.train.exponential_decay(
        learning_rate=start_lrate, global_step=global_step,
        decay_steps=decay_steps, decay_rate=LEARNING_RATE_DECAY_FACTOR,
        staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    optimizer = tf.train.AdamOptimizer(
        learning_rate).minimize(reg_loss, global_step=global_step)

    return reg_loss, optimizer


def define_model(data, training_flg=False):
    keep1 = (training_flg * keep_prob + (not training_flg)).tolist()

    # dataph = tf.placeholder(data.dtype, data.shape)
    # dataph.assign
    hidden_weights = [None] * len(patch_size)
    hidden_bias = [None] * len(patch_size)
    conv_l = [None] * len(patch_size)
    with tf.variable_scope('conv1') as scope:
        hidden_weights[0], hidden_bias[0] = conv_params(
            patch_size[0], num_channels, depth[0])
        conv_l[0] = define_conv_layer(data,
                                      hidden_weights[0], hidden_bias[0],
                                      1, pool_strides[0],
                                      want_pooling=want_pooling[0],
                                      keep_prob=keep1[0])
        conv_l[0] = local_normalization(
            conv_l[0], want_norm=want_norm[0])
        l2_regularize(hidden_weights[0], hidden_bias[0], weight_decay[0])

    for layer in range(1, len(patch_size)):
        with tf.variable_scope('conv' + str(layer + 1)) as scope:
            hidden_weights[layer], hidden_bias[layer] = conv_params(
                patch_size[layer], depth[layer - 1], depth[layer])
            conv_l[layer] = define_conv_layer(conv_l[layer - 1],
                                              hidden_weights[
                                                  layer], hidden_bias[layer],
                                              1, pool_strides[layer],
                                              want_pooling=want_pooling[layer],
                                              keep_prob=keep1[layer])
            conv_l[layer] = local_normalization(
                conv_l[layer], want_norm=want_norm[layer])
            l2_regularize(hidden_weights[layer], hidden_bias[
                layer], weight_decay[layer])

    with tf.variable_scope('out_layer') as scope:
        shape = conv_l[-1].get_shape().as_list()
        reshaped = tf.reshape(
            conv_l[-1], [shape[0], shape[1] * shape[2] * shape[3]])
        out_weights = tf.get_variable("weights",
                                      [(image_size // flattening_value) *
                                       (image_size // flattening_value) *
                                       depth[4], num_labels],
                                      initializer=tf.random_normal_initializer(
                                      )
                                      )
        out_bias = tf.get_variable("biases", [num_labels],
                                   initializer=tf.constant_initializer(0.1)
                                   )

    return tf.nn.bias_add(tf.matmul(reshaped, out_weights), out_bias, name='FULL-MODEL-Logits')


graph = tf.Graph()
with graph.as_default() as g:

    summary_writer = tf.summary.FileWriter('./log', g)
    tf_train_dataset_bat, tf_train_labels_bat = make_batch(
        batch_len, 'train', basepath='./cifar-10-batches-py')
    tf_valid_dataset_bat, tf_valid_dataset_labels_bat = make_batch(
        batch_len, 'validation', basepath='./cifar-10-batches-py')
    tf_test_dataset_bat, tf_test_dataset_labels_bat = make_batch(
        batch_len, 'test', basepath='./cifar-10-batches-py')
    # The op for initializing the variables.

    global_step = tf.Variable(0)
    init_learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    """
    tf_train_data = tf.placeholder(tf.float32, shape=(
        batch_len, image_size, image_size, num_channels), name='Train_data')
    tf_train_labels = tf.placeholder(tf.int64, shape=(
        batch_len), name='train_labels')

    tf_valid_dataset = tf.placeholder(tf.float32, shape=(
        examples_per_mode['validation'], image_size, image_size, num_channels),
        name='Valid_data')
    tf_test_dataset = tf.placeholder(tf.float32, shape=(
        examples_per_mode['test'], image_size, image_size, num_channels),
        name='Test_data')
	"""

    with tf.variable_scope('training') as scope:
        train_model = define_model(tf_train_dataset_bat, training_flg=True)
        loss, optimizer = define_training(
            train_model, tf_train_labels_bat, init_learning_rate, global_step)

        scope.reuse_variables()

        train_prediction = tf.nn.softmax(train_model)
        valid_prediction = tf.nn.softmax(define_model(tf_valid_dataset_bat))
        test_prediction = tf.nn.softmax(define_model(tf_test_dataset_bat))

        train_accuracy = accuracy(train_prediction, tf_train_labels_bat)
        valid_accuracy = accuracy(
            valid_prediction, tf_valid_dataset_labels_bat)
        test_accuracy = accuracy(test_prediction, tf_test_dataset_labels_bat)
    # tr_accuracy=tf.metrics.accuracy(predictions=tf.argmax(train_prediction, 1), labels=tf.argmax(tf_train_labels,1))
    # valid_accuracy=tf.metrics.accuracy(predictions=tf.argmax(valid_prediction, 1),labels=tf.argmax(tf_valid_dataset_labels,1))
    # test_accuracy=tf.metrics.accuracy(predictions=tf.argmax(test_prediction,
    # 1),labels=tf.argmax(tf_test_dataset_labels,1))

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    merged = tf.summary.merge_all()

"""
valid_data, valid_labels, test_data, test_labels = sess.run(
    [tf_valid_dataset_bat,
     tf_valid_dataset_labels_bat,
     tf_test_dataset_bat,
     tf_test_dataset_labels_bat])
"""

with tf.Session(graph=graph) as sess:

    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print('Initialized')
    tr_acc = []
    valid_acc = []
    test_acc = []
    loss_in_time = []
    val_pred = []
    # print(valid_data[0].dtype, np.array(valid_data).shape)
    for step in range(num_steps + 1):

        # batch_data, batch_labels = sess.run(
        #    [tf_train_dataset_bat, tf_train_labels_bat])
        # print(batch_labels.shape)
        feed_dict = {  # tf_train_data: batch_data,
            # tf_train_labels: batch_labels,
            # tf_valid_dataset: np.array(valid_data).astype(float),
            # tf_test_dataset: test_data,
            init_learning_rate: INIT_L_RATE}
        _, l, predictions = sess.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        # val_pred += [valid_model.eval()]
        # if len(val_pred) > 1:
        #    print(np.sum(val_pred[-1] - val_pred[-2]))
        if (step % 1000 == 0):
            # summary = sess.run([merged])
            # for var in tf.trainable_variables():
            #    print(np.sum(np.array(var.eval())))

            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()
            # summary_writer.add_summary(summary, step)

            # tr_a = accuracy(predictions, batch_labels)
            # val_a = accuracy(valid_prediction.eval(), valid_labels)
            tr_a, val_a = sess.run([train_accuracy, valid_accuracy])
            tr_acc += tr_a
            valid_acc += val_a
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % tr_a)
            print('Validation accuracy: %.1f%%' % val_a)
    # accuracy(test_prediction.eval(), test_labels)
    test_acc = sess.run([test_accuracy])
    print(test_acc)
    #print('Test accuracy: %.1f%%' % test_acc)
    coord.request_stop()
    coord.join(threads)
    plt.plot(np.array(tr_acc))


"""
with tf.Session(graph=graph) as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    final_image, final_lab = sess.run([tf_train_dataset_bat, tf_train_labels_bat])
    print(len(final_image), final_image.shape, final_lab.shape)
    print(final_lab)
    #for image in final_image:
    print(np.mean(final_image,axis=tuple(range(1,4))))
    plt.imshow(final_image[10, :, :, ::-1])
    plt.show()
    coord.request_stop()
    coord.join(threads)
"""
