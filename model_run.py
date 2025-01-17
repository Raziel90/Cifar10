"""
"""
import numpy as np
import tensorflow as tf
from CifarCNN_Architecture import define_training, define_model, accuracy
from TFRecord_load_Cifar import make_batch
# import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
num_steps = 100000
batch_len = 200
INIT_L_RATE = 1e-3

graph = tf.Graph()
with graph.as_default() as g:

    tf_train_dataset_bat, tf_train_labels_bat = make_batch(
        batch_len, 'train', basepath='./cifar-10-batches-py')
    tf_valid_dataset_bat, tf_valid_dataset_labels_bat = make_batch(
        batch_len, 'validation', basepath='./cifar-10-batches-py')
    tf_test_dataset_bat, tf_test_dataset_labels_bat = make_batch(
        batch_len, 'test', basepath='./cifar-10-batches-py')
    # The op for initializing the variables.

    global_step = tf.Variable(0, trainable=False)
    #  init_learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    with tf.variable_scope('training') as scope:
        train_model = define_model(tf_train_dataset_bat, training_flg=True)
        loss, optimizer = define_training(
            train_model, tf_train_labels_bat, INIT_L_RATE, global_step)

        scope.reuse_variables()

        train_prediction = tf.nn.softmax(define_model(tf_train_dataset_bat))
        valid_prediction = tf.nn.softmax(define_model(tf_valid_dataset_bat))
        test_prediction = tf.nn.softmax(define_model(tf_test_dataset_bat))

    train_accuracy = accuracy(train_prediction, tf_train_labels_bat)
    valid_accuracy = accuracy(valid_prediction, tf_valid_dataset_labels_bat)
    test_accuracy = accuracy(test_prediction, tf_test_dataset_labels_bat)

    with tf.name_scope('accuracy_summaries'):
        tf.summary.scalar('tr_accuracy', train_accuracy)
        tf.summary.scalar('test_accuracy', test_accuracy)
        tf.summary.scalar('tf_valid_accuracy', valid_accuracy)

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
    steps = []
    early_stop = {'max_count' : 10, 'count' : 0, 'best_accuracy' : 0.0 ,
                  'min_step' : 20000}
    # clean previous executions
    with open('dump/training', 'w') as myfile:
        myfile.write('')
    with open('dump/valid', 'w') as myfile:
        myfile.write('')
    with open('dump/valid', 'w') as myfile:
        myfile.write('')

    # print(valid_data[0].dtype, np.array(valid_data).shape)
    for step in range(num_steps + 1):

        feed_dict = {}
        _, l, predictions = sess.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)

        if (step % 100 == 0):
            steps += [step]
            tr_a, val_a, summary = sess.run(
                [train_accuracy, valid_accuracy, merged])
            tr_acc += tr_a
            valid_acc += val_a

            with open('dump/training', 'a') as myfile:
                myfile.write(str(step) + ',' + str(tr_a) + '\n')
            with open('dump/valid', 'a') as myfile:
                myfile.write(str(step) + ',' + str(val_a) + '\n')
            with open('dump/valid', 'a') as myfile:
                myfile.write(str(step) + ',' +
                             str(sess.run([test_accuracy])[0]) + '\n')

            summary_writer = tf.summary.FileWriter(
                'log/', sess.graph)
            summary_writer.add_summary(summary, step)
            summary_writer.flush()
            summary_writer.close()

            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % tr_a)
            print('Validation accuracy: %.1f%%' % val_a)
            if step > early_stop['min_step']:
                if val_a > early_stop['best_accuracy']:
                    early_stop['best_accuracy'] = val_a
                    early_stop['count'] = 0
                else:
                    early_stop['count']+=1
                    if early_stop['count'] > early_stop['max_count']:
                        print('Early stopping')

            """
            plt.plot(x=np.array(steps), y=np.array(tr_acc))
            plt.xlabel('batch_steps')
            plt.ylabel('train_accuracy')
            plt.axis([0, step + 1, 0, 101])
            plt.show(block=False)
            """
    # accuracy(test_prediction.eval(), test_labels)
    test_acc = sess.run([test_accuracy])
    # print(test_acc)
    print('Test accuracy: %.1f%%' % test_acc[0])
    coord.request_stop()
    coord.join(threads)
