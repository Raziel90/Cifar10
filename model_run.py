"""
"""
import tensorflow as tf
from CifarCNN import define_training, define_model, accuracy
from TFRecord_Cifar_load import make_batch
from matplotlib import plot, draw, show
num_steps = 10000
batch_len = 150
INIT_L_RATE = 1e-4

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

    global_step = tf.Variable(0, trainable=False)
    init_learning_rate = tf.placeholder(tf.float32, name='learning_rate')

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
        feed_dict = {
            init_learning_rate: INIT_L_RATE}
        _, l, predictions = sess.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        # val_pred += [valid_model.eval()]
        # if len(val_pred) > 1:
        #    print(np.sum(val_pred[-1] - val_pred[-2]))
        if (step % 500 == 0):
            # summary = sess.run([merged])
            # for var in tf.trainable_variables():
            #    print(np.sum(np.array(var.eval())))

            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()
            # summary_writer.add_summary(summary, step)
            tr_a, val_a = sess.run([train_accuracy, valid_accuracy])
            tr_acc += tr_a
            valid_acc += val_a
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % tr_a)
            print('Validation accuracy: %.1f%%' % val_a)
            plt.plot(x=range(step),y=np.array(tr_acc))
            show(block=False)
    # accuracy(test_prediction.eval(), test_labels)
    test_acc = sess.run([test_accuracy])
    print(test_acc)
    #print('Test accuracy: %.1f%%' % test_acc)
    coord.request_stop()
    coord.join(threads)

