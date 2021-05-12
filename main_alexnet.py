import os
from alexnet import AlexNet
from alexnet_datagenerator import ImageDataGenerator
from datetime import datetime
import glob
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pylab as plt
import numpy as np

def main():
    # 初始参数设置
    learning_rate = 1e-3
    num_epochs = 30
    train_batch_size = 64
    test_batch_size = 64
    dropout_rate = 0.5
    num_classes = 2
    display_step = 20

    image_format = 'png'
    file_name_of_class = ['normal',
                          'tumor']
    train_dataset_paths = ['/Users/xinxinyang/imagedata/train/normal/',
                           '/Users/xinxinyang/imagedata/train/tumor/']  #
    test_dataset_paths = ['/Users/xinxinyang/imagedata/predict/normal/',
                          '/Users/xinxinyang/imagedata/predict/tumor/']  #

    train_image_paths = []
    train_labels = []
    for train_dataset_path in train_dataset_paths:
        length = len(train_image_paths)
        train_image_paths[length:length] = np.array(
            glob.glob(train_dataset_path + '*.' + image_format)).tolist()
    for image_path in train_image_paths:
        for i in range(num_classes):
            if file_name_of_class[i] in image_path:
                train_labels.append(i)
                break

    test_image_paths = []
    test_labels = []

    for test_dataset_path in test_dataset_paths:
        length = len(test_image_paths)
        test_image_paths[length:length] = np.array(
            glob.glob(test_dataset_path + '*.' + image_format)).tolist()
    for image_path in test_image_paths:
        for i in range(num_classes):
            if file_name_of_class[i] in image_path:
                test_labels.append(i)
                break

    train_data = ImageDataGenerator(
        images=train_image_paths,
        labels=train_labels,
        batch_size=train_batch_size,
        num_classes=num_classes,
        image_format=image_format,
        shuffle=True)

    test_data = ImageDataGenerator(
        images=test_image_paths,
        labels=test_labels,
        batch_size=test_batch_size,
        num_classes=num_classes,
        image_format=image_format,
        shuffle=False)


    with tf.name_scope('input'):
        train_iterator = tf.data.Iterator.from_structure(
            train_data.data.output_types,
                                                 train_data.data.output_shapes)
        training_initalizer = train_iterator.make_initializer(train_data.data)

        test_iterator = tf.data.Iterator.from_structure(
            test_data.data.output_types,
                                                test_data.data.output_shapes)
        testing_initalizer = test_iterator.make_initializer(test_data.data)
        train_next_batch = train_iterator.get_next()
        test_next_batch = test_iterator.get_next()

    x = tf.placeholder(tf.float32, [None, 227, 227, 3])
    y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    # alexnet
    model = AlexNet(x, keep_prob, num_classes)
    output_y = model.fc8

    # loss
    with tf.name_scope('loss'):
        loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_y,
                                                       labels=y))
    # optimizer
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

    # accuracy
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.argmax(output_y, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    # Tensorboard
    tf.summary.scalar('loss', loss_op)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary = tf.summary.merge_all()

    train_batches_per_epoch = int(
            np.floor(len(train_labels) / train_batch_size))
    test_batches_per_epoch = int(
            np.floor(len(test_labels) / test_batch_size))

    print("train_batches_per_epoch = ", train_batches_per_epoch)
    print("test_batches_per_epoch = ", test_batches_per_epoch)

    # Start training
    with tf.Session() as sess:
        sess.run(init)

        list_train_loss = []
        list_train_acc = []
        list_test_loss = []
        list_test_acc = []
        for epoch in range(num_epochs):
            sess.run(training_initalizer)
            print(
                "Epoch number: {} start".format(epoch + 1))
            for step in range(train_batches_per_epoch):
                img_batch, label_batch = sess.run(train_next_batch)

                sess.run(train_op, feed_dict={x: img_batch, y: label_batch,
                                              keep_prob: dropout_rate})

                if step % display_step == 0:
                    s, train_acc, train_loss = sess.run(
                        [merged_summary, accuracy, loss_op],
                        feed_dict={x: img_batch, y: label_batch,
                                   keep_prob: 1.0})

            sess.run(testing_initalizer)
            val_acc = 0
            val_loss = 0
            test_count = 0
            for _ in range(test_batches_per_epoch):
                img_batch, label_batch = sess.run(test_next_batch)
                acc, val_batch_loss = sess.run([accuracy, loss_op],
                                               feed_dict={x: img_batch,
                                                          y: label_batch,
                                                          keep_prob: 1.0})
                val_acc += acc
                val_loss += val_batch_loss
                test_count += 1
            val_acc /= test_count
            val_loss /= test_count

            print(
                "%s epoch:%d,train acc:%.4f,train loss:%.4f,val acc:%.4f,val loss:%.4f"
                % (datetime.now(), epoch + 1, train_acc, train_loss, val_acc,
                   val_loss))


            list_train_loss.append(train_loss)
            list_train_acc.append(train_acc)
            list_test_loss.append(val_loss)
            list_test_acc.append(val_acc)

            # this epoch is over
            print("{}: Epoch number: {} end".format(datetime.now(), epoch + 1))

        train_accuracy = 100 * (np.array(list_train_acc))
        test_accuracy = 100 * (np.array(list_test_acc))

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_ylabel('Accuracy')
        ax1.plot(train_accuracy, color="tomato", linewidth=2,
                 label='train_acc')
        ax1.plot(test_accuracy, color="steelblue", linewidth=2,
                 label='test_acc')
        ax1.legend(loc=0)

        train_loss = np.array(list_train_loss)
        test_loss = np.array(list_test_loss)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Loss')
        ax2.plot(train_loss, '--', color="tomato", linewidth=2,
                 label='train_loss')
        ax2.plot(test_loss, '--', color="steelblue", linewidth=2,
                 label='test_loss')
        ax2.legend(loc=1)

        ax1.grid(True)

        plt.show()
        plt.clf()
        plt.close()


if __name__ == '__main__':
    main()