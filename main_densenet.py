from __future__ import print_function
import time
import json
import densenet
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob
import os


def run_denseset(batch_size,
                nb_epoch,
                depth,
                nb_dense_block,
                nb_filter,
                growth_rate,
                dropout_rate,
                learning_rate,
                weight_decay,
                plot_architecture):
    """
    :param batch_size: int -- batch size
    :param nb_epoch: int -- number of training epochs
    :param depth: int -- network depth
    :param nb_dense_block: int -- number of dense blocks
    :param nb_filter: int -- initial number of conv filter
    :param growth_rate: int -- number of new filters added by conv layers
    :param dropout_rate: float -- dropout rate
    :param learning_rate: float -- learning rate
    :param weight_decay: float -- weight decay
    :param plot_architecture: bool -- whether to plot network architecture
    """

    ###################
    # Data processing #
    ###################
    output0_path = "/Users/xinxinyang/data1/normal/"
    output1_path = "/Users/xinxinyang/data1/tumor/"

    X = []
    Y = []

    for img_file in glob(output0_path + '*.npy'):
        img = np.load(img_file, allow_pickle=True).astype(np.float32)
        X.append(img)
        Y.append(0)
    for img_file in glob(output1_path + '*.npy'):
        img = np.load(img_file, allow_pickle=True).astype(np.float32)
        X.append(img)
        Y.append(1)

    d1_X = np.array([np.reshape(x, (3, 128, 128)) for x in X])
    X_train, X_test, y_train, y_test = \
        train_test_split(d1_X, Y, test_size=0.2, random_state=20, stratify=Y)


    nb_classes = len(np.unique(y_train))
    img_dim = X_train.shape[1:]

    if K.image_data_format() == "channels_first":
        n_channels = X_train.shape[1]
    else:
        n_channels = X_train.shape[-1]

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Normalisation
    X = np.vstack((X_train, X_test))
    # 2 cases depending on the image ordering
    if K.image_data_format() == "channels_first":
        for i in range(n_channels):
            mean = np.mean(X[:, i, :, :])
            std = np.std(X[:, i, :, :])
            X_train[:, i, :, :] = (X_train[:, i, :, :] - mean) / std
            X_test[:, i, :, :] = (X_test[:, i, :, :] - mean) / std

    elif K.image_data_format() == "channels_last":
        for i in range(n_channels):
            mean = np.mean(X[:, :, :, i])
            std = np.std(X[:, :, :, i])
            X_train[:, :, :, i] = (X_train[:, :, :, i] - mean) / std
            X_test[:, :, :, i] = (X_test[:, :, :, i] - mean) / std

    ###################
    # Construct model #
    ###################

    model = densenet.DenseNet(nb_classes,
                               img_dim,
                               depth,
                               nb_dense_block,
                               growth_rate,
                               nb_filter,
                               dropout_rate=dropout_rate,
                               weight_decay=weight_decay)
    # Model output
    model.summary()

    # Build optimizer
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=["accuracy"])

    if plot_architecture:
        from keras.utils.visualize_util import plot
        plot(model, to_file='./figures/densenet_archi.png', show_shapes=True)

    ####################
    # Network training #
    ####################

    print("Training")

    list_train_loss = []
    list_test_loss = []
    list_learning_rate = []

    for e in range(nb_epoch):

        if e == int(0.5 * nb_epoch):
            K.set_value(model.optimizer.lr, np.float32(learning_rate / 10.))

        if e == int(0.75 * nb_epoch):
            K.set_value(model.optimizer.lr, np.float32(learning_rate / 100.))

        split_size = batch_size
        num_splits = X_train.shape[0] / split_size
        arr_splits = np.array_split(np.arange(X_train.shape[0]), num_splits)

        l_train_loss = []
        start = time.time()

        for batch_idx in arr_splits:

            X_batch, Y_batch = X_train[batch_idx], Y_train[batch_idx]
            train_logloss, train_acc = model.train_on_batch(X_batch, Y_batch)

            l_train_loss.append([train_logloss, train_acc])

        test_logloss, test_acc = model.evaluate(X_test,
                                                Y_test,
                                                verbose=0,
                                                batch_size=64)
        list_train_loss.append(np.mean(np.array(l_train_loss), 0).tolist())
        list_test_loss.append([test_logloss, test_acc])
        list_learning_rate.append(float(K.get_value(model.optimizer.lr)))
        # to convert numpy array to json serializable
        print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))

        d_log = {}
        d_log["train_loss"] = list_train_loss
        d_log["test_loss"] = list_test_loss

        json_file = os.path.join('./log/densenet.json')
        with open(json_file, 'w') as fp:
            json.dump(d_log, fp, indent=4, sort_keys=True)


if __name__ == '__main__':
    list_dir = ["./log", "./figures"]
    for d in list_dir:
        if not os.path.exists(d):
            os.makedirs(d)

    run_denseset(64,30,7,3, 16, 12, 0.2, 1E-3, 1e-4, False)
