from keras.models import Model

from keras.layers import Input, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K

from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D


def trans_layer(x, concat_axis, filter,rate=None, weight_decay=1E-4):
    #apply BatchNormalization
    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    # apply 1*1 Conv2D,
    x = Conv2D(filter, (1, 1),
               kernel_initializer="he_uniform",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    #dropout
    if rate:
        x = Dropout(rate)(x)
    # maxpooling2D
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x

#network in denseblock
def network(x, concat_axis, filter,
                 rate=None, weight_decay=1E-4):
    # apply BatchNormalization
    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    # apply 3*3 Conv2D,
    x = Conv2D(filter, (3, 3),
               kernel_initializer="he_uniform",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    # dropout
    if rate:
        x = Dropout(rate)(x)
    return x

 #build a model with nb_layers of network appended
def denseblock(x, concat_axis, nb_layers, filters, growth_rate,
               dropout_rate=None, weight_decay=1E-4):
    layers = [x]   # x: model
    for i in range(nb_layers):
        x = network(x, concat_axis, growth_rate,
                         dropout_rate, weight_decay)
        layers.append(x)
        x = Concatenate(axis=concat_axis)(layers)
        filters += growth_rate
    return x, filters


def DenseNet(classes, img, depth, dense_block, growth_rate,
             filters, dropout_rate=None, weight_decay=1E-4):

    if K.image_dim_ordering() == "th":
        concat_axis = 1
    elif K.image_dim_ordering() == "tf":
        concat_axis = -1

    model_input = Input(shape=img)

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    layers = int((depth - 4) / 3)

    # Initial convolution
    x = Conv2D(filters, (3, 3),
               kernel_initializer="he_uniform",
               padding="same",
               name="initial_conv2D",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(model_input)

    # Add dense blocks
    for block_idx in range(dense_block - 1):
        x, filters = denseblock(x, concat_axis, layers,
                                  filters, growth_rate,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
        # add transition
        x = trans_layer(x, filters, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)

    # The last denseblock does not have a transition
    x, filters = denseblock(x, concat_axis, layers,
                              filters, growth_rate,
                              dropout_rate=dropout_rate,
                              weight_decay=weight_decay)

    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D(data_format=K.image_data_format())(x)
    x = Dense(classes,
              activation='softmax',
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(x)

    densenet = Model(inputs=[model_input], outputs=[x], name="DenseNet")

    return densenet