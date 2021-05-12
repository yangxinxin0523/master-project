from __future__ import division

from keras.models import Model
from keras.layers import Input,Activation,Dense,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

#build a conv -> BN -> relu block
def conv_bn_relu(**conv_params):
    def f(input):
        conv = Conv2D(filters=conv_params["filters"],
                      kernel_size=conv_params["kernel_size"],
                      strides=conv_params.setdefault("strides", (1, 1)),
                      padding=conv_params.setdefault("padding", "same"),
                      kernel_initializer=conv_params.setdefault("kernel_initializer", "he_normal"),
                      kernel_regularizer=conv_params.setdefault("kernel_regularizer", l2(1.e-4)))(input)
        return Activation("relu")(BatchNormalization(axis=CHANNEL_AXIS)(conv))
    return f

# build a BN -> relu -> conv block.
def bn_relu_conv(**conv_params):
    def f(input):
        activation = Activation("relu")(BatchNormalization(axis=CHANNEL_AXIS)(input))
        return Conv2D(filters=conv_params["filters"],
                      kernel_size=conv_params["kernel_size"],
                      strides=conv_params.setdefault("strides", (1, 1)),
                      padding=conv_params.setdefault("padding", "same"),
                      kernel_initializer=conv_params.setdefault("kernel_initializer", "he_normal"),
                      kernel_regularizer=conv_params.setdefault("kernel_regularizer", l2(1.e-4)))(activation)

    return f

#Adds a shortcut between input and residual block and merges them with "sum"
def shortcut(input, residual):
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])

#Builds a residual block with repeating bottleneck blocks.
def residual_block(block_function, filters, repetitions, is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f

# Basic 3 X 3 convolution blocks for use on resnets
def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return shortcut(input, residual)

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3

class ResnetBuilder(object):
    def build(input_shape, num_outputs):
        repetitions = [2, 2, 2, 2]
        _handle_dim_ordering()

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])


        input = Input(shape=input_shape)
        conv1 = conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = residual_block(basic_block, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = Activation("relu")(BatchNormalization(axis=CHANNEL_AXIS)(
            block))

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax")(flatten1)

        model = Model(inputs=input, outputs=dense)
        return model
