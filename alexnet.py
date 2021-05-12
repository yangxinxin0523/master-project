'''
alexnet architecture is modified from kaggle dog and cat competition
https://zhuanlan.zhihu.com/p/106752935
'''
import tensorflow as tf

def conv(x,filter_height,filter_width,num_filters,stride_y,stride_x,name,padding="SAME",groups=1):
    #get input channel from tensor
    input = int(x.get_shape()[-1])

    convolve = lambda i,k:tf.nn.conv2d(i,k,strides=[1,stride_y,stride_x,1],padding=padding)
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable("weights",shape=[filter_height,filter_width,
                                                   input/groups,num_filters])
        biases = tf.get_variable("biases",shape=[num_filters])

    if groups == 1:
        conv = convolve(x,weights)
    else:
        input_groups = tf.split(axis=3,num_or_size_splits=groups,value=x)
        weight_groups = tf.split(axis=3,num_or_size_splits=groups,value=weights)
        output_groups = [convolve(i,k) for i,k in zip(input_groups,weight_groups)]
        #connect convolution layers
        conv = tf.concat(axis=3,values=output_groups)
    bias = tf.reshape(tf.nn.bias_add(conv,biases),tf.shape(conv))
    #relu activation function
    relu = tf.nn.relu(bias,name=scope.name)
    return relu

def fc(x,num_in,num_out,name,relu=True):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable("weights",shape=[num_in,num_out],trainable=True)
        biases = tf.get_variable("biases",[num_out],trainable=True)
        fc_out = tf.nn.xw_plus_b(x,weights,biases,name=scope.name)
    if relu:
        fc_out = tf.nn.relu(fc_out)
    return fc_out

class AlexNet(object):
    def __init__(self,x,keep_prob,num_classes,weights_path="default"):
        self.X = x
        self.KEEP_PROB = keep_prob
        self.NUM_CLASSES = num_classes
        self.WEIGHTS_PATH = weights_path
        self.create()

    def create(self):

        conv1 = conv(self.X,11,11,96,4,4,padding="VALID",name="conv1")
        norm1 = tf.nn.local_response_normalization(conv1, depth_radius=2,
                                                      alpha=2e-05, beta=0.75,
                                                      bias=1.0, name="norm1")
        pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1],
                                  strides=[1, 2, 2, 1],
                                  padding="VALID", name="pool1")


        conv2 = conv(pool1,5,5,256,1,1,groups=2,name="conv2")
        norm2 = tf.nn.local_response_normalization(conv2, depth_radius=2,
                                           alpha=2e-05, beta=0.75,
                                           bias=1.0, name="norm2")
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1],
                       padding="VALID", name="pool2")


        conv3 = conv(pool2,3,3,384,1,1,name="conv3")


        conv4 = conv(conv3,3,3,384,1,1,groups=2,name="conv4")


        conv5 = conv(conv4,3,3,256,1,1,groups=2,name="conv5")
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1],
                       padding="VALID", name="pool5")


        flattened = tf.reshape(pool5,[-1,6*6*256])
        fc6 = fc(flattened,6*6*256,4096,name="fc6")
        dropout6 = tf.nn.dropout(fc6, self.KEEP_PROB)


        fc7 = fc(dropout6,4096,4096,name="fc7")
        dropout7 = tf.nn.dropout(fc7, self.KEEP_PROB)

        self.fc8 = fc(dropout7,4096,self.NUM_CLASSES,relu=False,name="fc8")
