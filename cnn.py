import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob
import os
import json
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


sess = tf.Session()

# class0_path = "/Users/xinxinyang/data/normal_cut_1/"
output0_path = "/Users/xinxinyang/data1/normal/"

# class1_path = "/Users/xinxinyang/data/tumor_cut_1/"
output1_path = "/Users/xinxinyang/data1/tumor/"


# def save_image(input, output):
#     file_list=glob(input+"*.png")
#     for fcount, img_file in enumerate(file_list):
#         try:
#             image = cv2.imread(img_file)
#         except:
#             print("can't read file")
#         image = cv2.resize(image, (128, 128), cv2.INTER_LINEAR)
#         np.save(os.path.join(output,"images_%04d.npy" % fcount),image)
#
# save_image(class0_path, output0_path)
# save_image(class1_path, output1_path)

X = []
Y = []

for img_file in glob(output0_path + '*.npy'):
    img = np.load(img_file,allow_pickle=True).astype(np.float64)
    X.append(img)
    Y.append(0)
for img_file in glob(output1_path + '*.npy'):
    img = np.load(img_file,allow_pickle=True).astype(np.float64)
    X.append(img)
    Y.append(1)

d1_X = np.array([np.reshape(x,(128,128,3)) for x in X])
X_train, X_test, y_train, y_test = \
        train_test_split(d1_X, Y, test_size=0.2, random_state=20, stratify=Y)

batch_size = 100
learning_rate = 0.005
evaluation_size = 500
a = X_train[0]
image_width = X_train[0].shape[0]
image_height = X_train[0].shape[1]
target_size = max(y_train)+1
num_channels = 3
generations = 50
eval_every = 5
conv1_features = 25
conv2_features = 50
max_pool_size1 = 2
max_pool_size2 = 2
fully_connected_size1 = 100

x_input_shape = (batch_size, image_width, image_height, num_channels)
x_input = tf.placeholder(tf.float64, shape= [100,128,128,3])
y_target = tf.placeholder(tf.int32, shape= 100)
eval_input_shape = (evaluation_size, image_width, image_height,num_channels)
eval_input = tf.placeholder(tf.float64, shape=[500,128,128,3])
eval_target = tf.placeholder(tf.int32, shape= 500)

conv1_weight = tf.Variable(tf.truncated_normal([4, 4, num_channels,
                                                conv1_features], stddev=0.1,
                                               dtype = tf.float64))
conv1_bias = tf.Variable(tf.zeros([conv1_features], dtype=tf.float64))

conv2_weight = tf.Variable(tf.truncated_normal([4, 4, 25,
                                                conv2_features], stddev=0.1,
                                               dtype = tf.float64))
conv2_bias = tf.Variable(tf.zeros([conv2_features], dtype=tf.float64))

resulting_width = image_width // (max_pool_size1 * max_pool_size2)
resulting_height = image_height // (max_pool_size1 * max_pool_size2)

full1_input_size = resulting_width * resulting_height * conv2_features
full1_weight =tf.Variable(tf.truncated_normal([full1_input_size,
                                               fully_connected_size1],
                                              stddev=0.1, dtype=tf.float64))

full1_bias = tf.Variable(tf.truncated_normal([fully_connected_size1],
                                             stddev=0.1, dtype=tf.float64))

full2_weight = tf.Variable(tf.truncated_normal([fully_connected_size1,
                                                target_size], stddev=0.1,
                                               dtype=tf.float64))

full2_bias = tf.Variable(tf.truncated_normal([target_size], stddev=0.1,
                                             dtype=tf.float64))

def my_conv_net(input_data):
    conv1 = tf.nn.conv2d(input_data, conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
    max_pool1 = tf.nn.max_pool(relu1, ksize=[1, max_pool_size1,
                                             max_pool_size1, 1], strides=[1, max_pool_size1, max_pool_size1, 1], padding='SAME')


    conv2 = tf.nn.conv2d(max_pool1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
    max_pool2 = tf.nn.max_pool(relu2, ksize=[1, max_pool_size1,
                                             max_pool_size1, 1], strides=[1, max_pool_size2, max_pool_size2, 1], padding='SAME')

    final_conv_shape = max_pool2.get_shape().as_list()
    final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
    flat_output = tf.reshape(max_pool2, [final_conv_shape[0], final_shape])

    fully_connected1 = tf.nn.relu(tf.add(tf.matmul(flat_output,full1_weight), full1_bias))
    final_model_output = tf.add(tf.matmul(fully_connected1, full2_weight), full2_bias)

    return (final_model_output)

model_output = my_conv_net(x_input)
test_model_output = my_conv_net(eval_input)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=y_target))

prediction = tf.nn.softmax(model_output)

test_prediction = tf.nn.softmax(test_model_output)

def get_accuracy(logits, targets):
    batch_predictions = np.argmax(logits, axis=1)
    num_correct = np.sum(np.equal(batch_predictions, targets))
    return(100. * num_correct/batch_predictions.shape[0])

my_optimizer = tf.train.AdamOptimizer(learning_rate)

train_step = my_optimizer.minimize(loss)

# Initialize Variables
init = tf.initialize_all_variables()
sess.run(init)
train_loss = [ ]
train_acc = [ ]
test_acc = [ ]
test_loss = [ ]

for i in range(generations):
    rand_index = np.random.choice(len(X_train), size=batch_size)
    rand_x = np.array(X_train)[rand_index]
    rand_x = np.expand_dims(rand_x, 3)
    rand_x = np.reshape(rand_x,(-1,128,128,3))
    rand_y = np.array(y_train)[rand_index]
    train_dict = {x_input: rand_x, y_target: rand_y}
    sess.run(train_step, feed_dict=train_dict)

    temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict = train_dict)
    temp_train_acc = get_accuracy(temp_train_preds, rand_y)


    # if (i + 1) % eval_every == 0:
    eval_index = np.random.choice(len(X_test), size=evaluation_size)
    eval_x = X_test[eval_index]
    eval_x = np.expand_dims(eval_x, 3)
    eval_x = np.reshape(eval_x, (-1, 128, 128, 3))
    X_flat = tf.reshape(eval_x, [-1, 128 * 128 * 3])
    eval_y = np.array(y_test)[eval_index]
    test_dict = {eval_input: eval_x, eval_target: eval_y}
    test_preds = sess.run(test_prediction, feed_dict=test_dict)
    temp_test_acc = get_accuracy(test_preds, eval_y)





    train_loss.append(temp_train_loss)
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)
    test_loss.append(temp_train_loss)


    print('Generation # {}. Train Loss: ', train_loss, "  Train "
                                                                "Acc: ",
              train_acc, "  Test Loss: ", train_loss, "  Test Acc: ",
         test_acc)

print(train_loss)
print(train_acc)
d_log = {}
d_log["train_loss"] = train_loss
d_log["train_acc"] = train_acc
d_log["test_loss"] = test_loss
d_log["test_acc"] = test_acc

json_file = os.path.join('./log/cnn.json')
with open(json_file, 'w') as fp:
    json.dump(d_log, fp, indent=4, sort_keys=True)





