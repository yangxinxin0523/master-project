from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping

import resnet
import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob
import cv2
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)

batch_size = 32
nb_classes = 2
nb_epoch = 200
data_augmentation = False

# input image dimensions
img_rows, img_cols = 128, 128
img_channels = 3

sess = tf.Session()

class0_path = "/Users/xinxinyang/data/normal_cut_1/"
output0_path = "/Users/xinxinyang/data1/normal/"

class1_path = "/Users/xinxinyang/data/tumor_cut_1/"
output1_path = "/Users/xinxinyang/data1/tumor/"


def save_image(input, output):
    file_list=glob(input+"*.png")
    for fcount, img_file in enumerate(file_list):
        try:
            image = cv2.imread(img_file)
        except:
            print("can't read file")
        image = cv2.resize(image, (128, 128), cv2.INTER_LINEAR)
        np.save(os.path.join(output,"images_%04d.npy" % fcount),image)

save_image(class0_path, output0_path)
save_image(class1_path, output1_path)

X = []
Y = []

for img_file in glob(output0_path + '*.npy'):
    img = np.load(img_file,allow_pickle=True).astype(np.float32)
    X.append(img)
    Y.append(0)
for img_file in glob(output1_path + '*.npy'):
    img = np.load(img_file,allow_pickle=True).astype(np.float32)
    X.append(img)
    Y.append(1)

d1_X = np.array([np.reshape(x,(3,128,128)) for x in X])
X_train, X_test, Y_train, Y_test = \
        train_test_split(d1_X, Y, test_size=0.2, random_state=20, stratify=Y)


# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# subtract mean and normalize
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
X_train /= 128.
X_test /= 128.

model = resnet.ResnetBuilder.build((img_channels, img_rows,img_cols), nb_classes)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True,
              callbacks = [lr_reducer, early_stopper])
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(X_train)
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        validation_data=(X_test, Y_test),
                        epochs=nb_epoch, verbose=1, max_q_size=100,
                        callbacks=[lr_reducer, early_stopper])