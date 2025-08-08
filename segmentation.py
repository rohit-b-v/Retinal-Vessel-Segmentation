import cv2
import numpy as np
import keras
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
#from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
# %matplotlib inline

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

import tensorflow.compat.v1 as tf

def LeNet(input_shape = (64, 64, 1), classes = 6):

    X_input = Input(input_shape)

    X = Conv2D(6, (3,3), strides = (1,1), name = 'conv1', kernel_initializer = glorot_uniform())(X_input)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = AveragePooling2D((3, 3))(X)

    X = Conv2D(16, (3,3), strides = (1,1), name = 'conv2', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv2')(X)
    X = Activation('relu')(X)
    X = AveragePooling2D((3, 3))(X)

    X = Flatten()(X)
    X = Dense(120, activation='relu', name='fc2', kernel_initializer = glorot_uniform())(X)
    X = Dense(84, activation='relu', name='fc3', kernel_initializer = glorot_uniform())(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform())(X)

    model = Model(inputs = X_input, outputs = X, name='LeNet')

    return model

LeNet = LeNet(input_shape = (280, 320, 1), classes = 6)

LeNet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

X_train = X_train_orig/255.
X_test = X_test_orig/255.

Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

X_train1 = np.zeros((X_train.shape[0], 280, 320))
X_test1 = np.ones((X_test.shape[0], 280, 320))
rgb_weights = [0.2989, 0.5870, 0.1140]

X_train = np.dot(X_train[...,:3], rgb_weights)
for i in range(X_train.shape[0]):
    X_train1[i] = cv2.resize(X_train[i], (320,280))
X_test = np.dot(X_test[...,:3], rgb_weights)
for i in range(X_test.shape[0]):
    X_test1[i] = cv2.resize(X_test[i], (320,280))

#plt.imshow(X_train1[0], cmap='gray')
#plt.show()
#plt.imshow(X_test1[0], cmap='gray')
#plt.show()


print ("number of training examples = " + str(X_train1.shape[0]))
print ("number of test examples = " + str(X_test1.shape[0]))
print ("X_train shape: " + str(X_train1.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test1.shape))
print ("Y_test shape: " + str(Y_test.shape))

LeNet.fit(X_train1, Y_train, epochs = 25, batch_size = 32)

preds = LeNet.evaluate(X_test1, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

LeNet.save("LeNet_model")

import os
#import import_ipynb
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.config.run_functions_eagerly(True)

from keras import backend as K
from keras import objectives
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation, Flatten
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout, UpSampling2D, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

import tensorflow.compat.v1 as tf

os.environ['KERAS_BACKEND'] = 'tensorflow'
K.set_image_data_format('channels_last')

LeNet = load_model("LeNet_model")

LeNet.summary()

def generator(img_size, n_filters, name='g'):

    k=3
    s=2
    img_ch = 1
    out_ch = 1
    img_height, img_width = img_size[0], img_size[1]
    padding='same'

    inputs = Input((560, 640, img_ch))

    conv1 = Conv2D(n_filters, (k, k), padding=padding)(inputs)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(n_filters, (k, k),  padding=padding)(conv1)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)

    conv2 = Conv2D(2*n_filters, (k, k),  padding=padding)(pool1)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(2*n_filters, (k, k),  padding=padding)(conv2)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)

    conv3 = Conv2D(4*n_filters, (k, k),  padding=padding)(pool2)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(4*n_filters, (k, k),  padding=padding)(conv3)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)

    conv4 = Conv2D(8*n_filters, (k, k),  padding=padding)(pool3)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(8*n_filters, (k, k),  padding=padding)(conv4)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)

    conv5 = Conv2D(16*n_filters, (k, k),  padding=padding)(pool4)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(16*n_filters, (k, k),  padding=padding)(conv5)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)

    up1 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv5), conv4])
    conv6 = Conv2D(8*n_filters, (k, k),  padding=padding)(up1)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(8*n_filters, (k, k),  padding=padding)(conv6)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)

    up2 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv6), conv3])
    conv7 = Conv2D(4*n_filters, (k, k),  padding=padding)(up2)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(4*n_filters, (k, k),  padding=padding)(conv7)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
    conv7 = Activation('relu')(conv7)

    up3 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv7), conv2])
    conv8 = Conv2D(2*n_filters, (k, k),  padding=padding)(up3)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(2*n_filters, (k, k),  padding=padding)(conv8)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
    conv8 = Activation('relu')(conv8)

    up4 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv8), conv1])
    conv9 = Conv2D(n_filters, (k, k),  padding=padding)(up4)
    conv9 = BatchNormalization(scale=False, axis=3)(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(n_filters, (k, k),  padding=padding)(conv9)
    conv9 = BatchNormalization(scale=False, axis=3)(conv9)
    conv9 = Activation('relu')(conv9)

    outputs = Conv2D(out_ch, (1, 1), padding=padding, activation='sigmoid')(conv9)

    g = Model(inputs, outputs, name=name)

    return g

def discriminator_global(img_size, model_name= 'discriminator'):

    img_ch=1 # image channels
    img_height, img_width = img_size[0], img_size[1]
    padding = 'same'

    inputs = Input((560, 640, img_ch))

    X = Conv2D(64, kernel_size=(5, 5), strides=(2,2), padding=padding)(inputs)
    X = BatchNormalization(scale=False, axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(128, kernel_size=(5, 5), strides=(2,2), padding=padding)(X)
    X = BatchNormalization(scale=False, axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(256, kernel_size=(3, 3), strides=(2,2), padding=padding)(X)
    X = BatchNormalization(scale=False, axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(512, kernel_size=(3, 3), strides=(2,2), padding=padding)(X)
    X = BatchNormalization(scale=False, axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(512, kernel_size=(3, 3), strides=(2,2), padding=padding)(X)
    X = BatchNormalization(scale=False, axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(512, kernel_size=(3, 3), strides=(2,2), padding=padding)(X)
    X = BatchNormalization(scale=False, axis=3)(X)
    X = Activation('relu')(X)

    X = GlobalAveragePooling2D()(X)
    outputs = Dense(1, activation='sigmoid')(X)

    model = Model(inputs, outputs)

    model.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def discriminator_multi1(img_size, n_filters, model_name='dm1'):

    img_ch=1 # image channels
    img_height, img_width = img_size[0], img_size[1]
    padding = 'same'

    inputs = Input((560, 640, img_ch))

    X = AveragePooling2D((2,2), strides=(2,2))(inputs)

    X = Conv2D(64, kernel_size=(5, 5), strides=(2,2), padding=padding)(X)
    X = BatchNormalization(scale=False, axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(128, kernel_size=(5, 5), strides=(2,2), padding=padding)(X)
    X = BatchNormalization(scale=False, axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(256, kernel_size=(3, 3), strides=(2,2), padding=padding)(X)
    X = BatchNormalization(scale=False, axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(512, kernel_size=(3, 3), strides=(2,2), padding=padding)(X)
    X = BatchNormalization(scale=False, axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(512, kernel_size=(3, 3), strides=(2,2), padding=padding)(X)
    X = BatchNormalization(scale=False, axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(512, kernel_size=(3, 3), strides=(2,2), padding=padding)(X)
    X = BatchNormalization(scale=False, axis=3)(X)
    X = Activation('relu')(X)

    X = GlobalAveragePooling2D()(X)
    outputs = Dense(1, activation='sigmoid')(X)

    model = Model(inputs, outputs, name= model_name)

    model.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def discriminator_multi2(img_size, n_filters, model_name='dm2'):

    img_ch=1 # image channels
    img_height, img_width = img_size[0], img_size[1]
    padding = 'same'

    inputs = Input((560, 640, img_ch))

    X = AveragePooling2D((4,4), strides = (4,4))(inputs)

    X = Conv2D(64, kernel_size=(5, 5), strides=(2,2), padding=padding)(X)
    X = BatchNormalization(scale=False, axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(128, kernel_size=(5, 5), strides=(2,2), padding=padding)(X)
    X = BatchNormalization(scale=False, axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(256, kernel_size=(3, 3), strides=(2,2), padding=padding)(X)
    X = BatchNormalization(scale=False, axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(512, kernel_size=(3, 3), strides=(2,2), padding=padding)(X)
    X = BatchNormalization(scale=False, axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(512, kernel_size=(3, 3), strides=(2,2), padding=padding)(X)
    X = BatchNormalization(scale=False, axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(512, kernel_size=(3, 3), strides=(2,2), padding=padding)(X)
    X = BatchNormalization(scale=False, axis=3)(X)
    X = Activation('relu')(X)

    X = GlobalAveragePooling2D()(X)
    outputs = Dense(1, activation='sigmoid')(X)

    model = Model(inputs, outputs, name= model_name)

    model.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])

    return model

image_shape = (560,640,3)

def perceptual_loss(y_pred, y_true):
    loss = 0

    d_real = d(y_true)
    d_out1 = d(y_pred)

    loss = loss + K.mean(K.log(d_real) + K.log(1- d_out1))

    d_real = d3(y_true)
    d_out3 = d3(y_pred)

    loss = loss + K.mean(K.log(d_real) + K.log(1- d_out3))

    d_real = d2(y_true)
    d_out2 = d2(y_pred)

    loss = loss + K.mean(K.log(d_real) + K.log(1- d_out2))

    y_pred = AveragePooling2D((2,2), strides=(2,2))(tf.reshape(tf.image.convert_image_dtype(y_pred, dtype=tf.float32, saturate=False),[1,560,640,1]))
    y_true = AveragePooling2D((2,2), strides=(2,2))(tf.reshape(tf.image.convert_image_dtype(y_true, dtype=tf.float32, saturate=False),[1,560,640,1]))

    LeNet.trainable = False
    for l in LeNet.layers:
        l.trainable = False
    loss = loss + K.mean(K.square(LeNet(y_true) - LeNet(y_pred)))

    model = Model(inputs=LeNet.input, outputs = [layer.output for layer in LeNet.layers])
    model.trainable = False

    y_true = model(y_true)
    y_pred = model(y_pred)
    for i in range(len(y_true)):
#        mse = tf.keras.losses.MeanSquaredError()
        loss = loss + K.mean(K.square(y_pred[i] - y_true[i]))
        #    model = Model(inputs=LeNet.input, outputs=LeNet.get_layer('bn_conv2'))
#    model.trainable = False

#    loss = loss + K.mean(K.square(model(y_true)-model(y_pred)))

    return loss

def GAN(generator, discriminator, discriminator2, discriminator3, img_size, n_filters_g, n_filters_d, init_lr, name="gan"):
    """
    GAN (that binds generator and discriminator)
    """

    img_h, img_w=img_size[0], img_size[1]

    img_ch=1
    seg_ch=1

    fundus = Input((560, 640, img_ch))


    fake_vessel = generator(fundus)

    fake_pair = Concatenate(axis=3)([fundus, fake_vessel])

    d_out1 = discriminator(fake_vessel)
    d_out2 = discriminator2(fake_vessel)
    d_out3 = discriminator3(fake_vessel)

    gan= Model(fundus, outputs=fake_vessel, name = str('gan'))
    gan.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=perceptual_loss)

    return gan

img_size = (560, 640)
n_filters_d=32
n_filters_g=32
init_lr=2e-4
g = generator(img_size, n_filters_g)

d = discriminator_global(img_size, n_filters_d)

d2 = discriminator_multi1(img_size, n_filters_d)

d3 = discriminator_multi2(img_size, n_filters_d)

model = GAN(g, d, d2, d3, img_size, n_filters_g, n_filters_d, init_lr)

g.summary()

d.summary()

d2.summary()

d3.summary()

model.summary()

path = "dataset/DRIVE/Images/"
drive_images = []
for i in range(21, 41):
    img = cv2.imread(path + str(i) + "_training.tif", cv2.IMREAD_GRAYSCALE)

    resized_image = cv2.resize(img, (640,560))

    equalized_image = cv2.equalizeHist(resized_image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gamma_corrected = clahe.apply(equalized_image)

    drive_images.append(gamma_corrected)

path = "dataset/DRIVE/Vessels/"
drive_vessels = []
for i in range(21, 41):
    gif = cv2.VideoCapture(path + str(i) + "_manual1.gif")
    ret, frame = gif.read()
    img = Image.fromarray(frame)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(img, (640,560))

    equalized_image = cv2.equalizeHist(resized_image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gamma_corrected = clahe.apply(equalized_image)

    drive_vessels.append(gamma_corrected)

path = "dataset/STARE/Images/"
stare_images = []
for i in ['0001','0002','0003','0004','0005','0044','0077','0081','0082','0139']:
    img = cv2.imread(path + 'im' + str(i) + ".ppm", cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(img, (640,560))

    equalized_image = cv2.equalizeHist(resized_image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gamma_corrected = clahe.apply(equalized_image)

    stare_images.append(gamma_corrected)

path = "dataset/STARE/Vessels/"
stare_vessels = []
for i in ['0001','0002','0003','0004','0005','0044','0077','0081','0082','0139']:
    img = cv2.imread(path + 'im' + str(i) + ".ah.ppm", cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(img, (640,560))

    equalized_image = cv2.equalizeHist(resized_image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gamma_corrected = clahe.apply(equalized_image)

    stare_vessels.append(gamma_corrected)

drive_images = np.array(drive_images)
stare_images = np.array(stare_images)
drive_vessels = np.array(drive_vessels)
stare_vessels = np.array(stare_vessels)

print(drive_images.shape)
print(drive_vessels.shape)
print(stare_images.shape)
print(stare_vessels.shape)

def shuffled_dataset(X, Y):

    m = X.shape[0]
    np.random.seed()

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:]
    shuffled_Y = Y[permutation,:,:]

    return shuffled_X, shuffled_Y

X_stare, Y_stare = shuffled_dataset(stare_images, stare_vessels)
X_drive, Y_drive = shuffled_dataset(drive_images, drive_vessels)

plt.imshow(X_stare[0], cmap= 'gray')
plt.show()
plt.imshow(Y_stare[0], cmap= 'gray')
plt.show()

plt.imshow(X_drive[0], cmap= 'gray')
plt.show()
plt.imshow(Y_drive[0], cmap= 'gray')
plt.show()

X_train = []
Y_train = []
for i in range(drive_images.shape[0]):
    X_train.append(drive_images[i])
    Y_train.append(drive_vessels[i])
for i in range(stare_images.shape[0]):
    X_train.append(stare_images[i])
    Y_train.append(stare_vessels[i])

X_train = np.array(X_train)
Y_train = np.array(Y_train)

print(X_train.shape)
print(Y_train.shape)

for n_round in range(5):

    # train the discriminator
    d.trainable = True
    for l in d.layers:
        l.trainable = True
    d2.trainable = True
    for l in d2.layers:
        l.trainable = True
    d3.trainable = True
    for l in d3.layers:
        l.trainable = True

    real_imgs, real_vessels = shuffled_dataset(X_train, Y_train)
    fake_vessels = g.predict(real_imgs)
    fake_vessels = fake_vessels.reshape(real_vessels.shape[0], real_vessels.shape[1], real_vessels.shape[2])
    plt.imshow(fake_vessels[0], cmap='gray')
    plt.show()

    total_vessels = np.concatenate((real_vessels, fake_vessels))

    y_out = np.ones(len(total_vessels))
    y_out[len(real_vessels):] = 0

    d.fit(total_vessels.reshape([total_vessels.shape[0], 560, 640, 1]), y_out, epochs =2, batch_size = 8)

    d2.fit(total_vessels.reshape([total_vessels.shape[0], 560, 640, 1]), y_out, epochs =2, batch_size = 8)

    d3.fit(total_vessels.reshape([total_vessels.shape[0], 560, 640, 1]), y_out, epochs =2, batch_size = 8)


    # train G (freeze discriminator)
    d.trainable = False
    for l in d.layers:
        l.trainable = False
    d2.trainable = False
    for l in d2.layers:
        l.trainable = False
    d3.trainable = False
    for l in d3.layers:
        l.trainable = False


    real_imgs, real_vessels = shuffled_dataset(X_train, Y_train)
    model.fit(real_imgs, real_vessels, epochs = 2, batch_size = 1)

