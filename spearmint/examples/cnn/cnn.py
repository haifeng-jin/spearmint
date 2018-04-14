import keras
import numpy as np
from keras import Input, Model
from keras.datasets import cifar10
from keras.layers import Conv2D, BatchNormalization, Dropout, Flatten, Dense, Activation
from keras.regularizers import l2


def get_data():
    # Load the CIFAR10 data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)

    # Convert class vectors to binary class matrices.
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test


def cnn(params):
    print(params)
    print(type(params))
    ret = 0.0
    for key in params:
        ret += params[key]
    r = [params['regularizer1'], params['regularizer2'], params['regularizer3']]
    w = [params['width1'],
         params['width2'],
         params['width3']]
    d = [params['dropout1'],
         params['dropout2'],
         params['dropout3']]

    x_train, y_train, x_test, y_test = get_data()

    # Input image dimensions.
    input_shape = x_train.shape[1:]
    input_tensor = output_tensor = Input(shape=input_shape)
    for i in range(3):
        output_tensor = Conv2D(filters=w[i],
                               kernel_size=3,
                               kernel_regularizer=l2(r[i]),
                               kernel_initializer='he_normal')(output_tensor)
        output_tensor = BatchNormalization()(output_tensor)
        output_tensor = Activation('relu')(output_tensor)
        output_tensor = Dropout(d[i])(output_tensor)

    output_tensor = Flatten()(output_tensor)
    output_tensor = Dense(10, kernel_initializer='he_normal', activation='softmax')(output_tensor)

    model = Model(input_tensor, output_tensor)

    return ret


# Write a function like this called 'main'
def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #:', str(job_id)
    return cnn(params)
