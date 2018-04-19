import datetime

import keras
import tensorflow as tf
from keras import backend
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

import os
import constant
import GPUtil
from keras import Input, Model
from keras.callbacks import Callback, LearningRateScheduler, ReduceLROnPlateau
from keras.datasets import cifar10
from keras.layers import Conv2D, BatchNormalization, Dropout, Flatten, Dense, Activation, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2


def select_gpu():
    try:
        # Get the first available GPU
        DEVICE_ID_LIST = GPUtil.getFirstAvailable()
        DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list

        # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
    except EnvironmentError:
        print("GPU not found")


class NoImprovementError(Exception):
    def __init__(self, message):
        self.message = message


class EarlyStop(Callback):
    def __init__(self, max_no_improvement_num=constant.MAX_NO_IMPROVEMENT_NUM, min_loss_dec=constant.MIN_LOSS_DEC):
        super(EarlyStop, self).__init__()
        self.training_losses = []
        self.minimum_loss = None
        self._no_improvement_count = 0
        self._max_no_improvement_num = max_no_improvement_num
        self._done = False
        self._min_loss_dec = min_loss_dec

    def on_train_begin(self, logs=None):
        self.training_losses = []
        self._no_improvement_count = 0
        self._done = False
        self.minimum_loss = float('inf')

    def on_epoch_end(self, batch, logs=None):
        loss = logs.get('val_loss')
        self.training_losses.append(loss)
        if self._done and loss > (self.minimum_loss - self._min_loss_dec):
            raise NoImprovementError('No improvement for {} epochs.'.format(self._max_no_improvement_num))

        if loss > (self.minimum_loss - self._min_loss_dec):
            self._no_improvement_count += 1
        else:
            self._no_improvement_count = 0
            self.minimum_loss = loss

        if self._no_improvement_count > self._max_no_improvement_num:
            self._done = True


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr


class ModelTrainer:
    """A class that is used to train model

    This class can train a model with dataset and will not stop until getting minimum loss

    Attributes:
        model: the model that will be trained
        x_train: the input train data
        y_train: the input train data labels
        x_test: the input test data
        y_test: the input test data labels
        verbose: verbosity mode
    """

    def __init__(self, model, x_train, y_train, x_test, y_test, verbose):
        """Init ModelTrainer with model, x_train, y_train, x_test, y_test, verbose"""
        model.compile(loss=categorical_crossentropy,
                      optimizer=Adam(lr=lr_schedule(0)),
                      metrics=['accuracy'])
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.verbose = verbose
        if constant.DATA_AUGMENTATION:
            self.datagen = ImageDataGenerator(
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # randomly rotate images in the range (deg 0 to 180)
                rotation_range=0,
                # randomly shift images horizontally
                width_shift_range=0.1,
                # randomly shift images vertically
                height_shift_range=0.1,
                # randomly flip images
                horizontal_flip=True,
                # randomly flip images
                vertical_flip=False)
            self.datagen.fit(x_train)
        else:
            self.datagen = None

    def _converged(self, loss):
        """Return whether the training is converged"""

    def train_model(self):
        """Train the model with dataset and return the minimum_loss"""
        batch_size = min(self.x_train.shape[0], constant.MAX_BATCH_SIZE)
        terminator = EarlyStop()
        lr_scheduler = LearningRateScheduler(lr_schedule)

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)

        callbacks = [terminator, lr_scheduler, lr_reducer]

        try:
            if constant.DATA_AUGMENTATION:
                flow = self.datagen.flow(self.x_train, self.y_train, batch_size)
                self.model.fit_generator(flow,
                                         epochs=constant.MAX_ITER_NUM,
                                         validation_data=(self.x_test, self.y_test),
                                         callbacks=callbacks,
                                         verbose=self.verbose)
            else:
                self.model.fit(self.x_train, self.y_train,
                               batch_size=batch_size,
                               epochs=constant.MAX_ITER_NUM,
                               validation_data=(self.x_test, self.y_test),
                               callbacks=callbacks,
                               verbose=self.verbose)
        except NoImprovementError as e:
            if self.verbose:
                print('Training finished!')
                print(e.message)


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

    # Convert class vectors to binary class matrices.
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test


def cnn(params):
    if constant.LIMIT_MEMORY:
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)
        backend.set_session(sess)

    r = [params['regularizer1'][0],
         params['regularizer2'][0],
         params['regularizer3'][0],
         params['regularizer4'][0]]
    w = [params['width1'][0],
         params['width2'][0],
         params['width3'][0],
         params['width4'][0],
         params['width5'][0],
         params['width6'][0]]
    d = [params['dropout1'][0],
         params['dropout2'][0],
         params['dropout3'][0],
         params['dropout4'][0],
         params['dropout5'][0],
         params['dropout6'][0]]

    x_train, y_train, x_test, y_test = get_data()

    conv = Conv2D
    pool = MaxPooling2D
    # Input image dimensions.
    input_shape = x_train.shape[1:]
    input_tensor = output_tensor = Input(shape=input_shape)
    for i in range(4):
        output_tensor = conv(32,
                             kernel_size=3,
                             padding='same',
                             kernel_regularizer=l2(r[i]),
                             kernel_initializer='he_normal',
                             activation='linear')(output_tensor)
        output_tensor = BatchNormalization()(output_tensor)
        output_tensor = Activation('relu')(output_tensor)
        output_tensor = Dropout(constant.CONV_DROPOUT_RATE)(output_tensor)
        if i != 3:
            output_tensor = pool(padding='same')(output_tensor)

    output_tensor = Flatten()(output_tensor)
    output_tensor = Dense(w[4], kernel_initializer='he_normal', activation='relu')(output_tensor)
    output_tensor = Dropout(d[4])(output_tensor)
    output_tensor = Dense(w[5], kernel_initializer='he_normal', activation='relu')(output_tensor)
    output_tensor = Dropout(d[5])(output_tensor)
    output_tensor = Dense(10, activation='softmax')(output_tensor)
    Model(inputs=input_tensor, outputs=output_tensor)

    model = Model(input_tensor, output_tensor)

    x_train_new, x_val, y_train_new, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    ModelTrainer(model, x_train_new, y_train_new, x_val, y_val, False).train_model()
    loss, accuracy = model.evaluate(x_val, y_val, verbose=False)

    print(accuracy)

    return 1 - accuracy


# Write a function like this called 'main'
def main(job_id, params):
    select_gpu()
    print(params)
    print('Start time: ', datetime.datetime.now())
    print 'Anything printed here will end up in the output directory for job #:', str(job_id)
    result = cnn(params)
    print('End time: ', datetime.datetime.now())
    return result


if __name__ == '__main__':
    param = {'width1': [20],
             'width2': [20],
             'width3': [20],
             'width4': [20],
             'width5': [20],
             'width6': [20],
             'dropout1': [0.25],
             'dropout2': [0.25],
             'dropout3': [0.25],
             'dropout4': [0.25],
             'dropout5': [0.25],
             'dropout6': [0.25],
             'regularizer1': [0.05],
             'regularizer2': [0.05],
             'regularizer3': [0.05],
             'regularizer4': [0.05],
             }
    main(0, param)
