from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, Input, Flatten, Dense, Dropout, ZeroPadding2D, Convolution2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121
from tensorflow.keras.backend import concatenate
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters


class LossHistory(keras.callbacks.Callback):
    """
    用于绘制loss和acc曲线图的类
    """
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('val_acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('val_acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


'''
数据生成
数据增强
'''
datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

datagen_train = datagen.flow_from_directory(
    "archive/train",
    color_mode="rgb",
    target_size=(200, 200),
    class_mode="categorical",
    batch_size=16,
    shuffle=True,
    seed=7,
)

datagen_valid = datagen.flow_from_directory(
    "archive/val",
    color_mode="rgb",
    class_mode="categorical",
    target_size=(200, 200),
    batch_size=16,
    shuffle=True,
    seed=7,
)

hp = HyperParameters()


def build_model(hp):
    """
    建立模型
    :return: 模型
    """
    model = Sequential()
    model.add(Input(shape=(200, 200, 3)))
    # 添加层的代码块
    for i in range(hp.Int("num_conv_layers", 1, 3)):
        model.add(Conv2D(hp.Choice(f'num_filters_layer{i}', values=[16, 32, 64], default=16), (3, 3), padding='same'))
        model.add(Activation('relu'))
        # model.add(Dropout(hp.Choice(f'num_filters_layer_dropout{i}', values=[0.3, 0.5, 0.7], default=0.3)))
        model.add(MaxPooling2D(pool_size=2))
    #
    model.add(Flatten())
    model.add(Dense(hp.Int("hidden_units", 32, 128, step=32), activation='relu'))
    model.add(Dense(6, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model


def cnn():
    model = Sequential()
    model.add(Input(shape=(200, 200, 3)))
    # 添加层的代码块
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # model.add(Dropout(hp.Choice(f'num_filters_layer_dropout{i}', values=[0.3, 0.5, 0.7], default=0.3)))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # model.add(Dropout(hp.Choice(f'num_filters_layer_dropout{i}', values=[0.3, 0.5, 0.7], default=0.3)))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # model.add(Dropout(hp.Choice(f'num_filters_layer_dropout{i}', values=[0.3, 0.5, 0.7], default=0.3)))
    model.add(MaxPooling2D(pool_size=2))

    #
    model.add(Flatten())
    model.add(Dense(6, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model


# tuner = Hyperband(
#     build_model,
#     objective='val_acc',
#     max_epochs=20,
#     directory='cnn_param',
#     hyperparameters=hp,
#     project_name='cnn_project'
# )
# tuner.search(datagen_train, epochs=10, validation_data=datagen_valid)
# best_hps=tuner.get_best_hyperparameters(1)[0]
# print(best_hps.values)

model = cnn()
# 编译模型
history_callback = LossHistory()
checkpoint = ModelCheckpoint("lite_cnn_trained_model.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = model.fit_generator(
    generator=datagen_train,
    validation_data=datagen_valid,
    epochs=100,
    validation_freq=1,
    callbacks=[history_callback, checkpoint]
)
history_callback.loss_plot('epoch')