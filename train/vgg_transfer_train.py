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
        self.accuracy['batch'].append(logs.get('categorical_accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_categorical_accuracy'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('categorical_accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_categorical_accuracy'))

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
    height_shift_range=0.1,)


datagen_train= datagen.flow_from_directory(
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
    base_model = VGG16(weights = 'imagenet', include_top = False)
    for layer in base_model.layers:
        layer.trainable = False
    model.add(base_model)
    model.add(Flatten())
    for i in range(hp.Int("num_dense_layers", 1, 3)):
      model.add(Dense(hp.Int(f"hidden_units{i}", 32, 256,step=32), activation='relu'))
    model.add(Dense(6, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model


def vgg_transfer():
    model = Sequential()
    model.add(Input(shape=(200, 200, 3)))
    # 添加层的代码块
    base_model = VGG16(weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(160, activation='relu'))
    model.add(Dense(160, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model


tuner = Hyperband(
    build_model,
    objective='val_acc',
    max_epochs=20,
    directory='vgg_transfer_param',
    hyperparameters=hp,
    project_name='vgg_transfer_project'
)
tuner.search(datagen_train, epochs=10, validation_data=datagen_valid)
best_hps=tuner.get_best_hyperparameters(1)[0]
print(best_hps.values)
# model = build_model()
# # 编译模型
# # model.compile(optimizer='Adam',
# #             loss='categorical_crossentropy',
# #             metrics=[tf.keras.metrics.categorical_accuracy])
#
# history_callback = LossHistory()
# checkpoint = ModelCheckpoint("trained_model.h5", monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max', period=2)
# history = model.fit_generator(
#     generator=datagen_train,
#     validation_data=datagen_valid,
#     epochs=30,
#     validation_freq=1,
#     callbacks=[history_callback, checkpoint]
# )
# history_callback.loss_plot('epoch')