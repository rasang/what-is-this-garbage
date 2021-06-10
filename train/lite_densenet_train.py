from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, Input, Flatten, Dense, Dropout, ZeroPadding2D, Convolution2D, Add
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121
from tensorflow.keras.backend import concatenate
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters

gpuConfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)

# 限制一个进程使用 60% 的显存
gpuConfig.gpu_options.per_process_gpu_memory_fraction = 0.6
sess1 =tf.compat.v1.Session(config=gpuConfig)


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
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
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



def densenet(shape=None):
    image_input = layers.Input(shape)
    x = Conv2D(16, (3, 3), padding='same', activation='relu')(image_input)

    for i in range(3):
        conv = Conv2D(16 * (i + 1), (3, 3), padding='same', activation='relu')(x)
        conv = MaxPooling2D(pool_size=2, padding='same', strides=(1, 1))(conv)
        x = concatenate([x, conv])

    x = Flatten()(x)
    x = Dense(6, activation='softmax')(x)
    model = Model(image_input, x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model


model = densenet(shape=(200, 200, 3))
# 编译模型
history_callback = LossHistory()
checkpoint = ModelCheckpoint("lite_densenet_trained_model.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = model.fit_generator(
    generator=datagen_train,
    validation_data=datagen_valid,
    epochs=30,
    validation_freq=1,
    callbacks=[history_callback, checkpoint]
)
history_callback.loss_plot('epoch')