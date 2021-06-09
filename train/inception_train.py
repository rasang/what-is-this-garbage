from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, Input, Flatten, Dense, Dropout, ZeroPadding2D, Convolution2D, Add
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
    image_input = Input(shape=(200, 200, 3))
    layer1_1 = Conv2D(hp.Choice('num_filters_layer1_1', values=[16, 32, 64], default=16), (1, 1), activation='relu')(image_input)
    #TensorShape([None, 200, 200, 64])

    layer2_1 = Conv2D(hp.Choice('num_filters_layer2_1', values=[16, 32, 64], default=16), (1, 1), padding='same', activation='relu')(image_input)
    layer2_2 = Conv2D(hp.Choice('num_filters_layer2_2', values=[16, 32, 64], default=16), (3, 3), padding='same', activation='relu')(layer2_1)
    #TensorShape([None, 198, 198, 64])

    layer3_1 = Conv2D(hp.Choice('num_filters_layer3_1', values=[16, 32, 64], default=16), (1, 1), padding='same', activation='relu')(image_input)
    layer3_2 = Conv2D(hp.Choice('num_filters_layer3_2', values=[16, 32, 64], default=16), (5, 5), padding='same', activation='relu')(layer3_1)
    #TensorShape([None, 196, 196, 64])

    pooling = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(image_input)
    pooling_out = Conv2D(hp.Choice('pooling_out', values=[16, 32, 64], default=16), (1, 1), activation='relu')(pooling)
    #TensorShape([None, 66, 66, 64])

    out = concatenate([layer1_1, layer2_2, layer3_2, pooling_out], axis=-1)
    out = Flatten()(out)
    out = Dense(6, activation='softmax')(out)
    model = Model(image_input, out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model

def inception():
    image_input = Input(shape=(200, 200, 3))
    layer1_1 = Conv2D(16, (1, 1), activation='relu')(
        image_input)
    # TensorShape([None, 200, 200, 64])

    layer2_1 = Conv2D(32, (1, 1), padding='same',
                      activation='relu')(image_input)
    layer2_2 = Conv2D(32, (3, 3), padding='same',
                      activation='relu')(layer2_1)
    # TensorShape([None, 198, 198, 64])

    layer3_1 = Conv2D(64, (1, 1), padding='same',
                      activation='relu')(image_input)
    layer3_2 = Conv2D(64, (5, 5), padding='same',
                      activation='relu')(layer3_1)
    # TensorShape([None, 196, 196, 64])

    pooling = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(image_input)
    pooling_out = Conv2D(16, (1, 1), activation='relu')(pooling)
    # TensorShape([None, 66, 66, 64])

    out = concatenate([layer1_1, layer2_2, layer3_2, pooling_out], axis=-1)
    out = Flatten()(out)
    out = Dense(6, activation='softmax')(out)
    model = Model(image_input, out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model


# tuner = Hyperband(
#     build_model,
#     objective='val_acc',
#     max_epochs=20,
#     directory='inception_param',
#     hyperparameters=hp,
#     project_name='cnn_project'
# )
# tuner.search(datagen_train, epochs=10, validation_data=datagen_valid)
# best_hps=tuner.get_best_hyperparameters(1)[0]
# print(best_hps.values)

model = inception()
# 编译模型
history_callback = LossHistory()
checkpoint = ModelCheckpoint("inception_trained_model.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = model.fit_generator(
    generator=datagen_train,
    validation_data=datagen_valid,
    epochs=100,
    validation_freq=1,
    callbacks=[history_callback, checkpoint]
)
history_callback.loss_plot('epoch')