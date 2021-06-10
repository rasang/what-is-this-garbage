from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import regularizers


def conv_block(x, nb_filter, axis=-1, activation="relu", dropout_rate=None):
    x = layers.BatchNormalization(axis=axis)(x)
    x = layers.Activation(activation)(x)

    x = layers.Conv2D(nb_filter * 4, (1, 1), kernel_initializer="he_normal", padding='same', use_bias=False, kernel_regularizer=regularizers.l2())(x)
    x = layers.BatchNormalization(axis=axis)(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv2D(nb_filter, (3, 3), kernel_initializer="he_normal", padding='same', use_bias=False, kernel_regularizer=regularizers.l2())(x)
    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)
    return x


def dense_block(x, block_num, axis=-1, activation="relu", dropout_rate=None):
    for i in range(block_num):
        x = conv_block(x, 32,  axis=axis, activation=activation, dropout_rate=dropout_rate)
    return x


def transition_block(x, nb_filter, compression=1.0, axis=-1, activation="relu"):
    x = layers.BatchNormalization(axis=axis)(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv2D(int(nb_filter * compression), (1, 1), kernel_initializer="he_normal", padding='same', use_bias=False, kernel_regularizer=regularizers.l2())(x)
    x = layers.AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x


def densenet(shape=None, axis=-1, nb_filter=3, dropout_rate=None):
    image_input = layers.Input(shape)
    x = layers.Conv2D(64, (3, 3), use_bias=False, kernel_regularizer=regularizers.l2())(image_input)
    x = layers.BatchNormalization(axis=axis)(x)
    x = layers.Activation('relu')(x)
    x = dense_block(x, nb_filter, dropout_rate=dropout_rate)
    x = transition_block(x, nb_filter)
    x = dense_block(x, nb_filter, dropout_rate=dropout_rate)
    x = transition_block(x, nb_filter)
    x = dense_block(x, nb_filter, dropout_rate=dropout_rate)
    x = layers.BatchNormalization(axis=axis)(x)
    x = layers.Activation('relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(6, activation='softmax')(x)
    model = Model(inputs=image_input, outputs=x)
    return model