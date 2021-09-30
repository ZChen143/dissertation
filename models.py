from tensorflow import reverse
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Conv1D, Add, Input, concatenate, GlobalMaxPooling1D, \
    Softmax, Flatten
import tensorflow as tf



def conv_blocks(dimensions, blocks, length=None):

    inputs = Input((length, dimensions))
    f = 64

    # Conv block 1: receptive field = 16
    if 1 in blocks:
        block1 = Conv1D(f, 2, padding='causal', activation='relu', kernel_initializer='he_normal')(inputs)
        block1 = BatchNormalization()(block1)
        block1 = Conv1D(f, 2, dilation_rate=2, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block1)
        block1 = BatchNormalization()(block1)
        block1 = Conv1D(f, 2, dilation_rate=4, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block1)
        block1 = BatchNormalization()(block1)
        block1 = Conv1D(f, 2, dilation_rate=8, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block1)
        block1 = BatchNormalization()(block1)
        block1_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        block1 = Add()([block1_bypass, block1])
    else:
        block1 = None

    # Conv block 2: receptive field = 31
    if 2 in blocks:
        block2 = Conv1D(f, 3, padding='causal', activation='relu', kernel_initializer='he_normal')(inputs)
        block2 = BatchNormalization()(block2)
        block2 = Conv1D(f, 3, dilation_rate=2, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block2)
        block2 = BatchNormalization()(block2)
        block2 = Conv1D(f, 3, dilation_rate=4, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block2)
        block2 = BatchNormalization()(block2)
        block2 = Conv1D(f, 3, dilation_rate=8, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block2)
        block2 = BatchNormalization()(block2)
        block2_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        block2 = Add()([block2_bypass, block2])
    else:
        block2 = None

    # Conv block 3: receptive field = 46
    if 3 in blocks:
        block3 = Conv1D(f, 4, padding='causal', activation='relu', kernel_initializer='he_normal')(inputs)
        block3 = BatchNormalization()(block3)
        block3 = Conv1D(f, 4, dilation_rate=2, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block3)
        block3 = BatchNormalization()(block3)
        block3 = Conv1D(f, 4, dilation_rate=4, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block3)
        block3 = BatchNormalization()(block3)
        block3 = Conv1D(f, 4, dilation_rate=8, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block3)
        block3 = BatchNormalization()(block3)
        block3_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        block3 = Add()([block3_bypass, block3])
    else:
        block3 = None

    # Conv block 4: receptive field = 136
    if 4 in blocks:
        block4 = Conv1D(f, 10, padding='causal', activation='relu', kernel_initializer='he_normal')(inputs)
        block4 = BatchNormalization()(block4)
        block4 = Conv1D(f, 10, dilation_rate=2, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block4)
        block4 = BatchNormalization()(block4)
        block4 = Conv1D(f, 10, dilation_rate=4, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block4)
        block4 = BatchNormalization()(block4)
        block4 = Conv1D(f, 10, dilation_rate=8, padding='causal', activation='relu', kernel_initializer='he_normal')(
            block4)
        block4 = BatchNormalization()(block4)
        block4_bypass = Conv1D(f, 1, activation='relu', kernel_initializer='he_normal')(inputs)
        block4 = Add()([block4_bypass, block4])
    else:
        block4 = None

    # Concatenate conv blocks
    con = concatenate([i for i in [block1, block2, block3, block4] if i is not None])
    return Model(inputs=inputs, outputs=con)



def classification_model_2d(blocks=(1, 2, 3, 4)):
    inputs = Input((None, 2))

    # Convolutions - run through twice, flipping x and y dimensions on second run
    conv = conv_blocks(dimensions=2, blocks=blocks)
    c1 = GlobalMaxPooling1D()(conv(inputs))
    c2 = GlobalMaxPooling1D()(conv(reverse(inputs, axis=[2])))
    c = tf.math.maximum(c1, c2)  # max pool outputs from the two passes

    # Dense layers
    dense = Dense(512, activation='relu')(c)
    dense = Dense(256, activation='relu')(dense)
    out = Dense(3, activation='softmax')(dense)

    # Full model
    model = Model(inputs=inputs, outputs=out)
    return model



