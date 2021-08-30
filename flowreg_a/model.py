from tensorflow.keras.layers import Input, Conv3D, Concatenate, Dense, Flatten
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

from loss import correlation

# the Spatial Transformer layer used for the affine transform and resmapling was borrowed from 
# https://github.com/adalca/neurite orginally named neuron
from neuron.layers import SpatialTransformer

def affmodel(shape):
    fixedInput = Input(shape=shape, name='fixed')
    movingInput = Input(shape=shape, name='moving')

    inputs = Concatenate(axis=4, name='inputs')([fixedInput, movingInput])

    conv1 = Conv3D(filters=16, kernel_size=7, strides=(2, 2, 1), padding='SAME', name='conv1', activation='relu')(inputs)

    conv2 = Conv3D(filters=32, kernel_size=5, strides=(2, 2, 1), padding='SAME', name='conv2', activation='relu')(conv1)

    conv3 = Conv3D(filters=64, kernel_size=3, strides=2, padding='SAME', name='conv3', activation='relu')(conv2)

    conv4 = Conv3D(filters=128, kernel_size=3, strides=2, padding='SAME', name='conv4', activation='relu')(conv3)

    conv5 = Conv3D(filters=256, kernel_size=3, strides=2, padding='SAME', name='conv5', activation='relu')(conv4)

    conv6 = Conv3D(filters=512, kernel_size=3, strides=2, padding='SAME', name='conv6', activation='relu')(conv5)

    flat = Flatten()(conv6)
    fc6 = Dense(12, activation='linear', name='Dense')(flat)

    out = SpatialTransformer(interp_method='linear', indexing='ij')([movingInput, fc6])

    adam = Adam(lr=0.0001)
    model = Model(inputs=[fixedInput, movingInput], outputs=[out])
    model.compile(optimizer=adam, loss=correlation)
    return model