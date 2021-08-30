from keras.layers import Input, Conv2D, Conv2DTranspose, Lambda, LeakyReLU, concatenate
from keras import optimizers
from keras.models import Model

from loss import photometric_loss, smoothness_loss
from utils import rescale_tensors, warp_tensors

def flowmodelS(shape, batch_size):
    fixedinput = Input(shape=shape, batch_shape=(batch_size, 256, 256, 1), name='fixedinput')
    movinginput = Input(shape=shape, batch_shape=(batch_size, 256, 256, 1), name='movinginput')


    # encoder
    inputs = concatenate([fixedinput, movinginput], axis=3, name='inputs')
    conv1 = Conv2D(filters=64, kernel_size=7, strides=2, padding='same', name='conv1')(inputs)
    conv1 = LeakyReLU(alpha=0.1, name='LeakyReLu1')(conv1)
    conv2 = Conv2D(filters=128, kernel_size=5, strides=2, padding='same', name='conv2')(conv1)
    conv2 = LeakyReLU(alpha=0.1, name='LeakyReLu2')(conv2)
    conv3 = Conv2D(filters=256, kernel_size=5, strides=2, padding='same', name='conv3')(conv2)
    conv3 = LeakyReLU(alpha=0.1, name='LeakyReLu3')(conv3)
    conv3_1 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', name='conv3_1')(conv3)
    conv3_1 = LeakyReLU(alpha=0.1, name='LeakyReLu4')(conv3_1)
    conv4 = Conv2D(filters=512, kernel_size=3, strides=2, padding='same', name='conv4')(conv3_1)
    conv4 = LeakyReLU(alpha=0.1, name='LeakyReLu5')(conv4)
    conv4_1 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', name='conv4_1')(conv4)
    conv4_1 = LeakyReLU(alpha=0.1, name='LeakyReLu6')(conv4_1)
    conv5 = Conv2D(filters=512, kernel_size=3, strides=2, padding='same', name='conv5')(conv4_1)
    conv5 = LeakyReLU(alpha=0.1, name='LeakyReLu7')(conv5)
    conv5_1 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', name='conv5_1')(conv5)
    conv5_1 = LeakyReLU(alpha=0.1, name='LeakyReLu8')(conv5_1)
    conv6 = Conv2D(filters=1024, kernel_size=3, strides=2, padding='same', name='conv6')(conv5_1)
    conv6 = LeakyReLU(alpha=0.1, name='LeakyReLu9')(conv6)
    conv6_1 = Conv2D(filters=1024, kernel_size=3, strides=1, padding='same', name='conv6_1')(conv6)
    conv6_1 = LeakyReLU(alpha=0.1, name='LeakyReLu10')(conv6_1)

    # decoder
    flow6 = Conv2D(filters=2, kernel_size=3, strides=1, padding='same', name='flow6')(conv6_1)
    flow6_up = Conv2DTranspose(filters=2, kernel_size=4, strides=2, padding='same', name='flow6_up')(flow6)

    upconv5 = Conv2DTranspose(filters=512, kernel_size=4, strides=2, padding='same', name='upconv5')(conv6_1)
    upconv5 = LeakyReLU(alpha=0.1, name='LeakyReLu11')(upconv5)
    concat5 = concatenate([upconv5, conv5_1, flow6_up], axis=3, name='concat5')
    flow5 = Conv2D(filters=2, kernel_size=3, strides=1, padding='same', name='flow5')(concat5)
    flow5_up = Conv2DTranspose(filters=2, kernel_size=4, strides=2, padding='same', name='flow5_up')(flow5)

    upconv4 = Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same', name='upconv4')(concat5)
    upconv4 = LeakyReLU(alpha=0.1, name='LeakyReLu12')(upconv4)
    concat4 = concatenate([upconv4, conv4_1, flow5_up], axis=3, name='concat4')
    flow4 = Conv2D(filters=2, kernel_size=3, strides=1, padding='same', name='flow4')(concat4)
    flow4_up = Conv2DTranspose(filters=2, kernel_size=4, strides=2, padding='same', name='flow4_up')(flow4)

    upconv3 = Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same', name='upconv3')(concat4)
    upconv3 = LeakyReLU(alpha=0.1, name='LeakyReLu13')(upconv3)
    concat3 = concatenate([upconv3, conv3_1, flow4_up], axis=3, name='concat3')
    flow3 = Conv2D(filters=2, kernel_size=3, strides=1, padding='same', name='flow3')(concat3)
    flow3_up = Conv2DTranspose(filters=2, kernel_size=4, strides=2, padding='same', name='flow3_up')(flow3)

    upconv2 = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', name='upconv2')(concat3)
    upconv2 = LeakyReLU(alpha=0.1, name='LeakyReLu14')(upconv2)
    concat2 = concatenate([upconv2, conv2, flow3_up], axis=3, name='concat2')
    flow2 = Conv2D(filters=2, kernel_size=3, strides=1, padding='same', name='flow2')(concat2)
    flow2_up = Conv2DTranspose(filters=2, kernel_size=4, strides=2, padding='same', name='flow2_up')(flow2)

    upconv1 = Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='same', name='upconv1')(concat2)
    upconv1 = LeakyReLU(alpha=0.1, name='LeakyReLu15')(upconv1)
    concat1 = concatenate([upconv1, conv1, flow2_up], axis=3, name='concat1')
    flow1 = Conv2D(filters=2, kernel_size=3, strides=1, padding='same', name='flow1')(concat1)
    flow1_up = Conv2DTranspose(filters=2, kernel_size=4, strides=2, padding='same', name='flow1_up')(flow1)

    upconv0 = Conv2DTranspose(filters=16, kernel_size=4, strides=2, padding='same', name='upconv0')(concat1)
    upconv0 = LeakyReLU(alpha=0.1, name='LeakyReLu16')(upconv0)
    concat0 = concatenate([upconv0, inputs, flow1_up], axis=3, name='concat0')
    flow0 = Conv2D(filters=2, kernel_size=3, strides=1, padding='same', name='flow0')(concat0)


    rescaled_moving_input = Lambda(rescale_tensors, name='rescaling')(movinginput)
    out0 = Lambda(lambda x: warp_tensors(*x), name='out0')([rescaled_moving_input[0], flow0])
    out1 = Lambda(lambda x: warp_tensors(*x), name='out1')([rescaled_moving_input[1], flow1])
    out2 = Lambda(lambda x: warp_tensors(*x), name='out2')([rescaled_moving_input[2], flow2])
    out3 = Lambda(lambda x: warp_tensors(*x), name='out3')([rescaled_moving_input[3], flow3])
    out4 = Lambda(lambda x: warp_tensors(*x), name='out4')([rescaled_moving_input[4], flow4])
    out5 = Lambda(lambda x: warp_tensors(*x), name='out5')([rescaled_moving_input[5], flow5])
    out6 = Lambda(lambda x: warp_tensors(*x), name='out6')([rescaled_moving_input[6], flow6])

    outputs = [out0, out1, out2, out3, out4, out5, out6,
               flow0, flow1, flow2, flow3, flow4, flow5, flow6]
    loss = {
        'out0': photometric_loss,
        'out1': photometric_loss,
        'out2': photometric_loss,
        'out3': photometric_loss,
        'out4': photometric_loss,
        'out5': photometric_loss,
        'out6': photometric_loss,
        'flow0': smoothness_loss,
        'flow1': smoothness_loss,
        'flow2': smoothness_loss,
        'flow3': smoothness_loss,
        'flow4': smoothness_loss,
        'flow5': smoothness_loss,
        'flow6': smoothness_loss,
    }

    adam = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, decay=0.0005)
    model = Model(inputs=[fixedinput, movinginput], outputs=outputs)
    model.compile(optimizer=adam, loss=loss)
    return model