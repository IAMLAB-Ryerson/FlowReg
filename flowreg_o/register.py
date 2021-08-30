import glob
import os
import numpy as np
import scipy.io as sio
import tensorflow as tf
from skimage import transform
import argparse

from utils import normalize, rescale_img, rescale_imgs
from model import flowmodelS

img_size = (256, 256)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def warpMask(flow, mask):
    mask_tensor = mask.transpose(2, 0, 1)
    mask_tensor = tf.convert_to_tensor(mask_tensor.reshape(55, 256, 256, 1))
    regMask = tf.contrib.image.dense_image_warp(mask_tensor, flow, name='warpingmask')
    regMask = np.squeeze(regMask.eval(session=session)).transpose(1, 2, 0)
    regMask = np.where(regMask > 0.1, 1, 0)
    return regMask

def register(modelweights, fixed_vol, moving, brain_dir,  vent_dir, wml_dir, save_dir):
    fixed_vol = normalize(sio.loadmat(fixed_vol)['atlasFinal'])
    x, y, z = fixed_vol.shape

    vols = glob.glob(moving + "/*.mat")

    model = flowmodelS(shape=[256, 256, 1], batch_size=1)
    model.load_weights(modelweights)

    for j, vol in enumerate(vols):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        name = os.path.splitext(os.path.basename(vol))[0]

        if os.path.isfile(save_dir + "/" + name):
            print('Skip-------')
            print(name)
            continue

        volume = sio.loadmat(vol)
        moving_vol = volume['resized']
        moving_vol = normalize(transform.resize(moving_vol, (x, y, z)))

        regvol = np.empty((x, y, z))
        flowvol = np.empty((256, 256, z, 2))
        # flowvol128 = np.empty((128, 128, z, 2))
        # flowvol64 = np.empty((64, 64, z, 2))
        # flowvol32 = np.empty((32, 32, z, 2))
        # flowvol16 = np.empty((16, 16, z, 2))
        # flowvol8 = np.empty((8, 8, z, 2))
        # flowvol4 = np.empty((4, 4, z, 2))


        if 'brainMask' in volume and 'ventMask' in volume and 'wmlMask' in volume:
            brainMask = volume['brainMask'].astype(np.float32)
            ventMask = volume['ventMask'].astype(np.float32)
            wmlMask = volume['wmlMask'].astype(np.float32)
            regBrain = np.empty((x, y, z))
            regVent = np.empty((x, y, z))
            regWML = np.empty((x, y, z))


        for i in range(z):
            print('Registering Volume', j, name, ' Slice: ', i)

            fixed_img = rescale_imgs(fixed_vol[:, :, i], img_size=img_size)
            moving_img = rescale_img(moving_vol[:, :, i], img_size=img_size).reshape(1, 256, 256, 1)

            out = model.predict(x=[fixed_img[0], moving_img])
            reg_img = np.squeeze(out[0])
            flow = np.squeeze(out[7])
            # flow128 = np.squeeze(out[8])
            # flow64 = np.squeeze(out[9])
            # flow32 = np.squeeze(out[10])
            # flow16 = np.squeeze(out[11])
            # flow8 = np.squeeze(out[12])
            # flow4 = np.squeeze(out[13])

            regvol[:, :, i] = reg_img
            flowvol[:, :, i, :] = flow
            # flowvol256[:, :, i, :] = flow256
            # flowvol128[:, :, i, :] = flow128
            # flowvol64[:, :, i, :] = flow64
            # flowvol32[:, :, i, :] = flow32
            # flowvol16[:, :, i, :] = flow16
            # flowvol8[:, :, i, :] = flow8
            # flowvol4[:, :, i, :] = flow4


        if 'brainMask' in volume and 'ventMask' in volume and 'wmlMask' in volume:

            brainMask = volume['brainMask'].astype(np.float32)
            ventMask = volume['ventMask'].astype(np.float32)
            wmlMask = volume['wmlMask'].astype(np.float32)
            regBrain = np.empty((x, y, z))
            regVent = np.empty((x, y, z))
            regWML = np.empty((x, y, z))

            ft = tf.convert_to_tensor(flowvol.transpose(2, 0, 1, 3))

            regBrain = warpMask(ft, brainMask)
            regVent = warpMask(ft, ventMask)
            regWML = warpMask(ft, wmlMask)

            tf.keras.backend.clear_session()

            sio.savemat(os.path.join(save_dir, name), {'regvol': regvol, 'flow': flowvol,
                                                       'brainMask': regBrain, 'ventMask': regVent, 'wmlMask': regWML})
        else:
            sio.savemat(os.path.join(save_dir, name), {'regvol': regvol, 'flow': flowvol})

        print("Registered Volume Saved Successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""FlowReg-Affine (FlowReg-A) register""")

    parser.add_argument('-r', '--register', help='<string> training volumes directory', type=str, dest='moving')
    parser.add_argument('-f', '--fixed', help='<string> fixed volume directory', type=str, dest='fixed')
    parser.add_argument('-s', '--save', help='<string> results save directory', type=str, dest='save_dir')
    parser.add_argument('-b', '--brain', help='<string> brain masks directory', type=str, dest='brain_dir')
    parser.add_argument('-v', '--vent', help='<string> ventricle masks directory', type=str, dest='vent_dir')
    parser.add_argument('-w', '--wml', help='<string> wml masks directory', type=str, dest='wml_dir')
    parser.add_argument('-m', '--model', help='<string> trained model weights directory', dest='model')

    args = parser.parse_args()

    register(args.fixed, args.moving, args.save_dir, args.brain_dir, args.vent_dir, args.wml_dir, args.model)