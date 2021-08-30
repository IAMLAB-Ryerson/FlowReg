import scipy.io as sio
import numpy as np
import os
import glob
import argparse
from tensorflow.keras import Model
import tensorflow as tf
from skimage import transform

# the Spatial Transformer layer used for the affine transform and resmapling was borrowed from 
# https://github.com/adalca/neurite orginally named neuron
from neuron.layers import SpatialTransformer

from utils import normalize
from model import affmodel

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def warpMask(affmat, mask):
    affmat_tensor = tf.convert_to_tensor(affmat, dtype=tf.float32)
    mask_tensor = tf.convert_to_tensor(mask.reshape(1, 256, 256, 55, 1), dtype=tf.float32)
    regMask= SpatialTransformer(interp_method='linear', indexing='ij')([mask_tensor, affmat_tensor])
    regMask = np.squeeze(regMask.eval(session=session))
    regMask = np.where(regMask > 0.1, 1, 0)
    return regMask


def register(fixedDir, movingDir, saveDir, brainDir, ventDir, wmlDir, modelDir):

    fixedVol = sio.loadmat(fixedDir)['atlasFinal']
    fixedVol = np.reshape(normalize(fixedVol), [1, 256, 256, 55, 1])

    if os.path.isfile(movingDir):
        movingVols = [line.rstrip('\n') for line in open(movingDir)]
    elif os.path.isdir(movingDir):
        movingVols = glob.glob(movingDir + '/*.mat')
        movingVols.sort()

    model = affmodel([256, 256, 55, 1])
    modelh5path = modelDir
    model.load_weights(modelh5path)

    for i, movingVol in enumerate(movingVols):

        name = os.path.basename(movingVol)
        print('Registering volume ', i, name)

        movingVol = normalize(transform.resize(sio.loadmat(movingVol)['im']['vol'][0][0], (256, 256, 55)))
        movingVol = np.reshape(normalize(movingVol), [1, 256, 256, 55, 1])

        layer_name = 'Dense6'
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer(layer_name).output)
        affinemat = intermediate_layer_model.predict([fixedVol, movingVol])
        regvol = model.predict([fixedVol, movingVol])
        regvol = np.squeeze(regvol)

        if brainDir and ventDir and wmlDir:
            brainVol = sio.loadmat(brainDir + '/' + name)['brainMask'].astype('float32')
            ventVol = sio.loadmat(ventDir + '/' + name)['ventMask'].astype('float32')
            wmlVol = sio.loadmat(wmlDir + '/' + name)['wmlMask'].astype('float32')

            regBrain = warpMask(affinemat, brainVol)
            regVent = warpMask(affinemat, ventVol)
            regWml = warpMask(affinemat, wmlVol)
            sio.savemat(os.path.join(saveDir + name), {'regvol': regvol, 
                        'brainMask': regBrain, 'ventMask': regVent, 'wmlMask': regWml, 'affine': affinemat})

        else:
            sio.savemat(os.path.join(saveDir + name), {'regvol': regvol, 'affine': affinemat})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""FlowReg-Affine (FlowReg-A) register""")

    parser.add_argument('-r', '--register', help='<string> register volumes directory', type=str, dest='moving')
    parser.add_argument('-f', '--fixed', help='<string> fixed volume directory', type=str, dest='fixed')
    parser.add_argument('-s', '--save', help='<string> results save directory', type=str, dest='save_dir')
    parser.add_argument('-b', '--brain', help='<string> brain masks directory', type=str, dest='brain_dir')
    parser.add_argument('-v', '--vent', help='<string> ventricle masks directory', type=str, dest='vent_dir')
    parser.add_argument('-w', '--wml', help='<string> wml masks directory', type=str, dest='wml_dir')
    parser.add_argument('-m', '--model', help='<string> trained model weights directory', dest='model')

    args = parser.parse_args()

    register(args.fixed, args.moving, args.save_dir, args.brain_dir, args.vent_dir, args.wml_dir, args.model)