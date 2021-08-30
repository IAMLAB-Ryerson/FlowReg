import os
import glob
import numpy as np
import scipy.io as sio

from utils import rescale_imgs, rescale_img

def generatedata(moving_vols, fixed_vol, batch_size):

    batchcount = 0

    if os.path.isdir(moving_vols):
        vols = glob.glob(moving_vols + '/*.mat')
    elif os.path.isfile(moving_vols):
        vols = [line.rstrip('\n') for line in open(moving_vols)]
    else:
        print(
            "Invalid training data. Should be .txt file containing (training/validation) set location or directory of (training/validation) volumes")

    # loading and resizing the fixed volume, one slice at a time
    fixed = sio.loadmat(fixed_vol)['atlasFinal']
    fixed_vol_shape = fixed.shape
    fixed_reshaped_slices = []
    for i in range(fixed_vol_shape[2]):
        fixed_reshaped_slices.append(rescale_imgs(fixed[:, :, i], (256, 256)))

    f0 = np.zeros((batch_size+1, 256, 256, 1))
    f1 = np.zeros((batch_size+1, 128, 128, 1))
    f2 = np.zeros((batch_size+1, 64, 64, 1))
    f3 = np.zeros((batch_size+1, 32, 32, 1))
    f4 = np.zeros((batch_size+1, 16, 16, 1))
    f5 = np.zeros((batch_size+1, 8, 8, 1))
    f6 = np.zeros((batch_size+1, 4, 4, 1))

    z0 = np.zeros((batch_size+1, 256, 256, 1))
    z1 = np.zeros((batch_size+1, 128, 128, 1))
    z2 = np.zeros((batch_size+1, 64, 64, 1))
    z3 = np.zeros((batch_size+1, 32, 32, 1))
    z4 = np.zeros((batch_size+1, 16, 16, 1))
    z5 = np.zeros((batch_size+1, 8, 8, 1))
    z6 = np.zeros((batch_size+1, 4, 4, 1))

    m = np.zeros((batch_size+1, 256, 256, 1))

    while True:
        for s in range(55):
            for vol in vols:
                moving = sio.loadmat(vol)['regvol']
                moving_img = rescale_img(moving[:, :, s], (256, 256))


                f0[batchcount, :, :, :] = fixed_reshaped_slices[s][0]
                f1[batchcount, :, :, :] = fixed_reshaped_slices[s][1]
                f2[batchcount, :, :, :] = fixed_reshaped_slices[s][2]
                f3[batchcount, :, :, :] = fixed_reshaped_slices[s][3]
                f4[batchcount, :, :, :] = fixed_reshaped_slices[s][4]
                f5[batchcount, :, :, :] = fixed_reshaped_slices[s][5]
                f6[batchcount, :, :, :] = fixed_reshaped_slices[s][6]

                z0[batchcount, :, :, :] = fixed_reshaped_slices[s][7]
                z1[batchcount, :, :, :] = fixed_reshaped_slices[s][8]
                z2[batchcount, :, :, :] = fixed_reshaped_slices[s][9]
                z3[batchcount, :, :, :] = fixed_reshaped_slices[s][10]
                z4[batchcount, :, :, :] = fixed_reshaped_slices[s][11]
                z5[batchcount, :, :, :] = fixed_reshaped_slices[s][12]
                z6[batchcount, :, :, :] = fixed_reshaped_slices[s][13]

                m[batchcount, :, :, :] = moving_img

                batchcount += 1
                if batchcount > batch_size:
                    # print('f0 shape', f0.shape, 'batch_size', batch_size)
                    f0 = f0[0:batch_size, :, :, :]
                    f1 = f1[0:batch_size, :, :, :]
                    f2 = f2[0:batch_size, :, :, :]
                    f3 = f3[0:batch_size, :, :, :]
                    f4 = f4[0:batch_size, :, :, :]
                    f5 = f5[0:batch_size, :, :, :]
                    f6 = f6[0:batch_size, :, :, :]

                    z0 = z0[0:batch_size, :, :, :]
                    z1 = z1[0:batch_size, :, :, :]
                    z2 = z2[0:batch_size, :, :, :]
                    z3 = z3[0:batch_size, :, :, :]
                    z4 = z4[0:batch_size, :, :, :]
                    z5 = z5[0:batch_size, :, :, :]
                    z6 = z6[0:batch_size, :, :, :]

                    m = m[0:batch_size, :, :, :]

                    X = [f0, m]
                    y = [f0, f1, f2, f3, f4, f5, f6,
                         z0, z1, z2, z3, z4, z5, z6]
                    # print('input shape', X[0].shape, X[1].shape, 'output shape', y[0].shape, y[1].shape)
                    yield (X, y)
                    batchcount = 0
                    f0 = np.zeros((batch_size + 1, 256, 256, 1))
                    f1 = np.zeros((batch_size + 1, 128, 128, 1))
                    f2 = np.zeros((batch_size + 1, 64, 64, 1))
                    f3 = np.zeros((batch_size + 1, 32, 32, 1))
                    f4 = np.zeros((batch_size + 1, 16, 16, 1))
                    f5 = np.zeros((batch_size + 1, 8, 8, 1))
                    f6 = np.zeros((batch_size + 1, 4, 4, 1))

                    z0 = np.zeros((batch_size + 1, 256, 256, 1))
                    z1 = np.zeros((batch_size + 1, 128, 128, 1))
                    z2 = np.zeros((batch_size + 1, 64, 64, 1))
                    z3 = np.zeros((batch_size + 1, 32, 32, 1))
                    z4 = np.zeros((batch_size + 1, 16, 16, 1))
                    z5 = np.zeros((batch_size + 1, 8, 8, 1))
                    z6 = np.zeros((batch_size + 1, 4, 4, 1))

                    m = np.zeros((batch_size + 1, 256, 256, 1))