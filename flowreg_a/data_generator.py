import numpy as np
import scipy.io as sio
import glob, os
from tensorflow.python.keras.utils.data_utils import Sequence
from skimage import transform

from utils import normalize

class DataGenerator(Sequence):

    def __init__(self, vols, mvol_dir, fvol_dir, batch_size=4, shuffle=True, dim=(256, 256, 55), n_channels=1):
        self.vols = vols
        self.mvol_dir = mvol_dir
        self.fvol_dir = fvol_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dim = dim
        self.n_channels = n_channels
        self.on_epoch_end()

    def __len__(self):
        """ Denotes the number of batches per epoch"""
        return int(np.floor(len(self.vols) / self.batch_size))

    def __getitem__(self, index):
        """ Generates one batch of data """
        # Generates indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index+1) * self.batch_size]
        # find list of IDs
        list_IDs_temp = [self.vols[k] for k in indexes]
        # generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.vols))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """ Generates data containing batch_size samples""" # X: (n_samples, *dim, n_channels)
        fixed = np.empty((self.batch_size, *self.dim, self.n_channels))
        moving = np.empty((self.batch_size, *self.dim, self.n_channels))

        fixed_vol = normalize(sio.loadmat(self.fvol_dir)['atlasFinal'])

        if os.path.isdir(self.mvol_dir):
            moving_vols = glob.glob(self.mvol_dir + '*.mat')
        elif os.path.isfile(self.mvol_dir):
            moving_vols = [line.rstrip('\n') for line in open(self.mvol_dir)]
        else:
            print("Invalid training data. Should be .txt file containing (training/validation) set location or directory of (training/validation) volumes")


        for i, ID, in enumerate(list_IDs_temp):
            moving_vol = normalize(transform.resize(sio.loadmat(moving_vols[ID])['im']['vol'][0][0], (256,256,55)))

            moving[i, :, :, :, 0] = moving_vol
            fixed[i, :, :, :, 0] = fixed_vol

        X = [fixed, moving]
        y = fixed
        return X, y
