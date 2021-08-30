import numpy as np
import time
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger
import os
import datetime
import argparse
import glob

from model import affmodel
from data_generator import DataGenerator

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def train(batch_size, train, validation, fixed, checkpoint, epochs, save_loss, weights, model_save_dir):

    # to check if the train volumes are a directory or a text file
    if os.path.isfile(train):
        train_vol_names = [line.rstrip('\n') for line in open(train)]
    elif os.path.isdir(train):
        train_vol_names = glob.glob(train + '/*.mat')
        train_vol_names.sort()

    num_vols = len(train_vol_names)
    idx = np.arange(num_vols)
    np.random.shuffle(idx)
    train_ids = idx

    if os.path.isfile(validation):
        validation_vol_names = [line.rstrip('\n') for line in open(validation)]
    elif os.path.isdir(validation):
        validation_vol_names = glob.glob(validation + '/*.mat')
        validation_vol_names.sort()
    v_num_vols = len(validation_vol_names)
    idx = np.arange(v_num_vols)
    np.random.shuffle(idx)
    validation_ids = idx

    params = {'batch_size': batch_size,
              'dim': (256, 256, 55),
              'shuffle': True,
              'n_channels': 1
              }
    train_gen = DataGenerator(vols=train_ids, fvol_dir=fixed, mvol_dir=train, **params)
    valid_gen = DataGenerator(vols=validation_ids, fvol_dir=fixed, mvol_dir=validation, **params)

    timestr = time.strftime('%Y%m%d-%H%M%S')
    os.mkdir('../checkpoints/' + timestr)
    checkpoint = ModelCheckpoint(filepath='../checkpoints/' + timestr + '/weights-{epoch:02d}.h5',
                                 verbose=1, period=checkpoint,
                                 save_weights_only=True)
    
    if save_loss:
        csv_logger = CSVLogger('../losses/' + str(datetime.datetime.now().strftime('%Y-%m-%d')) + '.csv', separator=',')
        callbacks = [checkpoint, csv_logger]
    else:
        callbacks = [checkpoint]

    model = affmodel([256, 256, 55, 1])
    if os.path.isfile(weights):
        print("----Loading checkpoint weights----")
        model.load_weights(weights)

    model.fit_generator(train_gen, steps_per_epoch=len(train_ids)//batch_size,
                        validation_data=valid_gen, validation_steps=len(validation_ids)//batch_size,
                        verbose=1, epochs=epochs,
                        callbacks=callbacks)
    model.save(model_save_dir + timestr +'.h5')
    print("------------Model Saved---------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""FlowReg-Affine (FlowReg-A) training""")

    parser.add_argument('-t', '--train', help='<string> training volumes directory', type=str, dest='train')
    parser.add_argument('-v', '--validation', help='<string> validation volumes directory', type=str,
                        dest='validation')
    parser.add_argument('-f', '--fixed', help='<string> fixed volume directory', type=str, dest='fixed')
    parser.add_argument('-b', '--batch', help='<integer> batch size, default=4', type=int, dest='batch', default=4)
    parser.add_argument('-c', '--checkpoint', help='<integer> weights save checkpoint, default=00', type=int, dest='checkpoint', default=0)
    parser.add_argument('-e', '--epochs', help='<integer> number of training epochs, default=100', type=int, dest='epochs', default=100)
    parser.add_argument('-l', '--save_loss', help='<boolean> save loss across all epochs, default=TRUE', type=bool, dest='save_loss', default=True)
    parser.add_argument('-m', '--model_save', help='<string> model save directory', type=str, dest='model_save')

    args = parser.parse_args()

    train(args.batch, args.train, args.validation, args.fixed, args.checkpoint, args.epochs, args.save_loss, args.model_save)