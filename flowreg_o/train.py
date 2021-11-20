import glob
import os
import datetime
import time
import argparse

from keras.callbacks import ModelCheckpoint, CSVLogger

from data_generator import generatedata
from model import flowmodelS

def train(fixed, train, validation, batch_size, epochs, checkpoint, save_path, save_loss, alpha, weights):

    # get data from generator
    print('generating data')
    generate_train = generatedata(train, fixed, batch_size)
    generate_validation = generatedata(validation, fixed, batch_size)

    if os.path.isdir(train):
        train_vols = glob.glob(train + '/*.mat')
    elif os.path.isfile(train):
        train_vols = [line.rstrip('\n') for line in open(train)]
    else:
        print(
            "Invalid training data. Should be .txt file containing (training/validation) set location or directory of (training/validation) volumes")

    if os.path.isdir(train):
        validation_vols = glob.glob(validation + '/*.mat')
    elif os.path.isfile(train):
        validation_vols = [line.rstrip('\n') for line in open(validation)]
    else:
        print(
            "Invalid training data. Should be .txt file containing (training/validation) set location or directory of (training/validation) volumes")

    timestr = time.strftime('%Y%m%d-%H%M%S')
    checkpointdir = 'G:/My Drive/MASc/Code/python/flowReg/checkpoint/' + timestr + 'alpha' + alpha
    os.mkdir(checkpointdir)
    checkpoint = ModelCheckpoint(filepath=checkpointdir + '/weights-{epoch:02d}.h5',
                                 verbose=1, period=checkpoint,
                                 save_weights_only=True)
    datestr = str(
        datetime.datetime.now().strftime('%Y-%m-%d'))
    if save_loss:
        csv_logger = CSVLogger('G:/My Drive/MASc/Code/python/flowreg2d/losses/' + datestr + '.csv', separator=',')
        callbacks = [checkpoint, csv_logger]
    else:
        callbacks = [checkpoint]

    model = flowmodelS(shape=[256, 256, 1], batch_size=batch_size)
    model.summary()
    if weights:
        print("loading previously trained weights------------")
        model.load_weights(weights)
    model.fit_generator(generate_train, steps_per_epoch=len(train_vols) * 55 // batch_size,
                        validation_data=generate_validation, validation_steps=len(validation_vols) * 55 // batch_size,
                        verbose=1, epochs=epochs,
                        callbacks=callbacks)

    model.save(save_path + timestr + '.h5')
    print("------------Model Saved---------------")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""FlowReg-OpticalFlow (FlowReg-O) training""")

    parser.add_argument('-t', '--train', help='<string> training volumes directory', type=str, dest='train')
    parser.add_argument('-v', '--validation', help='<string> validation volumes directory', type=str,
                        dest='validation')
    parser.add_argument('-f', '--fixed', help='<string> fixed volume directory', type=str, dest='fixed')
    parser.add_argument('-b', '--batch', help='<integer> batch size, default=4', type=int, dest='batch_size', default=64)
    parser.add_argument('-c', '--checkpoint', help='<integer> weights save checkpoint, default=00', type=int, dest='checkpoint', default=0)
    parser.add_argument('-e', '--epochs', help='<integer> number of training epochs, default=100', type=int, dest='epochs', default=100)
    parser.add_argument('-l', '--save_loss', help='<boolean> save loss across all epochs, default=TRUE', type=bool, dest='save_loss', default=True)
    parser.add_argument('-m', '--model_save', help='<string> model save directory', type=str, dest='model_save')
    parser.add_argument('-a', '--alpha', help='<string> alpha value for loss function during training, default = 0.20', type=str, dest='alpha', default='0.20')
    parser.add_argument('-w', '--load_weights', help='<string> location of weights to load', type=str, dest='load_weights')

    args = parser.parse_args()

    train(args.fixed, args.train, args.validation, args.batch_size, args.epochs, args.checkpoint, args.model_save, args.save_loss, args.alpha, args.load_weights)
