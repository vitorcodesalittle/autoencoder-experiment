#!/usr/bin/python3
import os
import argparse
import torch
from os import path
from datetime import datetime
from torch import nn

from linear import AutoencoderLinear
from train import trainAutoEncoder
from data import load_data
from evaluate import plot_training_learning

description = """Train and evaluate autoencoders"""

DATAPATH = 'data'
MODELSPATH = '.models'
HISTORIESPATH = '.histories'

LINEAR_MODEL = 'linearae'
CONVAE_MODEL = 'convae'
CONVAE_CONVCL_MODEL = 'convae_convcl'
MODELS = [LINEAR_MODEL, CONVAE_MODEL, CONVAE_CONVCL_MODEL]

def getlastid(modelname):
    modelids = [ id for id in os.listdir(MODELSPATH) if modelname in id ]
    modelids.sort()
    historyids = [id for id in os.listdir(HISTORIESPATH) if modelname in id ]
    return modelids[-1], historyids[-1]

def main(models, skip_train, epochs=3, show_iteration_loss=False):
    trainloader, testloader = load_data()
    criterion = nn.MSELoss()
    if LINEAR_MODEL in models:
        if not skip_train:
            linearmodel = AutoencoderLinear()
            history = trainAutoEncoder(linearmodel, trainloader, epochs=epochs, momentum=0.9, debug=True, criterion=criterion)
            id = datetime.now().isoformat()
            torch.save(linearmodel, path.join(MODELSPATH, LINEAR_MODEL + id))
            torch.save(history, path.join(HISTORIESPATH, LINEAR_MODEL + id))
        else:
            modelid, historyid = getlastid(LINEAR_MODEL)
            print(f'modelid={modelid} historyid={historyid} - Skipped Training')
            linearmodel = torch.load(path.join(MODELSPATH, modelid))
            history = torch.load(path.join(HISTORIESPATH, modelid))
        filterkeys = ['iloss'] if not show_iteration_loss  else []
        plot_training_learning(history, filterkeys=filterkeys)

    if CONVAE_MODEL in models:
        print('Convolutional Autoencoder is not implemented')
    if CONVAE_CONVCL_MODEL in models:
            print('Convolutional Autoencoder with classifier not implemented')


parser = argparse.ArgumentParser(description=description)
parser.add_argument('models', choices=MODELS, nargs='*',
                    help='models to be trained or evaluated')
parser.add_argument('--skip-train', '-s', dest='skip_train', action='store_true',
                    default=False,
                    help='Skip training and just evaluate')
parser.add_argument('--epochs', '-e', dest='epochs', help='Epochs to iterate with optimizer', type=int, default=3)
parser.add_argument('--show-iteration-loss', help='Show iteration loss', dest="show_iteration_loss", action="store_true", default=False)

args = parser.parse_args()

if not path.isdir(MODELSPATH):
    os.mkdir(MODELSPATH)
if not path.isdir(HISTORIESPATH):
    os.mkdir(HISTORIESPATH)

main(args.models, epochs=args.epochs, skip_train=args.skip_train, show_iteration_loss=args.show_iteration_loss)

