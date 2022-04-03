#!/usr/bin/python3
import os
import argparse
import torch
from os import path
from datetime import datetime
from torch import nn
from torchsummary import summary

from models import AutoencoderConvClassifier, AutoencoderLinear, AutoencoderConv
from train import train_auto_encoder
from data import load_data
from evaluate import plot_training_learning, evaluate_autoencoder, visualize_autoencoded

DESCRIPTION = """Train and evaluate autoencoders"""
DATAPATH = 'data'
MODELSPATH = '.models'
HISTORIESPATH = '.histories'
LINEAR_MODEL = 'linearae' # I'd like to put model names in models.py later
CONVAE_MODEL = 'convae'
CONVAE_CONVCL_MODEL = 'convae_convcl'
INPUT_SIZE = (3, 32, 32)
MODELS = [LINEAR_MODEL, CONVAE_MODEL, CONVAE_CONVCL_MODEL]

def getlastid(modelname):
    modelids = [ id for id in os.listdir(MODELSPATH) if modelname in id ]
    modelids.sort()
    historyids = [id for id in os.listdir(HISTORIESPATH) if modelname in id ]
    historyids.sort()
    return modelids[-1], historyids[-1]

def runtrain(modelname, trainloader, epochs, criterion):
    model = create_model(modelname)
    summary(model, INPUT_SIZE)
    history = train_auto_encoder(model, trainloader,lr=5e-2, epochs=epochs, momentum=0.9, debug=True, criterion=criterion)
    id = datetime.now().isoformat()
    torch.save(model, path.join(MODELSPATH, LINEAR_MODEL + id))
    torch.save(history, path.join(HISTORIESPATH, LINEAR_MODEL + id))
    return model, history

def get_last_model_and_history(modelname):
    modelid, historyid = getlastid(modelname)
    print(f'modelid={modelid} historyid={historyid} - Skipped Training')
    model = torch.load(path.join(MODELSPATH, modelid))
    history = torch.load(path.join(HISTORIESPATH, modelid))
    return model, history

def create_model(modelName):
    if modelName == LINEAR_MODEL:
        return AutoencoderLinear()
    elif modelName == CONVAE_MODEL:
        return AutoencoderConv()
    elif modelName == CONVAE_CONVCL_MODEL:
        return AutoencoderConvClassifier()
    else:
        raise Exception(f'No model after "{modelName}"')

def main(models, skip_train, epochs=3, show_iteration_loss=False):
    trainloader, testloader = load_data()
    criterion = nn.MSELoss()
    for modelname in models:
        if not skip_train:
            model, history = runtrain(modelname, trainloader, epochs, criterion)
        else:
            model, history = get_last_model_and_history(modelname)
            summary(model, INPUT_SIZE)
        filterkeys = ['iloss'] if not show_iteration_loss else []
        plot_training_learning(history, filterkeys=filterkeys)
        batch = next(iter(testloader))
        input_batch = batch[0]
        input_batch = [ input_batch[0], input_batch[1] ]
        visualize_autoencoded(model,input_batch)

# setup
if not path.isdir(MODELSPATH):
    os.mkdir(MODELSPATH)
if not path.isdir(HISTORIESPATH):
    os.mkdir(HISTORIESPATH)

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('models', choices=MODELS, nargs='*',
                    help='models to be trained or evaluated')
parser.add_argument('--skip-train', '-s', dest='skip_train', action='store_true',
                    default=False,
                    help='Skip training and just evaluate')
parser.add_argument('--epochs', '-e', dest='epochs', help='Epochs to iterate with optimizer', type=int, default=3)
parser.add_argument('--show-iteration-loss', help='Show iteration loss', dest="show_iteration_loss", action="store_true", default=False)

args = parser.parse_args()

main(args.models, epochs=args.epochs, skip_train=args.skip_train, show_iteration_loss=args.show_iteration_loss)

