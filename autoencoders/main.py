#!/usr/bin/python3
import os
import argparse
import sys
import torch
from os import path
from datetime import datetime
from torch import nn
from torchsummary import summary

from models import AutoencoderLinear, AutoencoderConv, CodeClassifier
from train import train_classifier, train_auto_encoder
from data import load_data
from evaluate import plot_training_learning, visualize_autoencoded
from exceptions import ModelNotFoundException

DESCRIPTION = """Train and evaluate autoencoders"""
DATAPATH = 'data'
BATCH_SIZE=50
MODELSPATH = '.models'
HISTORIESPATH = '.histories'
LINEAR_MODEL = 'linearae' # I'd like to put model names in models.py later
CONVAE_MODEL = 'convae'
CONV_CODE_CLASSIFIER = 'codecl'
INPUT_SIZE = (3, 32, 32)
MODELS = [LINEAR_MODEL, CONVAE_MODEL, CONV_CODE_CLASSIFIER]
CLASSIFIER_MODELS = [CONV_CODE_CLASSIFIER]

def getlastid(modelname):
    modelids = [ filename for filename in os.listdir(MODELSPATH) if modelname in filename ]
    historyids = [filename for filename in os.listdir(HISTORIESPATH) if modelname in filename ]
    modelids.sort()
    historyids.sort()
    if len(modelids) == 0 or len(historyids) == 0:
        raise ModelNotFoundException(f'Missing model {modelname}')
    return modelids[-1], historyids[-1]

def runtrain(modelname, trainloader, epochs, testloader):
    criterion = nn.MSELoss()
    model = create_model(modelname)
    summary(model, INPUT_SIZE, batch_size=10)
    if modelname in CLASSIFIER_MODELS:
        print('Training a classifier')
        history = train_classifier(model, trainloader, testloader, lr=5e-2, epochs=epochs, momentum=0.9, debug=True)
    else:
        print('Training an autoencoder')
        history = train_auto_encoder(model, trainloader,lr=5e-2, epochs=epochs, momentum=0.9, debug=True, criterion=criterion)
    nowiso = datetime.now().isoformat()
    torch.save(model, path.join(MODELSPATH, LINEAR_MODEL + nowiso))
    torch.save(history, path.join(HISTORIESPATH, LINEAR_MODEL + nowiso))
    return model, history

def get_last_model_and_history(modelname):
    modelid, historyid = getlastid(modelname)
    print(f'modelid={modelid} historyid={historyid} - Skipped Training')
    model = torch.load(path.join(MODELSPATH, modelid))
    history = torch.load(path.join(HISTORIESPATH, modelid))
    return model, history

def panic(message):
    print(message)
    sys.exit(1)

def create_model(modelName) -> nn.Module:
    if modelName == LINEAR_MODEL:
        return AutoencoderLinear()
    elif modelName == CONVAE_MODEL:
        return AutoencoderConv()
    elif modelName == CONV_CODE_CLASSIFIER:
        try:
            model, _ = get_last_model_and_history(CONVAE_MODEL)
            return CodeClassifier(model)
        except ModelNotFoundException:
            panic(f'Failed to load most recently trained {CONVAE_MODEL}. Is it under {MODELSPATH}?')
    else:
        panic(f'No model {modelName}')

def main(models, skip_train, epochs=3, show_iteration_loss=False):
    trainloader, testloader = load_data(BATCH_SIZE)
    for modelname in models:
        if not skip_train:
            model, history = runtrain(modelname, trainloader, epochs, testloader)
        else:
            model, history = get_last_model_and_history(modelname)
            summary(model, INPUT_SIZE, BATCH_SIZE)
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

