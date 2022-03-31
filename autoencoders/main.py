#!/usr/bin/python3
from os import path
import argparse
import torch
from torch import nn

from linear import AutoencoderLinear
from train import trainAutoEncoder
from data import load_data

description = """Train and evaluate autoencoders"""
DATAPATH = 'data'
MODELSPATH = '.models'
LINEAR_MODEL = 'linearae'
CONVAE_MODEL = 'convae'
CONVAE_CONVCL_MODEL = 'convae_convcl'
MODELS = [LINEAR_MODEL, CONVAE_MODEL, CONVAE_CONVCL_MODEL]

def main(models, skip_train, epochs=3):
    trainloader, testloader = load_data()

    if not skip_train:
        criterion = nn.MSELoss()
        if LINEAR_MODEL in models:
            linearmodel = AutoencoderLinear()
            history = trainAutoEncoder(linearmodel, trainloader, epochs=epochs, momentum=0.9, debug=True, criterion=criterion)
            torch.save(linearmodel, path.join(MODELSPATH, LINEAR_MODEL))
            print(history)
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
args = parser.parse_args()
main(args.models, skip_train=args.skip_train)
