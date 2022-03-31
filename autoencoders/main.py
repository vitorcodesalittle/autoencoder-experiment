#!/usr/bin/python3
import argparse
from linear import AutoencoderLinear

description = """Train and evaluate autoencoders"""

LINEAR_MODEL = 'linearae'
CONVAE_MODEL = 'convae'
CONVAE_CONVCL_MODEL = 'convae_convcl'
MODELS = [LINEAR_MODEL, CONVAE_MODEL, CONVAE_CONVCL_MODEL]

def main(models, skip_train):
    if not skip_train:
        if LINEAR_MODEL in models:
            



parser = argparse.ArgumentParser(description=description)
parser.add_argument('models', choices=MODELS, nargs='+',
                    help='models to be trained or evaluated')
parser.add_argument('--skip-train', '-s', dest='skip_train', action='store_true',
                    default=False,
                    help='Skip training and just evaluate')

args = parser.parse_args()
print(args.models)
print(args.skip_train)
main(models, skip_train=args.skip_train)
