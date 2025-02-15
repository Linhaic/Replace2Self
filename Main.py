import argparse

import torch.backends.cudnn as cudnn
from Train import *
from utils import Parser
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

cudnn.benchmark = True
cudnn.fastest = True

## setup parse
parser = argparse.ArgumentParser(description='Train the unet network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--gpu_ids', default='0', dest='gpu_ids')

parser.add_argument('--mode', default='train', choices=['train', 'test'], dest='mode')
parser.add_argument('--train_continue', default='off', choices=['on', 'off'], dest='train_continue')

parser.add_argument('--scope', default='resnet', dest='scope')
parser.add_argument('--norm', type=str, default='bnorm', dest='norm')

parser.add_argument('--dir_checkpoint', default='checkpoints/simulation/', dest='dir_checkpoint')

parser.add_argument('--dir_data', default='data/simulation/train_test', dest='dir_data')
parser.add_argument('--dir_result', default='results/simulation/train_test/val/images', dest='dir_result')
parser.add_argument('--train_data', default='train/train_replace.npy', dest='train_data')
parser.add_argument('--train_data_reverse', default='train/train_replace_reverse.npy', dest='train_data_reverse')
parser.add_argument('--train_target', default='train/train_data.npy', dest='train_target')
parser.add_argument('--train_mask', default='train/train_mask.npy', dest='train_mask')
parser.add_argument('--train_replace_mask', default='train/train_replace_mask.npy', dest='train_replace_mask')
parser.add_argument('--train_changetable', default='train/train_changetable.npy', dest='train_changetable')
parser.add_argument('--train_changetable_reverse', default='train/train_changetable_reverse.npy', dest='train_changetable_reverse')
parser.add_argument('--val_data', default='val/val_data.npy', dest='val_data')
parser.add_argument('--val_mask', default='val/val_mask.npy', dest='val_mask')

parser.add_argument('--dir_loss', default='results/simulation/train_test', dest='dir_loss')

parser.add_argument('--num_epoch', type=int,  default=200, dest='num_epoch')
parser.add_argument('--batch_size', type=int, default=16, dest='batch_size')

parser.add_argument('--lr_G', type=float, default=1e-4, dest='lr_G')

parser.add_argument('--optim', default='adam', choices=['sgd', 'adam', 'rmsprop'], dest='optim')
parser.add_argument('--beta1', default=0.5, dest='beta1')

parser.add_argument('--ny_in', type=int, default=55, dest='ny_in')
parser.add_argument('--nx_in', type=int, default=55, dest='nx_in')
parser.add_argument('--nch_in', type=int, default=10, dest='nch_in')

parser.add_argument('--ny_load', type=int, default=55, dest='ny_load')
parser.add_argument('--nx_load', type=int, default=55, dest='nx_load')
parser.add_argument('--nch_load', type=int, default=10, dest='nch_load')

parser.add_argument('--ny_out', type=int, default=55, dest='ny_out')
parser.add_argument('--nx_out', type=int, default=55, dest='nx_out')
parser.add_argument('--nch_out', type=int, default=10, dest='nch_out')

parser.add_argument('--nch_ker', type=int, default=128, dest='nch_ker')

parser.add_argument('--data_type', default='float32', dest='data_type')

PARSER = Parser(parser)

def main():
    ARGS = PARSER.get_arguments()
    PARSER.print_args()
    TRAINER = Train(ARGS)
    TRAINER.train()

if __name__ == '__main__':
    main()