import argparse
from modules.extractor import Extractor
from modules.features_extraction import models
import os.path as osp
import random
import numpy as np
import torch


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    extractor = Extractor(args)
    extractor.extract()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing the model")
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    # model
    parser.add_argument('-a', '--arch', type=str, required=True, choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # testing configs
    parser.add_argument('--resume', type=str, required=True, metavar='PATH')
    parser.add_argument('--rerank', action='store_true', help="evaluation only")
    parser.add_argument('--hard_sample', action='store_true', help="evaluation only, strictly choose samples in clusters")
    parser.add_argument('--clusters', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',default=osp.join(working_dir, 'data'))
    parser.add_argument('--store-dir', type=str, metavar='PATH',default=osp.join(working_dir, 'store'))    
    main()