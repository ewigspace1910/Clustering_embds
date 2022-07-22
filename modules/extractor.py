import os.path as osp
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from .features_extraction import extract_features, FeatureDatabase,  create_model
from .features_extraction.dataloader import get_data
from .utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict

import numpy as np

class Extractor(object):

    def __init__(self, args) -> None:
        self.model = create_model(name=args.arch, pretrained=False, num_features=args.features, dropout=args.dropout, num_classes=0)
        self.data_loader = get_data(data_dir=args.dataset_dir, height=args.height, width=args.width, batch_size= args.batch_size, workers=args.workers)
        self.store_dir = args.store_dir
        # Load from checkpoint
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], self.model)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['best_mAP']
        print("=> Checkpoint of epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))
        print("=> Data size :", len(self.data_loader))


    def extract(self):
        features, _ = extract_features(self.model, self.data_loader)
        torch.save(osp.join(self.store_dir, "features.pth"))
        print("!save all features in {}".format(osp.join(self.store_dir, "features.pth")))

