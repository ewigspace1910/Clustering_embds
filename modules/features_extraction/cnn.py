from __future__ import absolute_import
from collections import OrderedDict
import torch

from ..utils import to_torch

def extract_cnn_feature(model, inputs, modules=None):
    model.eval()
    # with torch.no_grad():
    inputs = to_torch(inputs).cuda()
    if modules is None:
        outputs = model(inputs)
        outputs = outputs.data.cpu()
        return outputs
    # Register forward hook for each module
    outputs = OrderedDict()
    handles = []
    for m in modules:
        outputs[id(m)] = None
        def func(m, i, o): outputs[id(m)] = o.data.cpu()
        handles.append(m.register_forward_hook(func))
    model(inputs)
    for h in handles:
        h.remove()
    return list(outputs.values())

def extract_features(model, data_loader, print_freq=10, metric=None):
    model.eval()
    features = OrderedDict()
    camids = OrderedDict()
    with torch.no_grad():
        for i, (imgs, fnames, camid) in enumerate(data_loader):
            outputs = extract_cnn_feature(model, imgs)
            for fname, output, cid in zip(fnames, outputs, camid):
                features[fname] = output
                camids[fname] = cid
            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'.format(i + 1, len(data_loader)))
    return features, camids
