from __future__ import absolute_import

from .mobilenet import mobileNetL, mobileNetS
from .mnasnet import *
from .regnet import *
from .osnet import *
from .resnet import *
from .resnet_ibn import *


__factory = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnet_ibn50a': resnet_ibn50a,
    'resnet_ibn101a': resnet_ibn101a,

    #mobile net
    'mobilenetS': mobileNetS,
    'mobilenetL': mobileNetL,

    #RegNet
    "regnetY128gf"  : regnetY128gf,
    "regnetY32gf"   : regnetY32gf,
    "regnetY16gf"   : regnetY16gf,
    "regnetY3_2gf"  : regnetY3_2gf,
    "regnetY1_6gf"  : regnetY1_6gf,
    "regnetY400"    : regnetY400,
    "regnetY800"    : regnetY800,

    #MNastNet
    "mnasnet0_5": mnasnet0_5,
    "mnasnet0_75": mnasnet0_75,
    "mnasnet1_0": mnasnet1_0,
    "mnasnet1_3": mnasnet1_3,

    #OSnet
    "osnet0_5"      : osnet0_5,   
    "osnet0_75"     : osnet0_75,  
    "osnet1_0"      : osnet1_0,   
    "osnet1_0ibt"   : osnet1_0ibt,


    "":None
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    pretrained : bool, optional
        Only applied for 'resnet*' models. If True, will use ImageNet pretrained
        model. Default: True
    cut_at_pooling : bool, optional
        If True, will cut the model before the last global pooling layer and
        ignore the remaining kwargs. Default: False
    num_features : int, optional
        If positive, will append a Linear layer after the global pooling layer,
        with this number of output units, followed by a BatchNorm layer.
        Otherwise these layers will not be appended. Default: 256 for
        'inception', 0 for 'resnet*'
    norm : bool, optional
        If True, will normalize the feature to be unit L2-norm for each sample.
        Otherwise will append a ReLU layer after the above Linear layer if
        num_features > 0. Default: False
    dropout : float, optional
        If positive, will append a Dropout layer with this dropout rate.
        Default: 0
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
