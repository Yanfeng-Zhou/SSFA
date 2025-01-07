import sys
from models import *
import torch.nn as nn

def get_network(network, in_channels, num_classes):

    # 2d networks
    if network == 'unet':
        net = unet(in_channels, num_classes)
    elif network == 'unet_ta_v1':
        net = unet_ta_v1(in_channels, num_classes)
    elif network == 'unet_ta_v2':
        net = unet_ta_v2(in_channels, num_classes)
    elif network == 'unet_':
        net = unet_(in_channels, num_classes)
    else:
        print('the network you have entered is not supported yet')
        sys.exit()
    return net
