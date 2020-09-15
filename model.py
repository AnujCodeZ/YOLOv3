from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils import parse_cfg


yolov3_config = './config/yolov3.cfg'

def create_modules(cfgfile):
    blocks = parse_cfg(cfgfile)
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    for idx, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        
        # Convolutional layer
        if x['type'] == 'convolutional':
            activation = x['activation']
            try:
                batch_normalize = int(x['bacth_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            filters = int(x['filters'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])
            padding = int(x['pad'])
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            
            conv = nn.Conv2d(prev_filters, filters, kernel_size, 
                             stride, pad, bias=bias)
            module.add_module(f'conv_{idx}', conv)
            
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f'batch_norm_{idx}', bn)
            
            if activation == 'leaky':
                active = nn.LeakyReLU(0.1, inplace=True)
                module.add_module(f'leaky_{idx}', active)
        
        # Upsampling
        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            module.add_module(f'upsample_{idx}', upsample)
        
        # Route/shortcut
        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')
            
            start = int(x['layers'][0])
            
            try:
                end = int(x['layers'][1])
            except:
                end = 0
            
            if start > 0:
                start = start - idx
            if end > 0:
                end = end - idx
            
            route = EmptyLayer()
            module.add_module(f'route_{idx}', route)
            
            if end > 0:
                filters = output_filters[idx + start] + \
                    output_filters[idx + end]
            else:
                filters = output_filters[idx + start]
        
        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module(f'shortcut_{idx}', shortcut)
        
        # YOLO
        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]
            
            anchors = x['anchors'].split(',')
            anchors = [(anchors[i], anchors[i + 1]) 
                       for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            
            detection = Detection(anchors)
            module.add_module(f'detection_{idx}', detection)
        
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    
    return net_info, module_list

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class Detection(nn.Module):
    def __init__(self, anchors):
        super(Detection, self).__init__()
        self.anchors = anchors
# Test
# print(create_modules(yolov3_config))
