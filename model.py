import torch
import torch.nn as nn

from utils import parse_cfg, predict_transform
from module import create_modules

class Darknet(nn.Module):
    
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
    
    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}
        
        write = 0
        for i, module in enumerate(modules):
            m_type = (module['type'])
            
            if m_type == 'convolutional' or m_type == 'upsample':
                x = self.module_list[i](x)
                
            elif m_type == 'route':
                layers = module['layers']
                layers = [int(layer) for layer in layers]
                
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    
                    x = torch.cat((map1, map2), 1)
                    
            elif m_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i - 1] + outputs[i + from_]
            
            elif m_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                in_dim = int(self.net_info['height'])
                num_classes = int(module['classes'])
                
                x = x.data
                x = predict_transform(x, in_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)
                    
            outputs[i] = x
        
        return detections
            