import torch
import torch.nn as nn

from utils import parse_cfg


yolov3_config = './config/yolov3-tiny.cfg'

def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    output_filters = [int(net_info['channels'])]
    for idx, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        
        # Convolutional layer
        if x['type'] == 'convolutional':
            try:
                batch_normalize = int(x['bacth_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            prev_filters = output_filters[-1]
            filters = int(x['filters'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])
            pad = (kernel_size - 1) // 2
            
            conv = nn.Conv2d(prev_filters, filters, kernel_size, 
                             stride, pad, bias=bias)
            module.add_module(f'conv_{idx}', conv)
            
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f'batch_norm_{idx}', bn)
            
            if x['activation'] == 'leaky':
                active = nn.LeakyReLU(0.1, inplace=True)
                module.add_module(f'leaky_{idx}', active)
                
        elif x['type'] == 'maxpool':
            kernel_size = int(x['size'])
            stride = int(x['stride'])
            padding = (kernel_size - 1) // 2
            maxpool = nn.MaxPool2d(kernel_size, stride, padding)
            module.add_module(f'maxpool_{idx}', maxpool)
            
        # Upsampling
        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            module.add_module(f'upsample_{idx}', upsample)
        
        # Route/shortcut
        elif x['type'] == 'route':
            layers = [int(a) for a in x['layers'].split(',')]
            filters = sum([output_filters[1:][i] for i in layers])
            module.add_module(f'route_{idx}', EmptyLayer())
        
        elif x['type'] == 'shortcut':
            filters = output_filters[1:][int(x['from'])]
            module.add_module(f'shortcut_{idx}', EmptyLayer())
        
        # YOLO
        elif x['type'] == 'yolo':
            mask = [int(a) for a in x['mask'].split(',')]
            
            anchors = [int(a) for a in x['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) 
                       for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            
            num_classes = int(x['classes'])
            img_size = int(net_info['height'])
            
            detection = DetectionLayer(anchors, num_classes, img_size)
            module.add_module(f'detection_{idx}', detection)
        
        module_list.append(module)
        output_filters.append(filters)
    
    return net_info, module_list

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_size=416):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_size = img_size
        self.grid_size = 0
    
    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        self.stride = self.img_size // self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) 
                                    for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))
    
    def forward(self, x, targets=None, img_size=None):
        
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor
        
        self.img_size = img_size
        num_samples = x.size(0)
        grid_size = x.size(2)
        
        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, 
                   self.grid_size, self.grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
        
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        pred_conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])
        
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, x.is_cuda)
        
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h
        
        output = torch.cat(
            (
            pred_boxes.view(num_samples, -1, 4) * self.stride,
            pred_conf.view(num_samples, -1, 1),
            pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )
        
        return output

class Darknet(nn.Module):
    
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_cfg(config_path)
        self.net_info, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
    
    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'maxpool', 'upsample']:
                x = module(x)
            elif module_def['type'] == 'route':
                x = torch.cat([layer_outputs[int(layer_i)] 
                               for layer_i in module_def['layers'].split(',')], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'yolo':
                x = module[0](x, targets, img_dim)
                yolo_outputs.append(x)
            layer_outputs.append(x)
        return yolo_outputs
# Test modules
# print(create_modules(parse_cfg(yolov3_config))[1])

