import torch
import torch.nn as nn
import numpy as np

from utils.utils import *


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
        # Maxpool
        elif x['type'] == 'maxpool':
            kernel_size = int(x['size'])
            stride = int(x['stride'])
            if kernel_size == 2 and stride == 1:
                module.add_module(
                    f'_debug_padding_{idx}', nn.ZeroPad2d((0, 1, 0, 1)))
            padding = (kernel_size - 1) // 2
            maxpool = nn.MaxPool2d(kernel_size, stride, padding)
            module.add_module(f'maxpool_{idx}', maxpool)

        # Upsampling
        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
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
        self.ignore_thres = 0.5
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        self.stride = self.img_size // self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.grid_x = torch.arange(g).repeat(
            g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(
            g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride)
                                           for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view(
            (1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view(
            (1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_size=None):

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_size = img_size
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5,
                   grid_size, grid_size)
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

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                targets=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres
            )

        loss_x = self.mse(x[obj_mask], tx[obj_mask])
        loss_y = self.mse(y[obj_mask], ty[obj_mask])
        loss_w = self.mse(w[obj_mask], tw[obj_mask])
        loss_h = self.mse(h[obj_mask], th[obj_mask])
        loss_conf_obj = self.bce(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = self.bce(pred_conf[noobj_mask], tconf[noobj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
        loss_cls = self.bce(pred_cls[obj_mask], tcls[obj_mask])
        total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        cls_acc = 100 * class_mask[obj_mask].mean()
        conf_obj = pred_conf[obj_mask].mean()
        conf_noobj = pred_conf[noobj_mask].mean()
        conf50 = (pred_conf > 0.5).float()
        iou50 = (iou_scores > 0.5).float()
        iou75 = (iou_scores > 0.75).float()
        detected_mask = conf50 * class_mask * tconf
        precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
        recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
        recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

        self.metrics = {
            'loss': to_cpu(total_loss).item(),
            'x': to_cpu(loss_x).item(),
            'y': to_cpu(loss_y).item(),
            'w': to_cpu(loss_w).item(),
            'h': to_cpu(loss_h).item(),
            'conf': to_cpu(loss_conf).item(),
            'cls': to_cpu(loss_cls).item(),
            'cls_acc': to_cpu(cls_acc).item(),
            'recall50': to_cpu(recall50).item(),
            'recall75': to_cpu(recall75).item(),
            'precision': to_cpu(precision).item(),
            'conf_obj': to_cpu(conf_obj).item(),
            'conf_noobj': to_cpu(conf_noobj).item(),
            'grid_size': grid_size
        }

        return output, total_loss


class Darknet(nn.Module):

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_cfg(config_path)
        self.net_info, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs[1:], self.module_list)):
            if module_def['type'] in ['convolutional', 'maxpool', 'upsample']:
                x = module(x)
            elif module_def['type'] == 'route':
                x = torch.cat([layer_outputs[int(layer_i)]
                               for layer_i in module_def['layers'].split(',')], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'yolo':
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_weights(self, weights_path):

        with open(weights_path, 'rb') as f:
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header
            self.seen = header[3]
            weights = np.fromfile(f, dtype=np.float32)

        cutoff = None
        if 'darknet53.conv.74' in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs[1:], self.module_list)):
            if i == cutoff:
                break
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                if module_def['batch_normalize']:
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()

                    bn_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b

                    bn_w = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b

                    bn_rm = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b

                    bn_rv = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b

                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(
                    weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_weights(self, path, cutoff=-1):
        fp = open(path, 'wb')
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        for i, (module_def, module) in enumerate(zip(self.module_defs[1:cutoff], self.module_list[:cutoff])):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                if module_def['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
