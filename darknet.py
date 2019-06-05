from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def parse_cfg(cfgfile):
    """
    Args:
        cfgfile: The path of cfg file
    Returns:
        blocks: Each block describes a block in the neural network to be built.
                Block is represented as a dictionary in the list.
    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]  #get rid of blank lines
    lines = [x for x in lines if x[0] != '#']  #get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3  #input channels = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        # Check the type of the block
        # Create a new module for the block
        # Append to module_list

        # If convolution layer
        if x['type'] == 'convolutional':
            # Get info about the layer
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Add convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module('conv_{0}'.format(index), conv)

            # Add batch norm layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{0}'.format(index), bn)

            # Check activation
            # It is either Linear or Leaky ReLU in yolo
            if activation == 'leaky':
                acti = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{0}'.format(index), acti)

        # If upsampling layer
        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.UpsamplingNearest2d(scale_factor=stride)
            module.add_module('upsample_{}'.format(index), upsample)

        # If route layer
        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')
            start = int(x['layers'][0]) #start of the route
            try:
                end = int(x['layers'][1]) #end of the route if it exist
            except:
                end = 0
            # Positive anotation
            if start > 0:
                start -= index
            if end > 0:
                end -= index
            route = EmptyLayer()
            module.add_module('route_{0}'.format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        # Shortcut corresponds to skip connection
        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{}'.format(index), shortcut)

        # Yolo layer for detection
        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]

            anchors = x['anchors'].split(',')
            anchors = [int(x) for x in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('Detection_{}'.format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return net_info, module_list


class Darknet(nn.Module):
    def __init__(self,cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        detecttions = []
        modules = self.blocks[1:]
        outputs = {}  # cache the outputs for the route layer

        write = 0
        for i in range(len(modules)):
            module_type = modules[i]['type']
            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)
            elif module_type == 'route':
                layers = modules[i]['layers']
                layers = [int(x) for x in layers]

                if layers[0] > 0:
                    layers[0] -= i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                elif layers[1] > 0:
                    layers[1] -= i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
            elif module_type == 'shortcut':
                from_ = int(modules[i]['from'])
                x = outputs[i - 1] + outputs[i + from_]

            outputs[i] = x














