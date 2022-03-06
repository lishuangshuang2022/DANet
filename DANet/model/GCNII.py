#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F

class GraphConvolution_O(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution_O, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))#40,40
        self.att = Parameter(torch.FloatTensor(node_n, node_n))#48,48
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48,residual=False, variant=False,l=1):
        super(GraphConvolution, self).__init__()
        self.variant = True#variant#
        if self.variant:#
            self.in_features = 2 * in_features#
        else:#
            self.in_features = in_features#
        self.residual = residual  #
        self.l=l#
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features, out_features))#40,40
        self.att = Parameter(torch.FloatTensor(node_n, node_n))#48,48
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input,h0):
        lamda=1.4#0.5
        alpha=0.5

        theta = math.log(lamda / self.l + 1)

        hi = torch.matmul(self.att, input)#torch.Size([32, 60, 256])

        if self.variant:
            support = torch.cat([hi,h0],2)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.matmul(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + input

        # support = torch.matmul(input, self.weight)
        # output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48,l=1):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution_O(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias,l=l)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x,h0):
        y = self.gc2(x,h0)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y,h0)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution_O(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(self.num_stage):#12
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n,l=i+1))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = GraphConvolution_O(hidden_feature, input_feature, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()
        self.act_fn = nn.ReLU()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(input_feature, hidden_feature))
        self.fcs.append(nn.Linear(hidden_feature, input_feature))

    def forward(self, x):#torch.Size([32, 48, 40])
        # h0=F.dropout(x, 0.5)
        # h0=self.act_fn(self.fcs[0](h0))#torch.Size([32, 60,256])
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)
        h0=y

        for i in range(self.num_stage):
            y = self.gcbs[i](y,h0)

        y = self.gc7(y)#torch.Size([32, 48, 40])
        # y =self.fcs[-1](y)

        y = y + x

        return y
