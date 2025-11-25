#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/4/12 13:20
# @Author : hjh (modified by zhenyuguan)
# @File : MLP.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.inputs_dim = configs.enc_in
        self.hidden_dim = configs.MLP_hidden
        self.outputs_dim = configs.pred_len
        self.activation_name = configs.activation
        self.MLP_layers = configs.MLP_layers
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.weight_std = configs.weight_std

        self.Linear_1 = nn.Linear(self.seq_len, self.hidden_dim)
        self.alpha = torch.nn.Parameter(torch.tensor(configs.tau_hat_init))


        layers = []
        in_dim = self.inputs_dim * self.hidden_dim  

        if self.MLP_layers == 3:
            layers += [
                nn.Flatten(-2),
                nn.Linear(in_dim, self.hidden_dim * 2),
                nn.GELU(),
            ]
            
        elif self.MLP_layers == 4:
            layers += [
                nn.Flatten(-2),
                nn.Linear(in_dim, self.hidden_dim * 2),
                nn.GELU(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.GELU(),
            ]
        elif self.MLP_layers == 5:
            layers += [
                nn.Flatten(-2),
                nn.Linear(in_dim, self.hidden_dim * 2),
                nn.GELU(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim * 4),
                nn.GELU(),
                nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
                nn.GELU(),
            ]
        elif self.MLP_layers == 6:
            layers += [
                nn.Flatten(-2),
                nn.Linear(in_dim, self.hidden_dim * 2),
                nn.GELU(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim * 4),
                nn.GELU(),
                nn.Linear(self.hidden_dim * 4, self.hidden_dim * 4),
                nn.GELU(),
                nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
                nn.GELU(),
            ]
        elif self.MLP_layers == 7:
            layers += [
                nn.Flatten(-2),
                nn.Linear(in_dim, self.hidden_dim * 2),
                nn.GELU(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim * 4),
                nn.GELU(),
                nn.Linear(self.hidden_dim * 4, self.hidden_dim * 4),
                nn.GELU(),
                nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
                nn.GELU(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.GELU(),
            ]
        else:
            layers += [
                nn.Flatten(-2),
                nn.Linear(in_dim, self.hidden_dim),
                nn.GELU(),
            ]

        self.mlp = nn.Sequential(*layers)
        self.projection = nn.Linear(self.hidden_dim, int(configs.pred_len))

        self._select_activation(self.activation_name)
        self.dropout = nn.Dropout(configs.dropout)

        if self.task_name == 'classification':
            self.projection = nn.Linear(configs.enc_in * configs.seq_len, configs.num_class)
        else:
            self.projection = nn.Linear(self.hidden_dim, configs.pred_len)

    def _select_activation(self, _select_activation):
        if _select_activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif _select_activation == 'tanh':
            self.activation = nn.Tanh()
        elif _select_activation == 'elu':
            self.activation = nn.ELU()
        elif _select_activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif _select_activation == 'gelu':
            self.activation = nn.GELU()
        elif _select_activation == 'prelu':
            self.activation = nn.PReLU()
        else:
            self.activation = nn.Sigmoid()

    def regression(self, x_enc):
        x = x_enc.permute(0, 2, 1)  
        x = self.Linear_1(x)         
        x = self.mlp(x)             
        x = self.projection(x)
        return x

    def forward(self, x_enc, mask=None):
        if self.task_name in ['Long_term_forecasting', 'multiple_regression']:
            return self.regression(x_enc)
        return None