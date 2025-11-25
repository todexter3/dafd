#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/4/12 13:20
# @Author : hjh
# @File : MLP.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    def __init__(self, configs):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.inputs_dim = configs.enc_in
        # self.layers = configs.MLP_layers
        self.hidden_dim = configs.MLP_hidden
        # for i in range(self.layers):
        #     self.hidden_dim.insert(0, 32 * (2**(i)))
        self.outputs_dim = configs.pred_len
        self.activation_name = configs.activation

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.weight_std = configs.weight_std
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        # self.Linear_ = nn.ModuleList()
        # for i in range(2):
        #     if i == 0:
        #         self.Linear_.append(nn.Linear(128*configs.seq_len, self.hidden_dim//2))
        #     else:
        #         self.Linear_.append(nn.Linear(self.hidden_dim//2, self.hidden_dim//4))
        # self.Linear_ = nn.Sequential(  # 展平
        #         nn.Linear(128*configs.seq_len, self.hidden_dim*2),
        #         nn.GELU(),
        #         nn.Linear(self.hidden_dim*2, self.hidden_dim*4),
        #         nn.GELU(),
        #         # nn.Linear(self.hidden_dim*4, self.hidden_dim*4),
        #         # nn.GELU(),
        #         nn.Linear(self.hidden_dim*4, self.hidden_dim*2),
        #         nn.GELU(),
        #         nn.Linear(self.hidden_dim*2, self.hidden_dim),
        #         nn.GELU())
        self.out_ = nn.Linear(128*configs.seq_len, self.pred_len)
        self._select_activation(self.activation_name)

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)

        if self.task_name == 'multiple_regression' or self.task_name == 'predict_feature':
            self.act = nn.GELU()
            self.flatten = nn.Flatten(-2)
            self.down = nn.Linear(configs.enc_in, 128)
            self.dropout = nn.Dropout(configs.dropout)
            # self.out_ = nn.Linear(
            #     self.hidden_dim//4, configs.pred_len)
        self.apply(self.__init_normal_)

    def __init_normal_(self, m):
        if isinstance(m, nn.LSTM):
            nn.init.xavier_normal_(m.weight_ih_l0)
            nn.init.orthogonal_(m.weight_hh_l0)
            nn.init.constant_(m.bias_ih_l0, 0)
            nn.init.constant_(m.bias_hh_l0, 0)
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)

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

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        return self.encoder(x_enc)

    def classification(self, x_enc):
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def regression(self, x_enc):
        x_enc = self.down(x_enc)
        x_enc = self.flatten(x_enc)
        # x_enc = self.Linear_(x_enc)
        # for layer in self.Linear_:
        #     x_enc = layer(x_enc)
        #     if layer == self.Linear_[0]:
        #         x_enc = self.dropout(x_enc)
        #     x_enc = self.act(x_enc)
        output = self.out_(x_enc)
        return output

    def forward(self, x_enc, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':  # or self.task_name == 'multiple_regression':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        if self.task_name == 'multiple_regression' or self.task_name == 'predict_feature':
            dec_out = self.regression(x_enc)
            return dec_out  # [B, N]
        return None
