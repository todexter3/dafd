import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding_C_group
import copy

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)   
        self.linear = nn.Linear(nf, target_window) 
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  
        x = self.flatten(x)          
        x = self.linear(x)            
        x = self.dropout(x)
        return x


class MultiScaleConv(nn.Module):
    """轻量并行多尺度卷积（在 patch/time 维上提取短/中/长期特征）"""
    def __init__(self, d_model, kernel_sizes=(3,7,15)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=k, padding=k//2, bias=True)
            for k in kernel_sizes
        ])
        self.proj = nn.Linear(len(kernel_sizes)*d_model, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B_total, seq_len, d_model]
        x = x.transpose(1, 2)     
        outs = [conv(x) for conv in self.convs]   
        out = torch.cat(outs, dim=1)              
        out = out.transpose(1, 2)                  
        out = self.proj(out)                      
        return self.act(out)                      


class Model(nn.Module):

    #一次性 encoder + 多尺度卷积聚合 
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.alpha = nn.Parameter(torch.randn(1))
        padding = configs.stride
        patch_len = configs.patch_len
        stride = configs.stride
        self.configs = configs

        # feature groups
        if configs.data_path == '/data/stock_daily_2005_2021.feather':
            self.feature_group = [[0],[1,2,3,4,5],[6],[7],[8]] # 1,5,1,1,1
        else:
            self.feature_group = [[0],[1],[2,3,4,5,6],[7],[8]] # 1,1,5,1,1分组

        # 原始的 fea_fusion
        self.fea_fusion = nn.ModuleList()
        for i in range(len(self.feature_group)):
            layer = nn.Sequential(
                nn.Flatten(-2),
                nn.Linear(len(self.feature_group[i]) * configs.d_model, configs.d_model // 2),
                nn.Dropout(configs.dropout),
                nn.GELU()
            )
            self.fea_fusion.append(layer)

        self.patch_embedding = PatchEmbedding_C_group(
            configs.d_model, patch_len, stride, padding, configs.dropout, self.feature_group
        )

        # Encoder (共享 encoder)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )

        # 多尺度卷积
        self.multi_scale = MultiScaleConv(configs.d_model, kernel_sizes=(3,7,15))

        # Prediction Head 
        self.head_nf = int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name in ('Long_term_forecasting', 'short_term_forecast'):

            self.head = FlattenHead(configs.enc_in, configs.d_model * self.head_nf * len(self.feature_group), configs.pred_len,
                                    head_dropout=configs.dropout)
        elif self.task_name in ('imputation', 'anomaly_detection'):
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len, head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)
        elif self.task_name in ('multiple_regression','predict_feature'):
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.feature_out = nn.Linear(configs.enc_in * configs.d_model, configs.d_model // 2)
            # projection 输入维度等于 len(feature_group) * configs.d_model * head_nf
            self.projection = nn.Linear(len(self.feature_group) * configs.d_model * self.head_nf, configs.output_channels)


    def _encode_all_groups_once(self, x_enc):
        B = x_enc.shape[0]
        x_in = x_enc.permute(0, 2, 1) 
        # patch embedding -> 得到每个 group 的 enc_out 列表
        enc_out_list, n_vars = self.patch_embedding(x_in)
        #一次性 concat 到 batch 维度并送入 encoder
        split_sizes = [t.shape[0] for t in enc_out_list]  
        enc_inputs = torch.cat(enc_out_list, dim=0)      
        # encoder
        enc_outputs, attns = self.encoder(enc_inputs)    
        # 卷积
        enc_outputs = self.multi_scale(enc_outputs)      
        # split 回各 group 并 reshape成 [B, n_vars, patch_num, d_model]
        splits = torch.split(enc_outputs, split_sizes, dim=0)  
        
        group_tensors = []
        for s in splits:
            patch_num = s.shape[1]
            d_model = s.shape[2]
            s = s.contiguous().view(B, 1, patch_num, d_model)
            s = s.permute(0, 1, 3, 2)
            group_tensors.append(s)

        # 在 d_model方向把不同 group 拼接
        out_cat = torch.cat(group_tensors, dim=2)
        return out_cat, n_vars

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev
        # encode once
        out_cat, n_vars = self._encode_all_groups_once(x)  
        dec_out = self.head(out_cat)   
        dec_out = dec_out.permute(0, 2, 1) 
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # normalization your original way
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x = x_enc - means
        x = x.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x * x, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x = x / stdev

        out_cat, n_vars = self._encode_all_groups_once(x) 
        dec_out = self.head(out_cat)
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, dec_out.shape[1], 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, dec_out.shape[1], 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        out_cat, n_vars = self._encode_all_groups_once(x)
        dec_out = self.head(out_cat)
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, dec_out.shape[1], 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, dec_out.shape[1], 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        out_cat, n_vars = self._encode_all_groups_once(x)
        # flatten then projection to num_class
        output = self.flatten(out_cat) 
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def regression(self, x_enc):
        # normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        out_cat, n_vars = self._encode_all_groups_once(x)  

        # flatten across channel & patch and project
        output = self.flatten(out_cat)  
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  
        output = self.projection(output)  
        return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ('Long_term_forecasting', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :] 
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        if self.task_name == 'multiple_regression':
            return self.regression(x_enc)
        return None