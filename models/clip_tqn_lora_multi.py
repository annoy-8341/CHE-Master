'''
TJ, 2024-5-7
多lora
'''

import sys
# 加入父文件夹路径到sys.path中  
sys.path.append(sys.path[0].replace('models', ''))

import re
import logging
import math
import json
import pathlib
import numpy as np
import clip
from copy import deepcopy
from pathlib import Path
from einops import rearrange
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple, Union, Callable, Optional


import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.checkpoint import checkpoint

from transformers import AutoModel,BertConfig,AutoTokenizer
# from pytorch_pretrained_vit import ViT

# from visualizer import get_local
from models.transformer_decoder import *


from torch.autograd import Function
import timm
from models.resnet_multi import resnet50_lora, resnet101_lora, resnet152_lora
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
    

class DomainClassifier(nn.Module):
    '''一个单层分类器 带梯度反转层'''
    def __init__(self, domain_nums=4, feature_dims=768):
        super().__init__()
        self.domain_nums = domain_nums
        self.feature_dims = feature_dims
        self.fc = nn.Linear(feature_dims, domain_nums)

    def forward(self, x):
        reverse_x = ReverseLayerF.apply(x, 1.0)
        return self.fc(reverse_x)
    
class CLP_clinical(nn.Module):
    def __init__(self,
                bert_model_name: str,
                embed_dim: int = 768,
                freeze_layers:Union[Tuple[int, int], int] = None):
        super().__init__()
        self.bert_model = self._get_bert_basemodel(bert_model_name=bert_model_name, freeze_layers=freeze_layers)
        self.mlp_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.embed_dim = embed_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        for m in self.mlp_embed:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=self.embed_dim ** -0.5)

    def _get_bert_basemodel(self, bert_model_name, freeze_layers=None):#12
        try:
            print(bert_model_name)
            config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)#bert-base-uncased
            model = AutoModel.from_pretrained(bert_model_name, config=config)#, return_dict=True)
            print("Text feature extractor:", bert_model_name)
            print("bert encoder layers:",len(model.encoder.layer))
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model

    def encode_text(self, text):
        #input batch_size,token, return batch_size,dim 
        output = self.bert_model(input_ids = text['input_ids'],attention_mask = text['attention_mask'] )
        last_hidden_state, pooler_output, hidden_states = output[0],output[1],output[2]
        encode_out = self.mlp_embed(pooler_output)
        # encode_out = pooler_output
        return encode_out
    
    def forward(self, text):
        #input batch_size,token, return batch_size,dim 
        output = self.bert_model(input_ids = text['input_ids'],attention_mask = text['attention_mask'] )
        last_hidden_state, pooler_output, hidden_states = output[0],output[1],output[2]
        encode_out = self.mlp_embed(pooler_output)
        # encode_out = pooler_output
        return encode_out
    


class ModelRes_Lora_Multi(nn.Module):
    def __init__(self, res_base_model, r=16, lora_alpha=16):
        super(ModelRes_Lora_Multi, self).__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.resnet_dict = {
                            "lora_resnet50_multi": [resnet50_lora, 'ResNet50_Weights.IMAGENET1K_V2'],
                            "lora_resnet101_multi": [resnet101_lora, 'ResNet101_Weights.IMAGENET1K_V2'],
                            "lora_resnet152_multi": [resnet152_lora, 'ResNet152_Weights.IMAGENET1K_V2'],
                            }
        self.resnet = self._get_res_basemodel(res_base_model)

        num_ftrs = int(self.resnet.fc.in_features)
        self.res_features = nn.Sequential(*list(self.resnet.children())[:-2])
        # here num_ftrs = 2048 
        self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2 = nn.Linear(num_ftrs, 768)

    def _get_res_basemodel(self, res_model_name):
        try:
            
            res_list = self.resnet_dict[res_model_name]
            res_func = res_list[0]
            res_weight = res_list[1]
            res_model = res_func(r=self.r, lora_alpha=self.lora_alpha, weights=res_weight)
            print("Image feature extractor:", res_model_name)
            return res_model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, img):
        #return (batchsize, patch_num, dim)
        batch_size = img.shape[0]
        res_fea = self.res_features(img)
        # return res_fea
        # res_fea = F.adaptive_avg_pool2d(res_fea, (1, 1))
        res_fea = rearrange(res_fea,'b d n1 n2 -> b (n1 n2) d')
        h = rearrange(res_fea,'b n d -> (b n) d')
        x = self.res_l1(h)
        x = F.relu(x)
        x = self.res_l2(x)
        out_emb = rearrange(x,'(b n) d -> b n d',b=batch_size)
        out_pool = torch.mean(out_emb,dim=1)
        return out_emb,out_pool





if __name__ == "__main__":
            
    img = torch.randn(2,3,224,224)
    
    model = ModelRes_Lora(res_base_model = 'resnet152')
    out_emb, out_pool = model(img)
    
    print(out_emb.size(), out_pool.size())
    
    











