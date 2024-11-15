
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

# from io import BytesIO
# from petrel_client.client import Client

# conf_path = '~/petreloss.conf'
# client = Client(conf_path) 
from torch.autograd import Function
import timm

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
    
    # def forward(self,text1,text2):
    #     text1_features = self.encode_text(text1)
    #     text2_features = self.encode_text(text2)
    #     text1_features = F.normalize(text1_features, dim=-1)
    #     text2_features = F.normalize(text2_features, dim=-1)
    #     return text1_features, text2_features, self.logit_scale.exp()

class ModelRes(nn.Module):
    def __init__(self, res_base_model):
        super(ModelRes, self).__init__()
        self.resnet_dict = {
                            "resnet50": models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2'),
                            "resnet101": models.resnet101(weights='ResNet101_Weights.IMAGENET1K_V2'),
                            "resnet152": models.resnet152(weights='ResNet152_Weights.IMAGENET1K_V2'),
                            "resnet50_openai": None,
                            'resnet101_openai': None,
                            'resnet50x4_openai': None,
                            }
                            # "resnet50": models.resnet50(pretrained=True)}
        self.resnet = self._get_res_basemodel(res_base_model)
        # num_ftrs = int(self.resnet.fc.in_features/2)
        # self.res_features = nn.Sequential(*list(self.resnet.children())[:-3]) 224
        if 'openai' in res_base_model:
            # 重新定义res_features
            num_ftrs = int(self.resnet.attnpool.v_proj.in_features)
            self.res_features = nn.Sequential(*list(self.resnet.children())[:-1])
        else:
            num_ftrs = int(self.resnet.fc.in_features)
            self.res_features = nn.Sequential(*list(self.resnet.children())[:-2])
        # here num_ftrs = 2048 
        self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2 = nn.Linear(num_ftrs, 768)

    def _get_res_basemodel(self, res_model_name):
        try:
            if 'openai' in res_model_name:
                if res_model_name == 'resnet50_openai':
                    model, preprocess = clip.load("RN50", device='cpu')
                elif res_model_name == 'resnet101_openai':
                    model, preprocess = clip.load("RN101", device='cpu')
                elif res_model_name == 'resnet50x4_openai':
                    model, preprocess = clip.load("RN50x4", device='cpu')
                res_model = model.visual
            else:
                res_model = self.resnet_dict[res_model_name]
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


class ModelConvNeXt(nn.Module):
    def __init__(self, convnext_base_model):
        super(ModelConvNeXt, self).__init__()
        self.convnext_dict = {"convnext-tiny": timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k_384', pretrained=True, num_classes=1000),
                              "convnext-base": timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', pretrained=True, num_classes=1000),
                              }
        convnext = self._get_convnext_basemodel(convnext_base_model)
        num_ftrs = int(convnext.head.in_features)
        self.conv_features = nn.Sequential(*list(convnext.children())[:-2])
        self.conv_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.conv_l2 = nn.Linear(num_ftrs, 768)
        
        
    def _get_convnext_basemodel(self, convnext_model_name):
        try:
            convnext_model = self.convnext_dict[convnext_model_name]
            print("Image feature extractor:", convnext_model_name)
            return convnext_model
        except:
            raise ("Invalid model name. Check the config file and pass one of: convnext-tiny, convnext-small or convnext-base")

    def forward(self, img):
        #return (batchsize, patch_num, dim)
        batch_size = img.shape[0]
        conv_fea = self.conv_features(img)
        conv_fea = F.adaptive_avg_pool2d(conv_fea, (1, 1))
        conv_fea = rearrange(conv_fea,'b d n1 n2 -> b (n1 n2) d')
        h = rearrange(conv_fea,'b n d -> (b n) d')
        x = self.conv_l1(h)
        x = F.relu(x)
        x = self.conv_l2(x)
        out_emb = rearrange(x,'(b n) d -> b n d',b=batch_size)
        out_pool = torch.mean(out_emb,dim=1)
        return out_emb,out_pool
    

# class ModelConvNeXt(nn.Module):
#     def __init__(self, convnext_base_model):
#         super(ModelConvNeXt, self).__init__()
#         self.convnext_dict = {"convnext-tiny": models.convnext_tiny(weights='ConvNeXt_Tiny_Weights.DEFAULT'),
#                               "convnext-small": models.convnext_small(weights='ConvNeXt_Small_Weights.DEFAULT'),
#                               "convnext-base": models.convnext_base(weights='ConvNeXt_Base_Weights.DEFAULT'),
#                               }
#         convnext = self._get_convnext_basemodel(convnext_base_model)
#         num_ftrs = int(convnext.classifier[-1].in_features)
#         self.conv_features = nn.Sequential(*list(convnext.children())[:-2])
#         self.conv_l1 = nn.Linear(num_ftrs, num_ftrs)
#         self.conv_l2 = nn.Linear(num_ftrs, 768)
        
        
#     def _get_convnext_basemodel(self, convnext_model_name):
#         try:
#             convnext_model = self.convnext_dict[convnext_model_name]
#             print("Image feature extractor:", convnext_model_name)
#             return convnext_model
#         except:
#             raise ("Invalid model name. Check the config file and pass one of: convnext-tiny, convnext-small or convnext-base")

#     def forward(self, img):
#         #return (batchsize, patch_num, dim)
#         batch_size = img.shape[0]
#         conv_fea = self.conv_features(img)
#         conv_fea = F.adaptive_avg_pool2d(conv_fea, (1, 1))
#         conv_fea = rearrange(conv_fea,'b d n1 n2 -> b (n1 n2) d')
#         h = rearrange(conv_fea,'b n d -> (b n) d')
#         x = self.conv_l1(h)
#         x = F.relu(x)
#         x = self.conv_l2(x)
#         out_emb = rearrange(x,'(b n) d -> b n d',b=batch_size)
#         out_pool = torch.mean(out_emb,dim=1)
#         return out_emb,out_pool
    
# import open_clip
# class ModelCLIP(nn.Module):
#     def __init__(self, clip_base_model):
#         super(ModelCLIP, self).__init__()
#         # 根据clip_base_model加载不同的模型
#         if clip_base_model == 'openai_EVA02-B-16':
#             model, _, preprocess = open_clip.create_model_and_transforms('EVA02-B-16', pretrained='merged2b_s8b_b131k')
            
#         elif clip_base_model == 'openai_convnext_base_w':
#             model, _, preprocess = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k_augreg')
        
#         else:
#             raise ("Invalid model name. Check the config file and pass one of: EVA02-B-16 or convnext_base_w")

import clip
class ModelViT(nn.Module):
    def __init__(self, vit_base_model):
        'vit输出默认就是768维'
        super(ModelViT, self).__init__()
        self.vit_dict = {"vit_b_16": models.vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1'),
                         "vit_b_16_openai": None}
        self.vit_model = self._get_vit_basemodel(vit_base_model)
        if 'openai' in vit_base_model:
            self.vit_features = self.vit_model
        else:
            self.vit_features = self._get_vit_features
        self.vit_l1 = nn.Linear(512, 512)
        self.vit_l2 = nn.Linear(512, 768)
        self.vit_base_model = vit_base_model
        

    def _get_vit_features(self, x):
        x = self.vit_model._process_input(x)
        n = x.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.vit_model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.vit_model.encoder(x)
        x = x[:, 0]
        return x
        
    def _get_vit_basemodel(self, vit_model_name):
        if 'openai' in vit_model_name:
            model, preprocess = clip.load('ViT-B/16', device='cpu')
            return model.visual
        try:
            vit_model = self.vit_dict[vit_model_name]
            print("Image feature extractor:", vit_model_name)
            return vit_model
        except:
            raise ("Invalid model name. Check the config file and pass one of: vit_b_16")
        

    def forward(self, img):
        vit_fea = self.vit_features(img)
        if 'openai' in self.vit_base_model: # 换成768的维度
            vit_fea = self.vit_l1(vit_fea)
            vit_fea = F.relu(vit_fea)
            vit_fea = self.vit_l2(vit_fea)
        
        out_emb = vit_fea.unsqueeze(1)
        
        out_pool = torch.mean(out_emb,dim=1)
        return out_emb,out_pool
        
        
class ModelEfficientV2(nn.Module):
    def __init__(self, efficientv2_base_model):
        super(ModelEfficientV2, self).__init__()
        self.efficientv2_dict = {"efficientnet_v2_s": models.efficientnet_v2_s(weights='EfficientNet_V2_S_Weights.IMAGENET1K_V1'),}
        self.efficientv2_model = self._get_efficientv2_basemodel(efficientv2_base_model)
        num_ftrs = int(self.efficientv2_model.classifier[-1].in_features)
        self.efficientv2_features = nn.Sequential(*list(self.efficientv2_model.children())[:-2])
        self.efficientv2_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.efficientv2_l2 = nn.Linear(num_ftrs, 768)
        
    
    def _get_efficientv2_basemodel(self, efficientv2_model_name):
        try:
            efficientv2_model = self.efficientv2_dict[efficientv2_model_name]
            print("Image feature extractor:", efficientv2_model_name)
            return efficientv2_model
        except:
            raise ("Invalid model name. Check the config file and pass one of: efficientnetv2_rw_s")
    
    def forward(self, img):
        batch_size = img.shape[0]
        efficientv2_fea = self.efficientv2_features(img)
        # efficientv2_fea = F.adaptive_avg_pool2d(efficientv2_fea, (1, 1))
        # print(efficientv2_fea.shape)
        efficientv2_fea = rearrange(efficientv2_fea,'b d n1 n2 -> b (n1 n2) d')
        # print(efficientv2_fea.shape)
        h = rearrange(efficientv2_fea,'b n d -> (b n) d')
        # print(h.shape)
        x = self.efficientv2_l1(h)
        x = F.relu(x)
        x = self.efficientv2_l2(x)
        # print(x.shape)
        out_emb = rearrange(x,'(b n) d -> b n d',b=batch_size)
        out_pool = torch.mean(out_emb,dim=1)
        return out_emb,out_pool
        
        
    
class ModelDense(nn.Module):
    def __init__(self, dense_base_model):
        super(ModelDense, self).__init__()
        
        self.densenet_dict = {"densenet121": models.densenet121(weights='DenseNet121_Weights.IMAGENET1K_V1'),
                            "densenet161": models.densenet161(weights='DenseNet161_Weights.IMAGENET1K_V1'),
                            "densenet201": models.densenet201(weights='DenseNet201_Weights.IMAGENET1K_V1'),}
        self.densenet = self._get_dense_basemodel(dense_base_model)
        num_ftrs = int(self.densenet.classifier.in_features)
        self.dense_features = self.densenet.features
        self.dense_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.dense_l2 = nn.Linear(num_ftrs, 768)

    def _get_dense_basemodel(self, dense_base_model):
        try:
            dense_model = self.densenet_dict[dense_base_model]
            print("Image feature extractor:", dense_base_model)
            return dense_model
        except:
            raise ("Invalid model name. Check the config file and pass one of: densenet121 or densenet161")

    def forward(self, img):
        batch_size = img.shape[0]
        dense_fea = self.dense_features(img)#N, 1024, 7,7
        dense_fea = rearrange(dense_fea,'b d n1 n2 -> b (n1 n2) d')
        h = rearrange(dense_fea,'b n d -> (b n) d')
        x = self.dense_l1(h)
        x = F.relu(x)
        x = self.dense_l2(x)
        out_emb = rearrange(x,'(b n) d -> b n d',b=batch_size)
        out_pool = torch.mean(out_emb,dim=1)
        return out_emb,out_pool

class TQN_Model(nn.Module):
    def __init__(self, 
            embed_dim: int = 768, 
            class_num: int = 1, 
            lam: list = [1, 0]
            ):
        super().__init__()
        self.d_model = embed_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # decoder_layer = TransformerDecoderLayer(self.d_model, 4, 1024,
                                        # 0.1, 'relu',normalize_before=True)
        decoder_layerV1 = TransformerDecoderLayerV1(self.d_model, 4, 1024,
                                        0.1, 'relu', True, lam)
        self.decoder_norm = nn.LayerNorm(self.d_model)
        # self.decoder = TransformerDecoder(decoder_layer, 4, self.decoder_norm,
                                # return_intermediate=False)
        self.decoderV1 = TransformerDecoderV1(decoder_layerV1, 4, self.decoder_norm,
                                return_intermediate=False)
        
        self.dropout_feas = nn.Dropout(0.1)

        # class_num = 2
        self.mlp_head = nn.Sequential( # nn.LayerNorm(768),
            nn.Linear(embed_dim, class_num)
        )
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    # def forward(self, image_features, text_features):
    #     #image_features (batch_size,patch_num,dim)
    #     #text_features (query_num,dim)
    #     batch_size = image_features.shape[0]
    #     image_features = image_features.transpose(0,1)
    #     text_features = text_features.unsqueeze(1).repeat(1, batch_size, 1)
    #     image_features = self.decoder_norm(image_features)
    #     text_features = self.decoder_norm(text_features)
        
    #     # features = self.decoder(text_features, image_features,
    #     #         memory_key_padding_mask=None, pos=None, query_pos=None)
        
    #     image_features_pool = torch.mean(image_features,dim=0).unsqueeze(0)
    #     features = self.decoderV1(text_features, image_features, image_features_pool,
    #             memory_key_padding_mask=None, pos=None, query_pos=None)  
        

 
    #     features = self.dropout_feas(features).transpose(0,1)  #b,embed_dim
    #     out = self.mlp_head(features)  #(batch_size, query_num)
    #     # out = out.squeeze(-1)
    #     return out
    
    def forward(self, image_features, text_features, return_atten = False):
        #image_features (batch_size,patch_num,dim)
        #text_features (query_num,dim)
        batch_size = image_features.shape[0]
        image_features = image_features.transpose(0,1)
        text_features = text_features.unsqueeze(1).repeat(1, batch_size, 1)
        image_features = self.decoder_norm(image_features)
        text_features = self.decoder_norm(text_features)
        
        image_features_pool = torch.mean(image_features,dim=0).unsqueeze(0)
        features,atten_map = self.decoderV1(text_features, image_features, image_features_pool, 
                memory_key_padding_mask=None, pos=None, query_pos=None) 
        features = self.dropout_feas(features).transpose(0,1)  #b,embed_dim
        out = self.mlp_head(features)  #(batch_size, query_num)
        if return_atten:
            return out, atten_map
        else:
            return out



class TQN_Model_Ensemble(nn.Module):
    def __init__(self, 
            embed_dim: int = 768, 
            class_num: int = 1, 
            lam: list = [1, 0]
            ):
        super().__init__()
        self.d_model = embed_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        decoder_layerV1 = TransformerDecoderLayerV1(self.d_model, 4, 1024,
                                        0.1, 'relu', True, lam)
        self.decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder_norm_1 = nn.LayerNorm(self.d_model)
        self.decoder_norm_2 = nn.LayerNorm(self.d_model)
        self.decoderV1 = TransformerDecoderV1(decoder_layerV1, 4, self.decoder_norm,
                                return_intermediate=False)
        self.decoderV1_1 = TransformerDecoderV1(decoder_layerV1, 4, self.decoder_norm_1,
                                return_intermediate=False)
        self.decoderV1_2 = TransformerDecoderV1(decoder_layerV1, 4, self.decoder_norm_2,
                                return_intermediate=False)
        
        self.dropout_feas = nn.Dropout(0.1)

        # class_num = 2
        self.mlp_head = nn.Sequential(nn.Linear(embed_dim, class_num))
        self.mlp_head_1 = nn.Sequential(nn.Linear(embed_dim, class_num))
        self.mlp_head_2 = nn.Sequential(nn.Linear(embed_dim, class_num))
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def forward(self, image_features, text_features, return_atten = False):

        batch_size = image_features.shape[0]
        image_features = image_features.transpose(0,1)
        text_features = text_features.unsqueeze(1).repeat(1, batch_size, 1)
        image_features = self.decoder_norm(image_features)
        image_features_1 = self.decoder_norm_1(image_features)
        image_features_2 = self.decoder_norm_2(image_features)

        text_features = self.decoder_norm(text_features)
        text_features_1 = self.decoder_norm_1(text_features)
        text_features_2 = self.decoder_norm_2(text_features)
        
        image_features_pool = torch.mean(image_features,dim=0).unsqueeze(0)
        image_features_pool_1 = torch.mean(image_features_1,dim=0).unsqueeze(0)
        image_features_pool_2 = torch.mean(image_features_2,dim=0).unsqueeze(0)


        features,atten_map = self.decoderV1(text_features, image_features, image_features_pool, 
                memory_key_padding_mask=None, pos=None, query_pos=None) 
        features = self.dropout_feas(features).transpose(0,1)  
        out = self.mlp_head(features)

        features_1,atten_map_1 = self.decoderV1_1(text_features_1, image_features_1, image_features_pool_1, 
                memory_key_padding_mask=None, pos=None, query_pos=None) 
        features_1 = self.dropout_feas(features_1).transpose(0,1)  
        out_1 = self.mlp_head_1(features_1)

        features_2,atten_map_2 = self.decoderV1_2(text_features_2, image_features_2, image_features_pool_2, 
                memory_key_padding_mask=None, pos=None, query_pos=None) 
        features_2 = self.dropout_feas(features_2).transpose(0,1)  
        out_2 = self.mlp_head_2(features_2)     


        out_stack = torch.stack([out, out_1, out_2])
        out = torch.mean(out_stack, dim=0)               
        





        if return_atten:
            return out, atten_map
        else:
            return out



# MIMIC时，batch_size=32, query_num=41, patch_num=256, dim=768
# img 256, 32, 768
# txt   1, 32, 768
# query41, 32, 768
# fts 41, 32, 768
# out 41, 32, 1
# 未经过sigmoid！计算loss时sigmoid！


if __name__ == "__main__":
            
    #torch 1.10.2 to torch 1.12.1
    #torchvision-0.11.3 to torchvision-0.13.1

    # image = torch.randn(1, 3, 224, 224) 
    # image_encoder = ModelRes(res_base_model = 'resnet50')
    # # image_encoder = ModelDense(dense_base_model = 'densenet121')
    # # image_encoder = ModelViT(vit_base_model = 'vit_b_32')
    # image_encoder(image)

    # image = torch.randn(256, 1, 768)
    # query = torch.randn(41, 768)
    # model = TQN_Model()
    # out = model(image, query)
    
    
    # img = torch.randn(1,3,512,512)
    img = torch.randn(2,3,224,224)
    # model = ModelConvNeXt(convnext_base_model = 'convnext-base')
    # model = ModelEfficientV2(efficientv2_base_model = 'efficientnet_v2_s')
    # model = ModelRes(res_base_model = 'resnet50_openai')
    model = ModelViT(vit_base_model = 'vit_b_16_openai')
    out_emb, out_pool = model(img)
    
    print(out_emb.size(), out_pool.size())
    
    