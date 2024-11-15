'''
2023.10.16
只测试 考虑多种测试方式

'''

import sys
# 加入父文件夹路径到sys.path中  
sys.path.append(sys.path[0].replace('main', ''))

import argparse
import os

import logging
# import ruamel.yaml as yaml
import yaml
import numpy as np
import random
import time
import datetime
import json
import math
from pathlib import Path
from functools import partial
from sklearn.metrics import roc_auc_score

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, BertConfig, AutoTokenizer

from factory import utils
from engine.train_test2 import valid_on_padchest, valid_on_cheXpert, valid_on_chestxray14, valid_on_siimacr, valid_on_vindr,  valid_on_openi, valid_on_shenzhen
from models.clip_tqn import CLP_clinical, ModelRes, TQN_Model, ModelConvNeXt, ModelEfficientV2
from data.dataset_entity import MIMIC_Dataset, Mergetrain_Dataset, Chestxray14_Dataset, CheXpert_Dataset
from data.test_dataset import Vindr_Dataset, SIIMACR_Dataset
from main.main_utils import get_dataloader, get_ema_model, seed_torch, record_results
from models.clip_tqn_lora import ModelRes_Lora
from models.clip_tqn_lora_multi import ModelRes_Lora_Multi
from models.clip_tqn_lora_ensemble import ModelRes_Lora_Ensemble


def get_model(args, config):
    if '_ensemble' in config['image_encoder_name']:
        image_encoder = ModelRes_Lora_Ensemble(config['image_encoder_name'], config['lora_r'], config['lora_alpha']).cuda()

    elif 'lora_resnet' in config['image_encoder_name']:
        image_encoder = ModelRes_Lora(config['image_encoder_name'], config['lora_r'], config['lora_alpha']).cuda()
    elif 'resnet' in config['image_encoder_name']:
        image_encoder = ModelRes(config['image_encoder_name']).cuda()
    elif 'convnext' in config['image_encoder_name']:
        image_encoder = ModelConvNeXt(config['image_encoder_name']).cuda()
    elif 'efficientnet' in config['image_encoder_name']:
        image_encoder = ModelEfficientV2(config['image_encoder_name']).cuda()
    elif 'vit' in config['image_encoder_name']:
        image_encoder = ModelViT(config['image_encoder_name']).cuda()
    elif 'densenet' in config['image_encoder_name']:
        image_encoder = ModelDense(config['image_encoder_name']).cuda()
    else:
        raise NotImplementedError
    
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name, do_lower_case=True, local_files_only=True)
    text_encoder = CLP_clinical(bert_model_name=args.bert_model_name).cuda()


    if args.bert_pretrained:
        checkpoint = torch.load(args.bert_pretrained, map_location='cpu')
        state_dict = checkpoint["state_dict"]
        text_encoder.load_state_dict(state_dict,strict=False)
        if args.freeze_bert:
            for param in text_encoder.parameters():
                param.requires_grad = False

    if 'lam' in config:
        model = TQN_Model(class_num = args.class_num, lam = config['lam']).cuda()
    else:
        model = TQN_Model(class_num = args.class_num).cuda()  
        
    if args.distributed:
        # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True, device_ids=[args.local_rank], output_device=args.local_rank)
        model = torch.nn.DataParallel(model)
        image_encoder = torch.nn.DataParallel(image_encoder)
        
        
    return model, image_encoder, text_encoder, tokenizer


def load_checkpoint(model, image_encoder, args):


        checkpoint = torch.load(args.finetune, map_location='cpu')

        state_dict = checkpoint['model']   
        new_state_dict = OrderedDict()
        if 'module.' in list(model.state_dict().keys())[0] and 'module.' not in list(state_dict.keys())[0]:
            for k, v in state_dict.items():
                name =  'module.' + k
                new_state_dict[name] = v
        elif 'module.' not in list(model.state_dict().keys())[0] and 'module.' in list(state_dict.keys())[0]:
            for k, v in state_dict.items():
                name =  k.replace('module.', '')
                new_state_dict[name] = v
        else:
            new_state_dict = state_dict
        model.load_state_dict(new_state_dict, strict=False) 




        ckpt_cxr14 = torch.load('./results/checkpoint_15.pt', map_location='cpu')
        image_state_dict_cxr14 =  ckpt_cxr14['image_encoder']
        cxr14_state_dict = OrderedDict()
        for key, value in image_state_dict_cxr14.items():
                updated_key = key
                updated_key = updated_key.replace('module.', '')
                cxr14_state_dict[updated_key] = value

        ckpt_cxp = torch.load('./results/checkpoint_12.pt', map_location='cpu')
        image_state_dict_cxp =  ckpt_cxp['image_encoder']
        cxp_state_dict = OrderedDict()
        for key, value in image_state_dict_cxp.items():
                updated_key = key.replace('resnet.', 'resnet1.')
                updated_key = updated_key.replace('res_features.', 'res_features1.')
                updated_key = updated_key.replace('res_l1', 'res_l1_1')
                updated_key = updated_key.replace('res_l2', 'res_l2_1')
                updated_key = updated_key.replace('module.', '')
                cxp_state_dict[updated_key] = value
                
        ckpt_vindr = torch.load('./results/checkpoint_9.pt', map_location='cpu')
        image_state_dict_vindr =  ckpt_vindr['image_encoder']
        vindr_state_dict = OrderedDict()
        for key, value in image_state_dict_vindr.items():
                updated_key = key.replace('resnet.', 'resnet2.')
                updated_key = updated_key.replace('res_features.', 'res_features2.')
                updated_key = updated_key.replace('res_l1', 'res_l1_2')
                updated_key = updated_key.replace('res_l2', 'res_l2_2')
                updated_key = updated_key.replace('module.', '')
                vindr_state_dict[updated_key] = value

        
        state_dict_list = [cxr14_state_dict, cxp_state_dict, vindr_state_dict]
        merged_state_dict = OrderedDict()

        for state_dict in state_dict_list:
            for key, value in state_dict.items():
                merged_state_dict[key] = value

        image_encoder.load_state_dict(merged_state_dict, strict=True)


def test_all(model, dataset_list, image_encoder, text_encoder, tokenizer, val_dataloader, test_dataloader, test_dataloader_chexpert, test_dataloader_vinder, test_dataloader_siimacr, test_dataloader_openi, test_dataloader_shenzhen, test_dataloader_padchest, test_dataloader_mimic, device, args, config, writer, epoch=0, total_test=False):
    epoch = epoch
    total_metrics = {}
    with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
        f.write("Time: "+str(time.strftime("%Y-%m-%d-%H-%M", time.localtime()))+" Epoch: "+str(epoch)+"\n")
    if 'chexpert' in dataset_list:
        valid_on_cheXpert(model, image_encoder, text_encoder, tokenizer, test_dataloader_chexpert ,epoch, device, args, config, writer, total_test)


    if 'chestxray' in dataset_list:
        valid_on_chestxray14(model, image_encoder, text_encoder, tokenizer, test_dataloader, epoch, device, args, config, writer, total_test)

        
    
    if 'vindr' in dataset_list:        
        valid_on_vindr(model, image_encoder, text_encoder, tokenizer, test_dataloader_vinder, epoch, device, args, config, writer, total_test)

    if 'siimacr' in dataset_list:
        valid_on_siimacr(model, image_encoder, text_encoder, tokenizer, test_dataloader_siimacr, epoch, device, args, config, writer, total_test)

    
    if 'openi' in dataset_list:
        valid_on_openi(model, image_encoder, text_encoder, tokenizer, test_dataloader_openi, epoch, device, args, config, writer, total_test)

    
    if 'shenzhen' in dataset_list:
        valid_on_shenzhen(model, image_encoder, text_encoder, tokenizer, test_dataloader_shenzhen, epoch, device, args, config, writer, total_test)

    valid_on_padchest(model, image_encoder, text_encoder, tokenizer, test_dataloader_padchest, epoch, device, args, config, writer, total_test)


def main(args, config):
    '''Data准备'''
    train_dataloader, val_dataloader, test_dataloader, test_dataloader_chexpert, test_dataloader_vinder, test_dataloader_siimacr, test_dataloader_openi, test_dataloader_shenzhen, test_dataloader_padchest, test_dataloader_mimic, train_dataset, val_dataset, test_dataset, test_dataset_chexpert, test_dataset_vindr, test_dataset_siimacr, test_dataset_openi, test_dataset_shenzhen, test_dataset_padchest, test_dataset_mimic = get_dataloader(args, config)
    train_dataloader.num_samples = len(train_dataset)
    train_dataloader.num_batches = len(train_dataloader)  
       
    val_dataloader.num_samples = len(val_dataset)
    val_dataloader.num_batches = len(val_dataloader)     
    
    test_dataloader.num_samples = len(test_dataset)
    test_dataloader.num_batches = len(test_dataloader) 
      
    test_dataloader_chexpert.num_samples = len(test_dataset_chexpert)
    test_dataloader_chexpert.num_batches = len(test_dataloader_chexpert)
    
    test_dataloader_vinder.num_samples = len(test_dataset_vindr)
    test_dataloader_vinder.num_batches = len(test_dataloader_vinder)
    
    test_dataloader_siimacr.num_samples = len(test_dataset_siimacr)
    test_dataloader_siimacr.num_batches = len(test_dataloader_siimacr)
    
    test_dataloader_openi.num_samples = len(test_dataset_openi)
    test_dataloader_openi.num_batches = len(test_dataloader_openi)
    
    test_dataloader_shenzhen.num_samples = len(test_dataset_shenzhen)
    test_dataloader_shenzhen.num_batches = len(test_dataloader_shenzhen)
    
    test_dataloader_padchest.num_samples = len(test_dataset_padchest)
    test_dataloader_padchest.num_batches = len(test_dataloader_padchest)
    
    test_dataloader_mimic.num_samples = len(test_dataset_mimic)
    test_dataloader_mimic.num_batches = len(test_dataloader_mimic)
    
    '''Model准备'''
    model, image_encoder, text_encoder, tokenizer = get_model(args, config)
    
    writer = SummaryWriter(os.path.join(args.output_dir,  'log'))
    dataset_list = config['dataset_list']
    print(dataset_list)
    if os.path.isdir(args.finetune):
        # 获取所有带.pt的文件 然后一个个做valid
        checkpoint_list = [os.path.join(args.finetune, f) for f in os.listdir(args.finetune) if f.endswith('.pt')]
        checkpoint_list.sort()
        for checkpoint in checkpoint_list:
            epoch = checkpoint.split('/')[-1].split('_')[-1].split('.')[0]
            epoch = int(epoch)
            args.finetune = checkpoint
            load_checkpoint(model, image_encoder, args) 
            print('epoch: ', epoch)
            test_all(model, dataset_list, image_encoder, text_encoder, tokenizer, val_dataloader, test_dataloader, test_dataloader_chexpert, test_dataloader_vinder, test_dataloader_siimacr, test_dataloader_openi, test_dataloader_shenzhen, test_dataloader_padchest, test_dataloader_mimic, args.device, args, config, writer, epoch=epoch, total_test=True)
            
    else:    
        load_checkpoint(model, image_encoder, args) 
        test_all(model, dataset_list, image_encoder, text_encoder, tokenizer, val_dataloader, test_dataloader, test_dataloader_chexpert, test_dataloader_vinder, test_dataloader_siimacr, test_dataloader_openi, test_dataloader_shenzhen, test_dataloader_padchest, test_dataloader_mimic, args.device, args, config, writer, epoch=0, total_test=True)

if __name__ == '__main__':
    
    # 单卡速度太慢 考虑多卡
    
    # print(os.environ)
    abs_file_path = os.path.abspath(__file__)
    abs_file_path = abs_file_path.replace('main/test_v2.py', '')
    # print(abs_file_path)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument('--momentum', default=False, type=bool)
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--finetune', default='best_valid.pt')
    
    parser.add_argument('--freeze_bert', default=True, type=bool)
    parser.add_argument("--use_entity_features", default=True, type=bool)

    # Config
    parser.add_argument('--config', default=f'v3.0_base.yaml')

    parser.add_argument('--fourier', default=True, type=bool)
    parser.add_argument('--colourjitter', default=True, type=bool)
    
    parser.add_argument('--class_num', default=1, type=int) # FT1, FF2


    parser.add_argument('--ignore_index', default=True, type=bool)
    parser.add_argument('--add_dataset', default=True, type=bool) 


    # Path 指向相同的位置
    time_now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    parser.add_argument('--output_dir', default=f'{abs_file_path}/results/test-{time_now}')
    parser.add_argument('--aws_output_dir', default=f'{abs_file_path}/results/test-{time_now}')
    
    parser.add_argument('--bert_pretrained', default=f'{abs_file_path}/data/pretrained_weights/bert_pretrained/epoch_latest.pt')
    parser.add_argument('--bert_model_name', default=f'{abs_file_path}/data/pretrained_weights/UMLSBert_ENG')
    parser.add_argument('--max_length', default=256, type=int)
    parser.add_argument('--loss_ratio', default=1, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    # distributed training parameters
    parser.add_argument("--local_rank", type=int) # 单机多卡 master node的rank为0
    parser.add_argument('--distributed', action='store_true', default=False, help='Use multi-processing distributed training to launch ')
    
    parser.add_argument('--rho', default=0, type=float, help='gpu')
    parser.add_argument('--gpu', default=0, type=int, help='gpu')
    args = parser.parse_args()
    args.output_dir = args.output_dir + '-' + args.config.replace('.yaml', '')
    args.aws_output_dir = args.aws_output_dir + '-' + args.config.replace('.yaml', '')
    args.config = f'{abs_file_path}/configs/{args.config}'
    
    # print(args)
    # args.distributed = True
    
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    if config['finetune'] != '':
        args.finetune = config['finetune']
        args.checkpoint = config['finetune']
    
    args.loss_ratio = config['loss_ratio']
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.aws_output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))  


    seed_torch(args.seed)

    main(args, config)


