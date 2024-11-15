'''
2023.10.16
只测试 考虑多种测试方式

'''

import sys
# 加入父文件夹路径到sys.path中  
sys.path.append(sys.path[0].replace('main', ''))

from factory.loss import AsymmetricLoss, Ralloss
import torch.nn as nn
import os
import numpy as np
import random
import time
import json
import yaml
from pathlib import Path
import logging
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader

from transformers import AutoModel, BertConfig, AutoTokenizer

from engine.train import train,train_sam, valid_on_cheXpert, valid_on_chestxray14, valid_on_siimacr, valid_on_vindr, valid_on_openi, valid_on_shenzhen, valid_on_padchest, valid_on_mimic
from engine.mytrain import valid_nokad, valid_single_dataset_moe
from models.clip_tqn import CLP_clinical, ModelRes, TQN_Model, ModelConvNeXt, ModelEfficientV2, ModelDense, ModelViT
from models.clip_tqn_lora import ModelRes_Lora
from models.clip_tqn_lora_multi_end import ModelRes_Lora_Multi_End
from data.dataset_entity import MIMIC_Dataset, Mergetrain_Dataset, SIIMACR_train_Dataset
from data.dataset import Mergetrain_New_Dataset
from data.test_dataset import Vindr_Dataset, SIIMACR_Dataset, Openi_Dataset, Shenzhen_Dataset, Padchest_Dataset, Chestxray14_Dataset, CheXpert_Dataset, MIMIC_test_Dataset

import argparse

def get_args(log_prefix=''):
    abs_file_path = os.path.abspath(__file__)
    abs_file_path = abs_file_path.replace('main/main_utils.py', '')
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

    # V1-1: Trick 1: Data Augmentation
    parser.add_argument('--fourier', default=True, type=bool)
    parser.add_argument('--colourjitter', default=True, type=bool)
    
    # V1-1: Trick 2: ASL loss & DQN output_dim(原先CE Loss,所以class_num=2, 改为BCE Loss或ASL Loss后改为1)
    parser.add_argument('--class_num', default=1, type=int) # FT1, FF2

    # Dataset Enhance
    # V1-1: 两个都是False, V1-1+Data: 两个都是True
    parser.add_argument('--ignore_index', default=True, type=bool) #原始为false; +data时-1作为标记不算loss, 改成True
    parser.add_argument('--add_dataset', default=True, type=bool) 
    
    parser.add_argument('--use_dataset', type=str, default='all') # 考虑可以单独使用mimic,chex,vindr,cxr14 中间用-连接


    # Path 指向相同的位置
    time_now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    parser.add_argument('--output_dir', default=f'{abs_file_path}/results/{time_now}-{log_prefix}')
    parser.add_argument('--aws_output_dir', default=f'{abs_file_path}/results/{time_now}-{log_prefix}')
    
    # parser.add_argument('--image_encoder_name', default='resnet', choices=['resnet', 'convnext-base', 'convnext-small', 'convnext-tiny'])
    parser.add_argument('--bert_pretrained', default=f'{abs_file_path}/data/pretrained_weights/bert_pretrained/epoch_latest.pt')
    parser.add_argument('--bert_model_name', default=f'{abs_file_path}/data/pretrained_weights/UMLSBert_ENG')
    parser.add_argument('--max_length', default=256, type=int)
    parser.add_argument('--loss_ratio', default=1, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Use multi-processing distributed training to launch ')
    
    # EMA
    parser.add_argument('--ema', default=0.0, type=float, help='ema') # 0.0表示不使用ema
    parser.add_argument('--adv', default=1.0, type=float, help='adversial weight') # 0.0表示不使用ema
    
    
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
    
    if 'rho' in config:
        args.rho = config['rho']
    if 'use_dataset' in config:
        args.use_dataset = config['use_dataset']
    config['ema'] = args.ema
    config['adv'] = args.adv
    
    args.loss_ratio = config['loss_ratio']
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.aws_output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))  

    logging.info("Params:")
    params_file = os.path.join(args.output_dir, "params.txt")
    with open(params_file, "w") as f:
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")
            f.write(f"{name}: {val}\n")
    return args, config

def seed_torch(seed=42):
    # if os.environ['LOCAL_RANK'] == '0':
    print('=====> Using fixed random seed: ' + str(seed))
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloader(args, config):
    
    #### Dataset #### 
    # if args.distributed and os.environ['LOCAL_RANK'] == '0':
    print("Creating dataset")
    
    if args.add_dataset == True:
        train_dataset = Mergetrain_Dataset(config['train_entity_file'], config['train_fg_query_file_v1'], config['mrsty_file'],config['image_res'], args)

    else:
        train_dataset = MIMIC_Dataset(config['train_entity_file'], config['train_fg_query_file_v1'], config['mrsty_file'],config['image_res'], args)
    
    if config['ft_siimacr']:
        train_dataset = SIIMACR_train_Dataset(config['train_entity_file'], config['ft_siimacr'], config['mrsty_file'],config['image_res'], args)
    
    val_dataset = Chestxray14_Dataset(config['chestxray_valid_file'],config['image_res'])
    test_dataset = Chestxray14_Dataset(config['chestxray_test_file'],config['image_res'])
    test_dataset_chexpert = CheXpert_Dataset(config['chexpert_test_file'],config['image_res'])
    test_dataset_vindr = Vindr_Dataset(config['vindrcxr_test_file'],config['image_res'])
    test_dataset_siimacr = SIIMACR_Dataset(config['siimacr_file'],config['image_res'])
    test_dataset_openi = Openi_Dataset(config['openi_test_file'],config['image_res'])
    test_dataset_shenzhen = Shenzhen_Dataset(config['shenzhen_test_file'],config['image_res'])
    test_dataset_padchest = Padchest_Dataset(config['padchest_test_file'],config['image_res'])
    test_dataset_mimic = MIMIC_test_Dataset(config['mimic_test_file'],config['image_res'])
    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    #     train_dataloader = DataLoader(
    #         train_dataset,
    #         batch_size=config['batch_size'],
    #         num_workers=config["num_workers"],
    #         pin_memory=True,
    #         sampler=train_sampler, 
    #         # sampler=None,
    #         collate_fn=None,
    #         # shuffle=True,
    #         drop_last=True,
    #     ) 

    # else:
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config["test_num_workers"],
        pin_memory=True,
        collate_fn=None,
        shuffle=True,
        drop_last=True,
    )   

    val_dataloader =DataLoader(
            val_dataset,
            batch_size=config['test_batch_size'],
            num_workers=config["test_num_workers"],
            pin_memory=True,
            collate_fn=None,
            shuffle=False,
            drop_last=False,
        ) 
    
    test_dataloader =DataLoader(
            test_dataset,
            batch_size=config['test_batch_size'],
            num_workers=config["test_num_workers"],
            pin_memory=True,
            collate_fn=None,
            shuffle=False,
            drop_last=False,
        ) 
    
    test_dataloader_chexpert =DataLoader(
            test_dataset_chexpert,
            batch_size=config['test_batch_size'],
            num_workers=config["test_num_workers"],
            pin_memory=True,
            collate_fn=None,
            shuffle=False,
            drop_last=False,
        )
    
    test_dataloader_vinder = DataLoader(
            test_dataset_vindr,
            batch_size=config['test_batch_size'],
            num_workers=config["test_num_workers"],
            pin_memory=True,
            collate_fn=None,
            shuffle=False,
            drop_last=False,   
    )
    
    test_dataloader_siimacr = DataLoader(
            test_dataset_siimacr,
            batch_size=config['test_batch_size'],
            num_workers=config["test_num_workers"],
            pin_memory=True,
            collate_fn=None,
            shuffle=False,
            drop_last=False,   
    )
    
    test_dataloader_openi = DataLoader(
            test_dataset_openi,
            batch_size=config['test_batch_size'],
            num_workers=config["test_num_workers"],
            pin_memory=True,
            collate_fn=None,
            shuffle=False,
            drop_last=False,   
    )
    
    test_dataloader_shenzhen = DataLoader(
            test_dataset_shenzhen,
            batch_size=config['test_batch_size'],
            num_workers=config["test_num_workers"],
            pin_memory=True,
            collate_fn=None,
            shuffle=False,
            drop_last=False,   
    )
    
    test_dataloader_padchest = DataLoader(
        test_dataset_padchest,
            batch_size=config['test_batch_size'],
            num_workers=config["test_num_workers"],
            pin_memory=True,
            collate_fn=None,
            shuffle=False,
            drop_last=False,   
    )
    
    test_dataloader_mimic = DataLoader(
        test_dataset_mimic,
            batch_size=config['test_batch_size'],
            num_workers=config["test_num_workers"],
            pin_memory=True,
            collate_fn=None,
            shuffle=False,
            drop_last=False,
    )
    
    return train_dataloader, val_dataloader, test_dataloader, test_dataloader_chexpert, test_dataloader_vinder, test_dataloader_siimacr, test_dataloader_openi, test_dataloader_shenzhen, test_dataloader_padchest, test_dataloader_mimic, train_dataset, val_dataset, test_dataset, test_dataset_chexpert, test_dataset_vindr, test_dataset_siimacr, test_dataset_openi, test_dataset_shenzhen, test_dataset_padchest, test_dataset_mimic

def get_total_dataloader(args, config):
    
    #### Dataset #### 
    # if args.distributed and os.environ['LOCAL_RANK'] == '0':
    print("Creating dataset")
    
    if args.add_dataset == True:
        train_dataset = Mergetrain_New_Dataset(config['train_entity_file'], config['train_fg_query_file_v1'], config['mrsty_file'],config['image_res'], args)

    else:
        train_dataset = MIMIC_Dataset(config['train_entity_file'], config['train_fg_query_file_v1'], config['mrsty_file'],config['image_res'], args)
    
    
    val_dataset = Chestxray14_Dataset(config['chestxray_valid_file'],config['image_res'])
    test_dataset = Chestxray14_Dataset(config['chestxray_test_file'],config['image_res'])
    test_dataset_chexpert = CheXpert_Dataset(config['chexpert_test_file'],config['image_res'])
    test_dataset_vindr = Vindr_Dataset(config['vindrcxr_test_file'],config['image_res'])
    test_dataset_siimacr = SIIMACR_Dataset(config['siimacr_file'],config['image_res'])
    test_dataset_openi = Openi_Dataset(config['openi_test_file'],config['image_res'])
    test_dataset_shenzhen = Shenzhen_Dataset(config['shenzhen_test_file'],config['image_res'])
    test_dataset_padchest = Padchest_Dataset(config['padchest_test_file'],config['image_res'])
    test_dataset_mimic = MIMIC_test_Dataset(config['mimic_test_file'],config['image_res'])
    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    #     train_dataloader = DataLoader(
    #         train_dataset,
    #         batch_size=config['batch_size'],
    #         num_workers=config["num_workers"],
    #         pin_memory=True,
    #         sampler=train_sampler, 
    #         # sampler=None,
    #         collate_fn=None,
    #         # shuffle=True,
    #         drop_last=True,
    #     ) 

    # else:
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config["test_num_workers"],
        pin_memory=True,
        collate_fn=None,
        shuffle=True,
        drop_last=True,
    )   

    val_dataloader =DataLoader(
            val_dataset,
            batch_size=config['test_batch_size'],
            num_workers=config["test_num_workers"],
            pin_memory=True,
            collate_fn=None,
            shuffle=False,
            drop_last=False,
        ) 
    
    test_dataloader =DataLoader(
            test_dataset,
            batch_size=config['test_batch_size'],
            num_workers=config["test_num_workers"],
            pin_memory=True,
            collate_fn=None,
            shuffle=False,
            drop_last=False,
        ) 
    
    test_dataloader_chexpert =DataLoader(
            test_dataset_chexpert,
            batch_size=config['test_batch_size'],
            num_workers=config["test_num_workers"],
            pin_memory=True,
            collate_fn=None,
            shuffle=False,
            drop_last=False,
        )
    
    test_dataloader_vinder = DataLoader(
            test_dataset_vindr,
            batch_size=config['test_batch_size'],
            num_workers=config["test_num_workers"],
            pin_memory=True,
            collate_fn=None,
            shuffle=False,
            drop_last=False,   
    )
    
    test_dataloader_siimacr = DataLoader(
            test_dataset_siimacr,
            batch_size=config['test_batch_size'],
            num_workers=config["test_num_workers"],
            pin_memory=True,
            collate_fn=None,
            shuffle=False,
            drop_last=False,   
    )
    
    test_dataloader_openi = DataLoader(
            test_dataset_openi,
            batch_size=config['test_batch_size'],
            num_workers=config["test_num_workers"],
            pin_memory=True,
            collate_fn=None,
            shuffle=False,
            drop_last=False,   
    )
    
    test_dataloader_shenzhen = DataLoader(
            test_dataset_shenzhen,
            batch_size=config['test_batch_size'],
            num_workers=config["test_num_workers"],
            pin_memory=True,
            collate_fn=None,
            shuffle=False,
            drop_last=False,   
    )
    
    test_dataloader_padchest = DataLoader(
        test_dataset_padchest,
            batch_size=config['test_batch_size'],
            num_workers=config["test_num_workers"],
            pin_memory=True,
            collate_fn=None,
            shuffle=False,
            drop_last=False,   
    )
    
    test_dataloader_mimic = DataLoader(
        test_dataset_mimic,
            batch_size=config['test_batch_size'],
            num_workers=config["test_num_workers"],
            pin_memory=True,
            collate_fn=None,
            shuffle=False,
            drop_last=False,
    )
    
    return train_dataloader, val_dataloader, test_dataloader, test_dataloader_chexpert, test_dataloader_vinder, test_dataloader_siimacr, test_dataloader_openi, test_dataloader_shenzhen, test_dataloader_padchest, test_dataloader_mimic, train_dataset, val_dataset, test_dataset, test_dataset_chexpert, test_dataset_vindr, test_dataset_siimacr, test_dataset_openi, test_dataset_shenzhen, test_dataset_padchest, test_dataset_mimic

def load_checkpoint(model, image_encoder, args):
    if os.path.isfile(args.finetune):
        checkpoint = torch.load(args.finetune, map_location='cpu')
        image_state_dict =  checkpoint['image_encoder']
        new_image_state_dict = OrderedDict()
        if 'module.' in list(image_encoder.state_dict().keys())[0] and 'module.' not in list(image_state_dict.keys())[0]:
            for k, v in image_state_dict.items():
                name =  'module.' + k
                new_image_state_dict[name] = v
        elif 'module.' not in list(image_encoder.state_dict().keys())[0] and 'module.' in list(image_state_dict.keys())[0]:
            for k, v in image_state_dict.items():
                name =  k.replace('module.', '')
                new_image_state_dict[name] = v
        else:
            new_image_state_dict = image_state_dict
        image_encoder.load_state_dict(new_image_state_dict, strict=False)
        
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

def get_criterion(config):
    # 定义训练loss
    if config['criterion'] == 'bce':
        criterion = nn.BCEWithLogitsLoss()    
    elif config['criterion'] == 'asl':
        criterion = AsymmetricLoss(gamma_neg=6, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    elif config['criterion'] == 'ral':
        criterion = Ralloss(gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, lamb=1.5, epsilon_neg=0.0, epsilon_pos=1.0, epsilon_pos_pow=-2.5, disable_torch_grad_focal_loss=False)
    
    return criterion

def get_model(args, config):
    if 'end' in config['image_encoder_name']:
        image_encoder = ModelRes_Lora_Multi_End(config['image_encoder_name'], config['lora_r'], config['lora_alpha']).cuda()
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
    # if args.distributed:
        # text_encoder = torch.nn.DataParallel(text_encoder)


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

    

def record_results(epoch, losses, metrics, dataset_name, writer, args):
    auc = metrics['mean_auc']
    f1 = metrics['mean_f1']
    acc = metrics['mean_acc']
    ap = metrics['mean_ap']
    writer.add_scalar(f'loss/{dataset_name}_loss_epoch', losses, epoch)
    writer.add_scalar(f'loss/{dataset_name}_auc_epoch', auc, epoch)
    writer.add_scalar(f'loss/{dataset_name}_f1_epoch', f1, epoch)
    writer.add_scalar(f'loss/{dataset_name}_acc_epoch', acc, epoch)
    writer.add_scalar(f'loss/{dataset_name}_ap_epoch', ap, epoch)
    
    # 记录所有的metrics
    for key in metrics:
        writer.add_scalar(f'{dataset_name}/{key}', metrics[key], epoch)
    
    # log.txt记录val_loss和val_auc val_metrics
    log_stats = {'epoch': epoch, f'{dataset_name}_loss': losses.item(), f'{dataset_name}_mean_auc': auc, f'{dataset_name}_mean_f1': f1, f'{dataset_name}_mean_acc': acc, f'{dataset_name}_mean_ap': ap}
    with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
        f.write(json.dumps(log_stats) + "\n")


def test_all(model, image_encoder, text_encoder, tokenizer, val_dataloader, test_dataloader, test_dataloader_chexpert, test_dataloader_vinder, test_dataloader_siimacr, test_dataloader_openi, test_dataloader_shenzhen, test_dataloader_padchest, test_dataloader_mimic, device, args, config, writer, epoch=0, total_test=False):
    epoch = epoch
    # Log.txt记录时间和epoch数
    with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
        f.write("Time: "+str(time.strftime("%Y-%m-%d-%H-%M", time.localtime()))+" Epoch: "+str(epoch)+"\n")
    
    val_loss,val_auc,val_metrics = valid_on_chestxray14(model, image_encoder, text_encoder, tokenizer, val_dataloader, epoch, device, args, config, writer, total_test)
    record_results(epoch, val_loss, val_metrics, 'val', writer, args)

    chexpert_val_loss, chexpert_val_auc, chexpert_val_metrics = valid_on_cheXpert(model, image_encoder, text_encoder, tokenizer, test_dataloader_chexpert ,epoch, device, args, config, writer, total_test)
    record_results(epoch, chexpert_val_loss, chexpert_val_metrics, 'chexpert_val', writer, args)
                
    test_loss, test_auc, test_metrics = valid_on_chestxray14(model, image_encoder, text_encoder, tokenizer, test_dataloader, epoch, device, args, config, writer, total_test)
    record_results(epoch, test_loss, test_metrics, 'test', writer, args)

    vindr_test_loss, vindr_test_auc, vindr_test_metrics = valid_on_vindr(model, image_encoder, text_encoder, tokenizer, test_dataloader_vinder, epoch, device, args, config, writer, total_test)
    record_results(epoch, vindr_test_loss, vindr_test_metrics, 'vindr_test', writer, args)
        
    siimacr_test_loss, siimacr_test_auc, siimacr_test_metrics = valid_on_siimacr(model, image_encoder, text_encoder, tokenizer, test_dataloader_siimacr, epoch, device, args, config, writer, total_test)
    record_results(epoch, siimacr_test_loss, siimacr_test_metrics, 'siimacr_test', writer, args)
    
    openi_test_loss, openi_test_auc, openi_test_metrics = valid_on_openi(model, image_encoder, text_encoder, tokenizer, test_dataloader_openi, epoch, device, args, config, writer, total_test)
    record_results(epoch, openi_test_loss, openi_test_metrics, 'openi_test', writer, args)
    
    shenzhen_test_loss, shenzhen_test_auc, shenzhen_test_metrics = valid_on_shenzhen(model, image_encoder, text_encoder, tokenizer, test_dataloader_shenzhen, epoch, device, args, config, writer, total_test)
    record_results(epoch, shenzhen_test_loss, shenzhen_test_metrics, 'shenzhen_test', writer, args)
    
    # padchest_test_loss, padchest_test_auc, padchest_test_metrics = valid_on_padchest(model, image_encoder, text_encoder, tokenizer, test_dataloader_padchest, epoch, device, args, config, writer, total_test)
    # record_results(epoch, padchest_test_loss, padchest_test_metrics, 'padchest_test', writer, args)
    
    # mimic_test_loss, mimic_test_auc, mimic_test_metrics = valid_on_mimic(model, image_encoder, text_encoder, tokenizer, test_dataloader_mimic, epoch, device, args, config, writer, total_test)
    # record_results(epoch, mimic_test_loss, mimic_test_metrics, 'mimic_test', writer, args)
    
    padchest_test_auc = 0.
    mimic_test_auc = 0.
    # val_auc = 0.
    # chexpert_val_auc = 0.
    # test_auc = 0.
    # vindr_test_auc = 0.
    
    return val_auc, chexpert_val_auc, test_auc, vindr_test_auc, siimacr_test_auc, openi_test_auc, shenzhen_test_auc, padchest_test_auc, mimic_test_auc
 
def test_all_moe(model, moe_model_list, image_encoder, text_encoder, tokenizer, val_dataloader, test_dataloader, test_dataloader_chexpert, test_dataloader_vinder, test_dataloader_siimacr, test_dataloader_openi, test_dataloader_shenzhen, test_dataloader_padchest, test_dataloader_mimic, device, args, config, writer, epoch=0, total_test=False):
    epoch = epoch
    # Log.txt记录时间和epoch数
    with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
        f.write("Time: "+str(time.strftime("%Y-%m-%d-%H-%M", time.localtime()))+" Epoch: "+str(epoch)+"\n")
    
    val_loss,val_auc,val_metrics = valid_single_dataset_moe(model, moe_model_list, image_encoder, text_encoder, tokenizer, 'val_chestxray', val_dataloader, epoch, device, args, config, writer, total_test)
    record_results(epoch, val_loss, val_metrics, 'val', writer, args)

    chexpert_val_loss, chexpert_val_auc, chexpert_val_metrics = valid_single_dataset_moe(model, moe_model_list, image_encoder, text_encoder, tokenizer, 'chexpert', test_dataloader_chexpert ,epoch, device, args, config, writer, total_test)
    record_results(epoch, chexpert_val_loss, chexpert_val_metrics, 'chexpert_val', writer, args)
                
    test_loss, test_auc, test_metrics = valid_single_dataset_moe(model, moe_model_list, image_encoder, text_encoder, tokenizer, 'test_chestxray', test_dataloader, epoch, device, args, config, writer, total_test)
    record_results(epoch, test_loss, test_metrics, 'test', writer, args)

    vindr_test_loss, vindr_test_auc, vindr_test_metrics = valid_single_dataset_moe(model, moe_model_list, image_encoder, text_encoder, tokenizer, 'vindr', test_dataloader_vinder, epoch, device, args, config, writer, total_test)
    record_results(epoch, vindr_test_loss, vindr_test_metrics, 'vindr_test', writer, args)
        
    siimacr_test_loss, siimacr_test_auc, siimacr_test_metrics = valid_single_dataset_moe(model, moe_model_list, image_encoder, text_encoder, tokenizer, 'siimacr', test_dataloader_siimacr, epoch, device, args, config, writer, total_test)
    record_results(epoch, siimacr_test_loss, siimacr_test_metrics, 'siimacr_test', writer, args)
    
    openi_test_loss, openi_test_auc, openi_test_metrics = valid_single_dataset_moe(model, moe_model_list, image_encoder, text_encoder, tokenizer, 'openi', test_dataloader_openi, epoch, device, args, config, writer, total_test)
    record_results(epoch, openi_test_loss, openi_test_metrics, 'openi_test', writer, args)
    
    shenzhen_test_loss, shenzhen_test_auc, shenzhen_test_metrics = valid_single_dataset_moe(model, moe_model_list, image_encoder, text_encoder, tokenizer, 'shenzhen', test_dataloader_shenzhen, epoch, device, args, config, writer, total_test)
    record_results(epoch, shenzhen_test_loss, shenzhen_test_metrics, 'shenzhen_test', writer, args)
    
    # padchest_test_loss, padchest_test_auc, padchest_test_metrics = valid_on_padchest(model, image_encoder, text_encoder, tokenizer, test_dataloader_padchest, epoch, device, args, config, writer, total_test)
    # record_results(epoch, padchest_test_loss, padchest_test_metrics, 'padchest_test', writer, args)
    
    # mimic_test_loss, mimic_test_auc, mimic_test_metrics = valid_on_mimic(model, image_encoder, text_encoder, tokenizer, test_dataloader_mimic, epoch, device, args, config, writer, total_test)
    # record_results(epoch, mimic_test_loss, mimic_test_metrics, 'mimic_test', writer, args)
    
    padchest_test_auc = 0.
    mimic_test_auc = 0.
    # val_auc = 0.
    # chexpert_val_auc = 0.
    # test_auc = 0.
    # vindr_test_auc = 0.
    
    return val_auc, chexpert_val_auc, test_auc, vindr_test_auc, siimacr_test_auc, openi_test_auc, shenzhen_test_auc, padchest_test_auc, mimic_test_auc
 
 
def test_all_no_kad(model, val_dataloader, test_dataloader, test_dataloader_chexpert, test_dataloader_vinder, test_dataloader_siimacr, test_dataloader_openi, test_dataloader_shenzhen, test_dataloader_padchest, test_dataloader_mimic, device, args, config, writer, epoch=0, total_test=False):
    epoch = epoch
    # Log.txt记录时间和epoch数
    with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
        f.write("Time: "+str(time.strftime("%Y-%m-%d-%H-%M", time.localtime()))+" Epoch: "+str(epoch)+"\n")
    
    val_loss,val_auc,val_metrics = valid_nokad(model, 'val_chestxray', val_dataloader, epoch, device, args, config, writer, total_test)
    record_results(epoch, val_loss, val_metrics, 'val', writer, args)

    chexpert_val_loss, chexpert_val_auc, chexpert_val_metrics = valid_nokad(model, 'chexpert', test_dataloader_chexpert ,epoch, device, args, config, writer, total_test)
    record_results(epoch, chexpert_val_loss, chexpert_val_metrics, 'chexpert_val', writer, args)
                
    test_loss, test_auc, test_metrics = valid_nokad(model, 'test_chestxray', test_dataloader, epoch, device, args, config, writer, total_test)
    record_results(epoch, test_loss, test_metrics, 'test', writer, args)

    vindr_test_loss, vindr_test_auc, vindr_test_metrics = valid_nokad(model, 'vindr', test_dataloader_vinder, epoch, device, args, config, writer, total_test)
    record_results(epoch, vindr_test_loss, vindr_test_metrics, 'vindr_test', writer, args)
        
    siimacr_test_loss, siimacr_test_auc, siimacr_test_metrics = valid_nokad(model, 'siimacr', test_dataloader_siimacr, epoch, device, args, config, writer, total_test)
    record_results(epoch, siimacr_test_loss, siimacr_test_metrics, 'siimacr_test', writer, args)
    
    openi_test_loss, openi_test_auc, openi_test_metrics = valid_nokad(model, 'openi', test_dataloader_openi, epoch, device, args, config, writer, total_test)
    record_results(epoch, openi_test_loss, openi_test_metrics, 'openi_test', writer, args)
    
    shenzhen_test_loss, shenzhen_test_auc, shenzhen_test_metrics = valid_nokad(model, 'shenzhen', test_dataloader_shenzhen, epoch, device, args, config, writer, total_test)
    record_results(epoch, shenzhen_test_loss, shenzhen_test_metrics, 'shenzhen_test', writer, args)
    
    # padchest_test_loss, padchest_test_auc, padchest_test_metrics = valid_nokad(model, image_encoder, text_encoder, tokenizer, test_dataloader_padchest, epoch, device, args, config, writer, total_test)
    # record_results(epoch, padchest_test_loss, padchest_test_metrics, 'padchest_test', writer, args)
    
    # mimic_test_loss, mimic_test_auc, mimic_test_metrics = valid_nokad(model, image_encoder, text_encoder, tokenizer, test_dataloader_mimic, epoch, device, args, config, writer, total_test)
    # record_results(epoch, mimic_test_loss, mimic_test_metrics, 'mimic_test', writer, args)
    
    padchest_test_auc = 0.
    mimic_test_auc = 0.
    # val_auc = 0.
    # chexpert_val_auc = 0.
    # test_auc = 0.
    # vindr_test_auc = 0.
    
    return val_auc, chexpert_val_auc, test_auc, vindr_test_auc, siimacr_test_auc, openi_test_auc, shenzhen_test_auc, padchest_test_auc, mimic_test_auc
       
              
from copy import deepcopy
def get_ema_model(checkpoint_path, start_epoch, end_epoch):
    image_state_dict = {}
    model_state_dict = {}
    for epoch_num in range(start_epoch, end_epoch+1):
        checkpoint = os.path.join(checkpoint_path, f'checkpoint_{epoch_num}.pt')
        checkpoint = torch.load(checkpoint, map_location='cpu')
        if image_state_dict == {}:
            image_state_dict = deepcopy(checkpoint['image_encoder'])
            model_state_dict = deepcopy(checkpoint['model'])
        else:
            for k,v in checkpoint['image_encoder'].items():
                image_state_dict[k] = v + image_state_dict[k]

            for k,v in checkpoint['model'].items():
                model_state_dict[k] = v + model_state_dict[k]
        
    for k,v in image_state_dict.items():
        if 'num_batches_tracked' in k:
            continue
        image_state_dict[k] = v/float(end_epoch-start_epoch+1)

    for k,v in model_state_dict.items():
        if 'logit_scale' in k:
            continue
        model_state_dict[k] = v/float(end_epoch-start_epoch+1)

    return image_state_dict, model_state_dict

    
        
        

if __name__ == '__main__':
    pass


