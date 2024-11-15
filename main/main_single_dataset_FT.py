

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
from scheduler import create_scheduler
from optim import create_optimizer, create_optimizer_sam
from engine.train import train_sam, valid_on_cheXpert, valid_on_chestxray14, train_disam, valid_on_vindr, valid_on_siimacr, valid_on_openi, valid_on_shenzhen, AverageMeter, get_text_features, get_weight_mask, fourier_aug
from models.clip_tqn import CLP_clinical, ModelRes, TQN_Model, ModelConvNeXt, ModelEfficientV2
from data.dataset_entity import MIMIC_Dataset, Mergetrain_Dataset, Chestxray14_Dataset, CheXpert_Dataset
from configs.default import default_dataloder_kwargs
from factory.loss import AsymmetricLoss, Ralloss
from data.test_dataset import Vindr_Dataset, SIIMACR_Dataset
from main.main_utils import seed_torch, get_dataloader, get_model, load_checkpoint, get_criterion, record_results
from torch.cuda.amp import autocast
from factory.loss import ClipLoss

def train(model, criterion, image_encoder, text_encoder, tokenizer, data_loader, optimizer, epoch, warmup_steps, device, scheduler, args, config, writer):
    if config['weighted_criterion']:
        weight_dict = json.load(open(config['weight_dict']))['total']

        
    # clip_loss = ClipLoss(world_size=int(os.environ['WORLD_SIZE']), rank=int(os.environ['LOCAL_RANK'])) # 考虑对ASL loss是否也要针对DDP问题使用相同的gather feature操作
    clip_loss = ClipLoss()
    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_ce_m = AverageMeter()
    loss_ce_image_m = AverageMeter()
    loss_ce_text_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    model.train() 
    # model.eval() 
    image_encoder.train()  
    # image_encoder.eval()
    # text_encoder.train()
    # 尝试改为eval模式
    text_encoder.eval()
    # image_encoder.eval()
    # model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ce_image', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    if args.use_entity_features:
        metric_logger.add_meter('loss_ce_text', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_clip', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.update(loss=1.0)
    metric_logger.update(lr = scheduler._get_lr(epoch)[0])

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size 
    scalar_step = epoch*len(data_loader)
    num_batches_per_epoch = data_loader.num_batches
    sample_digits = math.ceil(math.log(data_loader.num_samples + 1, 10))

    
    if args.add_dataset:
            text_list = ['normal', 'pleural effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis',  'tube', 'consolidation','enlarged cardiomediastinum','tip', 'pneumonia','line','cardiomegaly', 'fracture','calcification',
            'device','engorgement',  'nodule', 'wire',  'pacemaker', 'pleural thicken', 'marking', 'scar', 'hyperinflate', 'blunt',  'collapse', 'emphysema', 'aerate', 'mass','infiltration', 'obscure', 'deformity', 'hernia',
            'drainage', 'distention', 'shift', 'stent', 'lesion', 'hardware', 'dilation',  'aspiration',
            'fibrosis',	'No Finding', 'Pleural Other', 'Support Devices', 'Aortic enlargement',
            'Clavicle fracture', 'Enlarged PA', 'ILD', 'Lung cavity', 'Lung cyst', 'Mediastinal shift',	
            'Nodule/Mass', 'Pulmonary fibrosis', 'Rib fracture', 'Other lesion', 'COPD', 'Lung tumor', 'Tuberculosis',
            'Other diseases']
    else:
        text_list = ['normal', 'pleural effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis',  'tube', 'consolidation','enlarged cardiomediastinum','tip', 'pneumonia','line','cardiomegaly', 'fracture','calcification',
        'device','engorgement',  'nodule', 'wire',  'pacemaker', 'pleural thicken', 'marking', 'scar', 'hyperinflate', 'blunt',  'collapse', 'emphysema', 'aerate', 'mass','infiltration', 'obscure', 'deformity', 'hernia',
        'drainage', 'distention', 'shift', 'stent', 'lesion', 'hardware', 'dilation',  'aspiration']
    with autocast():
        text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)

    # text_features这部分不能打散 所以要根据gpu的数量进行倍增
    device_num = torch.cuda.device_count()
    text_features = text_features.repeat(int(device_num),1)
        
    for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        if args.fourier:
            image = fourier_aug(sample['image'].to(device))
        else:
            image = sample['image'].to(device)  
        label = sample['label'].long().to(device)

        if args.ignore_index:
            pass
        else:
            label[label==-1]=0
        # caption = sample['caption'] #batch_size,len
        entity = sample['entity']
        # if args.add_dataset:
        #     dataset_label = sample['label_dataset']

        data_time_m.update(time.time() - end)

        optimizer.zero_grad()
        
        with autocast(): # 自动混合精度
            entity_features = get_text_features(text_encoder,entity,tokenizer,device,max_length=args.max_length)

            image_features,image_features_pool = image_encoder(image)

            pred_class_image = model(image_features,text_features)
            label = label.float()

            label_mask = (label != -1).squeeze()
            
            if config['weighted_criterion']:
                weight_mask = get_weight_mask(label, weight_dict)
                weight_mask = weight_mask[label_mask]
                if config['criterion'] == 'bce':
                    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight_mask)
            
            if args.add_dataset:
                pred_class_image = pred_class_image[label_mask]
                label_image = label[label_mask]
                if config['criterion'] == 'bce':
                    loss_ce_image = criterion(pred_class_image.view(-1),label_image.view(-1))
                else:
                    loss_ce_image = criterion(pred_class_image.view(-1,1),label_image.view(-1,1), use_weight=config['weighted_criterion'])
                    
                    if config['weighted_criterion']:
                        loss_ce_image = loss_ce_image[:,0] * weight_mask
                        loss_ce_image = -loss_ce_image.sum()
                    
            else:
                loss_ce_image = criterion(pred_class_image.view(-1,1),label.view(-1,1))
            
            if args.use_entity_features:
                pred_class_text = model(entity_features.unsqueeze(1),text_features)
                if args.add_dataset:
                    pred_class_text = pred_class_text[label_mask]
                    label_text = label[label_mask]
                    if config['criterion'] == 'bce':
                        loss_ce_text = criterion(pred_class_text.view(-1),label_text.view(-1))
                    else:
                        loss_ce_text = criterion(pred_class_text.view(-1,1),label_text.view(-1,1), use_weight=config['weighted_criterion'])
                        if config['weighted_criterion']:
                            loss_ce_text = loss_ce_text[:,0] * weight_mask
                            loss_ce_text = -loss_ce_text.sum()

                else:
                    loss_ce_text = criterion(pred_class_text.view(-1,1),label.view(-1,1))

                loss_ce = loss_ce_image  + loss_ce_text

            else:
                loss_ce = loss_ce_image

            # 要除以batch_size
            # loss_ce = loss_ce
            
            loss_clip = clip_loss(image_features_pool,entity_features)
            # loss_clip = torch.tensor(0.)
            # loss_ce_text = torch.tensor(0.)
            # loss_ce = loss_ce_image
            loss = loss_ce  + loss_clip * args.loss_ratio
        
        # loss的计算无法混合精度
        loss.backward()
        optimizer.step() 

        writer.add_scalar('loss/loss', loss, scalar_step)
        writer.add_scalar('loss/loss_ce', loss_ce, scalar_step)
        writer.add_scalar('loss/loss_ce_image', loss_ce_image, scalar_step)
        if args.use_entity_features:
            writer.add_scalar('loss/loss_ce_text', loss_ce_text, scalar_step)
        writer.add_scalar('loss/loss_clip', loss_clip, scalar_step)
        scalar_step += 1

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_ce=loss_ce.item())
        metric_logger.update(loss_ce_image=loss_ce_image.item())
        if args.use_entity_features:
            metric_logger.update(loss_ce_text=loss_ce_text.item())
        metric_logger.update(loss_clip=loss_clip.item())


        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        metric_logger.update(lr = scheduler._get_lr(epoch)[0])

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if i % 100 == 0:
            batch_size = len(image)
            num_samples = batch_count * batch_size
            samples_per_epoch = data_loader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(loss.item(), batch_size)
            loss_clip_m.update(loss_clip.item(), batch_size)
            loss_ce_m.update(loss_ce.item(), batch_size)
            loss_ce_image_m.update(loss_ce_image.item(), batch_size)
            if args.use_entity_features:
                loss_ce_text_m.update(loss_ce_text.item(), batch_size)
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Loss_clip: {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                    f"Loss_ce: {loss_ce_m.val:#.5g} ({loss_ce_m.avg:#.4g}) "
                    f"Loss_ce_image: {loss_ce_image_m.val:#.5g} ({loss_ce_image_m.avg:#.4g}) "
                    f"Loss_ce_text: {loss_ce_text_m.val:#.5g} ({loss_ce_text_m.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {batch_size/ batch_time_m.val:#g}/s "
                    f"LR: { scheduler._get_lr(epoch)[0]:5f} "
                )
            else:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Loss_clip: {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                    f"Loss_ce: {loss_ce_m.val:#.5g} ({loss_ce_m.avg:#.4g}) "
                    f"Loss_ce_image: {loss_ce_image_m.val:#.5g} ({loss_ce_image_m.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {batch_size/ batch_time_m.val:#g}/s "
                    f"LR: { scheduler._get_lr(epoch)[0]:5f} "
                )
        if i == 100:
            break
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} #,loss_epoch.mean()



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

    
    return val_auc, chexpert_val_auc, test_auc, vindr_test_auc, siimacr_test_auc, openi_test_auc, shenzhen_test_auc, padchest_test_auc, mimic_test_auc
 
 
def main(args, config):
    device = torch.device(args.device)
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    
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
    
    model, image_encoder, text_encoder, tokenizer = get_model(args, config)
    
    arg_opt = utils.AttrDict(config['optimizer'])
    if args.rho>0:
        optimizer = create_optimizer_sam(arg_opt, model, image_encoder, text_encoder, rho=args.rho)
    else:
        optimizer = create_optimizer(arg_opt, model, image_encoder, text_encoder)

    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer) 
    
    load_checkpoint(model, image_encoder, args)
            
    # if args.distributed and os.environ['LOCAL_RANK'] == '0':
    print("Start training")
    start_time = time.time()
    writer = SummaryWriter(os.path.join(args.output_dir,  'log'))


    best_epoch_dict = {'val_chestxray': 0, 'test_chestxray':0, 'chexpert': 0, 'vindr': 0, 'siimacr': 0, 'openi': 0, 'shenzhen': 0, 'padchest': 0, 'mimic': 0}
    best_results_dict = {'val_chestxray': 0, 'test_chestxray':0, 'chexpert': 0, 'vindr': 0, 'siimacr': 0, 'openi': 0, 'shenzhen': 0, 'padchest': 0, 'mimic': 0}

    
    # if args.distributed:
    #     dist.barrier()
        
    criterion = get_criterion(config)
    
    for epoch in range(start_epoch, max_epoch):
        # if args.distributed:
        #     dist.barrier()
        # if epoch>0:
        #     lr_scheduler.step(epoch+warmup_steps)

        if args.rho>0:
            print('Use SAM!')
            train_stats = train_disam(model, criterion, image_encoder, text_encoder, tokenizer, train_dataloader, optimizer, epoch, warmup_steps, device, lr_scheduler, args, config, writer) 

        else:
            train_stats = train(model, criterion, image_encoder, text_encoder, tokenizer, train_dataloader, optimizer, epoch, warmup_steps, device, lr_scheduler, args, config, writer) 

        
        for k, v in train_stats.items():
            if k == 'loss':
                train_loss_epoch = v
            elif k == 'loss_ce':
                train_loss_ce_epoch = v
            elif k == 'loss_clip':
                train_loss_clip_epoch = v
        
        writer.add_scalar('loss/train_loss_epoch', float(train_loss_epoch), epoch)
        writer.add_scalar('loss/train_loss_ce_epoch', float(train_loss_ce_epoch), epoch)
        writer.add_scalar('loss/train_loss_clip_epoch', float(train_loss_clip_epoch), epoch)
        writer.add_scalar('lr/leaning_rate',  lr_scheduler._get_lr(epoch)[0] , epoch)

        
        # if args.distributed:
        #     dist.barrier()
        
        
        # if utils.is_main_process():
            
        val_auc, chexpert_val_auc, test_auc, vindr_test_auc, siimacr_test_auc, openi_test_auc, shenzhen_test_auc, padchest_test_auc, mimic_test_auc = test_all(model, image_encoder, text_encoder, tokenizer, val_dataloader, test_dataloader, test_dataloader_chexpert, test_dataloader_vinder, test_dataloader_siimacr, test_dataloader_openi, test_dataloader_shenzhen, test_dataloader_padchest, test_dataloader_mimic, args.device, args, config, writer, epoch=epoch)

        
        if best_results_dict['chexpert'] < chexpert_val_auc:
            # 标记一下best val chexpert model
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(f"Save best valid chexpert model at epoch {epoch} with {chexpert_val_auc:.4f}.\n")

            best_results_dict['chexpert'] = chexpert_val_auc
            best_epoch_dict['chexpert'] = epoch
                
        if best_results_dict['val_chestxray'] < val_auc:
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(f"Save best valid chestxray14 model at epoch {epoch} with {val_auc:.4f}.\n")
            best_results_dict['val_chestxray'] = val_auc
            best_epoch_dict['val_chestxray'] = epoch
                
        if best_results_dict['test_chestxray'] < test_auc:
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(f"Save best test chestxray14 model at epoch {epoch} with {test_auc:.4f}.\n")
            best_results_dict['test_chestxray'] = test_auc
            best_epoch_dict['test_chestxray'] = epoch
            
        if best_results_dict['vindr'] < vindr_test_auc:
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(f"Save best valid vindr model at epoch {epoch} with {vindr_test_auc:.4f}.\n")
            best_results_dict['vindr'] = vindr_test_auc
            best_epoch_dict['vindr'] = epoch
                
        if best_results_dict['siimacr'] < siimacr_test_auc:
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(f"Save best valid siimacr model at epoch {epoch} with {siimacr_test_auc:.4f}.\n")
            best_results_dict['siimacr'] = siimacr_test_auc
            best_epoch_dict['siimacr'] = epoch
                
        if best_results_dict['openi'] < openi_test_auc:
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(f"Save best valid openi model at epoch {epoch} with {openi_test_auc:.4f}.\n")
            best_results_dict['openi'] = openi_test_auc
            best_epoch_dict['openi'] = epoch
                
        if best_results_dict['shenzhen'] < shenzhen_test_auc:
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(f"Save best valid shenzhen model at epoch {epoch} with {shenzhen_test_auc:.4f}.\n")
            best_results_dict['shenzhen'] = shenzhen_test_auc
            best_epoch_dict['shenzhen'] = epoch
                
        if best_results_dict['padchest'] < padchest_test_auc:
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(f"Save best valid padchest model at epoch {epoch} with {padchest_test_auc:.4f}.\n")
            best_results_dict['padchest'] = padchest_test_auc
            best_epoch_dict['padchest'] = epoch
        
        if best_results_dict['mimic'] < mimic_test_auc:
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(f"Save best valid mimic model at epoch {epoch} with {mimic_test_auc:.4f}.\n")
            best_results_dict['mimic'] = mimic_test_auc
            best_epoch_dict['mimic'] = epoch
        
        # 每个epoch最后都要汇报所有的情况
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write('\nSummary:\n')
            for key_name in best_epoch_dict.keys():
                f.write(f"Best {key_name} model at epoch {best_epoch_dict[key_name]} with {best_results_dict[key_name]:.4f}.\n")
            f.write('\n')
                
        
        # if utils.is_main_process():  
        save_obj = {
            'model': model.state_dict(),
            'image_encoder': image_encoder.state_dict(),
            # 'text_encoder':text_encoder.module.state_dict(),
            # 'text_encoder':text_encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'config': config,
            'epoch': epoch,
        }

        torch.save(save_obj, os.path.join(args.aws_output_dir, 'checkpoint_'+str(epoch)+'.pt'))
        
             

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # if args.distributed and os.environ['LOCAL_RANK'] == '0':
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    
    # 单卡速度太慢 考虑多卡
    
    # print(os.environ)
    abs_file_path = os.path.abspath(__file__)
    abs_file_path = abs_file_path.replace('main/main_single_dataset_FT.py', '')
    # print(abs_file_path)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument('--momentum', default=False, type=bool)
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--finetune', default=f'best_valid.pt')
    
    parser.add_argument('--freeze_bert', default=True, type=bool)
    parser.add_argument("--use_entity_features", default=True, type=bool)

    parser.add_argument('--config', default=f'v3.0_base.yaml')

    parser.add_argument('--fourier', default=False, type=bool)
    parser.add_argument('--colourjitter', default=False, type=bool)
    
    parser.add_argument('--class_num', default=1, type=int) 

    parser.add_argument('--ignore_index', default=True, type=bool) 
    parser.add_argument('--add_dataset', default=True, type=bool) 
    parser.add_argument('--use_dataset', type=str, default='all') 


    # Path 指向相同的位置
    time_now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    parser.add_argument('--output_dir', default=f'{abs_file_path}/results/{time_now}')
    parser.add_argument('--aws_output_dir', default=f'{abs_file_path}/results/{time_now}')
    
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
    
    if 'rho' in config:
        args.rho = config['rho']
    
    args.loss_ratio = config['loss_ratio']
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.aws_output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))  

    logging.info("Params:")
    params_file = os.path.join(args.output_dir, "params.txt")
    print(args.fourier, args.colourjitter)
    with open(params_file, "w") as f:
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")
            f.write(f"{name}: {val}\n")


    seed_torch(args.seed)

    main(args, config)


