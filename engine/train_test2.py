import sys
sys.path.append(sys.path[0].replace('engine', ''))
import json
import logging
import math
import os
import cv2
import time
import numpy as np
from torch.distributed import ReduceOp
import random

from PIL import Image
from contextlib import suppress
from itertools import chain
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score, accuracy_score, average_precision_score

import contextlib
from torch.cuda.amp import autocast, GradScaler

import torch
import torch.nn.functional as F
from torch import nn

from factory import utils
from factory.loss import ClipLoss
from configs.default import disease_text_dict

from factory.loss import AsymmetricLoss



def Shuffle_Batch_Data(data_in):

    len_total = len(data_in)
    idx_list = list(range(len_total))
    random.shuffle(idx_list)
    return data_in[idx_list]

def Combine_AmplitudeANDPhase(amp, phe):
    return torch.mul(amp, torch.exp(1j*phe))

def mixup_data(x, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    y = Shuffle_Batch_Data(x)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha) # beta分布
    else:
        lam = 1
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda() # 随机打乱
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :] # 按照比例混合
    y_a, y_b = y, y[index]
    return mixed_x

def FFT2_Amp_MixUp(data_original, data_aug, lamda):
    '''
    将fft_data_original和fft_data_aug按照lamda系数对幅值进行对应的扰动
    相位信息以fft_data_original为准
    '''
    # fft操作
    fft_data_original = torch.fft.fft2(data_original)
    fft_data_aug = torch.fft.fft2(data_aug)
    
    aug_amp = lamda*torch.abs(fft_data_original) + (1-lamda)*torch.abs(fft_data_aug)
    fft_mixup_data = torch.mul(aug_amp, torch.exp(1j*torch.angle(fft_data_original)))
    return torch.real(torch.fft.ifft2(fft_mixup_data))

def fourier_aug(batch_data, p=1.0):
    batch_x = batch_data
    batch_y = Shuffle_Batch_Data(batch_data)
    apply_p = np.random.rand()
    if apply_p<=p:
        lamda_vector = np.random.rand(batch_x.size(0))
        for i in range(batch_x.size(0)):
            batch_x[i] = FFT2_Amp_MixUp(batch_x[i], batch_y[i], lamda_vector[i])
        return batch_x
    else:
        return batch_x
    


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_text_features(model,text_list,tokenizer,device,max_length):
    text_token = tokenizer(list(text_list),add_special_tokens=True, padding='max_length', truncation=True, max_length= max_length, return_tensors="pt").to(device=device)
    text_features = model.encode_text(text_token)

    return text_features


def get_domain_loss(pred, label, label_mask, domain_label, criterion, weight_mask=None):
    if weight_mask is not None:
        use_weight = True
    else:
        use_weight = False
    # domain index是0-3
    domain_loss_list = []
    for domain_idx in range(4):
        domain_label_mask = (domain_label == domain_idx)
        if sum(domain_label_mask) == 0:
            continue
        single_domain_label_mask = label_mask[domain_label_mask]
        if use_weight:
            domain_weight_mask = weight_mask[domain_label_mask][single_domain_label_mask]
        
        single_domain_loss = criterion(pred[domain_label_mask][single_domain_label_mask].view(-1, 1), label[domain_label_mask][single_domain_label_mask].view(-1, 1), use_weight)
        if use_weight:
            single_domain_loss = single_domain_loss[:,0] * domain_weight_mask
            single_domain_loss = -single_domain_loss.mean()
        domain_loss_list.append(single_domain_loss)

    
    var_domain_loss = torch.std(torch.stack(domain_loss_list))
    return var_domain_loss
    


def get_weight_mask(label, weight_dict):
    weight_mask = torch.ones_like(label).cuda()
    for i in range(len(weight_dict)):
        if weight_dict[str(i)]['pos'] > 0:
            weight_mask[label[:,i]==1] = weight_dict[str(i)]['neg'] / weight_dict[str(i)]['pos']
    return weight_mask


        

def train_boost_chex(model, criterion, image_encoder, text_encoder, tokenizer, data_loader, optimizer, epoch, warmup_steps, device, scheduler, args, config, writer):
    if config['weighted_criterion']:
        weight_dict = json.load(open(config['weight_dict']))['total']

        
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
    image_encoder.train()  
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
        dataset_label = sample['label_dataset'].long().to(device)

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
            chex_mask = (dataset_label == 2).squeeze()
            pred_chex_image = pred_class_image[chex_mask]
            label_chex_mask = label[chex_mask]
            label_chex_mask_mask = (label_chex_mask != -1).squeeze()
            
            
            label_mask = (label != -1).squeeze()
            
            if config['weighted_criterion']:
                weight_mask = get_weight_mask(label, weight_dict)
                weight_mask = weight_mask[label_mask]
                if config['criterion'] == 'bce':
                    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight_mask)
            
            if args.add_dataset:
                pred_class_image = pred_class_image[label_mask]
                pred_class_chex_image = pred_chex_image[label_chex_mask_mask]
                label_image = label[label_mask]
                label_chex_image = label_chex_mask[label_chex_mask_mask]
                
                if config['criterion'] == 'bce':
                    loss_ce_image = criterion(pred_class_image.view(-1),label_image.view(-1))
                else:
                    loss_ce_image = criterion(pred_class_image.view(-1,1),label_image.view(-1,1), use_weight=config['weighted_criterion'])
                    
                    # 对dataset_label == 2的部分重新算一个loss 用来boost性能
                    if label_chex_mask_mask.sum() > 0:
                        loss_ce_image_boost = criterion(pred_class_chex_image.view(-1,1),label_chex_image.view(-1,1), use_weight=config['weighted_criterion'])
                    
                    loss_ce_image = loss_ce_image + 2.0 * loss_ce_image_boost
                    
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
            # loss = loss_clip
        
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

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} #,loss_epoch.mean()



def train_ft(model, criterion, image_encoder, text_encoder, tokenizer, data_loader, optimizer, epoch, warmup_steps, device, scheduler, args, config, writer):
    
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
    image_encoder.train()  
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

            loss = loss_ce + loss_clip * args.loss_ratio
        
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

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} #,loss_epoch.mean()



def valid_on_mimic(model, image_encoder, text_encoder, tokenizer, data_loader, epoch, device, args, config, writer, total_test=False):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    image_encoder.eval()
    text_encoder.eval()
    text_list = disease_text_dict['mimic']
    text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)
    # text_features这部分不能打散 所以要根据gpu的数量进行倍增
    device_num = torch.cuda.device_count()
    text_features = text_features.repeat(int(device_num),1)
    
    val_scalar_step = epoch*len(data_loader)
    val_losses = []

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    for i, sample in enumerate(data_loader):
        image = sample['image'].to(device, non_blocking=True)  
        label = sample['label'].long().to(device)
        label = label.float()
        
        gt = torch.cat((gt, label), 0)
        with torch.no_grad(): # 自动混合精度
            # with autocast():
            image_features,image_features_pool = image_encoder(image)
            
            pred_class = model(image_features,text_features)#b,14,2/1
            val_loss = F.binary_cross_entropy_with_logits(pred_class.view(-1,1),label.view(-1, 1))
            pred_class = torch.sigmoid(pred_class)
            pred = torch.cat((pred, pred_class[:,:,0]), 0)

            val_losses.append(val_loss.item())
            writer.add_scalar('val_loss/loss', val_loss, val_scalar_step)
            val_scalar_step += 1
    if total_test:
        metrics = compute_all_total(gt, pred, n_class = 15)
    else:
        metrics = compute_all(gt, pred, n_class = 15)
    AUROC_avg = metrics['mean_auc']
    avg_val_loss = np.array(val_losses).mean()
    return avg_val_loss,AUROC_avg,metrics


def valid_on_chestxray14(model, image_encoder, text_encoder, tokenizer, data_loader, epoch, device, args, config, writer, total_test=False):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    image_encoder.eval()
    text_encoder.eval()
    text_list = disease_text_dict['chestxray']
    text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)
    # text_features这部分不能打散 所以要根据gpu的数量进行倍增
    device_num = torch.cuda.device_count()
    text_features = text_features.repeat(int(device_num),1)
    
    val_scalar_step = epoch*len(data_loader)
    val_losses = []

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    for i, sample in enumerate(data_loader):
        image = sample['image'].to(device, non_blocking=True)  
        label = sample['label'].long().to(device)
        label = label.float()
        
        gt = torch.cat((gt, label), 0)
        with torch.no_grad(): # 自动混合精度
            # with autocast():
            image_features,image_features_pool = image_encoder(image)
            
            pred_class = model(image_features,text_features)#b,14,2/1
            val_loss = F.binary_cross_entropy_with_logits(pred_class.view(-1,1),label.view(-1, 1))
            pred_class = torch.sigmoid(pred_class)
            pred = torch.cat((pred, pred_class[:,:,0]), 0)

            val_losses.append(val_loss.item())
            writer.add_scalar('val_loss/loss', val_loss, val_scalar_step)
            val_scalar_step += 1




def valid_on_cheXpert(model,image_encoder,text_encoder,tokenizer,data_loader, epoch, device, args, config, writer, total_test=False):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    image_encoder.eval()
    text_encoder.eval()
    # text_list = ['atelectasis', 'cardiomegaly', 'consolidation', 'edema', 'pleural effusion']
    text_list = disease_text_dict['chexpert']
    text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)
    # text_features这部分不能打散 所以要根据gpu的数量进行倍增
    device_num = torch.cuda.device_count()
    text_features = text_features.repeat(int(device_num),1)
    
    val_scalar_step = epoch*len(data_loader)
    val_losses = []

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()
    ce_loss = nn.CrossEntropyLoss()
    for i, sample in enumerate(data_loader):
        image = sample['image'].to(device,non_blocking=True)  
        label = sample['label'].long().to(device)
        label = label.float()

        gt = torch.cat((gt, label), 0)
        # with autocast(): # 自动混合精度
        with torch.no_grad():
            image_features,image_features_pool = image_encoder(image)
        

            pred_class = model(image_features,text_features)#b,14,2/1
            val_loss = F.binary_cross_entropy_with_logits(pred_class.view(-1,1),label.view(-1, 1))
            # val_loss = ce_loss(pred_class.view(-1,2),label.view(-1))
            pred_class = torch.sigmoid(pred_class)
            pred = torch.cat((pred, pred_class[:,:,0]), 0)
            
            # else:
            #     val_loss = criterion(pred_class.view(-1,2),label.view(-1))
            #     pred_class = torch.softmax(pred_class, dim=-1)
            #     pred = torch.cat((pred, pred_class[:,:,1]), 0)
            
            val_losses.append(val_loss.item())
            writer.add_scalar('val_loss/loss', val_loss, val_scalar_step)
            val_scalar_step += 1


def valid_on_six(model,image_encoder,text_encoder,tokenizer,data_loader, epoch, device, args, config, writer, total_test=False):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    image_encoder.eval()
    text_encoder.eval()

    text_list = ['Pneumothorax', 'Aortic Enlargement', 'Rib Fracture', 'Mass', 'Atelectasis', 'Clavicle Fracture', 'Collapse', 'Mediastinal Shift', 'Pulmonary Fibrosis', 'Pneumonia', 'Blunt', 'Pleural Effusion', 'Cardiomegaly', 'Edema', 'Fibrosis', 'Lung Tumor', 'COPD', 'Fracture', 'Nodule', 'Lung Cavity']

    text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)
    device_num = torch.cuda.device_count()
    text_features = text_features.repeat(int(device_num),1)
    
    val_scalar_step = epoch*len(data_loader)
    val_losses = []

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()
    ce_loss = nn.CrossEntropyLoss()
    for i, sample in enumerate(data_loader):
        image = sample['image'].to(device,non_blocking=True)  
        label = sample['label'].long().to(device)
        label = label.float()

        gt = torch.cat((gt, label), 0)
        # with autocast(): # 自动混合精度
        with torch.no_grad():
            image_features,image_features_pool = image_encoder(image)
        

            pred_class = model(image_features,text_features)#b,14,2/1
            val_loss = F.binary_cross_entropy_with_logits(pred_class.view(-1,1),label.view(-1, 1))
            # val_loss = ce_loss(pred_class.view(-1,2),label.view(-1))
            pred_class = torch.sigmoid(pred_class)
            pred = torch.cat((pred, pred_class[:,:,0]), 0)
            
            # else:
            #     val_loss = criterion(pred_class.view(-1,2),label.view(-1))
            #     pred_class = torch.softmax(pred_class, dim=-1)
            #     pred = torch.cat((pred, pred_class[:,:,1]), 0)
            
            val_losses.append(val_loss.item())
            writer.add_scalar('val_loss/loss', val_loss, val_scalar_step)
            val_scalar_step += 1


def valid_on_cxr_lt(model,image_encoder,text_encoder,tokenizer,data_loader, epoch, device, args, config, writer, total_test=False):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    image_encoder.eval()
    text_encoder.eval()
    text_list = ['Atelectasis', 'Calcification of the Aorta', 'Cardiomegaly', 'Consolidation', 'Edema', 'Emphysema', 'Enlarged Cardiomediastinum', 'Fibrosis', 'Fracture', 'Hernia', 'Infiltration', 'Lung Lesion', 'Lung Opacity', 'Mass', 'Nodule', 'Pleural Effusion', 'Pleural Other', 'Pleural Thickening', 'Pneumomediastinum', 'Pneumonia', 'Pneumoperitoneum', 'Pneumothorax', 'Subcutaneous Emphysema', 'Support Devices', 'Tortuous Aorta', 'No Finding']

    # text_list = disease_text_dict['chexpert']
    text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)
    # text_features这部分不能打散 所以要根据gpu的数量进行倍增
    device_num = torch.cuda.device_count()
    text_features = text_features.repeat(int(device_num),1)
    
    val_scalar_step = epoch*len(data_loader)
    val_losses = []

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()
    ce_loss = nn.CrossEntropyLoss()
    for i, sample in enumerate(data_loader):
        image = sample['image'].to(device,non_blocking=True)  
        label = sample['label'].long().to(device)
        label = label.float()

        gt = torch.cat((gt, label), 0)
        # with autocast(): # 自动混合精度
        with torch.no_grad():
            image_features,image_features_pool = image_encoder(image)
        

            pred_class = model(image_features,text_features)#b,14,2/1
            val_loss = F.binary_cross_entropy_with_logits(pred_class.view(-1,1),label.view(-1, 1))
            # val_loss = ce_loss(pred_class.view(-1,2),label.view(-1))
            pred_class = torch.sigmoid(pred_class)
            pred = torch.cat((pred, pred_class[:,:,0]), 0)
            
            # else:
            #     val_loss = criterion(pred_class.view(-1,2),label.view(-1))
            #     pred_class = torch.softmax(pred_class, dim=-1)
            #     pred = torch.cat((pred, pred_class[:,:,1]), 0)
            
            val_losses.append(val_loss.item())
            writer.add_scalar('val_loss/loss', val_loss, val_scalar_step)
            val_scalar_step += 1



def valid_on_vindr(model, image_encoder, text_encoder, tokenizer, data_loader, epoch, device, args, config, writer, total_test=False):
    model.eval()
    image_encoder.eval()
    text_encoder.eval()
    text_list = disease_text_dict['vindr']
    text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)
    # text_features这部分不能打散 所以要根据gpu的数量进行倍增
    device_num = torch.cuda.device_count()
    text_features = text_features.repeat(int(device_num),1)
    
    val_scalar_step = epoch*len(data_loader)
    val_losses = []

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    for i, sample in enumerate(data_loader):
        image = sample['image'].to(device,non_blocking=True)  
        label = sample['label'].long().to(device)
        label = label.float()
        gt = torch.cat((gt, label), 0)
        # with autocast(): # 自动混合精度
        with torch.no_grad():
            image_features,image_features_pool = image_encoder(image)
            
            pred_class = model(image_features,text_features)#b,14,2/1
            val_loss = F.binary_cross_entropy_with_logits(pred_class.view(-1,1),label.view(-1, 1))
            pred_class = torch.sigmoid(pred_class)
            pred = torch.cat((pred, pred_class[:,:,0]), 0)
            # else:
            #     val_loss = criterion(pred_class.view(-1,2),label.view(-1))
            #     pred_class = torch.softmax(pred_class, dim=-1)
            #     pred = torch.cat((pred, pred_class[:,:,1]), 0)
            
            val_losses.append(val_loss.item())
            writer.add_scalar('val_loss/loss', val_loss, val_scalar_step)
            val_scalar_step += 1


def valid_on_siimacr(model, image_encoder, text_encoder, tokenizer, data_loader, epoch, device, args, config, writer, total_test=False):
    model.eval()
    image_encoder.eval()
    text_encoder.eval()
    text_list = ['pneumothorax']
    text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)
    # text_features这部分不能打散 所以要根据gpu的数量进行倍增
    device_num = torch.cuda.device_count()
    text_features = text_features.repeat(int(device_num),1)
    
    val_scalar_step = epoch*len(data_loader)
    val_losses = []

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    for i, sample in enumerate(data_loader):
        image = sample['image'].to(device,non_blocking=True)  
        label = sample['label'].long().to(device)
        label = label.float()
        
        gt = torch.cat((gt, label), 0)
        # with autocast(): # 自动混合精度
        with torch.no_grad():
            image_features,image_features_pool = image_encoder(image)
            
            pred_class = model(image_features,text_features)#b,14,2/1
            val_loss = F.binary_cross_entropy_with_logits(pred_class.view(-1,1),label.view(-1, 1))
            pred_class = torch.sigmoid(pred_class)
            pred = torch.cat((pred, pred_class[:,:,0]), 0)
            # else:
            #     val_loss = criterion(pred_class.view(-1,2),label.view(-1))
            #     pred_class = torch.softmax(pred_class, dim=-1)
            #     pred = torch.cat((pred, pred_class[:,:,1]), 0)
            
            val_losses.append(val_loss.item())
            writer.add_scalar('val_loss/loss', val_loss, val_scalar_step)
            val_scalar_step += 1
                



def valid_on_openi(model, image_encoder, text_encoder, tokenizer, data_loader, epoch, device, args, config, writer, total_test=False):
    model.eval()
    image_encoder.eval()
    text_encoder.eval()
    text_list = disease_text_dict['openi']
    text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)
    # text_features这部分不能打散 所以要根据gpu的数量进行倍增
    device_num = torch.cuda.device_count()
    text_features = text_features.repeat(int(device_num),1)
    
    val_scalar_step = epoch*len(data_loader)
    val_losses = []

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    for i, sample in enumerate(data_loader):
        image = sample['image'].to(device,non_blocking=True)  
        label = sample['label'].long().to(device)
        label = label.float()
        gt = torch.cat((gt, label), 0)
        # with autocast(): # 自动混合精度
        with torch.no_grad():
            image_features,image_features_pool = image_encoder(image)
            
            pred_class = model(image_features,text_features)
            val_loss = F.binary_cross_entropy_with_logits(pred_class.view(-1,1),label.view(-1, 1))
            pred_class = torch.sigmoid(pred_class)
            pred = torch.cat((pred, pred_class[:,:,0]), 0)
            
            val_losses.append(val_loss.item())
            writer.add_scalar('val_loss/loss', val_loss, val_scalar_step)
            val_scalar_step += 1



def valid_on_shenzhen(model, image_encoder, text_encoder, tokenizer, data_loader, epoch, device, args, config, writer, total_test=False):
    model.eval()
    image_encoder.eval()
    text_encoder.eval()
    text_list = ['Tuberculosis']
    text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)
    # text_features这部分不能打散 所以要根据gpu的数量进行倍增
    device_num = torch.cuda.device_count()
    text_features = text_features.repeat(int(device_num),1)
    
    val_scalar_step = epoch*len(data_loader)
    val_losses = []

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    for i, sample in enumerate(data_loader):
        image = sample['image'].to(device,non_blocking=True)  
        label = sample['label'].long().to(device)
        label = label.float()

        gt = torch.cat((gt, label), 0)
        # with autocast(): # 自动混合精度
        with torch.no_grad():
            image_features,image_features_pool = image_encoder(image)
            
            pred_class = model(image_features,text_features)
            val_loss = F.binary_cross_entropy_with_logits(pred_class.view(-1,1),label.view(-1, 1))
            pred_class = torch.sigmoid(pred_class)
            pred = torch.cat((pred, pred_class[:,:,0]), 0)
            
            val_losses.append(val_loss.item())
            writer.add_scalar('val_loss/loss', val_loss, val_scalar_step)
            val_scalar_step += 1



def valid_on_padchest(model, image_encoder, text_encoder, tokenizer, data_loader, epoch, device, args, config, writer, total_test=False):
    model.eval()
    image_encoder.eval()
    text_encoder.eval()
    text_list = ['normal', 'pulmonary fibrosis', 'chronic changes', 'kyphosis', 'pseudonodule', 'ground glass pattern', 'unchanged', 
                 'alveolar pattern', 'interstitial pattern', 'laminar atelectasis', 'pleural effusion', 'apical pleural thickening', 
                 'suture material', 'sternotomy', 'endotracheal tube', 'infiltrates', 'heart insufficiency', 'hemidiaphragm elevation', 
                 'superior mediastinal enlargement', 'aortic elongation', 'scoliosis', 'sclerotic bone lesion', 'supra aortic elongation', 
                 'vertebral degenerative changes', 'goiter', 'COPD signs', 'air trapping', 'descendent aortic elongation', 'aortic atheromatosis', 
                 'metal', 'hypoexpansion basal', 'abnormal foreign body', 'central venous catheter via subclavian vein', 'central venous catheter', 
                 'vascular hilar enlargement', 'pacemaker', 'atelectasis', 'vertebral anterior compression', 'hiatal hernia', 'pneumonia', 'diaphragmatic eventration', 
                 'consolidation', 'calcified densities', 'cardiomegaly', 'fibrotic band', 'tuberculosis sequelae', 'volume loss', 'bronchiectasis', 
                 'single chamber device', 'emphysema', 'vertebral compression', 'bronchovascular markings', 'bullas', 'hilar congestion', 'exclude', 
                 'axial hyperostosis', 'aortic button enlargement', 'calcified granuloma', 'clavicle fracture', 'pulmonary mass', 'dual chamber device', 
                 'increased density', 'surgery neck', 'osteosynthesis material', 'costochondral junction hypertrophy', 'segmental atelectasis', 
                 'costophrenic angle blunting', 'calcified pleural thickening', 'hyperinflated lung', 'callus rib fracture', 'pleural thickening', 
                 'mediastinal mass', 'nipple shadow', 'surgery heart', 'pulmonary artery hypertension', 'central vascular redistribution', 'tuberculosis', 
                 'nodule', 'cavitation', 'granuloma', 'osteopenia', 'lobar atelectasis', 'surgery breast', 'NSG tube', 'hilar enlargement', 'gynecomastia', 
                 'atypical pneumonia', 'cervical rib', 'mediastinal enlargement', 'major fissure thickening', 'surgery', 'azygos lobe', 'adenopathy', 'miliary opacities', 
                 'suboptimal study', 'dai', 'mediastinic lipomatosis', 'surgery lung', 'mammary prosthesis', 'humeral fracture', 'calcified adenopathy', 
                 'reservoir central venous catheter', 'vascular redistribution', 'hypoexpansion', 'heart valve calcified', 'pleural mass', 'loculated pleural effusion', 
                 'pectum carinatum', 'subacromial space narrowing', 'central venous catheter via jugular vein', 'vertebral fracture', 'osteoporosis', 'bone metastasis', 
                 'lung metastasis', 'cyst', 'humeral prosthesis', 'artificial heart valve', 'mastectomy', 'pericardial effusion', 'lytic bone lesion', 'subcutaneous emphysema', 
                 'pulmonary edema', 'flattened diaphragm', 'asbestosis signs', 'multiple nodules', 'prosthesis', 'pulmonary hypertension', 'soft tissue mass', 
                 'tracheostomy tube', 'endoprosthesis', 'post radiotherapy changes', 'air bronchogram', 'pectum excavatum', 'calcified mediastinal adenopathy', 
                 'central venous catheter via umbilical vein', 'thoracic cage deformation', 'obesity', 'tracheal shift', 'external foreign body', 'atelectasis basal', 
                 'aortic endoprosthesis', 'rib fracture', 'calcified fibroadenoma', 'pneumothorax', 'reticulonodular interstitial pattern', 'reticular interstitial pattern', 
                 'chest drain tube', 'minor fissure thickening', 'fissure thickening', 'hydropneumothorax', 'breast mass', 'blastic bone lesion', 'respiratory distress', 
                 'azygoesophageal recess shift', 'ascendent aortic elongation', 'lung vascular paucity', 'kerley lines', 'electrical device', 'artificial mitral heart valve', 
                 'artificial aortic heart valve', 'total atelectasis', 'non axial articular degenerative changes', 'pleural plaques', 'calcified pleural plaques', 
                 'lymphangitis carcinomatosa', 'lepidic adenocarcinoma', 'mediastinal shift', 'ventriculoperitoneal drain tube', 'esophagic dilatation', 'dextrocardia', 
                 'end on vessel', 'right sided aortic arch', 'Chilaiditi sign', 'aortic aneurysm', 'loculated fissural effusion', 'fracture', 'air fluid level', 
                 'round atelectasis', 'mass', 'double J stent', 'pneumoperitoneo', 'abscess', 'pulmonary artery enlargement', 'bone cement', 'pneumomediastinum', 
                 'catheter', 'surgery humeral', 'empyema', 'nephrostomy tube', 'sternoclavicular junction hypertrophy', 'pulmonary venous hypertension', 'gastrostomy tube', 'lipomatosis']

    text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)
    # text_features这部分不能打散 所以要根据gpu的数量进行倍增
    device_num = torch.cuda.device_count()
    text_features = text_features.repeat(int(device_num),1)
    
    val_scalar_step = epoch*len(data_loader)
    val_losses = []

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    for i, sample in enumerate(data_loader):
        image = sample['image'].to(device,non_blocking=True)  
        label = sample['label'].long().to(device)
        label = label.float()

        gt = torch.cat((gt, label), 0)
        # with autocast(): # 自动混合精度
        with torch.no_grad():
            image_features,image_features_pool = image_encoder(image)
            
            pred_class = model(image_features,text_features)
            val_loss = F.binary_cross_entropy_with_logits(pred_class.view(-1,1),label.view(-1, 1))
            pred_class = torch.sigmoid(pred_class)
            pred = torch.cat((pred, pred_class[:,:,0]), 0)
            
            val_losses.append(val_loss.item())
            writer.add_scalar('val_loss/loss', val_loss, val_scalar_step)
            val_scalar_step += 1

    
def compute_AUCs(gt, pred, n_class):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    metrics = {}
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    if n_class == 1:
        AUROCs.append(roc_auc_score(gt_np, pred_np))
    else:
        for i in range(n_class):
            if i == 6 and n_class == 28:
                continue
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    metrics[f"mean_auc"] = np.mean(np.array(AUROCs))
    if n_class == 1:
        metrics[f"auc/class_0"]=AUROCs[0]
        
    elif n_class == 28:
        for i in range(len(AUROCs)):
            if i >= 6:
                metrics[f"auc/class_{i+1}"]=AUROCs[i]
            else:
                metrics[f"auc/class_{i}"]=AUROCs[i]
    else:
        for i in range(n_class):
            metrics[f"auc/class_{i}"]=AUROCs[i]
    return metrics



def compute_mccs(gt, pred, n_class):
    # get a best threshold for all classes
    gt_np = gt 
    pred_np = pred 
    select_best_thresholds = []

    for i in range(n_class):
        if i == 6 and n_class == 28:
            select_best_thresholds.append(0.5)
            continue
        select_best_threshold_i = 0.0
        best_mcc_i = 0.0
        for threshold_idx in range(len(pred)):
            pred_np_ = pred_np.copy()
            thresholds = pred[threshold_idx]
            pred_np_[:,i][pred_np_[:,i]>=thresholds[i]]=1
            pred_np_[:,i][pred_np_[:,i]<thresholds[i]]=0
            mcc = matthews_corrcoef(gt_np[:, i], pred_np_[:, i])
            if mcc > best_mcc_i:
                best_mcc_i = mcc
                select_best_threshold_i = thresholds[i]
        select_best_thresholds.append(select_best_threshold_i)
            
    for i in range(n_class):
        if i == 6 and n_class == 28:
            continue
        pred_np[:,i][pred_np[:,i]>= select_best_thresholds[i]]=1
        pred_np[:,i][pred_np[:,i]< select_best_thresholds[i]]=0
    mccs = []
    for i in range(n_class):
        if i == 6 and n_class == 28:
            continue
        mccs.append(matthews_corrcoef(gt_np[:, i], pred_np[:, i]))
    return mccs,select_best_thresholds

def compute_F1s_threshold(gt, pred,threshold,n_class):
    gt_np = gt 
    pred_np = pred 
    
    F1s = []
    for i in range(n_class):
        if i == 6 and n_class == 28:
            continue
        pred_np[:,i][pred_np[:,i]>=threshold[i]]=1
        pred_np[:,i][pred_np[:,i]<threshold[i]]=0
        F1s.append(f1_score(gt_np[:, i], pred_np[:, i],average='binary'))
    return F1s

def compute_Accs_threshold(gt, pred,threshold,n_class):
    gt_np = gt 
    pred_np = pred 
    
    Accs = []
    for i in range(n_class):
        if i == 6 and n_class == 28:
            continue
        pred_np[:,i][pred_np[:,i]>=threshold[i]]=1
        pred_np[:,i][pred_np[:,i]<threshold[i]]=0
        Accs.append(accuracy_score(gt_np[:, i], pred_np[:, i]))
    return Accs

def compute_Mccs_threshold(gt, pred,threshold,n_class):
    gt_np = gt 
    pred_np = pred 
    
    Mccs = []
    for i in range(n_class):
        if i == 6 and n_class == 28:
            continue
        pred_np[:,i][pred_np[:,i]>=threshold[i]]=1
        pred_np[:,i][pred_np[:,i]<threshold[i]]=0
        Mccs.append(matthews_corrcoef(gt_np[:, i], pred_np[:, i]))
    return Mccs

def compute_all_total(gt, pred, n_class):
    """Computes average AUC, F1 score, ACC, and AP.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
        n_class: Number of classes.
    Returns:
        Dictionary of average AUC, F1 score, ACC, and AP.
    """
    metrics = {}
    AUROCs = []
    # F1_scores = []
    # ACCs = []
    APs = []

    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    
    if n_class == 1:
        AUROCs.append(roc_auc_score(gt_np, pred_np))
        # F1_scores.append(f1_score(gt_np, np.round(pred_np)))
        # ACCs.append(accuracy_score(gt_np, np.round(pred_np)))
        APs.append(average_precision_score(gt_np, pred_np))
        gt_np = gt_np.reshape(-1, 1)
        pred_np = pred_np.reshape(-1, 1)
        
    else:
        for i in range(n_class):
            if i == 6 and n_class == 28:
                continue
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
            # F1_scores.append(f1_score(gt_np[:, i], np.round(pred_np[:, i])))
            # ACCs.append(accuracy_score(gt_np[:, i], np.round(pred_np[:, i])))
            APs.append(average_precision_score(gt_np[:, i], pred_np[:, i]))

    mccs,threshold = compute_mccs(gt_np, pred_np,n_class)
    F1_scores = compute_F1s_threshold(gt_np, pred_np, threshold,n_class)
    ACCs = compute_Accs_threshold(gt_np, pred_np, threshold,n_class)
    MCCs = compute_Mccs_threshold(gt_np, pred_np, threshold,n_class)
    
    metrics["mean_auc"] = np.mean(np.array(AUROCs))
    metrics["mean_f1"] = np.mean(np.array(F1_scores))
    metrics["mean_acc"] = np.mean(np.array(ACCs))
    metrics["mean_ap"] = np.mean(np.array(APs))
    metrics["mean_mcc"] = np.mean(np.array(MCCs))
    
    if n_class == 1:
        metrics["auc/class_0"] = AUROCs[0]
        metrics["f1/class_0"] = F1_scores[0]
        metrics["acc/class_0"] = ACCs[0]
        metrics["ap/class_0"] = APs[0]
        metrics["mcc/class_0"] = MCCs[0]
        
    elif n_class == 28:
        for i in range(len(AUROCs)):
            if i >= 6:
                metrics[f"auc/class_{i+1}"] = AUROCs[i]
                metrics[f"f1/class_{i+1}"] = F1_scores[i]
                metrics[f"acc/class_{i+1}"] = ACCs[i]
                metrics[f"ap/class_{i+1}"] = APs[i]
                metrics[f"mcc/class_{i+1}"] = MCCs[i]
            else:
                metrics[f"auc/class_{i}"] = AUROCs[i]
                metrics[f"f1/class_{i}"] = F1_scores[i]
                metrics[f"acc/class_{i}"] = ACCs[i]
                metrics[f"ap/class_{i}"] = APs[i]
                metrics[f"mcc/class_{i}"] = MCCs[i]
    else:
        for i in range(n_class):
            metrics[f"auc/class_{i}"] = AUROCs[i]
            metrics[f"f1/class_{i}"] = F1_scores[i]
            metrics[f"acc/class_{i}"] = ACCs[i]
            metrics[f"ap/class_{i}"] = APs[i]
            metrics[f"mcc/class_{i}"] = MCCs[i]

    return metrics

def compute_all(gt, pred, n_class):
    """Computes average AUC, F1 score, ACC, and AP.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
        n_class: Number of classes.
    Returns:
        Dictionary of average AUC, F1 score, ACC, and AP.
    """
    metrics = {}
    AUROCs = []
    F1_scores = []
    ACCs = []
    APs = []
    MCCs = []

    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    
    if n_class == 1:
        AUROCs.append(roc_auc_score(gt_np, pred_np))
        F1_scores.append(f1_score(gt_np, np.round(pred_np)))
        ACCs.append(accuracy_score(gt_np, np.round(pred_np)))
        APs.append(average_precision_score(gt_np, pred_np))
        MCCs.append(matthews_corrcoef(gt_np, np.round(pred_np)))
    else:
        for i in range(n_class):
            if i == 6 and n_class == 28:
                continue
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
            F1_scores.append(f1_score(gt_np[:, i], np.round(pred_np[:, i])))
            ACCs.append(accuracy_score(gt_np[:, i], np.round(pred_np[:, i])))
            APs.append(average_precision_score(gt_np[:, i], pred_np[:, i]))
            MCCs.append(matthews_corrcoef(gt_np[:, i], np.round(pred_np[:, i])))
            
    # mccs,threshold = compute_mccs(gt_np, pred_np,n_class)
    # F1_scores = compute_F1s_threshold(gt_np, pred_np, threshold,n_class)
    # ACCs = compute_Accs_threshold(gt_np, pred_np, threshold,n_class)
    
    metrics["mean_auc"] = np.mean(np.array(AUROCs))
    metrics["mean_f1"] = np.mean(np.array(F1_scores))
    metrics["mean_acc"] = np.mean(np.array(ACCs))
    metrics["mean_ap"] = np.mean(np.array(APs))
    metrics["mean_mcc"] = np.mean(np.array(MCCs))
    
    if n_class == 1:
        metrics["auc/class_0"] = AUROCs[0]
        metrics["f1/class_0"] = F1_scores[0]
        metrics["acc/class_0"] = ACCs[0]
        metrics["ap/class_0"] = APs[0]
        metrics["mcc/class_0"] = MCCs[0]
    elif n_class == 28:
        for i in range(len(AUROCs)):
            if i >= 6:
                metrics[f"auc/class_{i+1}"] = AUROCs[i]
                metrics[f"f1/class_{i+1}"] = F1_scores[i]
                metrics[f"acc/class_{i+1}"] = ACCs[i]
                metrics[f"ap/class_{i+1}"] = APs[i]
                metrics[f"mcc/class_{i+1}"] = MCCs[i]
            else:
                metrics[f"auc/class_{i}"] = AUROCs[i]
                metrics[f"f1/class_{i}"] = F1_scores[i]
                metrics[f"acc/class_{i}"] = ACCs[i]
                metrics[f"ap/class_{i}"] = APs[i]
                metrics[f"mcc/class_{i}"] = MCCs[i]
    else:
        for i in range(n_class):
            metrics[f"auc/class_{i}"] = AUROCs[i]
            metrics[f"f1/class_{i}"] = F1_scores[i]
            metrics[f"acc/class_{i}"] = ACCs[i]
            metrics[f"ap/class_{i}"] = APs[i]
            metrics[f"mcc/class_{i}"] = MCCs[i]

    return metrics
