import sys
sys.path.append(sys.path[0].replace('engine', ''))
import json
import time
import math
import numpy as np
from engine.train import compute_all_total, compute_all
from factory import utils
from factory.loss import ClipLoss
from engine.train import AverageMeter, get_text_features, get_weight_mask, fourier_aug
import torch
import torch.nn.functional as F
from torch import nn
import logging
from torch.cuda.amp import autocast
from configs.default import disease_text_dict, redict_label_idx_dict
from engine.train import get_domain_loss

def MomentumUpdate(model, teacher, alpha=0.999):
    teacher_dict = teacher.state_dict()
    model_dict = model.state_dict()
    for k,v in teacher_dict.items():
        teacher_dict[k] = alpha * v + (1-alpha)*model_dict[k]
    teacher.load_state_dict(teacher_dict)
    

def coral_loss(x, y):
    mean_x = x.mean(0, keepdim=True)
    mean_y = y.mean(0, keepdim=True)
    cent_x = x - mean_x
    cent_y = y - mean_y
    cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
    cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

    mean_diff = (mean_x - mean_y).pow(2).mean()
    cova_diff = (cova_x - cova_y).pow(2).mean()

    return mean_diff + cova_diff


def batch_coral_loss(features, domain_labels):
    domain_list = torch.unique(domain_labels)
    if len(domain_list) == 1:
        return 0.
    else:
        for i in range(len(domain_list)):
            for j in range(len(domain_list)):
                if i > j:
                    domain_i = domain_list[i]
                    domain_j = domain_list[j]
                    index_i = (domain_labels == domain_i)
                    index_j = (domain_labels == domain_j)

                    if features[index_i].size(0) <= 1 or features[index_j].size(0) <= 1:
                        continue
                    coral_loss_i_j = coral_loss(features[index_i], features[index_j])
                    if i == 1 and j == 0:
                        total_coral_loss = coral_loss_i_j
                    else:
                        total_coral_loss += coral_loss_i_j
    return total_coral_loss / (len(domain_list) * (len(domain_list)-1) / 2)


def train_adv(model, criterion, image_encoder, text_encoder, domain_classifier, tokenizer, data_loader, optimizer, epoch, warmup_steps, device, scheduler, args, config, writer, early_stop=0):
    # clip_loss = ClipLoss(world_size=int(os.environ['WORLD_SIZE']), rank=int(os.environ['LOCAL_RANK'])) # 考虑对ASL loss是否也要针对DDP问题使用相同的gather feature操作
    clip_loss = ClipLoss()
    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_adv_m = AverageMeter()
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
    metric_logger.add_meter('loss_adv', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
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

    
    text_list = disease_text_dict['train']
    
    with autocast():
        text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)

    # text_features这部分不能打散 所以要根据gpu的数量进行倍增
    device_num = torch.cuda.device_count()
    text_features = text_features.repeat(int(device_num),1)
        
    for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if early_stop>0 and i>early_stop: # 每个step提前跳出
            break
        if args.fourier:
            image = fourier_aug(sample['image'].to(device))
        else:
            image = sample['image'].to(device)  
        label = sample['label'].long().to(device)
        domain_label = sample['label_dataset'].long().to(device)
        
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
            loss_adv = F.cross_entropy(domain_classifier(image_features_pool), domain_label)
            pred_class_image = model(image_features,text_features)
            label = label.float()

            label_mask = (label != -1).squeeze()
            
            
            if args.add_dataset:
                pred_class_image = pred_class_image[label_mask]
                label_image = label[label_mask]
                if config['criterion'] == 'bce':
                    loss_ce_image = criterion(pred_class_image.view(-1),label_image.view(-1))
                else:
                    loss_ce_image = criterion(pred_class_image.view(-1,1),label_image.view(-1,1), use_weight=False)
                    
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
                        loss_ce_text = criterion(pred_class_text.view(-1,1),label_text.view(-1,1), use_weight=False)

                else:
                    loss_ce_text = criterion(pred_class_text.view(-1,1),label.view(-1,1))

                loss_ce = loss_ce_image  + loss_ce_text

            else:
                loss_ce = loss_ce_image

            # 要除以batch_size
            # loss_ce = loss_ce
            loss_clip = clip_loss(image_features_pool,entity_features)
            # check 输入的部分是否有nan
            # loss_clip = torch.tensor(0.)
            # loss_ce_text = torch.tensor(0.)
            # loss_ce = loss_ce_image
            loss = loss_ce  + loss_clip * args.loss_ratio + args.adv * loss_adv

        # loss的计算无法混合精度
        loss.backward()
        optimizer.step()
            
        writer.add_scalar('loss/loss', loss, scalar_step)
        writer.add_scalar('loss/loss_ce', loss_ce, scalar_step)
        writer.add_scalar('loss/loss_ce_image', loss_ce_image, scalar_step)
        writer.add_scalar('loss/loss_adv', loss_adv, scalar_step)
        if args.use_entity_features:
            writer.add_scalar('loss/loss_ce_text', loss_ce_text, scalar_step)
        writer.add_scalar('loss/loss_clip', loss_clip, scalar_step)
        scalar_step += 1

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_ce=loss_ce.item())
        metric_logger.update(loss_ce_image=loss_ce_image.item())
        metric_logger.update(loss_adv=loss_adv.item())
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
            loss_adv_m.update(loss_adv.item(), batch_size)
            loss_ce_m.update(loss_ce.item(), batch_size)
            loss_ce_image_m.update(loss_ce_image.item(), batch_size)
            if args.use_entity_features:
                loss_ce_text_m.update(loss_ce_text.item(), batch_size)
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Loss_clip: {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                    f"Loss_coral: {loss_adv_m.val:#.5g} ({loss_adv_m.avg:#.4g}) "
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
                    f"Loss_coral: {loss_adv_m.val:#.5g} ({loss_adv_m.avg:#.4g}) "
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


def train_disam_adv(model, criterion, image_encoder, text_encoder, domain_classifier, tokenizer, data_loader, optimizer, epoch, warmup_steps, device, scheduler, args, config, writer, early_stop=0):
    # clip_loss = ClipLoss(world_size=int(os.environ['WORLD_SIZE']), rank=int(os.environ['LOCAL_RANK'])) # 考虑对ASL loss是否也要针对DDP问题使用相同的gather feature操作
    clip_loss = ClipLoss()
    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_adv_m = AverageMeter()
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
    metric_logger.add_meter('loss_adv', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
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

    
    text_list = disease_text_dict['train']
    
    with autocast():
        text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)

    # text_features这部分不能打散 所以要根据gpu的数量进行倍增
    device_num = torch.cuda.device_count()
    text_features = text_features.repeat(int(device_num),1)
        
    for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if early_stop>0 and i>early_stop: # 每个step提前跳出
            break
        if args.fourier:
            image = fourier_aug(sample['image'].to(device))
        else:
            image = sample['image'].to(device)  
        label = sample['label'].long().to(device)
        domain_label = sample['label_dataset'].long().to(device)
        
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
            loss_adv = F.cross_entropy(domain_classifier(image_features_pool), domain_label)
            pred_class_image = model(image_features,text_features)
            label = label.float()

            label_mask = (label != -1).squeeze()
                
            if args.add_dataset:
                domain_loss = get_domain_loss(pred_class_image, label, label_mask, domain_label, criterion)
                pred_class_image = pred_class_image[label_mask]
                label_image = label[label_mask]
                loss_ce_image = criterion(pred_class_image.view(-1,1),label_image.view(-1,1), use_weight=config['weighted_criterion'])
                loss_ce_image = loss_ce_image - 0.1 * domain_loss
                
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
                        loss_ce_text = criterion(pred_class_text.view(-1,1),label_text.view(-1,1), use_weight=False)

                else:
                    loss_ce_text = criterion(pred_class_text.view(-1,1),label.view(-1,1))

                loss_ce = loss_ce_image  + loss_ce_text

            else:
                loss_ce = loss_ce_image

            loss_clip = clip_loss(image_features_pool,entity_features)
            loss = loss_ce  + loss_clip * args.loss_ratio + args.adv * loss_adv

            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            entity_features = get_text_features(text_encoder,entity,tokenizer,device,max_length=args.max_length)

            image_features,image_features_pool = image_encoder(image)
            loss_adv = F.cross_entropy(domain_classifier(image_features_pool), domain_label)
            pred_class_image = model(image_features,text_features)
            label = label.float()

            label_mask = (label != -1).squeeze()
            
            
            if args.add_dataset:
                pred_class_image = pred_class_image[label_mask]
                label_image = label[label_mask]
                if config['criterion'] == 'bce':
                    loss_ce_image = criterion(pred_class_image.view(-1),label_image.view(-1))
                else:
                    loss_ce_image = criterion(pred_class_image.view(-1,1),label_image.view(-1,1), use_weight=False)
                    
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
                        loss_ce_text = criterion(pred_class_text.view(-1,1),label_text.view(-1,1), use_weight=False)

                else:
                    loss_ce_text = criterion(pred_class_text.view(-1,1),label.view(-1,1))

                loss_ce = loss_ce_image  + loss_ce_text

            else:
                loss_ce = loss_ce_image

            # 要除以batch_size
            # loss_ce = loss_ce
            loss_clip = clip_loss(image_features_pool,entity_features)
            # check 输入的部分是否有nan
            # loss_clip = torch.tensor(0.)
            # loss_ce_text = torch.tensor(0.)
            # loss_ce = loss_ce_image
            loss = loss_ce  + loss_clip * args.loss_ratio + args.adv * loss_adv

        # loss的计算无法混合精度
        loss.backward()
            
        optimizer.second_step(zero_grad=True)
            
        writer.add_scalar('loss/loss', loss, scalar_step)
        writer.add_scalar('loss/loss_ce', loss_ce, scalar_step)
        writer.add_scalar('loss/loss_ce_image', loss_ce_image, scalar_step)
        writer.add_scalar('loss/loss_adv', loss_adv, scalar_step)
        if args.use_entity_features:
            writer.add_scalar('loss/loss_ce_text', loss_ce_text, scalar_step)
        writer.add_scalar('loss/loss_clip', loss_clip, scalar_step)
        scalar_step += 1

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_ce=loss_ce.item())
        metric_logger.update(loss_ce_image=loss_ce_image.item())
        metric_logger.update(loss_adv=loss_adv.item())
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
            loss_adv_m.update(loss_adv.item(), batch_size)
            loss_ce_m.update(loss_ce.item(), batch_size)
            loss_ce_image_m.update(loss_ce_image.item(), batch_size)
            if args.use_entity_features:
                loss_ce_text_m.update(loss_ce_text.item(), batch_size)
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Loss_clip: {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                    f"Loss_coral: {loss_adv_m.val:#.5g} ({loss_adv_m.avg:#.4g}) "
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
                    f"Loss_coral: {loss_adv_m.val:#.5g} ({loss_adv_m.avg:#.4g}) "
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



# 考虑model的ema 包括image encoder和 model
def train_coral_ema(model, criterion, image_encoder, text_encoder, tokenizer, ema_model, ema_image_encoder, data_loader, optimizer, epoch, warmup_steps, device, scheduler, args, config, writer, early_stop=0):
    # clip_loss = ClipLoss(world_size=int(os.environ['WORLD_SIZE']), rank=int(os.environ['LOCAL_RANK'])) # 考虑对ASL loss是否也要针对DDP问题使用相同的gather feature操作
    clip_loss = ClipLoss()
    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_coral_m = AverageMeter()
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
    metric_logger.add_meter('loss_coral', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
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

    
    text_list = disease_text_dict['train']
    
    with autocast():
        text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)

    # text_features这部分不能打散 所以要根据gpu的数量进行倍增
    device_num = torch.cuda.device_count()
    text_features = text_features.repeat(int(device_num),1)
        
    for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if early_stop>0 and i>early_stop: # 每个step提前跳出
            break
        if args.fourier:
            image = fourier_aug(sample['image'].to(device))
        else:
            image = sample['image'].to(device)  
        label = sample['label'].long().to(device)
        domain_label = sample['label_dataset'].long()
        
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
            loss_coral = batch_coral_loss(image_features_pool, domain_label)
            pred_class_image = model(image_features,text_features)
            label = label.float()

            label_mask = (label != -1).squeeze()
            
            
            if args.add_dataset:
                pred_class_image = pred_class_image[label_mask]
                label_image = label[label_mask]
                if config['criterion'] == 'bce':
                    loss_ce_image = criterion(pred_class_image.view(-1),label_image.view(-1))
                else:
                    loss_ce_image = criterion(pred_class_image.view(-1,1),label_image.view(-1,1), use_weight=False)
                    
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
                        loss_ce_text = criterion(pred_class_text.view(-1,1),label_text.view(-1,1), use_weight=False)

                else:
                    loss_ce_text = criterion(pred_class_text.view(-1,1),label.view(-1,1))

                loss_ce = loss_ce_image  + loss_ce_text

            else:
                loss_ce = loss_ce_image

            # 要除以batch_size
            # loss_ce = loss_ce
            loss_clip = clip_loss(image_features_pool,entity_features)
            # check 输入的部分是否有nan
            # loss_clip = torch.tensor(0.)
            # loss_ce_text = torch.tensor(0.)
            # loss_ce = loss_ce_image
            loss = loss_ce  + loss_clip * args.loss_ratio + loss_coral

        # loss的计算无法混合精度
        loss.backward()
        optimizer.step()
        
        if args.ema > 0.0:
            MomentumUpdate(model, ema_model, args.ema)
            MomentumUpdate(image_encoder, ema_image_encoder, args.ema)
            
        writer.add_scalar('loss/loss', loss, scalar_step)
        writer.add_scalar('loss/loss_ce', loss_ce, scalar_step)
        writer.add_scalar('loss/loss_ce_image', loss_ce_image, scalar_step)
        writer.add_scalar('loss/loss_coral', loss_coral, scalar_step)
        if args.use_entity_features:
            writer.add_scalar('loss/loss_ce_text', loss_ce_text, scalar_step)
        writer.add_scalar('loss/loss_clip', loss_clip, scalar_step)
        scalar_step += 1

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_ce=loss_ce.item())
        metric_logger.update(loss_ce_image=loss_ce_image.item())
        metric_logger.update(loss_coral=loss_coral.item())
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
            loss_coral_m.update(loss_coral.item(), batch_size)
            loss_ce_m.update(loss_ce.item(), batch_size)
            loss_ce_image_m.update(loss_ce_image.item(), batch_size)
            if args.use_entity_features:
                loss_ce_text_m.update(loss_ce_text.item(), batch_size)
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Loss_clip: {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                    f"Loss_coral: {loss_coral_m.val:#.5g} ({loss_coral_m.avg:#.4g}) "
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
                    f"Loss_coral: {loss_coral_m.val:#.5g} ({loss_coral_m.avg:#.4g}) "
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

def train_disam_coral_moe(model, criterion, image_encoder, text_encoder, tokenizer, moe_model_list, data_loader, optimizer, epoch, warmup_steps, device, scheduler, args, config, writer,early_stop=0):
    # clip_loss = ClipLoss(world_size=int(os.environ['WORLD_SIZE']), rank=int(os.environ['LOCAL_RANK'])) # 考虑对ASL loss是否也要针对DDP问题使用相同的gather feature操作
    clip_loss = ClipLoss()
    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_coral_m = AverageMeter()
    loss_ce_m = AverageMeter()
    loss_ce_image_m = AverageMeter()
    loss_ce_text_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    model.train()  
    image_encoder.train()  
    text_encoder.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_coral', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
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

    
    text_list = disease_text_dict['train']
    with autocast():
        text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)

    # text_features这部分不能打散 所以要根据gpu的数量进行倍增
    device_num = torch.cuda.device_count()
    text_features = text_features.repeat(int(device_num),1)
        
    for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if early_stop>0 and i>early_stop: # 每个step提前跳出
            break
        if args.fourier:
            image = fourier_aug(sample['image'].to(device))
        else:
            image = sample['image'].to(device)  
        label = sample['label'].long().to(device)
        domain_label = sample['label_dataset'].long()
        
        if args.ignore_index:
            pass
        else:
            label[label==-1]=0
        entity = sample['entity']


        data_time_m.update(time.time() - end)

        optimizer.zero_grad()
        
        with autocast(): # 自动混合精度
            # with torch.no_grad():
            #     entity_features = get_text_features(text_encoder,entity,tokenizer,device,max_length=args.max_length)

            #     image_features,image_features_pool = image_encoder(image)
            # # loss_coral = batch_coral_loss(image_features_pool, domain_label)
            # loss_coral = 0.
            # pred_class_image = model(image_features,text_features)
            # moe_pred_class_image_list = [pred_class_image]
            # for moe_model in moe_model_list:
            #     moe_pred_class_image_list.append(moe_model(image_features,text_features))
            
            # # 求平均
            # pred_class_image = torch.mean(torch.stack(moe_pred_class_image_list),dim=0)
            
            # label = label.float()

            # label_mask = (label != -1).squeeze()
            
            # if args.add_dataset:
            #     domain_loss = get_domain_loss(pred_class_image, label, label_mask, domain_label, criterion)
            #     pred_class_image = pred_class_image[label_mask]
            #     label_image = label[label_mask]
            #     loss_ce_image = criterion(pred_class_image.view(-1,1),label_image.view(-1,1), use_weight=config['weighted_criterion'])
            #     loss_ce_image = loss_ce_image - 0.1 * domain_loss
                
            # else:
            #     loss_ce_image = criterion(pred_class_image.view(-1,1),label.view(-1,1))
            
            # if args.use_entity_features:
            #     pred_class_text = model(entity_features.unsqueeze(1),text_features)
            #     if args.add_dataset:
            #         pred_class_text = pred_class_text[label_mask]
            #         label_text = label[label_mask]
            #         if config['criterion'] == 'bce':
            #             loss_ce_text = criterion(pred_class_text.view(-1),label_text.view(-1))
            #         else:
            #             loss_ce_text = criterion(pred_class_text.view(-1,1),label_text.view(-1,1), use_weight=False)

            #     else:
            #         loss_ce_text = criterion(pred_class_text.view(-1,1),label.view(-1,1))

            #     loss_ce = loss_ce_image  + loss_ce_text

            # else:
            #     loss_ce = loss_ce_image

            # loss_clip = clip_loss(image_features_pool,entity_features)
            # loss = loss_ce  + loss_clip * args.loss_ratio + loss_coral

            # loss.backward()
            # optimizer.first_step(zero_grad=True)
            with torch.no_grad():
                entity_features = get_text_features(text_encoder,entity,tokenizer,device,max_length=args.max_length)

                image_features,image_features_pool = image_encoder(image)
            # loss_coral = batch_coral_loss(image_features_pool, domain_label)
            loss_coral = torch.tensor(0.)
            # pred_class_image = model(image_features,text_features)
            # moe_pred_class_image_list = [pred_class_image]
            moe_pred_class_image_list = []
            for moe_model in moe_model_list:
                moe_pred_class_image_list.append(moe_model(image_features,text_features))
            
            # 求平均
            pred_class_image = torch.mean(torch.stack(moe_pred_class_image_list),dim=0)
            
            label = label.float()

            label_mask = (label != -1).squeeze()
            
            if args.add_dataset:
                pred_class_image = pred_class_image[label_mask]
                label_image = label[label_mask]
                loss_ce_image = criterion(pred_class_image.view(-1,1),label_image.view(-1,1), use_weight=False)
                loss_ce_image = loss_ce_image
                
            else:
                loss_ce_image = criterion(pred_class_image.view(-1,1),label.view(-1,1))

            if args.use_entity_features:
                pred_class_text = model(entity_features.unsqueeze(1),text_features)
                if args.add_dataset:
                    pred_class_text = pred_class_text[label_mask]
                    label_text = label[label_mask]
                    loss_ce_text = criterion(pred_class_text.view(-1,1),label_text.view(-1,1), use_weight=False)

                else:
                    loss_ce_text = criterion(pred_class_text.view(-1,1),label.view(-1,1))

                loss_ce = loss_ce_image + loss_ce_text

            else:
                loss_ce = loss_ce_image


            
            loss_clip = clip_loss(image_features_pool,entity_features)

            loss = loss_ce + loss_clip * args.loss_ratio
            
            loss.backward()
            optimizer.step()
            # optimizer.second_step(zero_grad=True)
            
            
        writer.add_scalar('loss/loss', loss, scalar_step)
        writer.add_scalar('loss/loss_ce', loss_ce, scalar_step)
        writer.add_scalar('loss/loss_ce_image', loss_ce_image, scalar_step)
        writer.add_scalar('loss/loss_coral', loss_coral, scalar_step)
        if args.use_entity_features:
            writer.add_scalar('loss/loss_ce_text', loss_ce_text, scalar_step)
        writer.add_scalar('loss/loss_clip', loss_clip, scalar_step)
        scalar_step += 1

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_ce=loss_ce.item())
        metric_logger.update(loss_ce_image=loss_ce_image.item())
        metric_logger.update(loss_coral=loss_coral.item())
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
            loss_coral_m.update(loss_coral.item(), batch_size)
            loss_ce_m.update(loss_ce.item(), batch_size)
            loss_ce_image_m.update(loss_ce_image.item(), batch_size)
            if args.use_entity_features:
                loss_ce_text_m.update(loss_ce_text.item(), batch_size)
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Loss_clip: {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                    f"Loss_coral: {loss_coral_m.val:#.5g} ({loss_coral_m.avg:#.4g}) "
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
                    f"Loss_coral: {loss_coral_m.val:#.5g} ({loss_coral_m.avg:#.4g}) "
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



def train_total_ema(model, criterion, image_encoder, text_encoder, tokenizer, ema_model, ema_image_encoder, data_loader, optimizer, epoch, warmup_steps, device, scheduler, args, config, writer, early_stop=0):
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
    # 尝试改为eval模式
    text_encoder.eval()
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

    
    text_list = disease_text_dict['total_train']
    
    with autocast():
        text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)

    # text_features这部分不能打散 所以要根据gpu的数量进行倍增
    device_num = torch.cuda.device_count()
    text_features = text_features.repeat(int(device_num),1)
        
    for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if early_stop>0 and i>early_stop: # 每个step提前跳出
            break
        if args.fourier:
            image = fourier_aug(sample['image'].to(device))
        else:
            image = sample['image'].to(device)  
        label = sample['label'].long().to(device)
        domain_label = sample['label_dataset'].long()
        
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
            
            
            if args.add_dataset:
                pred_class_image = pred_class_image[label_mask]
                label_image = label[label_mask]
                if config['criterion'] == 'bce':
                    loss_ce_image = criterion(pred_class_image.view(-1),label_image.view(-1))
                else:
                    loss_ce_image = criterion(pred_class_image.view(-1,1),label_image.view(-1,1), use_weight=False)
                    
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
                        loss_ce_text = criterion(pred_class_text.view(-1,1),label_text.view(-1,1), use_weight=False)

                else:
                    loss_ce_text = criterion(pred_class_text.view(-1,1),label.view(-1,1))

                loss_ce = loss_ce_image  + loss_ce_text

            else:
                loss_ce = loss_ce_image

            # 要除以batch_size
            # loss_ce = loss_ce
            loss_clip = clip_loss(image_features_pool,entity_features)
            # check 输入的部分是否有nan
            # loss_clip = torch.tensor(0.)
            # loss_ce_text = torch.tensor(0.)
            # loss_ce = loss_ce_image
            loss = loss_ce  + loss_clip * args.loss_ratio

        # loss的计算无法混合精度
        loss.backward()
        optimizer.step()
        
        if args.ema > 0.0:
            MomentumUpdate(model, ema_model, args.ema)
            MomentumUpdate(image_encoder, ema_image_encoder, args.ema)
            
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


def train_disam_coral_ema(model, criterion, image_encoder, text_encoder, tokenizer, ema_model, ema_image_encoder, data_loader, optimizer, epoch, warmup_steps, device, scheduler, args, config, writer,early_stop=0):
    # clip_loss = ClipLoss(world_size=int(os.environ['WORLD_SIZE']), rank=int(os.environ['LOCAL_RANK'])) # 考虑对ASL loss是否也要针对DDP问题使用相同的gather feature操作
    clip_loss = ClipLoss()
    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_coral_m = AverageMeter()
    loss_ce_m = AverageMeter()
    loss_ce_image_m = AverageMeter()
    loss_ce_text_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    model.train()  
    image_encoder.train()  
    text_encoder.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_coral', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
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

    
    text_list = disease_text_dict['train']
    with autocast():
        text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)

    # text_features这部分不能打散 所以要根据gpu的数量进行倍增
    device_num = torch.cuda.device_count()
    text_features = text_features.repeat(int(device_num),1)
        
    for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if early_stop>0 and i>early_stop: # 每个step提前跳出
            break
        if args.fourier:
            image = fourier_aug(sample['image'].to(device))
        else:
            image = sample['image'].to(device)  
        label = sample['label'].long().to(device)
        domain_label = sample['label_dataset'].long()
        
        if args.ignore_index:
            pass
        else:
            label[label==-1]=0
        entity = sample['entity']


        data_time_m.update(time.time() - end)

        optimizer.zero_grad()
        
        with autocast(): # 自动混合精度
            entity_features = get_text_features(text_encoder,entity,tokenizer,device,max_length=args.max_length)

            image_features,image_features_pool = image_encoder(image)
            loss_coral = batch_coral_loss(image_features_pool, domain_label)
            pred_class_image = model(image_features,text_features)
            label = label.float()

            label_mask = (label != -1).squeeze()
            
            if args.add_dataset:
                domain_loss = get_domain_loss(pred_class_image, label, label_mask, domain_label, criterion)
                pred_class_image = pred_class_image[label_mask]
                label_image = label[label_mask]
                loss_ce_image = criterion(pred_class_image.view(-1,1),label_image.view(-1,1), use_weight=config['weighted_criterion'])
                loss_ce_image = loss_ce_image - 0.1 * domain_loss
                
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
                        loss_ce_text = criterion(pred_class_text.view(-1,1),label_text.view(-1,1), use_weight=False)

                else:
                    loss_ce_text = criterion(pred_class_text.view(-1,1),label.view(-1,1))

                loss_ce = loss_ce_image  + loss_ce_text

            else:
                loss_ce = loss_ce_image

            loss_clip = clip_loss(image_features_pool,entity_features)
            loss = loss_ce  + loss_clip * args.loss_ratio + loss_coral

            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            entity_features = get_text_features(text_encoder,entity,tokenizer,device,max_length=args.max_length)

            image_features,image_features_pool = image_encoder(image)
            loss_coral = batch_coral_loss(image_features_pool, domain_label)
            pred_class_image = model(image_features,text_features)

            label = label.float()

            label_mask = (label != -1).squeeze()
            
            if args.add_dataset:
                pred_class_image = pred_class_image[label_mask]
                label_image = label[label_mask]
                loss_ce_image = criterion(pred_class_image.view(-1,1),label_image.view(-1,1), use_weight=False)
                loss_ce_image = loss_ce_image
                
            else:
                loss_ce_image = criterion(pred_class_image.view(-1,1),label.view(-1,1))

            if args.use_entity_features:
                pred_class_text = model(entity_features.unsqueeze(1),text_features)
                if args.add_dataset:
                    pred_class_text = pred_class_text[label_mask]
                    label_text = label[label_mask]
                    loss_ce_text = criterion(pred_class_text.view(-1,1),label_text.view(-1,1), use_weight=False)

                else:
                    loss_ce_text = criterion(pred_class_text.view(-1,1),label.view(-1,1))

                loss_ce = loss_ce_image + loss_ce_text

            else:
                loss_ce = loss_ce_image


            
            loss_clip = clip_loss(image_features_pool,entity_features)

            loss = loss_ce + loss_clip * args.loss_ratio  + loss_coral
            
            loss.backward()
            
            optimizer.second_step(zero_grad=True)
            
        
        if args.ema > 0.0:
            MomentumUpdate(model, ema_model, args.ema)
            MomentumUpdate(image_encoder, ema_image_encoder, args.ema)
            
        writer.add_scalar('loss/loss', loss, scalar_step)
        writer.add_scalar('loss/loss_ce', loss_ce, scalar_step)
        writer.add_scalar('loss/loss_ce_image', loss_ce_image, scalar_step)
        writer.add_scalar('loss/loss_coral', loss_coral, scalar_step)
        if args.use_entity_features:
            writer.add_scalar('loss/loss_ce_text', loss_ce_text, scalar_step)
        writer.add_scalar('loss/loss_clip', loss_clip, scalar_step)
        scalar_step += 1

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_ce=loss_ce.item())
        metric_logger.update(loss_ce_image=loss_ce_image.item())
        metric_logger.update(loss_coral=loss_coral.item())
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
            loss_coral_m.update(loss_coral.item(), batch_size)
            loss_ce_m.update(loss_ce.item(), batch_size)
            loss_ce_image_m.update(loss_ce_image.item(), batch_size)
            if args.use_entity_features:
                loss_ce_text_m.update(loss_ce_text.item(), batch_size)
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Loss_clip: {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                    f"Loss_coral: {loss_coral_m.val:#.5g} ({loss_coral_m.avg:#.4g}) "
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
                    f"Loss_coral: {loss_coral_m.val:#.5g} ({loss_coral_m.avg:#.4g}) "
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





def train_1000(model, criterion, image_encoder, text_encoder, tokenizer, data_loader, optimizer, epoch, warmup_steps, device, scheduler, args, config, writer):
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
        if i >= 5000:
            break
        
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

def valid_single_dataset(model, image_encoder, text_encoder, tokenizer, dataset_name, data_loader, epoch, device, args, config, writer, total_test=False):
    text_list = disease_text_dict[dataset_name] # 获取类别名称
    num_classes = len(text_list)
    model.eval()
    image_encoder.eval()
    text_encoder.eval()
    text_features = get_text_features(text_encoder, text_list, tokenizer, device, max_length=args.max_length)
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
        metrics = compute_all_total(gt, pred, n_class = num_classes)
    else:
        metrics = compute_all(gt, pred, n_class = num_classes)
    AUROC_avg = metrics['mean_auc']
    avg_val_loss = np.array(val_losses).mean()
    
    return avg_val_loss,AUROC_avg,metrics


def valid_single_dataset_moe(model, moe_model_lists, image_encoder, text_encoder, tokenizer, dataset_name, data_loader, epoch, device, args, config, writer, total_test=False):
    text_list = disease_text_dict[dataset_name] # 获取类别名称
    num_classes = len(text_list)
    model.eval()
    image_encoder.eval()
    text_encoder.eval()
    text_features = get_text_features(text_encoder, text_list, tokenizer, device, max_length=args.max_length)
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
            
            # pred_class = model(image_features,text_features)#b,14,2/1
            
            # moe_pred_class_list = [pred_class]
            moe_pred_class_list = []
            for moe_model in moe_model_lists:
                moe_pred_class = moe_model(image_features,text_features)
                moe_pred_class_list.append(moe_pred_class)
            # average the prediction
            pred_class = torch.stack(moe_pred_class_list).mean(0)
            
            val_loss = F.binary_cross_entropy_with_logits(pred_class.view(-1,1),label.view(-1, 1))
            pred_class = torch.sigmoid(pred_class)
            pred = torch.cat((pred, pred_class[:,:,0]), 0)

            val_losses.append(val_loss.item())
            writer.add_scalar('val_loss/loss', val_loss, val_scalar_step)
            val_scalar_step += 1
    if total_test:
        metrics = compute_all_total(gt, pred, n_class = num_classes)
    else:
        metrics = compute_all(gt, pred, n_class = num_classes)
    AUROC_avg = metrics['mean_auc']
    avg_val_loss = np.array(val_losses).mean()
    
    return avg_val_loss,AUROC_avg,metrics




def train_nokad(model, criterion, data_loader, optimizer, epoch, warmup_steps, device, scheduler, args, config, writer):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    loss_m = AverageMeter()
    end = time.time()

    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.update(loss=1.0)
    metric_logger.update(lr = scheduler._get_lr(epoch)[0])
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size 
    scalar_step = epoch*len(data_loader)
    num_batches_per_epoch = data_loader.num_batches
    sample_digits = math.ceil(math.log(data_loader.num_samples + 1, 10))

        
    for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = sample['image'].to(device)  
        label = sample['label'].long().to(device)

        if args.ignore_index:
            pass
        else:
            label[label==-1]=0
            
        data_time_m.update(time.time() - end)

        optimizer.zero_grad()
        
        with autocast(): # 自动混合精度
            pred_class_image = model(image)
            
            label = label.float()

            label_mask = (label != -1).squeeze()
            pred_class_image = pred_class_image[label_mask]
            label_image = label[label_mask]
            if config['criterion'] == 'bce':
                loss = criterion(pred_class_image.view(-1),label_image.view(-1))
            else:
                loss = criterion(pred_class_image.view(-1,1),label_image.view(-1,1), use_weight=config['weighted_criterion'])
        
        loss.backward()
        optimizer.step() 

        writer.add_scalar('loss/loss', loss, scalar_step)
        
        metric_logger.update(loss=loss.item())

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

            if args.use_entity_features:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {batch_size/ batch_time_m.val:#g}/s "
                    f"LR: { scheduler._get_lr(epoch)[0]:5f} "
                )
            else:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {batch_size/ batch_time_m.val:#g}/s "
                    f"LR: { scheduler._get_lr(epoch)[0]:5f} "
                )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} #,loss_epoch.mean()


def valid_nokad(model, dataset_name, data_loader, epoch, device, args, config, writer, total_test=False):
    text_list = disease_text_dict[dataset_name] # 获取类别名称
    num_classes = len(text_list)
    model.eval()
    
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
            with autocast():
                pred_class = model(image)
            
            # label 重映射
            redirect_pred_class = pred_class[:,redict_label_idx_dict[dataset_name]]
            val_loss = F.binary_cross_entropy_with_logits(redirect_pred_class.view(-1,1),label.view(-1, 1))
            redirect_pred_class = torch.sigmoid(redirect_pred_class)
            pred = torch.cat((pred, redirect_pred_class), 0)
            val_losses.append(val_loss.item())
            writer.add_scalar('val_loss/loss', val_loss, val_scalar_step)
            val_scalar_step += 1
    if total_test:
        metrics = compute_all_total(gt, pred, n_class = num_classes)
    else:
        metrics = compute_all(gt, pred, n_class = num_classes)
    AUROC_avg = metrics['mean_auc']
    avg_val_loss = np.array(val_losses).mean()
    
    return avg_val_loss,AUROC_avg,metrics





