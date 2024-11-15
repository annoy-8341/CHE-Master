import sys
# 加入父文件夹路径到sys.path中  
sys.path.append(sys.path[0].replace('main', ''))
import os
import logging
import time
import datetime
import math
import torch
from torch.utils.tensorboard import SummaryWriter
from factory import utils
from scheduler import create_scheduler
from optim import create_optimizer, create_optimizer_sam
from engine.train import AverageMeter, get_text_features, get_weight_mask, fourier_aug
from main.main_utils import seed_torch, get_dataloader, get_model, load_checkpoint, get_criterion, test_all, get_args
from configs.default import disease_text_dict
from torch.cuda.amp import autocast
from factory.loss import ClipLoss




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

    # mean_domain_loss = torch.mean(torch.stack(domain_loss_list))
    
    var_domain_loss = torch.std(torch.stack(domain_loss_list))
    return var_domain_loss

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


def train_disam(model, criterion, image_encoder, text_encoder, tokenizer, data_loader, optimizer, epoch, warmup_steps, device, scheduler, args, config, writer,early_stop=0):
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
            
            domain_loss = get_domain_loss(pred_class_image, label, label_mask, domain_label, criterion)
            pred_class_image = pred_class_image[label_mask]
            label_image = label[label_mask]
            loss_ce_image = criterion(pred_class_image.view(-1,1),label_image.view(-1,1), use_weight=config['weighted_criterion'])
            loss_ce_image = loss_ce_image - 0.1 * domain_loss
            
            pred_class_text = model(entity_features.unsqueeze(1),text_features)
            pred_class_text = pred_class_text[label_mask]
            label_text = label[label_mask]
            if config['criterion'] == 'bce':
                loss_ce_text = criterion(pred_class_text.view(-1),label_text.view(-1))
            else:
                loss_ce_text = criterion(pred_class_text.view(-1,1),label_text.view(-1,1), use_weight=False)

            loss_ce = loss_ce_image  + loss_ce_text
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

            pred_class_image = pred_class_image[label_mask]
            label_image = label[label_mask]
            loss_ce_image = criterion(pred_class_image.view(-1,1),label_image.view(-1,1), use_weight=False)
            loss_ce_image = loss_ce_image

            pred_class_text = model(entity_features.unsqueeze(1),text_features)
            pred_class_text = pred_class_text[label_mask]
            label_text = label[label_mask]
            loss_ce_text = criterion(pred_class_text.view(-1,1),label_text.view(-1,1), use_weight=False)

            loss_ce = loss_ce_image + loss_ce_text

            loss_clip = clip_loss(image_features_pool,entity_features)

            loss = loss_ce + loss_clip * args.loss_ratio  + loss_coral
            
            loss.backward()
            
            optimizer.second_step(zero_grad=True)
            
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


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} #,loss_epoch.mean()


def train(model, criterion, image_encoder, text_encoder, tokenizer, data_loader, optimizer, epoch, warmup_steps, device, scheduler, args, config, writer,early_stop=0):
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
            
            domain_loss = get_domain_loss(pred_class_image, label, label_mask, domain_label, criterion)
            pred_class_image = pred_class_image[label_mask]
            label_image = label[label_mask]
            loss_ce_image = criterion(pred_class_image.view(-1,1),label_image.view(-1,1), use_weight=config['weighted_criterion'])
            loss_ce_image = loss_ce_image - 0.1 * domain_loss
            
            pred_class_text = model(entity_features.unsqueeze(1),text_features)
            pred_class_text = pred_class_text[label_mask]
            label_text = label[label_mask]
            if config['criterion'] == 'bce':
                loss_ce_text = criterion(pred_class_text.view(-1),label_text.view(-1))
            else:
                loss_ce_text = criterion(pred_class_text.view(-1,1),label_text.view(-1,1), use_weight=False)

            loss_ce = loss_ce_image  + loss_ce_text

            loss_clip = clip_loss(image_features_pool,entity_features)
            loss = loss_ce  + loss_clip * args.loss_ratio + loss_coral

            loss.backward()
            optimizer.step()
            
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

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} #,loss_epoch.mean()


def main(args, config):
    device = torch.device(args.device)
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    
    test_index = {'val_chestxray': 0, 'test_chestxray':2, 'chexpert': 1, 'vindr': 3, 'siimacr': 4, 'openi': 5, 'shenzhen': 6, 'padchest': 7, 'mimic': 8}
    best_epoch_dict = {'val_chestxray': 0, 'test_chestxray':0, 'chexpert': 0, 'vindr': 0, 'siimacr': 0, 'openi': 0, 'shenzhen': 0, 'padchest': 0, 'mimic': 0}
    best_results_dict = {'val_chestxray': 0, 'test_chestxray':0, 'chexpert': 0, 'vindr': 0, 'siimacr': 0, 'openi': 0, 'shenzhen': 0, 'padchest': 0, 'mimic': 0}

    data_index = {'train': 0, 'val_chestxray': 1, 'test_chestxray':2, 'chexpert': 3, 'vindr': 4, 'siimacr': 5, 'openi': 6, 'shenzhen': 7, 'padchest': 8, 'mimic': 9}
    '''Data准备'''
    # 获得所有的dataloader和dataset
    dataloaders_and_datasets = get_dataloader(args, config)
    dataloaders = dataloaders_and_datasets[:10]
    datasets = dataloaders_and_datasets[10:]

    # 使用zip函数和循环来设置每个dataloader的num_samples和num_batches
    for dataloader, dataset in zip(dataloaders, datasets):
        dataloader.num_samples = len(dataset)
        dataloader.num_batches = len(dataloader)
    
    model, image_encoder, text_encoder, tokenizer = get_model(args, config)
    
    arg_opt = utils.AttrDict(config['optimizer'])
    if args.rho>0:
        optimizer = create_optimizer_sam(arg_opt, model, image_encoder, text_encoder, rho=args.rho)
    else:
        optimizer = create_optimizer(arg_opt, model, image_encoder, text_encoder)

    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer) 
    
    load_checkpoint(model, image_encoder, args)
            
    print("Start training")
    start_time = time.time()
    writer = SummaryWriter(os.path.join(args.output_dir,  'log'))
        
    criterion = get_criterion(config)
    best_unseen_auc = 0
    for epoch in range(start_epoch, max_epoch):
        # if args.distributed:
        #     dist.barrier()
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)

        if args.rho>0:
            print('Use SAM!')
            train_stats = train_disam(model, criterion, image_encoder, text_encoder, tokenizer, dataloaders[data_index['train']], optimizer, epoch, warmup_steps, device, lr_scheduler, args, config, writer) 

        else:
            train_stats = train(model, criterion, image_encoder, text_encoder, tokenizer, dataloaders[data_index['train']], optimizer, epoch, warmup_steps, device, lr_scheduler, args, config, writer) 
        
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

        results = test_all(model, image_encoder, text_encoder, tokenizer, 
                           dataloaders[data_index['val_chestxray']], dataloaders[data_index['test_chestxray']], dataloaders[data_index['chexpert']], 
                           dataloaders[data_index['vindr']], dataloaders[data_index['siimacr']], dataloaders[data_index['openi']], 
                           dataloaders[data_index['shenzhen']], dataloaders[data_index['padchest']], dataloaders[data_index['mimic']], 
                           args.device, args, config, writer, epoch=epoch)

        for dataset_name in best_results_dict.keys():
            single_auc = results[test_index[dataset_name]]
            if best_results_dict[dataset_name] < single_auc:
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(f"Save best {dataset_name} model at epoch {epoch} with {single_auc:.4f}.\n")
                best_results_dict[dataset_name] = single_auc
                best_epoch_dict[dataset_name] = epoch
        
        # 标记出unseen dataset上总计最好的结果
        unseen_auc = sum(results[3:])
        
        if unseen_auc > best_unseen_auc:
            best_unseen_auc = unseen_auc
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(f"Best unseen dataset model at epoch {epoch} with {unseen_auc:.4f}.\n")
        
        # 每个epoch最后都要汇报所有的情况
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write('\nSummary:\n')
            for key_name in best_epoch_dict.keys():
                f.write(f"Best {key_name} model at epoch {best_epoch_dict[key_name]} with {best_results_dict[key_name]:.4f}.\n")
            f.write('\n')
                        
        save_obj = {
            'model': model.state_dict(),
            'image_encoder': image_encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'config': config,
            'epoch': epoch,
        }
        torch.save(save_obj, os.path.join(args.aws_output_dir, 'checkpoint_'+str(epoch)+'.pt'))
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args, config = get_args()
    seed_torch(args.seed)
    main(args, config)

