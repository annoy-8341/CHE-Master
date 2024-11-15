""" Optimizer Factory w/ Custom Weight Decay
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch import optim as optim

from .adafactor import Adafactor
from .adahessian import Adahessian
from .adamp import AdamP
from .lookahead import Lookahead
from .nadam import Nadam
from .novograd import NovoGrad
from .nvnovograd import NvNovoGrad
from .radam import RAdam
from .rmsprop_tf import RMSpropTF
from .sgdp import SGDP
from collections import defaultdict
try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False

import torch
import math

class AdaptSAM(torch.optim.Optimizer):
    # 这里fedsam代码中给的rho是0.1
    def __init__(self, params, base_optimizer, rho=0.1, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(AdaptSAM, self).__init__(params, defaults)
        if base_optimizer == torch.optim.SGD:
            self.base_optim_type = 'sgd_momentum'
        elif base_optimizer == torch.optim.Adam:
            self.base_optim_type = 'adam'
        elif base_optimizer == torch.optim.AdamW:
            self.base_optim_type = 'adamw'
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        
    @torch.no_grad()
    def first_step(self, zero_grad=False, epoch=None):
        # 能否在向上的这个梯度步骤中加入momentum 或者二阶估计 让上升的扰动也更加的有效？
        
        # 首先所有的grad都加上momentum
        for group in self.param_groups:
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None: continue
                # print(self.state[p])
                self.state[p]["old_p"] = p.data.clone()
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    self.state[p]['momentum_buffer'] = torch.clone(p.grad).detach()
                else:
                    buf = self.state[p]['momentum_buffer']
                    buf.mul_(momentum)
                    p.grad.add_(buf)
                    self.state[p]['momentum_buffer'] = torch.clone(p.grad).detach()
                    
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = 1.0 * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        
        
class SGDM_SAM(AdaptSAM):
    def __init__(self, params, base_optimizer, rho=0.1, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert base_optimizer == torch.optim.SGD, f"Invalid base_optimizer, should be SGD: {base_optimizer}"
        super(SGDM_SAM, self).__init__(params, base_optimizer, rho, adaptive, **kwargs)

class ADAMW_SAM(AdaptSAM):
    def __init__(self, params, base_optimizer, rho=0.1, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert base_optimizer == torch.optim.AdamW, f"Invalid base_optimizer, should be AdamW: {base_optimizer}"
        super(ADAMW_SAM, self).__init__(params, base_optimizer, rho, adaptive, **kwargs)
        self.base_optim_type = 'adamw'
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.betas = self.base_optimizer.defaults['betas']
        self.weight_decay = self.base_optimizer.defaults['weight_decay']
        self.eps = self.base_optimizer.defaults['eps']
        self.rho = rho

    
    @torch.no_grad()
    def first_step(self, zero_grad=False, epoch=None):
        # 首先计算adam的梯度
        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    self.state[p]["old_p"] = p.data.clone()
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                    state = self.state[p]
                    # Lazy state initialization
                    if 'step' not in state:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    # update the steps for each param group update
                    state['step'] += 1
                    
                    # 计算AdamW部分
                    grad = p.grad
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    step = state['step']

                    # Perform stepweight decay
                    # p.mul_(1 - self.rho * self.weight_decay)

                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)

                    step_size = 1.0 / bias_correction1
                    
                    p.grad = step_size * exp_avg / denom
                    self.state[p]["exp_avg"] = exp_avg
                    self.state[p]["exp_avg_sq"] = exp_avg_sq
                    self.state[p]["step"] = step
                    
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = 1.0 * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()
        
def add_weight_decay(model, image_encoder,text_encoder, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    for name, param in image_encoder.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    # for name, param in text_encoder.named_parameters():
    #     if not param.requires_grad:
    #         continue  # frozen weights
    #     if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
    #         no_decay.append(param)
    #     else:
    #         decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]
    
class BaseSAM(torch.optim.Optimizer):
    # 这里fedsam代码中给的rho是0.1
    def __init__(self, params, base_optimizer, rho=0.1, adaptive=False, max_decay_epoch=-1, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(BaseSAM, self).__init__(params, defaults)
        self.max_decay_epoch = max_decay_epoch
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False, epoch=0.):
        grad_norm = self._grad_norm()
        if self.max_decay_epoch <= 0.:
            decay_ratio = 1.0
        else: # 测试线性decay
            decay_ratio = (self.max_decay_epoch - epoch) / self.max_decay_epoch
        
        for group in self.param_groups:
            scale = decay_ratio * group["rho"] / (grad_norm + 1e-12)
            # print(decay_ratio, grad_norm.item(), scale.item(), scale * grad_norm) 
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                # print(torch.pow(p, 2) * scale.to(p)) # adaptive的scale比rho好得多
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        
class SAM():
    def __init__(self, optimizer, parameters, rho=0.5):
        self.optimizer = optimizer
        self.parameters = parameters
        self.param_groups=self.optimizer.param_groups
        self.rho = rho
        #self.eta = eta
        self.state = defaultdict(dict)
    @torch.no_grad()
    def ascent_step(self):
        
        grads = []
        for param in self.parameters:
            for p in param["params"]:

                
                if p.grad is None:
                    continue

                grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for param in self.parameters:
            if param["params"]:
                for p in param["params"]:
                    if p.grad is None:
                        continue
                    eps = self.state[p].get("eps")
                    if eps is None:
                        eps = torch.clone(p).detach()
                        self.state[p]["eps"] = eps
                    eps[...] = p.grad[...]
                    eps.mul_(self.rho / grad_norm)
                    p.add_(eps)
        self.optimizer.zero_grad()
    @torch.no_grad()
    def descent_step(self):
        for param in self.parameters:
            for p in param["params"]:
            
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()
def create_optimizer_sam(args, model, image_encoder,text_encoder, filter_bias_and_bn=True,rho=0.5, domain_classifier=None):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = add_weight_decay(model,image_encoder,text_encoder, weight_decay, skip)
        #print(parameters.shape)
        weight_decay = 0.
    else:
        parameters = [filter(lambda p: p.requires_grad, model.parameters()),filter(lambda p: p.requires_grad, image_encoder.parameters())]
        
        #model.parameters()
    if domain_classifier is not None:
        parameters += [{'params':filter(lambda p: p.requires_grad, domain_classifier.parameters()), 'weight_decay':weight_decay}]
        
    # print(parameters)
    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas
    if hasattr(args, 'opt_args') and args.opt_args is not None:
        opt_args.update(args.opt_args)

    # opt_split = opt_lower.split('_')
    # opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
        
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'sam_my_adamw':
        optimizer = ADAMW_SAM(parameters, optim.AdamW, rho=rho, **opt_args)
    elif opt_lower == 'sam_my_sgdm':
        optimizer = SGDM_SAM(parameters, optim.SGD, rho=rho, **opt_args)
    elif opt_lower == 'sam':
        optimizer = BaseSAM(parameters, optim.SGD, rho=rho, **opt_args)
    elif opt_lower == 'sam_adamw':
        optimizer = BaseSAM(parameters, optim.AdamW, rho=rho, **opt_args)
    
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':        
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':        
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'novograd':
        optimizer = NovoGrad(parameters, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    # if len(opt_split) > 1:
    #     if opt_split[0] == 'lookahead':
    #         optimizer = Lookahead(optimizer)
    # optimizer_sam=SAM(optimizer,parameters,rho)
    
    return optimizer


def create_optimizer_sam_moe(args, moe_model_list, filter_bias_and_bn=True,rho=0.5, domain_classifier=None):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(moe_model_list[0], 'no_weight_decay'):
            skip = moe_model_list[0].no_weight_decay()
        #print(parameters.shape)
        weight_decay = 0.
        decay = []
        no_decay = []
        if len(moe_model_list) >= 1:
            for name, param in moe_model_list[0].named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or name in skip:
                    no_decay.append(param)
                else:
                    decay.append(param)
        if len(moe_model_list) >= 2:
            for name, param in moe_model_list[1].named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or name in skip:
                    no_decay.append(param)
                else:
                    decay.append(param)
        if len(moe_model_list) >= 3:
            for name, param in moe_model_list[2].named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or name in skip:
                    no_decay.append(param)
                else:
                    decay.append(param)
        parameters = [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]
    else:
        parameters = []
        for i in range(len(moe_model_list)):
            parameters += [filter(lambda p: p.requires_grad, moe_model_list[i].parameters())]

        #model.parameters()
    if domain_classifier is not None:
        parameters += [{'params':filter(lambda p: p.requires_grad, domain_classifier.parameters()), 'weight_decay':weight_decay}]
        
    # print(parameters)
    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas
    if hasattr(args, 'opt_args') and args.opt_args is not None:
        opt_args.update(args.opt_args)

    # opt_split = opt_lower.split('_')
    # opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
        
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'sam_my_adamw':
        optimizer = ADAMW_SAM(parameters, optim.AdamW, rho=rho, **opt_args)
    elif opt_lower == 'sam_my_sgdm':
        optimizer = SGDM_SAM(parameters, optim.SGD, rho=rho, **opt_args)
    elif opt_lower == 'sam':
        optimizer = BaseSAM(parameters, optim.SGD, rho=rho, **opt_args)
    elif opt_lower == 'sam_adamw':
        optimizer = BaseSAM(parameters, optim.AdamW, rho=rho, **opt_args)
    
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':        
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':        
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'novograd':
        optimizer = NovoGrad(parameters, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    # if len(opt_split) > 1:
    #     if opt_split[0] == 'lookahead':
    #         optimizer = Lookahead(optimizer)
    # optimizer_sam=SAM(optimizer,parameters,rho)
    
    return optimizer




def create_optimizer(args, model, image_encoder,text_encoder, filter_bias_and_bn=True, domain_classifier=None):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = add_weight_decay(model,image_encoder,text_encoder, weight_decay, skip)
        weight_decay = 0.
    else:
        # parameters = [filter(lambda p: p.requires_grad, model.parameters()),filter(lambda p: p.requires_grad, image_encoder.parameters()),filter(lambda p: p.requires_grad, text_encoder.parameters())]
        parameters = [filter(lambda p: p.requires_grad, model.parameters()),filter(lambda p: p.requires_grad, image_encoder.parameters())]
        #model.parameters()
    if domain_classifier is not None:
        parameters += [{'params':filter(lambda p: p.requires_grad, domain_classifier.parameters()), 'weight_decay':weight_decay}]
    # print(parameters)
    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas
    if hasattr(args, 'opt_args') and args.opt_args is not None:
        opt_args.update(args.opt_args)

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=0.9, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':        
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':        
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'novograd':
        optimizer = NovoGrad(parameters, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer
