import os
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from types import MethodType
import torch
from tqdm import tqdm
import json
from datetime import datetime

def save_act_max(model, bit):
    max_act = {}
    for name, module in model.named_modules():
        if hasattr(module, "act_scale"):
            m_name = module.own_name
            act_scale = module.act_scale
            max_act[m_name] = act_scale.item() * (2**(bit-1)-1)
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    # import ipdb;ipdb.set_trace()
    with open(f"{formatted_time}_max_act_w{bit}a{bit}_w4a{bit}.json","w") as fp:
        json.dump(max_act, fp)

def quant_activation(x, bit, act_scale):
    n = 2 ** (bit - 1) - 1
    aint = (x / act_scale).round().clamp(-n-1,n)
    x = aint * act_scale
    return x

def quant_weight(w, bit, mode="channel_wise", symmetric=True):
    if mode=="channel_wise" and symmetric:
        n = 2 ** (bit - 1) - 1
        # conv
        if len(w.shape)==4:
            shape = w.shape
            scale_channel_wise = w.view(shape[0],-1).abs().max(dim=1,keepdim=True)[0] / n
            scale_channel_wise = scale_channel_wise.view(shape[0],1,1,1)
        elif len(w.shape)==2:
            scale_channel_wise = w.abs().max(dim=1,keepdim=True)[0] / n
        else:
            raise NotImplementedError
        wint = (w/scale_channel_wise).round().clamp(-n-1,n)
        wq = wint * scale_channel_wise
    else:
        raise NotImplementedError

    return wq , scale_channel_wise
def linear_percentile_search(x, w, bias, z0, bit, search_space=200):
    """percentile method to determine clipping point

    Args:
        x : raw activation
        w : raw/quanted weight
        bias : raw bias
        z0 : raw x@w
        search_space (int, optional): Defaults to 200.
    """
    absmax = x.abs().max()
    min_loss = None
    best_clip = None
    pbar = tqdm(range(search_space, 0, -1), desc="search clip")
    for ii in pbar:
        clip_value = absmax/search_space*ii
        act_scale = clip_value/(2**(bit-1)-1)
        z = F.linear(quant_activation(x.clamp(-clip_value, clip_value), bit, act_scale), w, bias)
        loss = ((z-z0)**2).mean()
        if min_loss is None:
            min_loss = loss
            best_clip = clip_value
        elif loss < min_loss:
            min_loss = loss
            best_clip = clip_value
            best_act_scale = act_scale
        pbar.set_postfix(
            loss=f"{loss.item():.2e}", 
            loss_min=0 if min_loss is None else f"{min_loss:.2e}",
            absmax=f"{absmax.item():.2e}",
            best_clip=0 if best_clip is None else f"{best_clip.item():.2e}"
        )

    return best_act_scale

def conv_percentile_search(x, w, bias, z0, bit, stride, padding, dilation, groups, search_space=200):
    absmax = x.abs().max()
    min_loss = None
    best_clip = None
    pbar = tqdm(range(search_space, 0, -1), desc="search clip")
    for ii in pbar:
        clip_value = absmax/search_space*ii
        act_scale = clip_value/(2**(bit-1)-1)
        z = F.conv2d(
            quant_activation(x.clamp(-clip_value, clip_value), bit, act_scale), 
            w, bias, stride, padding, dilation, groups
        )
        loss = ((z-z0)**2).mean()
        if min_loss is None:
            min_loss = loss
            best_clip = clip_value
            best_act_scale = act_scale
        elif loss < min_loss:
            min_loss = loss
            best_clip = clip_value
            best_act_scale = act_scale
        pbar.set_postfix(
            loss=f"{loss.item():.2e}", 
            loss_min=0 if min_loss is None else f"{min_loss:.2e}",
            absmax=f"{absmax.item():.2e}",
            best_clip=0 if best_clip is None else f"{best_clip.item():.2e}"
        )

    return best_act_scale

def quant_linear_forward(self, x: Tensor) -> Tensor:
    # NOTE: the 1st forward should be determine absmax from calibration!!
    if self.clip_search:
        z0 = F.linear(x, self.weight, self.bias)
        self.act_scale = conv_percentile_search(x, self.weight, self.bias, z0, self.bit, search_space=1000)
        self.clip_search = False
        xq = quant_activation(x, self.bit, self.act_scale)
        return F.linear(xq, self.weight, self.bias)
    elif self.noisy_search: 
        if self.with_noisy_quant:
            # NoisyQunat implementation
            criterion = torch.nn.MSELoss()
            noisy_bias = (torch.randn_like(x[:1,:1,:])*2-1)*self.act_scale
            search_space_mean = 200
            search_space_range = 1000
            
            if self.search_mean:
                # determine mean of noisy bias
                loss_min = 1e6
                best_candidate = torch.tensor([0.0])
                pbar = tqdm(range(-search_space_mean,search_space_mean),desc=f"noisy mean: {self.own_name}")
                for ii in pbar:
                    candidate = self.act_scale * ii/search_space_mean
                    xq = quant_activation(x + candidate, bit=self.bit, act_scale=self.act_scale)
                    xq -= candidate
                    zq = F.linear(xq, self.weight, self.bias)
                    z = F.linear(x, self.weight, self.bias)
                    loss = criterion(zq, z)
                    pbar.set_postfix(
                        loss=f"{loss.item():.2e}", 
                        loss_min=f"{loss_min:.2e}",
                        best_mean=f"{best_candidate.item():.2e}"
                    )
                    if loss < loss_min:
                        loss_min = loss
                        best_candidate = candidate
                self.noisy_bias = best_candidate
                best_noisy_mean = best_candidate
            else:
                self.noisy_bias = 0.
                best_noisy_mean = 0.
            
            if self.search_noisy:
                # determine range of noisy bias
                loss_min = 1e6
                best_candidate_scale = 0
                pbar = tqdm(range(0,search_space_range*2),desc=f"noisy range: {self.own_name}")
                for ii in pbar:
                    candidate = best_noisy_mean + noisy_bias * ii/search_space_range
                    xq = quant_activation(x + candidate, bit=self.bit, act_scale=self.act_scale)
                    xq -= candidate
                    zq = F.linear(xq, self.weight, self.bias)
                    z = F.linear(x, self.weight, self.bias)
                    loss = criterion(zq, z)
                    pbar.set_postfix(
                        loss=f"{loss.item():.2e}", 
                        loss_min=f"{loss_min:.2e}",
                        best_range=f"{best_candidate_scale:.2e}"
                    )
                    if loss < loss_min:
                        loss_min = loss
                        best_candidate = candidate
                        best_candidate_scale = ii/search_space_range
                self.noisy_bias = best_candidate
            else:
                self.noisy_bias = best_noisy_mean
            self.add_noise = True
            self.noisy_search = False
            x = quant_activation(x + self.noisy_bias, bit=self.bit, act_scale=self.act_scale)
            x -= self.noisy_bias
        else:
            # vanilla quant with no tricks, e.g., clipping, zero-shifting, bias-correction ...
            x = quant_activation(x, bit=self.bit, act_scale=self.act_scale)
    else:
        if self.add_noise:
            x = x + self.noisy_bias
        x = quant_activation(x, bit=self.bit, act_scale=self.act_scale)
        if self.add_noise:
            x = x - self.noisy_bias
    return F.linear(x, self.weight, self.bias)

def conv_weight_search(x,w,scale,bit,z0, bias,stride, padding, dilation, groups):
    n = 2 ** (bit - 1) - 1
    pbar = tqdm(range(scale.shape[0]), desc="weight search")
    for ii in pbar:
        min_loss = 1e5
        scale_try = scale
        for jj in range(0,101):
            mul = 0.5 + jj*0.005
            scale_try[ii,0,0,0] = scale[ii,0,0,0]*mul
            # import ipdb;ipdb.set_trace()
            wint = (w/scale_try).round().clamp(-n-1,n)
            wq = wint * scale_try
            z = F.conv2d(
                x, wq, bias, 
                stride, padding, dilation, groups
            )
            loss = ((z-z0)**2).mean()
            if loss<min_loss:
                min_loss = loss
                best_scale = scale_try
            pbar.set_postfix(
                loss=f"{loss.item():.2e}", 
                loss_min=0 if min_loss is None else f"{min_loss:.2e}",
            )
        scale = best_scale
    wint = (w/scale).round().clamp(-n-1,n)
    wq = wint * scale
    return wq
            

def quant_conv_forward(self, x: Tensor):
    # if 1:
    #     z0 = F.conv2d(
    #         x, self.original_weight, self.bias, 
    #         self.stride, self.padding, self.dilation, self.groups
    #     )
    #     wq = conv_weight_search(
    #         x,self.original_weight,self.scale_channel_wise,
    #         self.wbit,z0, self.bias,
    #         self.stride, self.padding, self.dilation, self.groups
    #     )
    #     self.weight.data = wq.data
            
    if self.clip_search:
        z0 = F.conv2d(
            x, self.weight, self.bias, 
            self.stride, self.padding, self.dilation, self.groups
        )
        self.act_scale = conv_percentile_search(
            x, self.weight, self.bias, z0, self.bit, 
            self.stride, self.padding, self.dilation, self.groups, 
            search_space=300
        )
        self.clip_search = False
        xq = quant_activation(x, self.bit, self.act_scale)
        return F.conv2d(
            xq, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )
    else:
        if self.act_scale is None:
            self.act_scale = x.abs().max() / (2**(self.bit-1)-1)
        xq = quant_activation(x, bit=self.bit, act_scale=self.act_scale)
        return F.conv2d(
            xq, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )

def fast_quant(model, bit=8, clip_search=False):
    num_4bit = 0
    name_4bit = []
    for name, module in tqdm(model.named_modules(), desc="Quantize weights"):
        module.own_name = name
        if isinstance(module, nn.Conv2d):
            w = module.weight.data.clone()
            num = w[:,:,0,0].numel()
            module.bit = bit # activation bit width
            if num > 512*512 or name.endswith("model.16.conv"):
                if (
                    name.endswith("model.12.conv") or
                    name.endswith("model.14.conv") or
                    name.endswith("model.16.conv")
                    
                ):
                    wbit = 6
                else:
                    wbit = 4 # weight bit width
                # print(name)
                num_4bit += 1
                name_4bit += [name.replace("model.model","model")]
            else:
                wbit = bit
            module.wbit = wbit
            print("|", name,"|", f"bit=W{wbit}A{module.bit}")
            wq, scale_channel_wise = quant_weight(w, wbit, mode="channel_wise", symmetric=True)
            # import ipdb;ipdb.set_trace()
            module.original_weight = w
            module.scale_channel_wise = scale_channel_wise
            module.weight.data = wq.data
            module.act_scale = None
            module.clip_search = clip_search
            module.forward = MethodType(quant_conv_forward, module)
        # elif isinstance(module, nn.SiLU):
        #     new_module = nn.LeakyReLU(inplace=True)
        #     update_module(model, name, new_module=new_module)
    print(model)
    print("-"*40)
    print(name_4bit)
    print(f"4bit layer numbers: {num_4bit}")
    return model

def update_module(model, module_name, new_module):
    super_module, leaf_module = get_module_by_name(model, module_name)
    setattr(super_module, module_name.split('.')[-1], new_module)
def get_module_by_name(model, module_name):
    name_list = module_name.split(".")
    for name in name_list[:-1]:
        if hasattr(model, name):
            model = getattr(model, name)
        else:
            return None, None
    if hasattr(model, name_list[-1]):
        leaf_module = getattr(model, name_list[-1])
        return model, leaf_module
    else:
        return None, None
