import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
from datetime import datetime

import numpy as np
'''
def quantize(number,bitwidth):
    temp=1/bitwidth
    if number>0:
        for i in range(1,bitwidth):
            if number<=temp*i:
                return 2*i-1
        return 2*bitwidth-1
    else:
        for i in range(1,bitwidth):
            if number>=-temp*i:
                return -(2*i-1)
        return -(2*bitwidth-1)
'''


def Binarize(tensor,quant_mode='det',bitwidth=1):
    if quant_mode == 'weight':
        #temp = torch.floor(tensor.div_(2)).mul_(2).add_(1).mul_(tensor).div_(tensor)
        temp = torch.floor(tensor.mul_(2**bitwidth).div_(2)).mul_(2).add_(1).mul_(tensor).div_(tensor).div_(2**bitwidth)
        temp[temp!=temp]=0
        return temp
    elif quant_mode == 'input':
        #return torch.round(tensor.mul_(45))#tensor.mul_(45).div_(128)
        return torch.clamp(tensor.mul_(47).div_(128),min=-0.99,max=0.99)
    elif quant_mode=='multi':
        #tensor_clone = tensor.clone()
        #return tensor.sign()
        #temp = torch.floor(tensor.div_(2)).mul_(2).add_(1).mul_(tensor).div_(tensor)
        tensor = torch.clamp(tensor, -1+1e-4, 1-1e-4)
        temp = torch.floor(tensor.mul_(2**bitwidth).div_(2)).mul_(2).add_(1).mul_(tensor).div_(tensor).div_(2**bitwidth)
        #temp = torch.floor(tensor_clone.mul_(2**bitwidth).div_(2)).mul_(2).add_(1).mul_(tensor_clone).div_(tensor_clone).div_(2**bitwidth)
        temp[temp!=temp]=0
        temp[temp==0] = 1/(torch.pow(torch.tensor(2.), bitwidth))
        return temp
    elif quant_mode=='riptide':
        if bitwidth==1:
            return tensor.sign()
        temp = torch.floor(tensor.mul_(2**bitwidth-1)).div_(2**bitwidth)
    elif quant_mode=='det':
        return tensor.sign()
    elif quant_mode=='bin':
        return (tensor>=0).type(type(tensor))*2-1
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

def Ternarize(tensor, mult = 0.7, mask = None, permute_list = None, pruned = False, align = False, pack = 32):
    if type(mask) == type(None):
        mask = torch.ones_like(tensor)
    
    # Fix permutation. Tensor needs to be permuted
    if not pruned:
        tensor_masked = utils_own.permute_from_list(tensor, permute_list)
        if len(tensor_masked.size())==4:
            tensor_masked = tensor_masked.permute(0,2,3,1)
       
        if not align:
            tensor_flat = torch.abs(tensor_masked.contiguous().view(-1)).contiguous()
            tensor_split = torch.split(tensor_flat, pack, dim=0)
            tensor_split = torch.stack(tensor_split, dim=0)
            tensor_sum = torch.sum(tensor_split, dim=1)
            tensor_size = tensor_sum.size(0)
            tensor_sorted, _ = torch.sort(tensor_sum)
            thres = tensor_sorted[int(mult*tensor_size)]
            tensor_flag = torch.ones_like(tensor_sum)
            tensor_flag[tensor_sum.ge(-thres) * tensor_sum.le(thres)] = 0
            tensor_flag = tensor_flag.repeat(pack).reshape(pack,-1).transpose(1,0).reshape_as(tensor_masked)
            
        else:
            tensor_flat = torch.abs(tensor_masked.reshape(tensor_masked.size(0),-1)).contiguous()
            tensor_split = torch.split(tensor_flat, pack, dim=1)
            tensor_split = torch.stack(tensor_split, dim=1)
            tensor_sum = torch.sum(tensor_split, dim=2)
            tensor_size = tensor_sum.size(1)
            tensor_sorted, _ = torch.sort(tensor_sum, dim=1)
            tensor_sorted = torch.flip(tensor_sorted, [1])
            multiplier = 32./pack
            index = int(torch.ceil((1-mult)*tensor_size/multiplier)*multiplier)
            thres = tensor_sorted[:, index-1].view(-1,1)
            tensor_flag = torch.zeros_like(tensor_sum)
            tensor_flag[tensor_sum.ge(thres)] = 1
            tensor_flag[tensor_sum.le(-thres)] = 1
            tensor_flag = tensor_flag.repeat(1,pack).reshape(tensor_flag.size(0),pack,-1).transpose(2,1).reshape_as(tensor_masked)

        if len(tensor_masked.size())==4:
            tensor_flag = tensor_flag.permute(0,3,1,2)            
        tensor_flag = utils_own.permute_from_list(tensor_flag, permute_list, transpose=True)
        tensor_bin = tensor.sign() * tensor_flag
            
    else:
        tensor_bin = tensor.sign() * mask
        
    return tensor_bin

class BinarizeLinear(nn.Linear):

    def __init__(self, input_bit=1, output_bit=1, weight_bit=1, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
        self.input_bit=input_bit
        self.output_bit=output_bit
        self.weight_bit = weight_bit

    def forward(self, input):
        #commented out below condition since multi binarization produces wrong result
        input.data=Binarize(input.data,quant_mode='multi',bitwidth=self.input_bit)
        weight_org_clone = self.weight_org.clone()
        weight_data=Binarize(self.weight_org, quant_mode="multi", bitwidth=self.weight_bit)
        self.weight.data=torch.clamp(weight_data,min=-0.99,max=0.99)
        self.weight_org = weight_org_clone #weight_org modified by Binarize function, we want it to stay the same
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BWLinear(nn.Linear):

    def __init__(self, weight_bit=1, *kargs, **kwargs):
        super(BWLinear, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
        self.weight_bit = weight_bit

    def forward(self, input):
        weight_org_clone = self.weight_org.clone()
        weight_data=Binarize(self.weight_org, quant_mode="multi", bitwidth=self.weight_bit)
        self.weight.data=torch.clamp(weight_data,min=-0.99,max=0.99)
        self.weight_org = weight_org_clone #weight_org modified by Binarize function, we want it to stay the same
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out
    
class TernarizeLinear(nn.Linear):

    def __init__(self, thres, input_bit=1, output_bit=1, *kargs, **kwargs):
        try:
            pack = kwargs['pack']
        except:
            pack = 32
        else:
            del(kwargs['pack'])
        try:
            permute = kwargs['permute']
        except:
            permute = 1
        else:
            del(kwargs['permute'])
        try:
            self.align=kwargs['align']
        except:
            self.align=True
        else:
            del(kwargs['align'])
        super(TernarizeLinear, self).__init__(*kargs, **kwargs)
        self.input_bit = input_bit
        self.output_bit = output_bit
        permute = min(permute, self.weight.size(0))
        self.register_buffer('pack', torch.LongTensor([pack]))
        self.register_buffer('thres', torch.FloatTensor([thres]))
        self.register_buffer('mask', torch.ones_like(self.weight.data))
        self.register_buffer('permute_list', torch.LongTensor(np.tile(range(self.weight.size(1)), (permute,1))))
        self.register_buffer('weight_org', self.weight.data.clone())

    def forward(self, input, pruned=False):
        if (input.size(1) != 768) and (input.size(1) != 3072): # 784->768
            input.data=Binarize(input.data,quant_mode='multi',bitwidth=self.input_bit)
        else:
            input.data = Binarize(input.data, quant_mode='det')
        self.weight.data=Ternarize(self.weight_org, self.thres, self.mask, self.permute_list, pruned, align=self.align, pack=self.pack.item())
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)
        return out
class CustomLeakyRelu(torch.autograd.Function):
    
    @staticmethod
    def forward(self,input,slope,output_bitwidth):
        self.neg = input < 0
        self.slope=slope
        prev=0
        for i in range(1,2**(output_bitwidth-1)):
            mask=torch.logical_and(input<prev,input>-i/(2**(output_bitwidth-1))/slope)
            input[mask]=-i/(2**(output_bitwidth))
            prev=-i/(2**(output_bitwidth-1))/slope
        mask=input<prev
        input[mask]=-1+1/(2**output_bitwidth)
        return input
    
    @staticmethod
    def backward(self,grad_output):
        grad_input = grad_output.clone()
        grad_input[self.neg] *= self.slope
        return grad_input,None,None
#REF:https://github.com/cornell-zhang/dnn-gating/blob/31666fadf35789b433c79eec8669a3a2df818bd4/utils/pg_utils.py
class Floor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # ctx.save_for_backward(input)
        return torch.floor(input)
    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        The backward behavior of the floor function is defined as the identity function.
        """
        # input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input
class GreaterThan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        return torch.Tensor.float(torch.gt(input, threshold))
    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        The backward behavior of the floor function is defined as the identity function.
        """
        grad_input = grad_output.clone()
        return grad_input, None
class TorchTruncate(nn.Module):
    """ 
    Quantize an input tensor to a b-bit fixed-point representation, and
    remain the bh most-significant bits.
        Args:
        input: Input tensor
        b:  Number of bits in the fixed-point
        bh: Number of most-significant bits remained
    """
    def __init__(self, input_bit=8, out_msb=4):
        super(TorchTruncate, self).__init__()
        self.input_bit = input_bit
        self.out_msb = out_msb

    def forward(self, input):
        #print(input)
        """ extract the sign of each element """
        sign = torch.sign(input).detach()
        """ get the mantessa bits """
        input = torch.abs(input)
        #scaling = (2.0**(self.input_bit)-1.0)/(2.0**self.input_bit) + self.epsilon
        #input = torch.clamp( input/scaling ,0.0, 1.0 )
        """ round the mantessa bits to the required precision """
        input = Floor.apply( input * (2.0**self.out_msb) )
        """ truncate the mantessa bits """
        input = Floor.apply( input/2.0 ).mul_(2).add_(1)
        """ rescale """
        input /= (2.0**self.out_msb)
        #print(input*sign)
        return input * sign
def force_pack(tensor,bitwidth):
    s = tensor.shape
    for i in range(int(s[1]/8)):
        tensor_temp = torch.narrow(tensor,1,i*8,8)
        tensor_mean = torch.mean(tensor_temp,1,keepdim=True)
        #temp=torch.floor(tensor_mean.mul_(2**bitwidth).div_(2)).mul_(2).add_(1).mul_(tensor_mean).div_(tensor_mean).div_(2**bitwidth)
        #temp[temp!=temp]=0
        tensor_mean.expand(s[0],8,s[2],s[3])
        tensor[:,i*8:(i+1)*8,:,:] = tensor_mean
    return tensor
class BinarizeConv2d(nn.Conv2d):

    def __init__(self, input_bit=1, output_bit=1, weight_bit=1, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
        self.input_bit = input_bit
        self.output_bit = output_bit
        self.weight_bit = weight_bit
        #self.exp=True

    def forward(self, input):
        mask = [None]*self.input_bit
        input_bin = Binarize(input.data,quant_mode='multi',bitwidth=self.input_bit)
        #input.data = force_pack(input.data,self.input_bit)
        self.weight.data=torch.clamp(Binarize(self.weight_org.clone(), quant_mode="multi", bitwidth=self.weight_bit), min=-0.99, max=0.99)
        '''
        out_msb = nn.functional.conv2d(self.trunc[0](input),
                        self.weight, None, self.stride, self.padding, 
                        self.dilation, self.groups)
        """ Calculate the mask """
        mask[0] = self.gt(torch.sigmoid((self.threshold[0] - out_msb)), 0.5)
            
        """ combine outputs """
        out = out_msb  # + mask * out_lsb
        for i in range(self.input_bit - 1):
            out_msb = nn.functional.conv2d(self.trunc[i+1](input) - self.trunc[i](input),
                                            self.weight, None, self.stride, self.padding,
                                            self.dilation, self.groups)
            out += mask[i] * out_msb
            """ Calculate the mask """
            mask[i+1] = self.gt(torch.sigmoid((self.threshold[i + 1] - out)), 0.5)*mask[i]
            """ perform LSB convolution """
        '''
        out = nn.functional.conv2d(input_bin, self.weight, None, self.stride,self.padding, self.dilation, self.groups)
        #self.exp=True
        #if self.exp:
        #    now=datetime.now().time()
        #    temp=input.detach().numpy()
        #    with open ("./"+str(now.minute)+str(now.second)+str(now.microsecond)+".npy","wb") as f:
        #        np.save(f,temp)


        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
        #if self.exp:
        #    now=datetime.now().time()
        #    temp=out.detach().numpy()
        #    with open("./" + str(now.minute) + str(now.second) + str(now.microsecond) + ".npy", "wb") as f:
        #        np.save(f, temp)

        return out
class BWConv2d(nn.Conv2d):
    def __init__(self, weight_bit=1, *kargs, **kwargs):
        super(BWConv2d, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())
        self.weight_bit = weight_bit

    def forward(self, input):
        self.weight.data=torch.clamp(Binarize(self.weight_org.clone(), quant_mode="multi", bitwidth=self.weight_bit), min=-0.99, max=0.99)
        out = nn.functional.conv2d(input, self.weight, None, self.stride,self.padding, self.dilation, self.groups)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
        
        return out
class Bias(nn.Module):
    def __init__(self, dim):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        if len(x.shape)==2:
            x = x + self.bias.view(1,-1)
        else:
            x = x + self.bias.view(1,-1,1,1)#.expand_as(x)
        return x
class TernarizeConv2d(nn.Conv2d):

    def __init__(self, thres, input_bit=1, output_bit=1,*kargs, **kwargs):
        try:
            pack = kwargs['pack']
        except:
            pack = 32
        else:
            del(kwargs['pack'])
        try:
            permute = kwargs['permute']
        except:
            permute = 1
        else:
            del(kwargs['permute'])
        try:
            self.align=kwargs['align']
        except:
            self.align=True
        else:
            del(kwargs['align'])
        self.input_bit = input_bit
        self.output_bit = output_bit
        super(TernarizeConv2d, self).__init__(*kargs, **kwargs)
        permute = min(permute, self.weight.size(0))
        self.register_buffer('pack', torch.LongTensor([pack]))
        self.register_buffer('thres', torch.FloatTensor([thres]))
        self.register_buffer('mask', torch.ones_like(self.weight.data))
        self.register_buffer('permute_list', torch.LongTensor(np.tile(range(self.weight.size(1)), (permute,1))))
        self.register_buffer('weight_org', self.weight.data.clone())

    def forward(self, input, pruned=False):
        if input.size(1) != 3 and input.size(1) != 1:
            input.data = Binarize(input.data,quant_mode='multi',bitwidth=self.input_bit)
        else:
            input.data = Binarize(input.data, quant_mode='input', bitwidth=self.input_bit)
            input.data = Binarize(input.data,quant_mode='multi',bitwidth=self.input_bit)
        self.weight.data=Ternarize(self.weight_org, self.thres, self.mask, self.permute_list, pruned, align=self.align, pack=self.pack.item())
        #self.exp=True
        #if self.exp:
        #   now=datetime.now().time()
        #   temp=input.detach().numpy()
        #   with open ("./"+str(now.minute)+str(now.second)+str(now.microsecond)+".npy","wb") as f:
        #       np.save(f,temp)
        out = nn.functional.conv2d(input, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
        #if self.exp:
        #    now=datetime.now().time()
        #    temp=out.detach().numpy()
        #    with open("./" + str(now.minute) + str(now.second) + str(now.microsecond) + ".npy", "wb") as f:
        #        np.save(f, temp)
        return out

def quantize(input, mask=None, quant=False, pruned=False, mult=0, bitwidth=8):
    if pruned:
        input = input * mask
    if quant:
        input = (input * np.power(2, bitwidth-1)).floor()/(np.power(2, bitwidth-1))
    if mult>0:
        input_flat = torch.abs(input.reshape(-1))
        input_size = input_flat.size(0)
        input_sorted, _ = torch.sort(input_flat)
        thres = input_sorted[int(mult*input_size)]
        input_flag = torch.ones_like(input_flat)
        input_flag[input_flat.ge(-thres) * input_flat.le(thres)] = 0
        mask = input_flag.reshape_as(input)
        input = input * mask
        return input, mask
    else:
        return input, torch.ones_like(input)
    
class QuantizeConv2d(nn.Conv2d):
    '''
    Quantized conv2d with mask for pruning
    '''
    def __init__(self, *kargs, bitwidth=8, weight_bitwidth=8, **kwargs):
        super(QuantizeConv2d, self).__init__(*kargs, **kwargs)
        self.bitwidth=bitwidth
        self.weight_bitwidth = weight_bitwidth
        #print("QuantizeConv2d initialized with bitwidth", self.bitwidth)
        self.thres = 0
        self.register_buffer('mask', torch.ones_like(self.weight.data))
        self.register_buffer('weight_org', self.weight.data.clone())
    
    def forward(self, input, quant=True, pruned=False):
        # If mult exists, overwrites pruning
        input.data, _ = quantize(input.data, quant=quant, bitwidth=self.bitwidth)
        self.weight.data, self.mask=quantize(self.weight_org, mask=self.mask, quant=quant, pruned=pruned, mult=self.thres, bitwidth=self.weight_bitwidth)
        out = F.conv2d(input, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
        return out
    
class QuantizeLinear(nn.Linear):
    '''
    Quantized Linear with mask for pruning
    '''
    def __init__(self, *kargs, bitwidth=8, weight_bitwidth=8, **kwargs):
        super(QuantizeLinear, self).__init__(*kargs, **kwargs)
        self.bitwidth=bitwidth
        self.weight_bitwidth = weight_bitwidth
        #print("QuantizeLinear initialized with bitwidth", self.bitwidth)
        self.thres = 0
        self.register_buffer('mask', torch.ones_like(self.weight.data))
        self.register_buffer('weight_org', self.weight.data.clone())
        
    def forward(self, input, quant=True, pruned=False, mult=None):
        input.data, _ = quantize(input.data, quant=quant, bitwidth=self.bitwidth)
        self.weight.data, self.mask=quantize(self.weight_org, mask=self.mask, quant=quant, pruned=pruned, mult=self.thres, bitwidth=self.weight_bitwidth)
        out = F.linear(input, self.weight)
        return out

class ClampFloatConv2d(nn.Conv2d):
    '''
    Full precision Conv2d with weights clamped between -1 and 1
    '''
    def __init__(self, *kargs, **kwargs):
        super(ClampFloatConv2d, self).__init__(*kargs, **kwargs)
    
    def forward(self, input, quant=True, pruned=False):
        self.weight.data=torch.clamp(self.weight.data, -0.99, 0.99)
        out = F.conv2d(input, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
        return out

class ClampFloatLinear(nn.Linear):
    '''
    Full precision Linear with weights clamped between -1 and 1
    '''
    def __init__(self, *kargs, **kwargs):
        super(ClampFloatLinear, self).__init__(*kargs, **kwargs)
        
    def forward(self, input, quant=True, pruned=False, mult=None):
        self.weight.data=torch.clamp(self.weight.data, -0.99, 0.99)
        out = F.linear(input, self.weight)
        return out

def get_activation(activation_type, input_shape=(1,1,1,1), leaky_relu_slope=0.1):
    if activation_type=="htanh":
        return nn.Hardtanh()
    elif activation_type=="leaky_relu":
        return nn.LeakyReLU(negative_slope=leaky_relu_slope)
    elif activation_type=="2bn":
        return nn.Sequential(
            nn.BatchNorm2d(input_shape[1]),
            nn.ReLU(),
            nn.BatchNorm2d(input_shape[1])
        )
    elif activation_type=="relu_bias":
        return nn.Sequential(
            nn.ReLU(),
            Bias(input_shape[1])
        )
    elif activation_type=="relu":
        return nn.ReLU()

def get_all_modules(model):
    all_modules = []
    for p in model.modules():
        all_modules.append(p)
    return all_modules

def copy_org_to_data(model):
    all_modules = get_all_modules(model)
    for p in all_modules:
        if hasattr(p, 'weight_org'):
            p.weight_org.copy_(p.weight.data.clamp_(-1,1))

def copy_data_to_org(model):
    all_modules = get_all_modules(model)
    for p in all_modules:
        if hasattr(p, 'weight_org'):
            p.weight.data.copy_(p.weight_org)