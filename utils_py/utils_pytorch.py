import os
import torch
import numpy as np
import random
import torch.distributed as dist
import math

# ********************** Pytorch Distributed Training ******************
def getoneNode():
    nodelist = os.environ['SLURM_JOB_NODELIST']
    nodelist = nodelist.strip().split(',')[0]
    import re
    text = re.split('[-\[\]]',nodelist)
    if ('' in text):
        text.remove('')
    return text[0] + '-' + text[1] + '-' + text[2]


def dist_init(host_addr, rank, local_rank, world_size, port=23456):
    host_addr_full = 'tcp://' + host_addr + ':' + str(port)
    dist.init_process_group("nccl", init_method=host_addr_full,
                                         rank=rank, world_size=world_size)
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    assert dist.is_initialized()


def average_gradients(model, world_size):
    size = float(world_size)
    for param in model.parameters():
        if(param.requires_grad and param.grad is not None):
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size

# ********************** Pytorch Distributed Training ******************


def reduce_lr(optimizer, initial_lr, final_lr, current_iter, max_iter, coeff=1.0):
    current_lr = coeff * math.exp((current_iter / max_iter) * math.log(final_lr / initial_lr)) * initial_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr


def get_reduce_lr(initial_lr, final_lr, current_iter, max_iter):
    current_lr = math.exp((current_iter / max_iter) * math.log(final_lr / initial_lr)) * initial_lr
    return current_lr


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def increase_margin(initial_margin, final_margin, current_iter, max_iter):
    initial_val = 1.0
    final_val = 0.02

    ratio = 1.0 - math.exp((current_iter / max_iter) * math.log(final_val / (initial_val+1e-6))) * initial_val
    return initial_margin + (final_margin - initial_margin) * ratio


def merge_image_uttfeat(image, feat):
    '''
    :param image: numpy array, 3 x H x W
    :param feat: numpy array, frame_num x feat_dim
    :return:
    '''

    _, h, w = image.shape
    frame_num, feat_dim = feat.shape

    h_max = max(h, frame_num)
    w_max = max(w, feat_dim)

    image_h_left = (h_max - h) // 2
    image_h_right = h_max - h - image_h_left
    image_w_left = (w_max - w) // 2
    image_w_right = w_max - w - image_w_left

    feat_h_left = (h_max - frame_num) // 2
    feat_h_right = h_max - frame_num - feat_h_left
    feat_w_left = (w_max - feat_dim) // 2
    feat_w_right = w_max - feat_dim - feat_w_left

    image = np.pad(image, ((0, 0), (image_h_left, image_h_right), (image_w_left, image_w_right)), 'constant')
    feat = np.pad(feat, ((feat_h_left, feat_h_right), (feat_w_left, feat_w_right)), 'constant')
    feat = feat.reshape((1, h_max, w_max))

    stack_res = np.concatenate((image, feat))
    return stack_res


def set_seed(seed=66):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def load_torchvision_resnet(resnet_selfdefine, torchresnet_statedict):
    '''
    The resnet in torchvision's module has different name with our
    self-defined resnet. So it will be a little complex to load 
    parameters from it.
    '''
    torchresnet_keys = list(torchresnet_statedict.keys())

    state_dict = resnet_selfdefine.state_dict()
    keys_list = list(state_dict.keys())

    # Because all the weight and bias parameter are in same order
    # and all the batchnorm parameter are in same order
    # we restore them respectively.
    weight_bias_keys = []
    batchnorm_keys = []
    weight_bias_dict = {}
    batchnorm_dict = {}

    for key in keys_list:
        if 'weight' in key or 'bias' in key:
            weight_bias_keys.append(key)
        elif 'running' in key or 'batches_tracked' in key:
            batchnorm_keys.append(key)
    

    weight_key_count = 0
    batchnorm_key_count = 0
    for key in torchresnet_keys:
        if 'weight' in key or 'bias' in key:
            if weight_key_count >= len(weight_bias_keys):
                continue
            weight_bias_dict[weight_bias_keys[weight_key_count]] = torchresnet_statedict[key]
            weight_key_count += 1
        elif 'running' in key or 'batches_tracked' in key:
            if batchnorm_key_count >= len(batchnorm_keys):
                continue
            batchnorm_dict[batchnorm_keys[batchnorm_key_count]] = torchresnet_statedict[key]
            batchnorm_key_count += 1
    
    state_dict.update(batchnorm_dict)
    state_dict.update(weight_bias_dict)
    resnet_selfdefine.load_state_dict(state_dict)
    return resnet_selfdefine


if __name__ == '__main__':
    image = np.random.randn(3, 112, 96)
    feat = np.random.randn(400, 40)
    data = merge_image_uttfeat(image, feat)
    print(data.shape)