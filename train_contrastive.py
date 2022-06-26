'''
In this version, the data set is changed,
'''
import os
import argparse
import configparser
import numpy as np
from tqdm import tqdm
import random
import h5py
import math

import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils_py.utils_pytorch import reduce_lr, increase_margin

from utils import get_logger_2, check_dir
from models.fusion import System


def save_checkpoint(model, checkpoint_format, epoch):
    filepath = checkpoint_format.format(epoch)
    torch.save(model.state_dict(), filepath)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def write_conf(read_path, store_path):
    '''
    store the config file in exp dir
    :param read_path: the conf path to read
    :param store_path: the conf path to write
    '''
    with open(read_path,'r') as f_r, open(store_path,'w') as f_w:
        lines = f_r.readlines()
        f_w.writelines(lines)


def select_threshold(score, ratio):
    '''
    :param score: The cosine score between vectors, shape:[batch_size]
    :param ratio: The score ratio above the threshold
    :return: The selected threshold
    '''
    score_detach = score.detach()
    length = score.shape[0]

    # # ------------------------ original version ------------------------
    # threshold = -1.0
    # score_descend = torch.sort(score_detach, descending=True)[0]

    # for i in range(length):
    #     if (i+1) / length > ratio:
    #         threshold = score_descend[i]
    #         break
    # index = score_detach > threshold
    # # ------------------------ original version ------------------------

    # --------------------------- new version --------------------------
    score_descend, index = torch.sort(score_detach, descending=True)
    select_len = math.ceil(ratio * length)
    threshold = score_descend[select_len - 1]
    index = index[:select_len]
    # --------------------------- new version --------------------------

    return threshold, index


def loss_calculator(input1, input2, label, ratio):
    pos_input1 = input1[label == 1]
    pos_input2 = input2[label == 1]
    neg_input1 = input1[label == 0]
    neg_input2 = input2[label == 0]

    pos_score = F.cosine_similarity(pos_input1, pos_input2) * (-1.0)
    neg_score = F.cosine_similarity(neg_input1, neg_input2)
    total_loss = torch.mean(neg_score) + torch.mean(pos_score) + 2.0

    pos_threshold, pos_index = select_threshold(pos_score, ratio)
    neg_threshold, neg_index = select_threshold(neg_score, ratio)

    loss = torch.mean(neg_score[neg_index]) + torch.mean(pos_score[pos_index]) + 2.0  # in [0.0, 4.0]

    return loss, total_loss, pos_threshold, neg_threshold


class Dataset_H5(Dataset):
    def __init__(self, face_h5, utt_h5, length=64 * 10000):
        super(Dataset_H5, self).__init__()

        self.length = length

        # get common speaker list
        face_h5_f = h5py.File(face_h5, 'r')
        utt_h5_f = h5py.File(utt_h5, 'r')
        face_spk_list = list(face_h5_f.keys())
        utt_spk_list = list(utt_h5_f.keys())
        face_h5_f.close()
        utt_h5_f.close()
        self.spk_list = list(set(face_spk_list) & set(utt_spk_list))
        self.spk_num = len(self.spk_list)

        self.face_h5 = face_h5
        self.utt_h5 = utt_h5
        self.face_h5_f = None
        self.utt_h5_f = None

    def __getitem__(self, idx):
        if self.face_h5_f is None:
            self.face_h5_f = h5py.File(self.face_h5, 'r')
            self.utt_h5_f = h5py.File(self.utt_h5, 'r')

        choice = random.randint(0, 1)  # to generate positive or negative pair
        if choice == 0:  # negative pair
            spk1, spk2 = random.sample(self.spk_list, 2)
            face_group1 = self.face_h5_f[spk1]
            utt_group1 = self.utt_h5_f[spk1]
            face_group2 = self.face_h5_f[spk2]
            utt_group2 = self.utt_h5_f[spk2]

            face_list1 = list(face_group1.keys())
            utt_list1 = list(utt_group1.keys())
            face_list2 = list(face_group2.keys())
            utt_list2 = list(utt_group2.keys())

            face_emb1 = face_group1[random.sample(face_list1, 1)[0]][()]
            utt_emb1 = utt_group1[random.sample(utt_list1, 1)[0]][()]
            face_emb2 = face_group2[random.sample(face_list2, 1)[0]][()]
            utt_emb2 = utt_group2[random.sample(utt_list2, 1)[0]][()]
        else:  # positive pair
            spk = random.sample(self.spk_list, 1)[0]
            face_group = self.face_h5_f[spk]
            utt_group = self.utt_h5_f[spk]

            face_list = list(face_group.keys())
            utt_list = list(utt_group.keys())

            face_emb1 = face_group[random.sample(face_list, 1)[0]][()]
            utt_emb1 = utt_group[random.sample(utt_list, 1)[0]][()]
            face_emb2 = face_group[random.sample(face_list, 1)[0]][()]
            utt_emb2 = utt_group[random.sample(utt_list, 1)[0]][()]

        label = choice

        return label, face_emb1, utt_emb1, face_emb2, utt_emb2

    def __len__(self):
        return self.length


def train(model, optimizer, logger, data_params, args, scheduler):
    model.train()

    max_iter = args.max_epoch * data_params['h5_file_num']
    for epoch in range(args.max_epoch):
        with tqdm(total=data_params['batch_iters'] * data_params['h5_file_num'], ncols=100) as pbar:
            epoch_total_loss_avg = 0.0
            epoch_loss_avg = 0.0
            epoch_pos_threshold = 0.0
            epoch_neg_threshold = 0.0
            for h5_idx in range(data_params['h5_file_num']):
                h5_idx_str = str(h5_idx).zfill(3)
                face_h5_idx_str = str(random.randint(0, data_params['h5_file_num']-1)).zfill(3)
                face_h5 = os.path.join(data_params['face_h5_dir'],'data_' + face_h5_idx_str + '.h5')
                utt_h5 = os.path.join(data_params['utt_h5_dir'],'data_' + h5_idx_str + '.h5')

                dataset = Dataset_H5(face_h5, utt_h5, length=data_params['batch_size'] * data_params['batch_iters'])
                train_loader = DataLoader(dataset, batch_size=data_params['batch_size'], shuffle=False,
                                          num_workers=data_params['num_workers'])

                h5_total_loss_avg = 0.0
                h5_loss_avg = 0.0
                pos_threshold_avg = 0.0
                neg_threshold_avg = 0.0
                for labels, face_data1, utt_data1, face_data2, utt_data2 in train_loader:
                    optimizer.zero_grad()

                    face_input = torch.cat((face_data1, face_data2), dim=0)
                    utt_input = torch.cat((utt_data1, utt_data2), dim=0)

                    if args.use_gpu:
                        face_input = face_input.cuda()
                        utt_input = utt_input.cuda()
                        labels = labels.cuda()

                    out, _, _ = model(face_input, utt_input)
                    pair_num = out.shape[0] // 2
                    assert pair_num == data_params['batch_size']

                    loss, total_loss, pos_threshold, neg_threshold = loss_calculator(out[0:pair_num], out[pair_num:], labels, args.ratio)
                    pos_threshold = - pos_threshold

                    if args.partial_loss:
                        loss.backward()
                    else:
                        total_loss.backward()
                    optimizer.step()

                    h5_loss_avg += loss.item()
                    h5_total_loss_avg += total_loss.item()
                    pos_threshold_avg += pos_threshold
                    neg_threshold_avg += neg_threshold
                    pbar.update()

                h5_loss_avg /= data_params['batch_iters']
                h5_total_loss_avg /= data_params['batch_iters']
                pos_threshold_avg /= data_params['batch_iters']
                neg_threshold_avg /= data_params['batch_iters']

                current_iter = epoch * data_params['h5_file_num'] + h5_idx
                if args.use_scheduler:
                    if args.scheduler_type == 0:
                        scheduler.step(h5_loss_avg)
                    else:
                        reduce_lr(optimizer, args.lr, args.final_lr, current_iter, max_iter)
                if args.update_ratio:
                    args.ratio = increase_margin(args.initial_ratio, args.final_ratio, current_iter, max_iter)

                epoch_loss_avg += h5_loss_avg
                epoch_total_loss_avg += h5_total_loss_avg
                epoch_pos_threshold += pos_threshold_avg
                epoch_neg_threshold += neg_threshold_avg

                if h5_idx % 10 == 0:
                    lr = get_lr(optimizer)
                    logger.info("Epoch-{} H5-{} Partial_loss:{:.4f} Total_loss: {:.4f} Ratio:{:.3f} Pos_thres:{:.4f} Neg_thres:{:.4f} lr:{}".format(epoch,h5_idx,
                                    h5_loss_avg, h5_total_loss_avg, args.ratio, pos_threshold_avg, neg_threshold_avg, lr))

                if optimizer.param_groups[0]['lr'] < 1e-7:
                    logger.info("Stop training, the learning rate is too small")
                    save_checkpoint(model, args.checkpoint_format, epoch)
                    exit()

            epoch_loss_avg /= data_params['h5_file_num']
            epoch_total_loss_avg /= data_params['h5_file_num']
            epoch_pos_threshold /= data_params['h5_file_num']
            epoch_neg_threshold /= data_params['h5_file_num']

            lr = get_lr(optimizer)
            logger.info("Epoch-{} Partial_loss:{:.4f} Total_loss: {:.4f} Ratio:{:.3f} Pos_thres:{:.4f} Neg_thres:{:.4f} lr:{}".format(epoch,
                    epoch_loss_avg, epoch_total_loss_avg, args.ratio, epoch_pos_threshold, epoch_neg_threshold, lr))
            save_checkpoint(model, args.checkpoint_format, epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='conf/train.conf', type=str,
                        help='the config file')
    parser.add_argument('--pretrained', default=None, type=str,
                        help='the pretrained model')
    args=parser.parse_args()

    # *********************** process config ***********************
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.optionxform=str
    config.read(args.config)

    exp_section = config['exp']
    model_section = config['model']
    loss_section = config['loss']
    data_section = config['data']
    optim_section = config['optimization']

    exp_params = {k: eval(v) for k, v in exp_section.items()}
    model_params = {k: eval(v) for k, v in model_section.items()}
    loss_params = {k: eval(v) for k, v in loss_section.items()}
    data_params = {k: eval(v) for k, v in data_section.items()}
    optim_params = {k: eval(v) for k, v in optim_section.items()}
    # *********************** process config ***********************

    # setup logger
    check_dir(exp_params['exp_dir'])
    logger = get_logger_2(os.path.join(exp_params['exp_dir'],'train.log'))

    # write config file to expdir
    store_path = os.path.join(exp_params['exp_dir'], 'train.conf')
    write_conf(args.config, store_path)

    # model init
    args.use_gpu = optim_params['use_gpu'] and torch.cuda.is_available()
    model = System(**model_params)

    # load pretrained model
    if args.pretrained is not None:
        logger.info('Load pretrained model from {}'.format(args.pretrained))
        state_dict = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)

    if args.use_gpu:
        model = model.cuda()

    if optim_params['type'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=optim_params['lr'], momentum=optim_params['momentum'],
                              weight_decay=optim_params['weight_decay'], nesterov=True)
    else:
        optimizer = optim.Adam(model.parameters(), lr=optim_params['lr'],  betas=(0.9, 0.999),
                              weight_decay=optim_params['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=optim_params['factor'],
                                  patience=optim_params['patience'])

    checkpoint_dir = os.path.join(exp_params['exp_dir'],'models')
    check_dir(checkpoint_dir)
    checkpoint_format = os.path.join(checkpoint_dir,'epoch-{}.th')

    args.max_epoch = optim_params['max_epoch']
    args.checkpoint_format = checkpoint_format
    args.use_scheduler = optim_params['use_scheduler']
    args.scheduler_type = optim_params['scheduler_type']
    args.lr = optim_params['lr']
    args.final_lr = optim_params['final_lr']
    args.ratio = optim_params['ratio']
    args.initial_ratio = args.ratio
    args.final_ratio = optim_params['final_ratio']
    args.update_ratio = optim_params['update_ratio']
    args.partial_loss = loss_params['partial_loss']

    num_params = sum(param.numel() for param in model.parameters())
    logger.info('Number of parmeters:{}'.format(num_params))

    train(model, optimizer, logger, data_params, args, scheduler)


if __name__ == '__main__':
    main()
