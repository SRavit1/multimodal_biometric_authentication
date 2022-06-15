'''
The combination of contrastive_2 and classification
'''
import os
import argparse
import configparser
from tqdm import tqdm
import random
import h5py
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils_py.utils_pytorch import reduce_lr, increase_margin
from utils_py.utils_common import add_noise_to_embedding, add_noise_to_embedding_v2, load_mean_vec, spk2id, param2args
from models.projections import ArcMarginProduct
from train_classify import stats

from utils import get_logger_2, check_dir
from models.fusion import System


def save_checkpoint(state_dict, checkpoint_format, epoch):
    filepath = checkpoint_format.format(epoch)
    torch.save(state_dict, filepath)


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
    select_len = int(ratio * length)
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
    def __init__(self, face_h5, utt_h5, spk2id_dict, length=64 * 10000, pollute_prob=None,
                 face_mean_vec=None, utt_mean_vec=None, noise_dir=None):
        super(Dataset_H5, self).__init__()

        self.length = length
        self.spk2id_dict = spk2id_dict
        self.pollute_prob = pollute_prob  # the probability add noise to one modality
        self.noise_dir = noise_dir
        if self.noise_dir is not None:
            self.get_noise()

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

        if face_mean_vec is not None:
            self.face_mean_vec = load_mean_vec(face_mean_vec)
            self.utt_mean_vec = load_mean_vec(utt_mean_vec)
        else:
            self.face_mean_vec = None
            self.utt_mean_vec = None

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

            label1 = self.spk2id_dict[spk1]
            label2 = self.spk2id_dict[spk2]
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

            label1 = self.spk2id_dict[spk]
            label2 = self.spk2id_dict[spk]

        if self.face_mean_vec is not None:
            face_emb1 = face_emb1 - self.face_mean_vec
            face_emb2 = face_emb2 - self.face_mean_vec
            utt_emb1 = utt_emb1 - self.utt_mean_vec
            utt_emb2 = utt_emb2 - self.utt_mean_vec

        if self.pollute_prob is not None:
            choice = random.randint(0, 1)
            if choice == 0:
                if self.noise_dir is None:
                    face_emb1 = add_noise_to_embedding(face_emb1, self.pollute_prob)
                else:
                    face_emb1 = add_noise_to_embedding_v2(face_emb1, self.face_mean_list, self.face_std_list,
                                                          self.pollute_prob)
            else:
                if self.noise_dir is None:
                    utt_emb1 = add_noise_to_embedding(utt_emb1, self.pollute_prob)
                else:
                    utt_emb1 = add_noise_to_embedding_v2(utt_emb1, self.utt_mean_list, self.utt_std_list,
                                                         self.pollute_prob)
            choice = random.randint(0, 1)
            if choice == 0:
                if self.noise_dir is None:
                    face_emb2 = add_noise_to_embedding(face_emb2, self.pollute_prob)
                else:
                    face_emb2 = add_noise_to_embedding_v2(face_emb2, self.face_mean_list, self.face_std_list,
                                                          self.pollute_prob)
            else:
                if self.noise_dir is None:
                    utt_emb2 = add_noise_to_embedding(utt_emb2, self.pollute_prob)
                else:
                    utt_emb2 = add_noise_to_embedding_v2(utt_emb2, self.utt_mean_list, self.utt_std_list,
                                                         self.pollute_prob)

        return label1, label2, face_emb1.astype(np.dtype('float32')), utt_emb1.astype(np.dtype('float32')), face_emb2.astype(np.dtype('float32')), utt_emb2.astype(np.dtype('float32'))

    def __len__(self):
        return self.length

    def get_noise(self):
        face_noise_type = ['gaussian', 'motion_vertical', 'motion_horizontal']
        utt_noise_type = ['babble', 'music', 'noise']

        self.face_mean_list = []
        self.face_std_list = []
        for val in face_noise_type:
            mean_path = os.path.join(self.noise_dir, '{}_mean.npy'.format(val))
            std_path = os.path.join(self.noise_dir, '{}_std.npy'.format(val))
            self.face_mean_list.append(np.load(mean_path))
            self.face_std_list.append(np.load(std_path))
        self.face_mean_list.append(0.0)  # for zero embedding
        self.face_std_list.append(0.0)

        self.utt_mean_list = []
        self.utt_std_list = []
        for val in utt_noise_type:
            mean_path = os.path.join(self.noise_dir, '{}_mean.npy'.format(val))
            std_path = os.path.join(self.noise_dir, '{}_std.npy'.format(val))
            self.utt_mean_list.append(np.load(mean_path))
            self.utt_std_list.append(np.load(std_path))
        self.utt_mean_list.append(0.0)  # for zero embedding
        self.utt_std_list.append(0.0)


def train(model, projection, optimizer, logger, args, scheduler, spk2id_dict):
    model.train()

    max_iter = args.max_epoch * args.h5_file_num
    for epoch in range(args.max_epoch):

        # tqdm bar initial
        tbar = tqdm(total=args.batch_iters * args.h5_file_num, ncols=100)

        epoch_total_loss_avg = 0.0
        epoch_loss_avg = 0.0
        epoch_pos_threshold = 0.0
        epoch_neg_threshold = 0.0
        epoch_acc_avg = 0.0
        epoch_cla_loss_avg = 0.0
        for h5_idx in range(args.h5_file_num):
            h5_idx_str = str(h5_idx).zfill(3)
            face_h5_idx_str = str(random.randint(0, args.h5_file_num-1)).zfill(3)
            face_h5 = os.path.join(args.face_h5_dir, 'data_' + face_h5_idx_str + '.h5')
            utt_h5 = os.path.join(args.utt_h5_dir, 'data_' + h5_idx_str + '.h5')

            dataset = Dataset_H5(face_h5, utt_h5, spk2id_dict,
                                 length=args.batch_size * args.batch_iters,
                                 pollute_prob=args.pollute_prob,
                                 face_mean_vec=args.face_mean_vec, utt_mean_vec=args.utt_mean_vec)
            train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers)

            h5_total_loss_avg = 0.0  # total contrastive loss
            h5_loss_avg = 0.0  # partial contrastive loss
            pos_threshold_avg = 0.0
            neg_threshold_avg = 0.0

            h5_cla_loss_avg = 0.0
            items_h5_count = 0
            matches_h5_count = 0
            for labels1, labels2, face_data1, utt_data1, face_data2, utt_data2 in train_loader:
                optimizer.zero_grad()

                face_input = torch.cat((face_data1, face_data2), dim=0)
                utt_input = torch.cat((utt_data1, utt_data2), dim=0)
                labels = torch.cat((labels1, labels2), dim=0)

                if args.use_gpu:
                    face_input = face_input.cuda()
                    utt_input = utt_input.cuda()
                    labels = labels.cuda()

                out, face_trans, utt_trans = model(face_input, utt_input)

                # -------------- projection part --------------
                if args.project_type == 'linear':
                    project_out = projection(out)
                else:
                    project_out = projection(out, labels)

                classify_loss = F.cross_entropy(project_out, labels, reduction='mean')

                num_matches, num_items = stats(project_out, labels)
                items_h5_count += num_items
                matches_h5_count += num_matches
                # -------------- projection part --------------

                # -------------- contrastive part --------------
                pair_num = out.shape[0] // 2
                assert pair_num == args.batch_size

                contras_label = (labels[0:pair_num] == labels[pair_num:]).long()
                loss, total_loss, pos_threshold, neg_threshold = loss_calculator(out[0:pair_num], out[pair_num:], contras_label, args.ratio)
                pos_threshold = - pos_threshold
                # -------------- contrastive part --------------

                if args.partial_loss:
                    loss_back = args.loss_ratio * loss + classify_loss
                else:
                    loss_back = args.loss_ratio * total_loss + classify_loss
                loss_back.backward()
                optimizer.step()

                h5_loss_avg += loss.item()
                h5_total_loss_avg += total_loss.item()
                pos_threshold_avg += pos_threshold
                neg_threshold_avg += neg_threshold
                h5_cla_loss_avg += classify_loss.item()

                tbar.update()

            h5_loss_avg /= args.batch_iters
            h5_total_loss_avg /= args.batch_iters
            pos_threshold_avg /= args.batch_iters
            neg_threshold_avg /= args.batch_iters
            h5_acc_avg = 1.0 * matches_h5_count / items_h5_count
            h5_cla_loss_avg /= args.batch_iters

            current_iter = epoch * args.h5_file_num + h5_idx
            if args.use_scheduler:
                if args.scheduler_type == 0:
                    scheduler.step(h5_loss_avg)
                else:
                    reduce_lr(optimizer, args.lr, args.final_lr, current_iter, max_iter)
            if args.update_ratio:
                args.ratio = increase_margin(args.initial_ratio, args.final_ratio, current_iter, max_iter)
            if args.update_margin:
                current_margin = increase_margin(args.margin, args.final_margin, current_iter, max_iter)
                projection.update(margin=current_margin)

            epoch_loss_avg += h5_loss_avg
            epoch_total_loss_avg += h5_total_loss_avg
            epoch_pos_threshold += pos_threshold_avg
            epoch_neg_threshold += neg_threshold_avg
            epoch_acc_avg += h5_acc_avg
            epoch_cla_loss_avg += h5_cla_loss_avg

            if h5_idx % 10 == 0:
                lr = get_lr(optimizer)
                logger.info(
                    "Epoch:{:02d} H5:{:03d} Con_Partial_Loss:{:.4f} Con_Total_Loss:{:.4f} Class_Loss:{:.4f} Acc:{:.4f}".format(
                        epoch, h5_idx,
                        h5_loss_avg, h5_total_loss_avg, h5_cla_loss_avg, h5_acc_avg))
                logger.info(
                    "Epoch:{:02d} H5:{:03d} Margin:{:.4f} Ratio:{:.3f} Pos_thres:{:.4f} Neg_thres:{:.4f} lr:{}".format(
                        epoch, h5_idx,
                        projection.margin, args.ratio, pos_threshold_avg, neg_threshold_avg, lr))
            if optimizer.param_groups[0]['lr'] < 1e-7:
                logger.info("Stop training, the learning rate is too small")
                state_dict = {
                    'model': model.state_dict(),
                    'projection': projection.state_dict()
                }
                save_checkpoint(state_dict, args.checkpoint_format, epoch)
                exit()

        epoch_loss_avg /= args.h5_file_num
        epoch_total_loss_avg /= args.h5_file_num
        epoch_pos_threshold /= args.h5_file_num
        epoch_neg_threshold /= args.h5_file_num
        epoch_acc_avg /= args.h5_file_num
        epoch_cla_loss_avg /= args.h5_file_num

        lr = get_lr(optimizer)
        logger.info(
            "Epoch:{:02d} Con_Partial_Loss:{:.4f} Con_Total_Loss:{:.4f} Class_Loss:{:.4f} Acc:{:.4f}".format(
                epoch,
                epoch_loss_avg, epoch_total_loss_avg, epoch_cla_loss_avg, epoch_acc_avg))
        logger.info(
            "Epoch:{:02d} Margin:{:.4f} Ratio:{:.3f} Pos_thres:{:.4f} Neg_thres:{:.4f} lr:{}".format(
                epoch,
                projection.margin, args.ratio, epoch_pos_threshold, epoch_neg_threshold, lr))
        state_dict = {
            'model': model.state_dict(),
            'projection': projection.state_dict()
        }
        save_checkpoint(state_dict, args.checkpoint_format, epoch)

        tbar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='conf/tmp.conf', type=str,
                        help='the config file')
    parser.add_argument('--pretrained', default=None, type=str,
                        help='the pretrained model')
    args = parser.parse_args()

    # *********************** process config ***********************
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.optionxform = str
    config.read(args.config)

    exp_section = config['exp']
    model_section = config['model']
    loss_section = config['loss']
    projection_section = config['projection']
    data_section = config['data']
    optim_section = config['optimization']

    exp_params = {k: eval(v) for k, v in exp_section.items()}
    model_params = {k: eval(v) for k, v in model_section.items()}
    loss_params = {k: eval(v) for k, v in loss_section.items()}
    projection_params = {k: eval(v) for k, v in projection_section.items()}
    data_params = {k: eval(v) for k, v in data_section.items()}
    optim_params = {k: eval(v) for k, v in optim_section.items()}

    # store config to args
    param2args(args, exp_params)
    param2args(args, model_params)
    param2args(args, loss_params)
    param2args(args, projection_params)
    param2args(args, data_params)
    param2args(args, optim_params)
    # *********************** process config ***********************

    # setup logger
    check_dir(exp_params['exp_dir'])
    logger = get_logger_2(os.path.join(exp_params['exp_dir'], 'train.log'))

    # write config file to expdir
    store_path = os.path.join(exp_params['exp_dir'], 'train.conf')
    write_conf(args.config, store_path)

    # model init
    args.use_gpu = optim_params['use_gpu'] and torch.cuda.is_available()
    model = System(**model_params)

    if model_params['system_type'] == 'D':
        model_params['emb_dim_out'] = model_params['emb_dim_out'] * 2
    if projection_params['project_type'] == 'linear':
        projection = nn.Linear(model_params['emb_dim_out'],projection_params['num_class'])
    else:
        projection = ArcMarginProduct(model_params['emb_dim_out'],projection_params['num_class'],
                        projection_params['scale'], projection_params['margin'],projection_params['easy_margin'])

    # load pretrained model
    if args.pretrained is not None:
        logger.info('Load pretrained model from {}'.format(args.pretrained))
        checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'])
        projection.load_state_dict(checkpoint['projection'])

    if args.use_gpu:
        model = model.cuda()
        projection = projection.cuda()

    if optim_params['type'] == 'SGD':
        optimizer = optim.SGD([{'params': model.parameters()}, {'params': projection.parameters()}],
                              lr=optim_params['lr'], momentum=optim_params['momentum'],
                              weight_decay=optim_params['weight_decay'], nesterov=True)
    else:
        optimizer = optim.Adam([{'params': model.parameters()}, {'params': projection.parameters()}],
                               lr=optim_params['lr'], betas=(0.9, 0.999),
                               weight_decay=optim_params['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=optim_params['factor'],
                                  patience=optim_params['patience'])

    checkpoint_dir = os.path.join(exp_params['exp_dir'], 'models')
    check_dir(checkpoint_dir)
    args.checkpoint_format = os.path.join(checkpoint_dir, 'epoch-{}.th')

    num_params = sum(param.numel() for param in model.parameters())
    logger.info(model)
    logger.info('Number of parmeters:{}'.format(num_params))

    spk2id_dict = spk2id(data_params['spk_file'])
    args.initial_ratio = args.ratio
    train(model, projection, optimizer, logger, args, scheduler, spk2id_dict)


if __name__ == '__main__':
    main()
