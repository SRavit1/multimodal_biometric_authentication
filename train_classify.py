import os
import argparse
import configparser
from tqdm import tqdm
import h5py
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import get_logger_2, check_dir, spk2id
from models.fusion import System, ContrastiveLoss
from models.projections import ArcMarginProduct
from utils_py.utils_pytorch import reduce_lr, increase_margin
from utils_py.utils_common import add_noise_to_embedding, add_noise_to_embedding_v2, load_mean_vec, param2args
from train_contrastive import Dataset_H5 as Eval_Dataset


def save_checkpoint(state_dict, save_path):
    torch.save(state_dict, save_path)


def worker_init_fn(worker_id):
    random.seed(worker_id)


def stats(output, label):
    if output.size(0) != 0:
        maxval, prediction = output.max(len(output.size()) - 1)
        num_matches = torch.sum(torch.eq(label, prediction)).item()
        num_items = label.size(0)
    else:
        num_matches, num_items = 0, 0
    return num_matches, num_items

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


class Dataset_H5(Dataset):
    def __init__(self, face_h5, utt_h5, spk2id_dict, length=64 * 10000, pollute_prob=None,
                 face_mean_vec=None, utt_mean_vec=None, noise_dir=None):
        super(Dataset_H5, self).__init__()

        self.length = length
        self.pollute_prob = pollute_prob  # the probability add noise to one modality
        self.noise_dir = noise_dir
        if self.noise_dir is not None:
            self.get_noise()

        # get common speaker list
        face_h5_f = h5py.File(face_h5,'r')
        utt_h5_f = h5py.File(utt_h5, 'r')
        face_spk_list = list(face_h5_f.keys())
        utt_spk_list = list(utt_h5_f.keys())
        face_h5_f.close()
        utt_h5_f.close()
        self.spk_list = list(set(face_spk_list) & set(utt_spk_list))

        self.face_h5 = face_h5
        self.utt_h5 = utt_h5
        self.spk2id_dict = spk2id_dict
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
            self.face_h5_f = h5py.File(self.face_h5,'r')
            self.utt_h5_f = h5py.File(self.utt_h5, 'r')

        spk = random.sample(self.spk_list, 1)[0]
        face_group = self.face_h5_f[spk]
        utt_group = self.utt_h5_f[spk]

        face_list = list(face_group.keys())
        utt_list = list(utt_group.keys())

        face_emb = face_group[random.sample(face_list, 1)[0]][()]
        utt_emb = utt_group[random.sample(utt_list, 1)[0]][()]
        if self.face_mean_vec is not None:
            face_emb = face_emb - self.face_mean_vec
            utt_emb = utt_emb - self.utt_mean_vec

        if self.pollute_prob is not None:
            choice = np.random.randint(0, 2)
            if choice == 0:
                if self.noise_dir is None:
                    face_emb = add_noise_to_embedding(face_emb, self.pollute_prob)
                else:
                    face_emb = add_noise_to_embedding_v2(face_emb, self.face_mean_list, self.face_std_list,
                                                         self.pollute_prob)
            else:
                if self.noise_dir is None:
                    utt_emb = add_noise_to_embedding(utt_emb, self.pollute_prob)
                else:
                    utt_emb = add_noise_to_embedding_v2(utt_emb, self.utt_mean_list, self.utt_std_list,
                                                        self.pollute_prob)

        spk_id = self.spk2id_dict[spk]

        return spk_id, face_emb.astype(np.dtype('float32')), utt_emb.astype(np.dtype('float32'))

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



def train(model, projection, optimizer, logger, spk2id_dict, args, scheduler):

    # train process
    model.train()

    max_iter = args.max_epoch * args.h5_file_num
    for epoch in range(args.max_epoch):
        with tqdm(total=args.batch_iters * args.h5_file_num,ncols=100) as pbar:
            epoch_acc_avg = 0.0
            epoch_loss_avg = 0.0
            epoch_artho_loss_avg = 0.0
            for h5_idx in range(args.h5_file_num):
                h5_idx_str = str(h5_idx).zfill(3)
                face_h5 = os.path.join(args.face_h5_dir, 'data_' + h5_idx_str + '.h5')
                utt_h5 = os.path.join(args.utt_h5_dir, 'data_' + h5_idx_str + '.h5')

                dataset = Dataset_H5(face_h5, utt_h5, spk2id_dict,
                                     length=args.batch_size * args.batch_iters,
                                     pollute_prob=args.pollute_prob,
                                     face_mean_vec=args.face_mean_vec, utt_mean_vec=args.utt_mean_vec)
                train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers)

                items_h5_count = 0
                matches_h5_count = 0
                h5_loss_avg = 0.0
                h5_artho_loss_avg = 0.0
                for labels, face_data, utt_data in train_loader:
                    optimizer.zero_grad()

                    if args.usegpu:
                        labels = labels.cuda()
                        face_data = face_data.cuda()
                        utt_data = utt_data.cuda()

                    out, face_trans, utt_trans = model(face_data, utt_data)
                    if args.project_type == 'linear':
                        project_out = projection(out)
                    else:
                        project_out = projection(out, labels)

                    num_matches, num_items = stats(project_out, labels)
                    items_h5_count += num_items
                    matches_h5_count += num_matches

                    loss = F.cross_entropy(project_out, labels, reduction='mean')
                    # orthogonal loss
                    if args.ortho_or_same:  # make the embedding pair orthogonal
                        ortho_loss = torch.mean(torch.abs(F.cosine_similarity(face_trans, utt_trans)))
                    else:
                        if args.ortho_loss_type == 'cos':
                            ortho_loss = 1.0 - torch.mean(F.cosine_similarity(face_trans, utt_trans))
                        else:
                            ortho_loss = torch.mean(F.pairwise_distance(face_trans, utt_trans))

                    (loss + args.ortho_loss_ratio * ortho_loss).backward()
                    optimizer.step()

                    h5_loss_avg += loss.item()
                    h5_artho_loss_avg += ortho_loss.item()
                    pbar.update()

                h5_acc_avg = 1.0 * matches_h5_count / items_h5_count
                h5_loss_avg /= args.batch_iters
                h5_artho_loss_avg /= args.batch_iters

                if h5_idx % (args.h5_file_num // 5 + 1) == 0:
                    lr = get_lr(optimizer)
                    logger.info('Epoch-{}-H5-{} Acc:{:.3f} Loss:{:.2f} Artho Loss:{:.2f} Margin:{:.3f} lr:{}'.format(epoch, h5_idx, h5_acc_avg,
                                                                         h5_loss_avg, h5_artho_loss_avg, projection.margin, lr))


                current_iter = epoch * args.h5_file_num + h5_idx
                if args.use_scheduler:
                    if args.scheduler_type == 0:
                        scheduler.step(h5_loss_avg)
                    else:
                        reduce_lr(optimizer, args.lr, args.final_lr, current_iter, max_iter)
                if args.update_margin:
                    current_margin = increase_margin(args.margin, args.final_margin, current_iter, max_iter)
                    projection.update(margin=current_margin)

                epoch_acc_avg += h5_acc_avg
                epoch_loss_avg += h5_loss_avg
                epoch_artho_loss_avg += h5_artho_loss_avg
                
                if optimizer.param_groups[0]['lr'] < 1e-7:
                    logger.info("Stop training, the learning rate is too small")
                    state_dict = {
                        'model': model.state_dict(),
                        'projection': projection.state_dict()
                    }
                    save_path = args.checkpoint_format.format(epoch)
                    save_checkpoint(state_dict, save_path)
                    exit()

            epoch_acc_avg /= args.h5_file_num
            epoch_loss_avg /= args.h5_file_num
            epoch_artho_loss_avg /= args.h5_file_num
            lr = get_lr(optimizer)
            logger.info('Epoch-{} Acc:{:.3f} Loss:{:.2f} Artho Loss:{:.2f} Margin:{:.3f} lr:{}'.format(epoch, epoch_acc_avg,
                                                                         epoch_loss_avg, epoch_artho_loss_avg, projection.margin, lr))

            state_dict = {
                'model': model.state_dict(),
                'projection': projection.state_dict()
            }
            save_path = args.checkpoint_format.format(epoch)
            save_checkpoint(state_dict, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='conf/tmp_classify_h5.conf',type=str,
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
    logger = get_logger_2(os.path.join(exp_params['exp_dir'],'train.log'))

    # write config file to expdir
    store_path = os.path.join(exp_params['exp_dir'], 'train.conf')
    write_conf(args.config, store_path)

    # model init
    args.usegpu = torch.cuda.is_available()
    if args.usegpu:
        print('Use {} gpus.'.format(torch.cuda.device_count()), flush=True)
    else:
        print('Use cpu', flush=True)
    model = System(**model_params)

    if model_params['system_type'] == 'D':
        model_params['emb_dim_out'] = model_params['emb_dim_out'] * 2

    if model_params['system_type'] == 'F':
        model_params['emb_dim_out'] = model_params['cbp_dim']
        
    if args.project_type == 'linear':
        projection = nn.Sequential(nn.BatchNorm1d(model_params['emb_dim_out']),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(model_params['emb_dim_out'], projection_params['num_class']))
    else:
        projection = ArcMarginProduct(model_params['emb_dim_out'],projection_params['num_class'],
                        projection_params['scale'],projection_params['margin'],projection_params['easy_margin'])

    # load pretrained model
    if args.pretrained is not None:
        logger.info('Load pretrained model from {}'.format(args.pretrained))
        checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        if 'fusion_model' in checkpoint:
            model.load_state_dict(checkpoint['fusion_model'])
        else:
            model.load_state_dict(checkpoint['model'])
        projection.load_state_dict(checkpoint['projection'])

    if(args.usegpu):
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

    logger.info(model)
    logger.info(projection)
    num_params = sum(param.numel() for param in model.parameters())
    logger.info('Number of parmeters:{}'.format(num_params))

    spk2id_dict = spk2id(data_params['spk_file'])
    train(model, projection, optimizer, logger, spk2id_dict, args, scheduler)


if __name__ == '__main__':
    main()
