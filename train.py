import os
import argparse
import configparser
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import get_logger_2, check_dir
from utils_py.utils_common import write_conf
from models.fusion import System
import models.resnet as resnet
from dataset import MultimodalContrastiveDataset
from loss import loss_calculator

def train_fusion(model, face_model, audio_model, optimizer, logger, data_params, args, scheduler):
    model.train()

    max_iter = args.max_epoch * data_params['h5_file_num']
    for epoch in tqdm(range(args.max_epoch), position=0):
        dataset = MultimodalContrastiveDataset(length=data_params['batch_size'] * data_params['batch_iters'])
        train_loader = DataLoader(dataset, batch_size=data_params['batch_size'], shuffle=False,
                                    num_workers=data_params['num_workers'])

        for labels, face1, utt1, face2, utt2 in tqdm(train_loader, position=1):
            optimizer.zero_grad()
            
            faces = torch.cat((face1, face2), dim=0)
            utts = torch.cat((utt1, utt2), dim=0)

            face_data = face_model(faces)
            utt_data = audio_model(utts)

            half_index = int(len(face_data)/2)
            face_data1 = face_data[:half_index]
            face_data2 = face_data[half_index:]
            utt_data1 = utt_data[:half_index]
            utt_data2 = utt_data[half_index:]

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

            if args.partial_loss:
                loss.backward()
            else:
                total_loss.backward()
            optimizer.step()
            scheduler.step(loss)


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

    face_model = resnet.resnet18(num_classes=512)
    audio_model = resnet.resnet18(num_classes=512, input_channels=1)
    #train_audio(audio_model, audio_optimizer, logger, audio_scheduler)
    #train_face()
    train_fusion(model, face_model, audio_model, optimizer, logger, data_params, args, scheduler)

if __name__ == '__main__':
    main()
