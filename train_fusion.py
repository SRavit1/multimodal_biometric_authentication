import os
import argparse
import configparser
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from utils import get_logger_2, check_dir
from utils_py.utils_common import write_conf
from models.fusion import System
import models.resnet as resnet

from dataset import MultimodalPairDataset
from loss import loss_calculator
from evaluate import evaluate_multimodal

def train_fusion(fusion_model, face_model, audio_model, optimizer, logger, scheduler, test_loader, params):
    fusion_model.train()

    for epoch in tqdm(range(params["optim_params"]["max_epoch"]), position=0):
        dataset = MultimodalPairDataset(length=params["data_params"]['batch_size'] * params["data_params"]['batch_iters'])
        train_loader = DataLoader(dataset, batch_size=params["data_params"]['batch_size'], shuffle=False,
                                    num_workers=params["data_params"]['num_workers'])

        for labels, face1, utt1, face2, utt2 in tqdm(train_loader, position=1):
            optimizer.zero_grad()

            if params["optim_params"]['use_gpu']:
                face1 = face1.cuda()
                utt1 = utt1.cuda()
                face2 = face2.cuda()
                utt2 = utt2.cuda()
                labels = labels.cuda()
            
            faces = torch.cat((face1, face2), dim=0)
            utts = torch.cat((utt1, utt2), dim=0)

            with torch.no_grad():
                face_data = face_model(faces)
                utt_data = audio_model(utts)

            half_index = int(len(face_data)/2)
            face_data1 = face_data[:half_index]
            face_data2 = face_data[half_index:]
            utt_data1 = utt_data[:half_index]
            utt_data2 = utt_data[half_index:]

            face_input = torch.cat((face_data1, face_data2), dim=0)
            utt_input = torch.cat((utt_data1, utt_data2), dim=0)

            out, _, _ = fusion_model(face_input, utt_input)
            pair_num = out.shape[0] // 2
            assert pair_num == params["data_params"]['batch_size']

            loss, total_loss, pos_threshold, neg_threshold = loss_calculator(out[0:pair_num], out[pair_num:], labels, params["optim_params"]['ratio'])

            if params["loss_params"]['partial_loss']:
                loss.backward()
            else:
                total_loss.backward()
            optimizer.step()
            scheduler.step(loss)

        eer = evaluate_multimodal(fusion_model, face_model, audio_model, test_loader, params)

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

    params = {
        "exp_params": {k: eval(v) for k, v in config['exp'].items()},
        "model_params": {k: eval(v) for k, v in config['model'].items()},
        "loss_params": {k: eval(v) for k, v in config['loss'].items()},
        "data_params": {k: eval(v) for k, v in config['data'].items()},
        "optim_params": {k: eval(v) for k, v in config['optimization'].items()}
    }
    # *********************** process config ***********************

    # setup logger
    check_dir(params["exp_params"]['exp_dir'])
    logger = get_logger_2(os.path.join(params["exp_params"]['exp_dir'], 'train.log'))

    # write config file to expdir
    store_path = os.path.join(params["exp_params"]['exp_dir'], 'train.conf')
    write_conf(args.config, store_path)

    # models init
    params["optim_params"]['use_gpu'] = params["optim_params"]['use_gpu'] and torch.cuda.is_available()
    fusion_model = System(**params["model_params"])
    face_model = resnet.resnet18(num_classes=512)
    audio_model = resnet.resnet18(num_classes=512, input_channels=1)

    # load pretrained model
    if args.pretrained is not None:
        logger.info('Load pretrained model from {}'.format(args.pretrained))
        state_dict = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        fusion_model.load_state_dict(state_dict)

    # convert models to cuda
    if params["optim_params"]['use_gpu']:
        fusion_model = fusion_model.cuda()
        face_model = face_model.cuda()
        audio_model = audio_model.cuda()
    # intialize optimizers
    if params["optim_params"]['type'] == 'SGD':
        fusion_optimizer = optim.SGD(fusion_model.parameters(), lr=params["optim_params"]['lr'], momentum=params["optim_params"]['momentum'],
                              weight_decay=params["optim_params"]['weight_decay'], nesterov=True)
    else:
        fusion_optimizer = optim.Adam(fusion_model.parameters(), lr=params["optim_params"]['lr'],  betas=(0.9, 0.999),
                              weight_decay=params["optim_params"]['weight_decay'])
    # initialize schedulers
    fusion_scheduler = ReduceLROnPlateau(fusion_optimizer, 'min', factor=params["optim_params"]['factor'],
                                  patience=params["optim_params"]['patience'])

    # check existence of checkpoint dir
    checkpoint_dir = os.path.join(params["exp_params"]['exp_dir'],'models')
    check_dir(checkpoint_dir)

    fusion_num_params = sum(param.numel() for param in fusion_model.parameters())
    logger.info('Fusion number of parmeters:{}'.format(fusion_num_params))

    fusion_test_dataset = MultimodalPairDataset(length=params["data_params"]['batch_size'] * params["data_params"]['batch_iters'])
    fusion_test_loader = DataLoader(fusion_test_dataset, batch_size=params["data_params"]['batch_size'], shuffle=False,
                                num_workers=params["data_params"]['num_workers'])
    train_fusion(fusion_model, face_model, audio_model, fusion_optimizer, logger, fusion_scheduler, fusion_test_loader, params)

if __name__ == '__main__':
    main()
