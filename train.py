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

from dataset import MultimodalPairDataset, FaceDataset
from loss import loss_calculator, AngularPenaltySMLoss
from evaluate import evaluate_single_modality

emb_size = 512
num_classes = 1211 #5994

def train_audio(model, optimizer, logger, scheduler, params):
    model.train()

    for epoch in tqdm(range(params["optim_params"]['audio_end_epoch']), position=0):
        dataset = MultimodalPairDataset(length=params["data_params"]['audio_batch_size'] * params["data_params"]['audio_batch_iters'], select_face=False)
        train_loader = DataLoader(dataset, batch_size=params["data_params"]['audio_batch_size'], shuffle=False,
                                    num_workers=params["data_params"]['num_workers'])

        for labels, utt1, utt2 in tqdm(train_loader, position=1):
            if params["optim_params"]['use_gpu']:
                labels = labels.cuda()
                utt1 = utt1.cuda()
                utt2 = utt2.cuda()

            optimizer.zero_grad()
            
            utts = torch.cat((utt1, utt2), dim=0)

            utt_data = model(utts)

            half_index = int(len(utt_data)/2)
            utt_data1 = utt_data[:half_index]
            utt_data2 = utt_data[half_index:]

            loss = torch.mean(utt_data1-utt_data2) #...

            loss.backward()
            optimizer.step()
            scheduler.step(loss)

def train_face(model, optimizer, criterion, scheduler, train_loader, test_loader, logger, params):
    model.train()

    for epoch in tqdm(range(params["optim_params"]['face_end_epoch']), position=0):
        for faces, labels in tqdm(train_loader, position=1):
            if params["optim_params"]['use_gpu']:
                faces = faces.cuda()
                labels = labels.cuda()
            
            optimizer.zero_grad()

            face_embeddings = model(faces)

            loss = criterion(face_embeddings, labels)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
        eer = evaluate_single_modality(model, test_loader, params)

def train_fusion(model, face_model, audio_model, optimizer, logger, scheduler, params):
    model.train()

    max_iter = params["optim_params"]["max_epoch"] * params["data_params"]['h5_file_num']
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

            out, _, _ = model(face_input, utt_input)
            pair_num = out.shape[0] // 2
            assert pair_num == params["data_params"]['batch_size']

            loss, total_loss, pos_threshold, neg_threshold = loss_calculator(out[0:pair_num], out[pair_num:], labels, params["optim_params"]['ratio'])

            if params["loss_params"]['partial_loss']:
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

    face_criterion = AngularPenaltySMLoss(emb_size, num_classes).cuda()

    # intialize optimizers
    if params["optim_params"]['type'] == 'SGD':
        fusion_optimizer = optim.SGD(fusion_model.parameters(), lr=params["optim_params"]['lr'], momentum=params["optim_params"]['momentum'],
                              weight_decay=params["optim_params"]['weight_decay'], nesterov=True)
    else:
        fusion_optimizer = optim.Adam(fusion_model.parameters(), lr=params["optim_params"]['lr'],  betas=(0.9, 0.999),
                              weight_decay=params["optim_params"]['weight_decay'])
    audio_optimizer = optim.Adam(audio_model.parameters(), lr=params["optim_params"]['audio_lr'])
    face_optimizer = optim.Adam(list(set(face_model.parameters()) | set(face_criterion.fc.parameters())), lr=params["optim_params"]['face_lr'])

    # initialize schedulers
    fusion_scheduler = ReduceLROnPlateau(fusion_optimizer, 'min', factor=params["optim_params"]['factor'],
                                  patience=params["optim_params"]['patience'])
    audio_scheduler = CosineAnnealingLR(audio_optimizer, params["optim_params"]['audio_end_epoch'], 
        params["optim_params"]['audio_lr_min'])
    face_scheduler = CosineAnnealingLR(audio_optimizer, params["optim_params"]['face_end_epoch'], 
        params["optim_params"]['face_lr_min'])

    face_train_dataset = FaceDataset(length=params["data_params"]['face_batch_size'] * params["data_params"]['batch_iters'])
    face_train_loader = DataLoader(face_train_dataset, batch_size=params["data_params"]['face_batch_size'], shuffle=False,
                                num_workers=params["data_params"]['num_workers'])
    face_test_dataset = MultimodalPairDataset(length=params["data_params"]['face_batch_size'] * params["data_params"]['batch_iters'],
                                select_face=True, select_audio=False)
    face_test_loader = DataLoader(face_test_dataset, batch_size=params["data_params"]['face_batch_size'], shuffle=False,
                                num_workers=params["data_params"]['num_workers'])

    # check existence of checkpoint dir
    checkpoint_dir = os.path.join(params["exp_params"]['exp_dir'],'models')
    check_dir(checkpoint_dir)

    fusion_num_params = sum(param.numel() for param in fusion_model.parameters())
    logger.info('Fusion number of parmeters:{}'.format(fusion_num_params))
    audio_num_params = sum(param.numel() for param in audio_model.parameters())
    logger.info('Audio number of parmeters:{}'.format(audio_num_params))
    face_num_params = sum(param.numel() for param in face_model.parameters())
    logger.info('Fusion number of parmeters:{}'.format(face_num_params))
    
    #train_audio(audio_model, audio_optimizer, logger, audio_scheduler, params)
    train_face(face_model, face_optimizer, face_criterion, face_scheduler, face_train_loader, face_test_loader, logger, params)
    #train_fusion(fusion_model, face_model, audio_model, fusion_optimizer, logger, fusion_scheduler, params)

if __name__ == '__main__':
    main()
