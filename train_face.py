import os
import argparse
import configparser
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import get_logger_2, check_dir
from utils_py.utils_common import write_conf
from models.fusion import System
import models.resnet as resnet

from dataset import MultimodalPairDataset, FaceDataset
from loss import AngularPenaltySMLoss
from evaluate import evaluate_single_modality

emb_size = 512
num_classes = 1211 #5994

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='conf/face_train.conf', type=str,
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

    # load pretrained model
    if args.pretrained is not None:
        logger.info('Load pretrained model from {}'.format(args.pretrained))
        state_dict = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        fusion_model.load_state_dict(state_dict)

    # convert models to cuda
    if params["optim_params"]['use_gpu']:
        face_model = face_model.cuda()

    face_criterion = AngularPenaltySMLoss(emb_size, num_classes).cuda()

    # intialize optimizers
    face_optimizer = optim.Adam(list(set(face_model.parameters()) | set(face_criterion.fc.parameters())), lr=params["optim_params"]['face_lr'])

    # initialize schedulers
    face_scheduler = CosineAnnealingLR(face_optimizer, params["optim_params"]['face_end_epoch'], 
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

    face_num_params = sum(param.numel() for param in face_model.parameters())
    logger.info('Fusion number of parmeters:{}'.format(face_num_params))
    
    train_face(face_model, face_optimizer, face_criterion, face_scheduler, face_train_loader, face_test_loader, logger, params)
    
if __name__ == '__main__':
    main()
