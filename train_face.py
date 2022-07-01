import os
import argparse
import configparser
from tqdm import tqdm
import time
import shutil

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils import get_logger_2, check_dir, create_logger, save_checkpoint, AverageMeter
from utils_py.utils_common import write_conf
from models.fusion import System
import models.resnet as resnet

from dataset import MultimodalPairDataset, FaceDataset
from loss import AngularPenaltySMLoss
from evaluate import evaluate_single_modality

emb_size = 512
num_classes = 1211 #5994

def train_face(model, optimizer, criterion, scheduler, train_loader, test_loader, logger, log_dir, params):
    best_acc1 = 0
    best_acc5 = 0
    best_eer = 1
    for epoch in tqdm(range(params["optim_params"]['end_epoch']), position=0):
        model.train()
        
        losses = AverageMeter("Loss")
        top1 = AverageMeter("Acc@1")
        top5 = AverageMeter("Acc@5")
        
        for batch_no, (faces, labels) in enumerate(train_loader):
            if params["optim_params"]['use_gpu']:
                faces = faces.cuda()
                labels = labels.cuda()
            
            optimizer.zero_grad()

            face_embeddings = model(faces)

            loss, acc1, acc5 = criterion(face_embeddings, labels)
            losses.update(float(loss))
            top1.update(float(acc1))
            top5.update(float(acc5))

            loss.backward()
            optimizer.step()
            if batch_no % params["optim_params"]["print_frequency_batch"] == 0:
                logger.info("Epoch [{}] Batch {}/{} {} {} {}".format(epoch, batch_no, len(train_loader), str(losses), str(top1), str(top5)))
        
        scheduler.step()

        if epoch % params["optim_params"]["val_frequency_epoch"] == 0:
            model.eval()
            eer = evaluate_single_modality(model, test_loader, params)
            logger.info("Validation EER: {:.4f}".format(float(eer)))
        
        if eer < best_eer:
            is_best = True
            best_eer = eer
        else:
            is_best = False
        logger.info("is_best {}. Saving checkpoint.".format(is_best))

        checkpoint_path = os.path.join(log_dir, "checkpoint")
        best_checkpoint_path = os.path.join(log_dir, "best_checkpoint")
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                #'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            },
            is_best,
            checkpoint_path + ".pth",
            best_checkpoint_path + ".pth")
        
        dummy_input = torch.zeros((1, 3, 224, 224))
        if params["optim_params"]['use_gpu']:
            dummy_input = dummy_input.cuda()
        torch.onnx.export(model, dummy_input, checkpoint_path + ".onnx")
        if is_best:
            shutil.copyfile(checkpoint_path + ".onnx", best_checkpoint_path + ".onnx")

def main():
    # *********************** process config ***********************
    conf_name = "face_train.conf"
    config_path = os.path.join('conf', conf_name)
    
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.optionxform=str
    config.read(config_path)

    params = {
        "exp_params": {k: eval(v) for k, v in config['exp'].items()},
        "data_params": {k: eval(v) for k, v in config['data'].items()},
        "optim_params": {k: eval(v) for k, v in config['optimization'].items()}
    }
    # *********************** process config ***********************

    # setup logger
    check_dir(params["exp_params"]['exp_dir'])
    #logger = get_logger_2(os.path.join(params["exp_params"]['exp_dir'], 'train.log'))
    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    model_name = "face_" + time_str
    log_dir = os.path.join(params["exp_params"]['exp_dir'], model_name)
    os.mkdir(log_dir)
    logger = create_logger(log_dir)

    # write config file to expdir
    store_path = os.path.join(params["exp_params"]['exp_dir'], conf_name)
    write_conf(config_path, store_path)

    # models init
    params["optim_params"]['use_gpu'] = params["optim_params"]['use_gpu'] and torch.cuda.is_available()
    face_model = resnet.resnet18(num_classes=512)

    # load pretrained model
    if params["exp_params"]["pretrained"]:
        logger.info('Load pretrained model from {}'.format(params["exp_params"]["pretrained"]))
        state_dict = torch.load(params["exp_params"]["pretrained"], map_location=lambda storage, loc: storage)
        face_model.load_state_dict(state_dict)

    # convert models to cuda
    if params["optim_params"]['use_gpu']:
        face_model = face_model.cuda()

    face_criterion = AngularPenaltySMLoss(emb_size, num_classes).cuda()

    # intialize optimizers
    face_optimizer = optim.Adam(list(set(face_model.parameters()) | set(face_criterion.fc.parameters())), lr=params["optim_params"]['lr'])

    # initialize schedulers
    face_scheduler = StepLR(face_optimizer, step_size=30, gamma=0.1)

    face_train_dataset = datasets.ImageFolder(
        params["data_params"]["train_face_dir"],
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    face_train_loader = DataLoader(face_train_dataset, batch_size=params["data_params"]['batch_size'], shuffle=True,
                                num_workers=params["data_params"]['num_workers'])
    face_test_dataset = MultimodalPairDataset(length=params["data_params"]['batch_size'] * params["data_params"]['batch_iters'],
                                select_face=True, select_audio=False)
    face_test_loader = DataLoader(face_test_dataset, batch_size=params["data_params"]['batch_size'], shuffle=False,
                                num_workers=params["data_params"]['num_workers'])

    # check existence of checkpoint dir
    checkpoint_dir = os.path.join(params["exp_params"]['exp_dir'],'models')
    check_dir(checkpoint_dir)

    face_num_params = sum(param.numel() for param in face_model.parameters())
    logger.info('Face number of parmeters:{}'.format(face_num_params))
    
    train_face(face_model, face_optimizer, face_criterion, face_scheduler, face_train_loader, face_test_loader, logger, log_dir, params)
    
if __name__ == '__main__':
    main()