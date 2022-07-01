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

from utils import get_logger_2, check_dir, create_logger, save_checkpoint, AverageMeter, plot_distances_labels, plot_far_frr
from utils_py.utils_common import write_conf
from models.fusion import System
import models.resnet as resnet

from dataset import MultimodalPairDataset, FaceDataset
import loss as loss_utils
from loss import AngularPenaltySMLoss, ArcFace
from evaluate import evaluate_single_modality

emb_size = 512
num_classes = 1211 #5994

def train_face(model, classifier, optimizer, criterion, scheduler, train_loader, test_loader, logger, log_dir, params):
    best_eer = 100
    distances_labels_hists_dir = os.path.join(log_dir, "distances_labels_hists")
    if not os.path.exists(distances_labels_hists_dir):
        os.mkdir(distances_labels_hists_dir)
    
    far_frr_curves_dir = os.path.join(log_dir, "far_frr_curves")
    if not os.path.exists(far_frr_curves_dir):
        os.mkdir(far_frr_curves_dir)

    ArcFaceLayer = ArcFace()
    
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
            face_logits = classifier(face_embeddings)
            face_logits_mod = ArcFaceLayer(face_logits, labels)
            loss = criterion(face_logits_mod, labels)
            acc1, acc5 = loss_utils.accuracy(face_logits, labels, topk=(1, 5))

            losses.update(float(loss))
            top1.update(float(acc1))
            top5.update(float(acc5))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(set(model.parameters()) | set(classifier.parameters())), 5)
            optimizer.step()
            if batch_no % params["optim_params"]["print_frequency_batch"] == 0:
                logger.info("Epoch [{}] Batch {}/{} {} {} {}".format(epoch, batch_no, len(train_loader), str(losses), str(top1), str(top5)))
        
        scheduler.step()
        if epoch % params["optim_params"]["val_frequency_epoch"] == 0:
            model.eval()
            distances, labels, fprs, tprs, thresholds, eer = evaluate_single_modality(model, test_loader, params)
            plot_distances_labels(distances, labels, os.path.join(distances_labels_hists_dir, "distances_labels_epoch_{}.png".format(epoch)), epoch)
            plot_far_frr(thresholds, fprs, tprs, os.path.join(far_frr_curves_dir, "far_frr_epoch_{}.png".format(epoch)), epoch)
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
    store_path = os.path.join(log_dir, conf_name)
    write_conf(config_path, store_path)

    # models init
    params["optim_params"]['use_gpu'] = params["optim_params"]['use_gpu'] and torch.cuda.is_available()
    face_model = resnet.resnet18(num_classes=emb_size)
    face_classifier = torch.nn.Linear(emb_size, num_classes)

    # load pretrained model
    if params["exp_params"]["pretrained"]:
        logger.info('Load pretrained model from {}'.format(params["exp_params"]["pretrained"]))
        state_dict = torch.load(params["exp_params"]["pretrained"], map_location=lambda storage, loc: storage)
        face_model.load_state_dict(state_dict)

    # convert models to cuda
    if params["optim_params"]['use_gpu']:
        face_model = face_model.cuda()
        face_classifier = face_classifier.cuda()

    #face_criterion = AngularPenaltySMLoss(emb_size, num_classes, log_eps=params["optim_params"]["log_eps"]).cuda()
    face_criterion = torch.nn.CrossEntropyLoss()

    # intialize optimizers
    face_optimizer = optim.Adam(list(set(face_model.parameters()) | set(face_classifier.parameters())), lr=params["optim_params"]['lr'], weight_decay=params["optim_params"]['weight_decay'])

    # initialize schedulers
    face_scheduler = StepLR(face_optimizer, step_size=30, gamma=0.1)

    face_train_dataset = datasets.ImageFolder(
        os.path.join(params["data_params"]["train_dir"], "face"),
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    face_train_loader = DataLoader(face_train_dataset, batch_size=params["data_params"]['batch_size'], shuffle=True,
                                num_workers=params["data_params"]['num_workers'])
    face_test_dataset = MultimodalPairDataset(params["data_params"]["train_dir"], length=params["data_params"]['batch_size'] * params["data_params"]['batch_iters'],
                                select_face=True, select_audio=False)
    face_test_loader = DataLoader(face_test_dataset, batch_size=params["data_params"]['batch_size'], shuffle=False,
                                num_workers=params["data_params"]['num_workers'])

    face_num_params = sum(param.numel() for param in face_model.parameters())
    logger.info('Face number of parmeters:{}'.format(face_num_params))
    
    train_face(face_model, face_classifier, face_optimizer, face_criterion, face_scheduler, face_train_loader, face_test_loader, logger, log_dir, params)
    
if __name__ == '__main__':
    main()