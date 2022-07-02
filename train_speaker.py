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

from dataset import MultimodalPairDataset, Vox1ValDataset, FaceDataset, utt_path_to_utt
import loss as loss_utils
from loss import AngularPenaltySMLoss, ArcFace
from evaluate import evaluate_single_modality

def train_speaker(model, classifier, optimizer, criterion, scheduler, train_loader, val_loader, logger, log_dir, params):
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
        
        for batch_no, (utts, labels) in enumerate(train_loader):
            if params["optim_params"]['use_gpu']:
                utts = utts.cuda()
                labels = labels.cuda()
            
            optimizer.zero_grad()

            utt_embeddings = model(utts)
            utt_logits = classifier(utt_embeddings)
            utt_logits = utt_logits.clamp(-1, 1)
            acc1, acc5 = loss_utils.accuracy(utt_logits, labels, topk=(1, 5))
            utt_logits_mod = ArcFaceLayer(utt_logits, labels)
            loss = criterion(utt_logits_mod, labels)

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
            distances, labels, fprs, tprs, thresholds, eer = evaluate_single_modality(model, val_loader, params)
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
                'best_eer': best_eer,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            },
            is_best,
            checkpoint_path + ".pth",
            best_checkpoint_path + ".pth")
        
        dummy_input = torch.zeros((1, 1, 128, 94))
        if params["optim_params"]['use_gpu']:
            dummy_input = dummy_input.cuda()
        torch.onnx.export(model, dummy_input, checkpoint_path + ".onnx")
        if is_best:
            shutil.copyfile(checkpoint_path + ".onnx", best_checkpoint_path + ".onnx")

def main():
    # *********************** process config ***********************
    conf_name = "speaker_train.conf"
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
    model_name = "speaker_" + time_str
    log_dir = os.path.join(params["exp_params"]['exp_dir'], model_name)
    os.mkdir(log_dir)
    logger = create_logger(log_dir)

    # write config file to expdir
    store_path = os.path.join(log_dir, conf_name)
    write_conf(config_path, store_path)

    # models init
    params["optim_params"]['use_gpu'] = params["optim_params"]['use_gpu'] and torch.cuda.is_available()
    speaker_model = resnet.resnet18(num_classes=params["exp_params"]["emb_size"], input_channels=1)
    speaker_classifier = torch.nn.Linear(params["exp_params"]["emb_size"], params["exp_params"]["num_classes"])

    # load pretrained model
    if params["exp_params"]["pretrained"]:
        logger.info('Load pretrained model from {}'.format(params["exp_params"]["pretrained"]))
        state_dict = torch.load(params["exp_params"]["pretrained"], map_location=lambda storage, loc: storage)["state_dict"]
        speaker_model.load_state_dict(state_dict)

    # convert models to cuda
    if params["optim_params"]['use_gpu']:
        speaker_model = speaker_model.cuda()
        speaker_classifier = speaker_classifier.cuda()

    speaker_criterion = torch.nn.CrossEntropyLoss()

    # intialize optimizer
    opt_params = [{"params": speaker_model.parameters()}, {"params": speaker_classifier.parameters()}]
    lr = params["optim_params"]['lr']
    weight_decay = params["optim_params"]['weight_decay']
    speaker_optimizer = optim.AdamW(opt_params, lr=lr, weight_decay=weight_decay)
    #speaker_optimizer = optim.SGD(opt_params, lr=lr, momentum=0.9, weight_decay=weight_decay)

    # initialize scheduler
    speaker_scheduler = StepLR(speaker_optimizer, step_size=30, gamma=0.1)

    speaker_train_dataset = datasets.DatasetFolder(
        os.path.join(params["data_params"]["train_dir"], "utt"),
        utt_path_to_utt,
        "wav"
    )
    speaker_train_loader = DataLoader(speaker_train_dataset, batch_size=params["data_params"]['batch_size'], shuffle=True,
                                num_workers=params["data_params"]['num_workers'])
    speaker_val_dataset = Vox1ValDataset(os.path.join(params["data_params"]["test_dir"], os.pardir),
        select_face=False, select_audio=True, dataset=params["data_params"]["val_dataset"])
    speaker_val_loader = DataLoader(speaker_val_dataset, batch_size=params["data_params"]['batch_size'], shuffle=False,
                                num_workers=params["data_params"]['num_workers'])

    speaker_num_params = sum(param.numel() for param in speaker_model.parameters())
    logger.info('Face number of parmeters:{}'.format(speaker_num_params))
    
    train_speaker(speaker_model, speaker_classifier, speaker_optimizer, speaker_criterion, speaker_scheduler, speaker_train_loader, speaker_val_loader, logger, log_dir, params)
    
if __name__ == '__main__':
    main()