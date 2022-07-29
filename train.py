import os
import argparse
import configparser
from tqdm import tqdm
import time
import shutil

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
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
from scheduler import PolyScheduler
from binarized_modules import copy_data_to_org, copy_org_to_data

def train(model, classifier, optimizer, criterion, scheduler, train_loader, val_loader, logger, log_dir, params):
    best_eer = 100
    distances_labels_hists_dir = os.path.join(log_dir, "distances_labels_hists")
    if not os.path.exists(distances_labels_hists_dir):
        os.mkdir(distances_labels_hists_dir)
    
    far_frr_curves_dir = os.path.join(log_dir, "far_frr_curves")
    if not os.path.exists(far_frr_curves_dir):
        os.mkdir(far_frr_curves_dir)

    logger.info("LR {}".format(optimizer.param_groups[0]['lr']))

    ArcFaceLayer = ArcFace()
    for epoch in tqdm(range(params["optim_params"]['end_epoch']), position=0):
        model.train()

        logger.info("LR {}".format(optimizer.param_groups[0]['lr']))
        
        losses = AverageMeter("Loss")
        top1 = AverageMeter("Acc@1")
        top5 = AverageMeter("Acc@5")

        if epoch in params["exp_params"]['prec_config_schedule'].keys():
            prec_config = params["exp_params"]['prec_config_schedule'][epoch]
            model.update_prec_config(prec_config, save_weights=True)
            if params["optim_params"]['use_gpu']:
                model = model.cuda(params["optim_params"]['device'])
            logger.info("New precision config: " + str(prec_config))
        
        for batch_no, (samples, labels) in enumerate(train_loader):
            if params["optim_params"]['use_gpu']:
                samples = samples.cuda(params["optim_params"]['device'])
                labels = labels.cuda(params["optim_params"]['device'])

            embeddings = model(samples)
            logits = classifier(embeddings)
            logits = logits.clamp(-1, 1)
            acc1, acc5 = loss_utils.accuracy(logits, labels, topk=(1, 5))
            if params['exp_params']['useArcFace']:
                logits_mod = ArcFaceLayer(logits, labels)
            else:
                logits_mod = logits
            loss = criterion(logits_mod, labels)

            losses.update(float(loss))
            top1.update(float(acc1))
            top5.update(float(acc5))
            

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(set(model.parameters()) | set(classifier.parameters())), 5)
            copy_org_to_data(model)
            optimizer.step()
            copy_data_to_org(model)

            if batch_no % params["optim_params"]["print_frequency_batch"] == 0:
                logger.info("Epoch [{}] Batch {}/{} {} {} {}".format(epoch, batch_no, len(train_loader), str(losses), str(top1), str(top5)))
        
            if not params['optim_params']['scheduler'] == "" and not params['optim_params']['scheduler'] in ["ReduceLROnPlateau"]:
                scheduler.step()
        if params['optim_params']['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(losses.avg)
        
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
        scheduler_state_dict = scheduler.state_dict() if scheduler else None
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_eer': best_eer,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler_state_dict
            },
            is_best,
            checkpoint_path + ".pth",
            best_checkpoint_path + ".pth")
        
        if params["exp_params"]["model_type"] == "face":
            dummy_input = torch.zeros((1, 3, params["data_params"]["input_dim"], params["data_params"]["input_dim"]))
        elif params["exp_params"]["model_type"] == "speaker":
            dummy_input = torch.zeros((1, 1, params["data_params"]["input_dim"], params["data_params"]["input_dim"]))
        
        if params["optim_params"]['use_gpu']:
            dummy_input = dummy_input.cuda(params["optim_params"]['device'])
        torch.onnx.export(model, dummy_input, checkpoint_path + ".onnx")
        if is_best:
            shutil.copyfile(checkpoint_path + ".onnx", best_checkpoint_path + ".onnx")

def main():
    # *********************** process config ***********************
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--conf", type=str, default="face_train_xnor.conf", help="config file to use")
    args = parser.parse_args()
    config_path = os.path.join('conf', args.conf)
    
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.optionxform=str
    config.read(config_path)

    params = {
        "exp_params": {k: eval(v) for k, v in config['exp'].items()},
        "data_params": {k: eval(v) for k, v in config['data'].items()},
        "optim_params": {k: eval(v) for k, v in config['optimization'].items()}
    }
    # *********************** process config ***********************
    model_type = params["exp_params"]["model_type"]
    assert model_type in ["face", "speaker"]

    # setup logger
    check_dir(params["exp_params"]['exp_dir'])
    #logger = get_logger_2(os.path.join(params["exp_params"]['exp_dir'], 'train.log'))
    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    model_name = model_type + "_" + time_str
    log_dir = os.path.join(params["exp_params"]['exp_dir'], model_name)
    os.mkdir(log_dir)
    logger = create_logger(log_dir)

    # write config file to expdir
    store_path = os.path.join(log_dir, args.conf)
    write_conf(config_path, store_path)

    # models init
    params["optim_params"]['use_gpu'] = params["optim_params"]['use_gpu'] and torch.cuda.is_available()
    input_channels = 3 if model_type == "face" else 1
    model = resnet.resnet18(num_classes=params["exp_params"]["emb_size"],
        prec_config=params["exp_params"]["prec_config"], input_channels=input_channels,
        normalize_output=params["exp_params"]["normalize_output"])
    logger.info("Initial precision config: " + str(model.prec_config))

    classifier = torch.nn.Linear(params["exp_params"]["emb_size"], params["exp_params"]["num_classes"])

    # load pretrained model
    if params["exp_params"]["pretrained"]:
        logger.info('Load pretrained model from {}'.format(params["exp_params"]["pretrained"]))
        checkpoint = torch.load(params["exp_params"]["pretrained"], map_location=lambda storage, loc: storage)
        state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        for p in model.modules():
            if hasattr(p, 'weight_org'):
                p.weight_org.copy_(p.weight.data.clamp_(-1,1))

    # convert models to cuda
    if params["optim_params"]['use_gpu']:
        model = model.cuda(params["optim_params"]['device'])
        classifier = classifier.cuda(params["optim_params"]['device'])

    criterion = torch.nn.CrossEntropyLoss()

    # intialize optimizer
    opt_params = [{"params": model.parameters()}, {"params": classifier.parameters(), "lr": params["optim_params"]['classifier_lr']}]
    lr = params["optim_params"]['lr']
    weight_decay = params["optim_params"]['weight_decay']
    if params['optim_params']['optimizer'] == 'adam':
        optimizer = optim.Adam(opt_params, lr=lr, weight_decay=weight_decay)
    elif params['optim_params']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(opt_params, lr=lr, weight_decay=weight_decay)
    elif params['optim_params']['optimizer'] == 'sgd':
        optimizer = optim.SGD(opt_params, lr=lr, momentum=params['optim_params']['momentum'], weight_decay=weight_decay)

    if model_type == "face" and params["data_params"]["small_dataset"]:
        dir_ext = "face-small"
    else:
        dir_ext = "face" if model_type=="face" else "utt"
    
    if model_type == "face":
        train_dataset = datasets.ImageFolder(
            os.path.join(params["data_params"]["train_dir"], dir_ext),
            transforms.Compose([
                transforms.RandomResizedCrop(params["data_params"]["input_dim"]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]))
    elif model_type == "speaker":
        train_dataset = datasets.DatasetFolder(
            os.path.join(params["data_params"]["train_dir"], "utt"),
            utt_path_to_utt,
            ("wav", "m4a")
        )
    train_loader = DataLoader(train_dataset, batch_size=params["data_params"]['batch_size'], shuffle=True,
                                num_workers=params["data_params"]['num_workers'])
    val_dataset = Vox1ValDataset(os.path.join(params["data_params"]["test_dir"], os.pardir),
        select_face=(model_type=="face"), select_audio=(model_type=="speaker"), dataset=params["data_params"]["val_dataset"], face_dim=params["data_params"]["input_dim"])
    val_loader = DataLoader(val_dataset, batch_size=params["data_params"]['batch_size'], shuffle=False,
                                num_workers=params["data_params"]['num_workers'])

    # initialize scheduler
    if params['optim_params']['scheduler'] == "":
        scheduler = None
    elif params['optim_params']['scheduler'] == 'Poly':
        scheduler = PolyScheduler(
            optimizer=optimizer,
            base_lr=params["optim_params"]['lr'],
            max_steps=params["optim_params"]['end_epoch'] * len(train_loader),
            warmup_steps=params["optim_params"]['warmup_epoch'] * len(train_loader),
            last_epoch=-1
        )
    elif params['optim_params']['scheduler'] == 'StepLR':
        scheduler = StepLR(optimizer, step_size=params['optim_params']['scheduler_step_size'] * len(train_loader), gamma=params['optim_params']['scheduler_gamma'])
    elif params['optim_params']['scheduler'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, threshold=5e-2)
    else:
        raise Exception ("Invalid scheduler choice {}".format(params['optim_params']['scheduler']))

    num_params = sum(param.numel() for param in model.parameters())
    logger.info('Number of parmeters:{}'.format(num_params))
    
    train(model, classifier, optimizer, criterion, scheduler, train_loader, val_loader, logger, log_dir, params)
    
if __name__ == '__main__':
    main()