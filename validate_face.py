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

def main():
    # *********************** process config ***********************
    conf_name = "face_val.conf"
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

    # models init
    params["optim_params"]['use_gpu'] = params["optim_params"]['use_gpu'] and torch.cuda.is_available()
    face_model = resnet.resnet18(num_classes=params["exp_params"]["emb_size"])

    # load pretrained model
    if params["exp_params"]["pretrained"]:
        state_dict = torch.load(params["exp_params"]["pretrained"], map_location=lambda storage, loc: storage)["state_dict"]
        face_model.load_state_dict(state_dict)
    else:
        raise Exception("Please specify pretrained checkpoint to load from in config file.")

    # convert models to cuda
    if params["optim_params"]['use_gpu']:
        face_model = face_model.cuda()
    face_val_dataset = MultimodalPairDataset(params["data_params"]["train_dir"], length=params["data_params"]['val_dataset_len'],
                                select_face=True, select_audio=False)
    face_val_loader = DataLoader(face_val_dataset, batch_size=params["data_params"]['batch_size'], shuffle=False,
                                num_workers=params["data_params"]['num_workers'])

    all_distances, all_labels, fprs, tprs, thresholds, eer = evaluate_single_modality(face_model, face_val_loader, params)
    print(eer)
    
if __name__ == '__main__':
    main()