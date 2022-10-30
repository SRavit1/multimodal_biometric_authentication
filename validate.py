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

def main():
    # *********************** process config ***********************
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--conf", type=str, default="face_val_xnor.conf", help="config file to use")
    args = parser.parse_args()
    config_path = os.path.join('/home/sravit/multimodal/multimodal_biometric_authentication/conf', args.conf)
    
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

    # models init
    input_channels = 3 if model_type == "face" else 1
    model = resnet.resnet18(num_classes=params["exp_params"]["emb_size"],
        prec_config=params["exp_params"]["prec_config"], input_channels=input_channels,
        normalize_output=params["exp_params"]["normalize_output"])

    # load pretrained model
    if params["exp_params"]["pretrained"]:
        checkpoint = torch.load(params["exp_params"]["pretrained"], map_location=lambda storage, loc: storage)
        state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        for p in model.modules():
            if hasattr(p, 'weight_org'):
                p.weight_org.copy_(p.weight.data.clamp_(-1,1))

    # convert models to cuda
    params["optim_params"]['use_gpu'] = params["optim_params"]['use_gpu'] and torch.cuda.is_available()
    if params["optim_params"]['use_gpu']:
        model = model.cuda(params["optim_params"]['device'])
    
    val_dataset = Vox1ValDataset(os.path.join(params["data_params"]["test_dir"], os.pardir),
        select_face=(model_type=="face"), select_audio=(model_type=="speaker"), dataset=params["data_params"]["val_dataset"], face_dim=params["data_params"]["input_dim"])
    val_loader = DataLoader(val_dataset, batch_size=params["data_params"]['batch_size'], shuffle=False,
                                num_workers=params["data_params"]['num_workers'])
    num_params = sum(param.numel() for param in model.parameters())

    all_distances, all_labels, fprs, tprs, thresholds, eer = evaluate_single_modality(model, val_loader, params)
    print("Validation EER on {} dataset: {:.2f}".format(params["data_params"]["val_dataset"], eer))
    
if __name__ == '__main__':
    main()