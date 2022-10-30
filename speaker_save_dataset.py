import torchaudio.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import dataset
import torch
from torch.utils.data import DataLoader
import os
from pathlib import Path
import cv2

train_dataset_path = "/home/sravit/datasets/VoxCeleb-multimodal/VoxCeleb2/dev/"
val_dataset_path = "/home/sravit/datasets/VoxCeleb-multimodal/VoxCeleb1/test/"

# adapted from dataset.py: utt_path_to_utt
def save_image(utt_path, clip_len=3):
    new_path = utt_path.replace("/utt/", "/utt-im/")
    new_path = new_path.replace(".wav", ".png")
    """
    if os.path.exists(new_path) and not os.path.getsize(new_path) == 0:
        return np.zeros((224, 224, 3))
    """
    new_pardir = Path(new_path).parent.absolute()
    """
    if not os.path.isdir(new_pardir):
        os.makedirs(new_pardir)
    """
    # Hopefully the following statement is thread-safe
    Path(new_pardir).mkdir(parents=True, exist_ok=True)

    utt = dataset.utt_path_to_utt(utt_path)
    utt = np.transpose((utt*255.).numpy(), (1, 2, 0))
    cv2.imwrite(new_path, utt)

    return utt

train_dataset = datasets.DatasetFolder(
    os.path.join(train_dataset_path, "utt"),
    transforms.Compose([
        save_image
    ]),
    ("wav", "m4a")
)
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=128)

val_dataset = datasets.DatasetFolder(
    os.path.join(val_dataset_path, "utt"),
    transforms.Compose([
        save_image
    ]),
    ("wav", "m4a")
)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=128)

"""
for i, _ in enumerate(val_loader):
    if i % 100 == 0:
        print("Val loader batch", i)
"""
for i, _ in enumerate(train_loader):
    if i % 100 == 0:
        print("Train loader batch", i)