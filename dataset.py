import wave
import torch
import torchaudio.functional as F
from torch.utils.data import Dataset
import torchaudio
import torchvision.transforms as transforms
T = transforms
import torchaudio.transforms as T_a

import random
import os

import cv2
import numpy as np
import librosa

dataset_path = "/home/sravit/datasets/VoxCeleb-multimodal"
train_path = "/home/sravit/datasets/VoxCeleb-multimodal/VoxCeleb1/test"
face_dir = os.path.join(train_path, "face")
utt_dir = os.path.join(train_path, "utt")

# ******* FACE UTILS *******
def motion_blur(img, kernel_size=15, vertical=True):
    '''
    reference: https://www.geeksforgeeks.org/opencv-motion-blur-in-python/
    :param img: cv2 image
    :param kernel_size: the blur kernel size
    :param vertical: do vertical blur or horizontal blur
    :return: motion blured img
    '''

    # Create the kernel.
    kernel = np.zeros((kernel_size, kernel_size))

    # Fill the middle row with ones.
    if vertical:
        kernel[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    else:
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)

    # Normalize.
    kernel /= kernel_size

    # Apply the vertical kernel.
    mb_img = cv2.filter2D(img, -1, kernel)

    return mb_img

def gaussian_blur(img, kernel_size=7, sigma=3):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

def blur_image(cv2_img, blur_type):
    if blur_type == 0:
        return cv2_img
    elif blur_type == 1:
        blur_img = gaussian_blur(cv2_img, kernel_size=7, sigma=3)
    elif blur_type == 2:
        blur_img = motion_blur(cv2_img, kernel_size=20, vertical=True)
    else:
        blur_img = motion_blur(cv2_img, kernel_size=25, vertical=False)
    return blur_img
# ******* FACE UTILS *******

def face_path_to_face(face_path):
    face = cv2.imread(face_path)
    face = blur_image(face, random.randint(0, 3))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    
    #making image square through padding
    height, width, _ = face.shape
    max_dim = np.max([height, width])
    face = cv2.copyMakeBorder(face, max_dim-height, 0, max_dim-width, 0, cv2.BORDER_CONSTANT, 0)

    #resizing to 224x224
    mean = np.mean(face)
    std = np.std(face)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std, inplace=True),
        transforms.Resize((224, 224)),
    ])
    face = transform(face)
    return face

"""
clip_len: desired length of audio clip in seconds
"""
def utt_path_to_utt(utt_path, clip_len=3):
    #audio, sr = librosa.load(utt_path, sr=22050, mono=True)
    audio, sr = torchaudio.load(utt_path)
    target_sr = 8000
    audio = F.resample(
                audio,
                sr,
                target_sr
            )
    waveform_len = target_sr*clip_len
    
    # Removing silence (src: https://www.tutorialexample.com/python-remove-silence-in-wav-using-librosa-librosa-tutorial/)
    """
    clips = librosa.effects.split(audio, top_db=10)
    audio = []
    for c in clips:
        print(c)
        data = audio[c[0]: c[1]]
        audio.extend(data)
    """
    
    # Fixing length of audio clip
    audio_len = audio.shape[1]
    if audio_len <= waveform_len:
        #audio += [0]*(waveform_len-len(audio))
        audio = torch.nn.functional.pad(audio, ((0, 0), (0, waveform_len-audio_len)))
    else:
        start = random.randint(0, audio_len-waveform_len)
        audio = audio[:, start:start+waveform_len]

    assert audio.shape[1] == waveform_len

    #utt = librosa.feature.mfcc(audio, sr=sr, n_mfcc=40)
    #utt = librosa.feature.melspectrogram(audio, sr=sr, hop_length=256, n_fft=2048)
    mel_spectrogram = T_a.MelSpectrogram(sample_rate=sr, hop_length=256, n_fft=2048)
    #utt = mel_spectrogram(audio)
    #mean = np.mean(utt)
    #std = np.std(utt)
    transform = transforms.Compose([
        T_a.MelSpectrogram(sample_rate=sr, hop_length=256, n_fft=2048),
        #transforms.ToTensor(),
        #transforms.Normalize(mean, std, inplace=True)
    ])
    utt = transform(audio)
    return utt

def get_random_face_audio(spk):
    spk_face_dir = os.path.join(train_path, "face", spk)
    spk_utt_dir = os.path.join(train_path, "utt", spk)

    video_id = random.choice(os.listdir(spk_face_dir))
    spk_video_face_dir = os.path.join(spk_face_dir, video_id)
    spk_video_utt_dir = os.path.join(spk_utt_dir, video_id)
    
    face_path = os.path.join(spk_video_face_dir, random.choice(os.listdir(spk_video_face_dir)))
    utt_path = os.path.join(spk_video_utt_dir, random.choice(os.listdir(spk_video_utt_dir)))

    face = face_path_to_face(face_path)
    utt = utt_path_to_utt(utt_path)
    return face, utt

class MultimodalContrastiveDataset(Dataset):
    def __init__(self, face_dir=face_dir, utt_dir=utt_dir, length=64 * 10000):
        super(MultimodalContrastiveDataset, self).__init__()

        self.face_dir = face_dir
        self.utt_dir = utt_dir
        self.length = length

        self.spk_list = list(set(os.listdir(face_dir)) & set(os.listdir(utt_dir)))
        self.spk_num = len(self.spk_list)

        self.counter = 0

    def __getitem__(self, idx):
        choice = random.randint(0, 1)  # to generate positive or negative pair

        if choice == 0:  # negative pair
            spk1, spk2 = random.sample(self.spk_list, 2)
        else:  # positive pair
            spk1 = random.sample(self.spk_list, 1)[0]
            spk2 = spk1

        label = choice
        face1, utt1 = get_random_face_audio(spk1)
        face2, utt2 = get_random_face_audio(spk2)

        self.counter += 1
        return label, face1, utt1, face2, utt2

    def __len__(self):
        return self.length