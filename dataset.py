import wave
import torch
import torchaudio.functional as F
from torch.utils.data import Dataset
import torchaudio
import torchvision.transforms as transforms
T = transforms
import torchaudio.transforms as T_a

from PIL import Image
import random
import os

import cv2
import numpy as np
import librosa
from pydub import AudioSegment

def image_path_to_image(image_path, dim=224):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((dim, dim)),
        transforms.ToTensor()
    ])
    image = transform(image)
    return image

def face_path_to_face(face_path, random_transform=False, dim=224):
    face = Image.open(face_path)
    if random_transform:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(dim),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((dim, dim)),
            transforms.ToTensor()
        ])
    face = transform(face)
    return face

"""
clip_len: desired length of audio clip in seconds
"""
def utt_path_to_utt(utt_path, clip_len=3):
    target_sr = 22050
    ext = utt_path.split(".")[-1]
    if ext == "wav":
        audio, sr = torchaudio.load(utt_path, normalize=True)
        audio = F.resample(
                    audio,
                    sr,
                    target_sr
                )
    """
    else:
        audio = AudioSegment.from_file(utt_path).resample(sample_rate_Hz=target_sr, sample_width=2, channels=1)
        #audio, sr = librosa.load(utt_path, sr=target_sr)
    """
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
    utt = torch.unsqueeze(torch.tensor(np.absolute(librosa.stft(audio.numpy()[0], hop_length=128, n_fft=1024))), 0)
    #utt = mel_spectrogram(audio)
    #transform = T_a.MelSpectrogram(sample_rate=target_sr)#, hop_length=256, n_fft=2048)
    #transform = T_a.Spectrogram(hop_length=256, n_fft=2048)
    #transform = T_a.MFCC(sample_rate=target_sr, n_mfcc=20)
    #utt = transform(audio)
    #utt /= 100
    utt = utt - utt.min() + 1e-3
    utt = torch.log(utt)
    utt = (utt-utt.min())/(utt.max()-utt.min())
    utt = utt.repeat(3, 1, 1)

    t = transforms.Resize((224, 224))
    utt = t(utt)

    return utt

# gets a random image (face or spectrogram)
def get_random_image(spk_face_dir, video_id, dim=224):
    spk_video_face_dir = os.path.join(spk_face_dir, video_id)
    image_path = os.path.join(spk_video_face_dir, random.choice(os.listdir(spk_video_face_dir)))
    image = image_path_to_image(image_path, dim=dim)
    return image

def get_random_audio(spk_utt_dir, video_id):
    spk_video_utt_dir = os.path.join(spk_utt_dir, video_id)
    
    utt_path = os.path.join(spk_video_utt_dir, random.choice(os.listdir(spk_video_utt_dir)))

    utt = utt_path_to_utt(utt_path)
    return utt

def get_random_spk_sample(spk, dataset_path, select_face=True, select_audio=True):
    spk_face_dir = os.path.join(dataset_path, "face", spk)
    spk_utt_dir = os.path.join(dataset_path, "utt-im", spk)

    video_id = None
    while video_id is None or \
        len(os.listdir(os.path.join(spk_face_dir, video_id)))==0 or \
        len(os.listdir(os.path.join(spk_utt_dir, video_id)))==0:
        video_id = random.choice(os.listdir(spk_face_dir))
    face = None
    utt = None

    if select_face:
        face = get_random_image(spk_face_dir, video_id)
    if select_audio:
        utt = get_random_image(spk_utt_dir, video_id)
    return face, utt

# VoxCeleb1
class Vox1ValDataset(Dataset):
    def __init__(self, vox1_dir, select_face=True, select_audio=True, dataset='vox1-o', random_transform=False, dim=224):
        super(Vox1ValDataset, self).__init__()
        self.vox1_dir = vox1_dir
        self.select_face = select_face
        self.select_audio = select_audio
        self.dataset = dataset
        self.random_transform = random_transform
        self.dim = dim
        print("Using VoxCeleb1 evaluation dataset {}".format(dataset))

        if dataset == "vox1-o":
            self.file = os.path.join(vox1_dir, "veri_test.txt")
        elif dataset == "vox1-h":
            self.file = os.path.join(vox1_dir, "list_test_hard.txt")
        elif dataset == "vox1-e":
            self.file = os.path.join(vox1_dir, "list_test_all.txt")
        else:
            raise Exception("Invalid validation dataset choice {}.".format(dataset))

        with open(self.file, 'r') as f:
            self.dataset = [line[:-1].split(" ") for line in f.readlines()]
            self.dataset = [[int(entry[0]), entry[1], entry[2]] for entry in self.dataset]
        self.length = len(self.dataset)

    def __getitem__(self, idx):
        label, utt_path1, utt_path2 = self.dataset[idx]
        
        spk1, spk1_vid, _ = utt_path1.split("/")
        spk2, spk2_vid, _ = utt_path2.split("/")

        if self.select_face:
            face1 = get_random_image(os.path.join(self.vox1_dir, "test", "face", spk1), spk1_vid, dim=self.dim)
            face2 = get_random_image(os.path.join(self.vox1_dir, "test", "face", spk2), spk2_vid, dim=self.dim)
        if self.select_audio:
            utt1 = get_random_image(os.path.join(self.vox1_dir, "test", "utt-im", spk1), spk1_vid, dim=self.dim)
            utt2 = get_random_image(os.path.join(self.vox1_dir, "test", "utt-im", spk2), spk2_vid, dim=self.dim)
        # if reading from raw audio files, use following lines
        """
        if self.select_audio:
            utt1 = utt_path_to_utt(os.path.join(self.vox1_dir, "test", "utt", utt_path1))
            utt2 = utt_path_to_utt(os.path.join(self.vox1_dir, "test", "utt", utt_path1))
        """

        if self.select_face and self.select_audio:
            return label, face1, utt1, face2, utt2
        elif self.select_face:
            return label, face1, face2
        elif self.select_audio:
            return label, utt1, utt2
        else:
            return None

    def __len__(self):
        return self.length

class MultimodalPairDataset(Dataset):
    def __init__(self, dataset_dir, length=64 * 10000, select_face=True, select_audio=True):
        super(MultimodalPairDataset, self).__init__()

        self.dataset_dir = dataset_dir
        self.face_dir = os.path.join(dataset_dir, "face")
        self.utt_dir = os.path.join(dataset_dir, "utt")
        self.length = length

        self.spk_list = list(set(os.listdir(self.face_dir)) & set(os.listdir(self.utt_dir)))
        self.spk_num = len(self.spk_list)

        self.select_face = select_face
        self.select_audio = select_audio

    def __getitem__(self, idx):
        choice = random.randint(0, 1)  # to generate positive or negative pair

        if choice == 0:  # negative pair
            spk1, spk2 = random.sample(self.spk_list, 2)
        else:  # positive pair
            spk1 = random.sample(self.spk_list, 1)[0]
            spk2 = spk1

        label = choice
        face1, utt1 = get_random_spk_sample(spk1, self.dataset_dir, select_face=self.select_face)
        face2, utt2 = get_random_spk_sample(spk2, self.dataset_dir, select_audio=self.select_audio)

        if self.select_face and self.select_audio:
            return label, face1, utt1, face2, utt2
        elif self.select_face:
            return label, face1, face2
        elif self.select_audio:
            return label, utt1, utt2
        else:
            return None

    def __len__(self):
        return self.length

class CombinedDataset(Dataset):
    def __init__(self, dataset_dir, length=5994*150):
        super(CombinedDataset, self).__init__()

        self.dataset_dir = dataset_dir
        self.face_dir = os.path.join(dataset_dir, "face")
        self.utt_dir = os.path.join(dataset_dir, "utt-im")

        self.spk_list = os.listdir(self.face_dir)
        self.spk_dict = dict((spk, id) for (id, spk) in enumerate(self.spk_list))
        self.spk_num = len(self.spk_list)

        self.length = length

    def __getitem__(self, idx):
        spk = random.choice(self.spk_list)
        id = self.spk_dict[spk]
        face, spectrogram = get_random_spk_sample(spk, self.dataset_dir, select_face=True, select_audio=True)

        return face, spectrogram, id

    def __len__(self):
        return self.length

class FaceDataset(Dataset):
    def __init__(self, face_dir, length=64 * 10000):
        super(FaceDataset, self).__init__()

        self.face_dir = face_dir
        self.length = length

        self.spk_list = os.listdir(face_dir)
        self.spk_dict = dict((spk, id) for (id, spk) in enumerate(self.spk_list))
        self.spk_num = len(self.spk_list)

    def __getitem__(self, idx):
        spk = random.choice(self.spk_list)
        id = self.spk_dict[spk]
        face, _ = get_random_spk_sample(spk, os.path.join(self.face_dir, os.pardir), select_face=True, select_audio=False)

        return face, id

    def __len__(self):
        return self.length