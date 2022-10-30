import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve
from tqdm import tqdm

# Adapted from https://github.com/VITA-Group/AutoSpeech/blob/master/utils.py
def compute_eer(distances, labels):
    # Calculate evaluation metrics
    fprs, tprs, thresholds = roc_curve(labels, distances)
    eer = fprs[np.nanargmin(np.absolute((1 - tprs) - fprs))]
    return fprs*100, tprs*100, thresholds, eer*100

def normalize_embedding(x):
    x_norm = torch.sqrt(torch.sum(torch.mul(x,x), dim=1))  #torch.linalg.norm(x)
    x_norm = torch.unsqueeze(x_norm, 1)
    x = torch.div(x, x_norm)
    return x

def evaluate_single_modality(model, val_loader, params):
    model.eval()
    all_distances = None
    all_labels = None
    for i, (labels, inputs1, inputs2) in tqdm(enumerate(val_loader), position=1, desc='evaluation'):
        if params["optim_params"]['use_gpu']:
            inputs1 = inputs1.cuda(params["optim_params"]['device'])
            inputs2 = inputs2.cuda(params["optim_params"]['device'])
        with torch.no_grad():
            embeddings1 = normalize_embedding(model(inputs1))
            embeddings2 = normalize_embedding(model(inputs2))
        #distances = -1.*F.cosine_similarity(embeddings1, embeddings2).cpu().numpy()

        distances = torch.sqrt(torch.sum(torch.pow(embeddings2-embeddings1, 2), dim=1)).cpu().numpy()
        labels = labels.numpy()

        if all_distances is not None:
            all_distances = np.concatenate((all_distances, distances))
            all_labels = np.concatenate((all_labels, labels))
        else:
            all_distances = distances
            all_labels = labels

    # compute_eer expects array of similarity values
    # because we have distance values, we negate all_distances
    assert np.min(all_distances) >= 0 and np.min(all_distances) <= 2
    assert np.max(all_distances) >= 0 and np.max(all_distances) <= 2
    fprs, tprs, thresholds, eer = compute_eer(-1.*all_distances, all_labels)
    thresholds *= -1.
    assert np.max(thresholds) >= 0 and np.max(thresholds) <= 2
    return all_distances, all_labels, fprs, tprs, thresholds, eer

def evaluate_multimodal(model, val_loader, params):
    model.eval()
    all_distances = None
    all_labels = None
    for i, (labels, face1, utt1, face2, utt2) in tqdm(enumerate(val_loader), position=1, desc='evaluation'):
        #TODO: Remove
        if i==50:
            break
        
        if params["optim_params"]['use_gpu']:
            face1 = face1.cuda(params["optim_params"]['device'])
            face2 = face2.cuda(params["optim_params"]['device'])
            spectrogram1 = utt1.cuda(params["optim_params"]['device'])
            spectrogram2 = utt2.cuda(params["optim_params"]['device'])
        with torch.no_grad():
            embeddings1 = model(face1, spectrogram1)
            embeddings2 = model(face2, spectrogram2)

        distances = torch.sqrt(torch.sum(torch.pow(embeddings2-embeddings1, 2), dim=1)).cpu().numpy()
        labels = labels.numpy()

        if all_distances is not None:
            all_distances = np.concatenate((all_distances, distances))
            all_labels = np.concatenate((all_labels, labels))
        else:
            all_distances = distances
            all_labels = labels

    # compute_eer expects array of similarity values
    # because we have distance values, we negate all_distances
    assert np.min(all_distances) >= 0 and np.min(all_distances) <= 2
    assert np.max(all_distances) >= 0 and np.max(all_distances) <= 2
    fprs, tprs, thresholds, eer = compute_eer(-1.*all_distances, all_labels)
    thresholds *= -1.
    assert np.max(thresholds) >= 0 and np.max(thresholds) <= 2
    return all_distances, all_labels, fprs, tprs, thresholds, eer