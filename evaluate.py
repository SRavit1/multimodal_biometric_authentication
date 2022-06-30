import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve
from tqdm import tqdm

# Adapted from https://github.com/VITA-Group/AutoSpeech/blob/master/utils.py
def compute_eer(distances, labels):
    # Calculate evaluation metrics
    fprs, tprs, _ = roc_curve(labels, distances)
    eer = fprs[np.nanargmin(np.absolute((1 - tprs) - fprs))]
    return eer

def evaluate_single_modality(model, test_loader, params):
    all_distances = None
    all_labels = None
    for labels, inputs1, inputs2 in tqdm(test_loader, position=1, desc='evaluation'):
        if params["optim_params"]['use_gpu']:
            inputs1 = inputs1.cuda()
            inputs2 = inputs2.cuda()
        with torch.no_grad():
            embeddings1 = model(inputs1)
            embeddings2 = model(inputs2)
        distances = F.cosine_similarity(embeddings1, embeddings2).cpu().numpy()
        labels = labels.numpy()

        if all_distances is not None:
            all_distances = np.concatenate((all_distances, distances))
            all_labels = np.concatenate((all_labels, labels))
        else:
            all_distances = distances
            all_labels = labels

    eer = compute_eer(all_distances, all_labels)
    return eer

def evaluate_multimodal():
    return