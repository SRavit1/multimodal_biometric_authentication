import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve
from tqdm import tqdm
from matplotlib import pyplot as plt

# Adapted from https://github.com/VITA-Group/AutoSpeech/blob/master/utils.py
def compute_eer(distances, labels):
    # Calculate evaluation metrics
    fprs, tprs, thresholds = roc_curve(labels, distances)
    eer = fprs[np.nanargmin(np.absolute((1 - tprs) - fprs))]
    return fprs*100, tprs*100, thresholds, eer*100

def compute_threshold(distances, labels):
    fprs, tprs, thresholds = roc_curve(labels, distances)
    return thresholds[np.nanargmin(np.absolute((1 - tprs) - fprs))]

def normalize_embedding(x):
    x_norm = torch.sqrt(torch.sum(torch.mul(x,x), dim=1))  #torch.linalg.norm(x)
    x_norm = torch.unsqueeze(x_norm, 1)
    x = torch.div(x, x_norm)
    return x

def evaluate_single_modality(model, val_loader, params, log_dir, calc_agreement=False):
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

    if calc_agreement:
        threshold = -1*compute_threshold(-1.*all_distances, all_labels)

        tables_dir = os.path.join(log_dir, "tables")
        if not os.path.exists(tables_dir):
            os.mkdir(tables_dir)
        evaluate_confusion_single_modality(model, val_loader, params, threshold, tables_dir)

    return all_distances, all_labels, fprs, tprs, thresholds, eer

def evaluate_multimodal(model, val_loader, params, log_dir, calc_agreement=False):
    model.eval()

    face_all_distances = None
    speaker_all_distances = None
    all_distances = None
    all_labels = None
    for i, (labels, face1, utt1, face2, utt2) in tqdm(enumerate(val_loader), position=1, desc='evaluation'):
        if params["optim_params"]['use_gpu']:
            face1 = face1.cuda(params["optim_params"]['device'])
            face2 = face2.cuda(params["optim_params"]['device'])
            spectrogram1 = utt1.cuda(params["optim_params"]['device'])
            spectrogram2 = utt2.cuda(params["optim_params"]['device'])
        with torch.no_grad():
            face_embeddings1, speaker_embeddings1, embeddings1 = model(face1, spectrogram1)
            face_embeddings2, speaker_embeddings2, embeddings2 = model(face2, spectrogram2)

        face_distances = torch.sqrt(torch.sum(torch.pow(face_embeddings2-face_embeddings1, 2), dim=1)).cpu().numpy()
        speaker_distances = torch.sqrt(torch.sum(torch.pow(speaker_embeddings2-speaker_embeddings1, 2), dim=1)).cpu().numpy()
        distances = torch.sqrt(torch.sum(torch.pow(embeddings2-embeddings1, 2), dim=1)).cpu().numpy()
        labels = labels.numpy()
        
        if all_distances is not None:
            face_all_distances = np.concatenate((face_all_distances, face_distances))
            speaker_all_distances = np.concatenate((speaker_all_distances, speaker_distances))
            all_distances = np.concatenate((all_distances, distances))
            all_labels = np.concatenate((all_labels, labels))
        else:
            face_all_distances = face_distances
            speaker_all_distances = speaker_distances
            all_distances = distances
            all_labels = labels

    # compute_eer expects array of similarity values
    # because we have distance values, we negate all_distances
    assert np.min(face_all_distances) >= 0 and np.max(face_all_distances) <= 2
    assert np.min(speaker_all_distances) >= 0 and np.max(speaker_all_distances) <= 2
    assert np.min(all_distances) >= 0 and np.max(all_distances) <= 2
    
    fprs, tprs, thresholds, eer = compute_eer(-1.*all_distances, all_labels)
    thresholds *= -1.
    assert np.max(thresholds) >= 0 and np.max(thresholds) <= 2
    
    if calc_agreement:
        face_threshold = -1*compute_threshold(-1.*face_all_distances, all_labels)
        speaker_threshold = -1*compute_threshold(-1.*speaker_all_distances, all_labels)
        combined_threshold = -1*compute_threshold(-1.*all_distances, all_labels)

        tables_dir = os.path.join(log_dir, "tables")
        if not os.path.exists(tables_dir):
            os.mkdir(tables_dir)
        print(face_threshold, speaker_threshold, combined_threshold)
        evaluate_confusion_multimodal(model, val_loader, params, face_threshold, speaker_threshold, combined_threshold, tables_dir)
    
    return all_distances, all_labels, fprs, tprs, thresholds, eer

def evaluate_confusion_multimodal(model, val_loader, params, face_thresh, speaker_thresh, combined_thresh, tables_dir):
    # array of (face_pred, speaker_pred, combined_pred, label)
    results = [np.array([]), np.array([]), np.array([]), np.array([])]
    
    for i, (labels, face1, utt1, face2, utt2) in tqdm(enumerate(val_loader), position=1, desc='evaluation'):
        if params["optim_params"]['use_gpu']:
            face1 = face1.cuda(params["optim_params"]['device'])
            face2 = face2.cuda(params["optim_params"]['device'])
            spectrogram1 = utt1.cuda(params["optim_params"]['device'])
            spectrogram2 = utt2.cuda(params["optim_params"]['device'])
        with torch.no_grad():
            face_embeddings1, speaker_embeddings1, embeddings1 = model(face1, spectrogram1)
            face_embeddings2, speaker_embeddings2, embeddings2 = model(face2, spectrogram2)
        
        face_distances = torch.sqrt(torch.sum(torch.pow(face_embeddings2-face_embeddings1, 2), dim=1)).cpu().numpy()
        speaker_distances = torch.sqrt(torch.sum(torch.pow(speaker_embeddings2-speaker_embeddings1, 2), dim=1)).cpu().numpy()
        distances = torch.sqrt(torch.sum(torch.pow(embeddings2-embeddings1, 2), dim=1)).cpu().numpy()
        labels = labels.numpy()

        results[0] = np.append(results[0], face_distances < face_thresh)
        results[1] = np.append(results[1], speaker_distances < speaker_thresh)
        results[2] = np.append(results[2], distances < combined_thresh)
        results[3] = np.append(results[3], labels)

    agreement_matrix = np.zeros((4, 4))
    for r in range(len(results[0])):
        for i in range(4):
            for j in range(i, 4):
                if results[i][r] == results[j][r]:
                    agreement_matrix[i][j] += 1
                    if i < j:
                        agreement_matrix[j][i] += 1
    agreement_matrix /= agreement_matrix[0][0]

    fig = plt.gcf()
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.box(on=None)

    plt.table(agreement_matrix, rowLabels=["Face", "Speaker", "Fusion", "Label"], colLabels=["Face", "Speaker", "Fusion", "Label"], loc='center')
    plt.title("Agreement Percent Between Classes")
    plt.savefig(os.path.join(tables_dir, "agreement.png"))

    return agreement_matrix

def evaluate_confusion_single_modality():
    # TODO
    pass