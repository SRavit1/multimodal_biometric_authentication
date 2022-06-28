import math

import torch
import torch.nn.functional as F

def select_threshold(score, ratio):
    '''
    :param score: The cosine score between vectors, shape:[batch_size]
    :param ratio: The score ratio above the threshold
    :return: The selected threshold
    '''
    score_detach = score.detach()
    length = score.shape[0]

    # # ------------------------ original version ------------------------
    # threshold = -1.0
    # score_descend = torch.sort(score_detach, descending=True)[0]

    # for i in range(length):
    #     if (i+1) / length > ratio:
    #         threshold = score_descend[i]
    #         break
    # index = score_detach > threshold
    # # ------------------------ original version ------------------------

    # --------------------------- new version --------------------------
    score_descend, index = torch.sort(score_detach, descending=True)
    select_len = math.ceil(ratio * length)
    threshold = score_descend[select_len - 1]
    index = index[:select_len]
    # --------------------------- new version --------------------------

    return threshold, index

def loss_calculator(input1, input2, label, ratio):
    pos_input1 = input1[label == 1]
    pos_input2 = input2[label == 1]
    neg_input1 = input1[label == 0]
    neg_input2 = input2[label == 0]

    pos_score = F.cosine_similarity(pos_input1, pos_input2) * (-1.0)
    neg_score = F.cosine_similarity(neg_input1, neg_input2)
    total_loss = torch.mean(neg_score) + torch.mean(pos_score) + 2.0

    pos_threshold, pos_index = select_threshold(pos_score, ratio)
    neg_threshold, neg_index = select_threshold(neg_score, ratio)

    loss = torch.mean(neg_score[neg_index]) + torch.mean(pos_score[pos_index]) + 2.0  # in [0.0, 4.0]

    return loss, total_loss, pos_threshold, neg_threshold