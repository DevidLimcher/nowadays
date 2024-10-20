import torch
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = F.cosine_similarity(anchor, positive)
    neg_dist = F.cosine_similarity(anchor, negative)
    loss = F.relu(margin + neg_dist - pos_dist).mean()
    return loss
