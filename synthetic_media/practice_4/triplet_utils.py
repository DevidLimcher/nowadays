import torch
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()

def cosine_similarity(a, b):
    """
    Calculate the cosine similarity between two tensors.
    Args:
        a (torch.Tensor): The first tensor.
        b (torch.Tensor): The second tensor.
    Returns:
        torch.Tensor: The cosine similarity between the two tensors.
    """
    cos_sim = F.cosine_similarity(a, b)
    return cos_sim.mean()
