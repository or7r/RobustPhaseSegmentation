import torch
import torch.nn.functional as F


def get_distance_func(distance):
    if distance == "euclidean":
        distance = torch.nn.PairwiseDistance(p=2, eps=0)
    elif distance == "cosine":
        # distance = torch.nn.CosineSimilarity(dim=1)
        def distance(x1, x2): return 1 - F.cosine_similarity(x1, x2, 1, 0)
    else:
        raise ValueError("Distance metric not supported")

    return distance
