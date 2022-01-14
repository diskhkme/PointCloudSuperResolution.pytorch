import torch
import model.grouping_util_simple as gutil

def cd_loss(pred, gt, radius, alpha=1.0): # author proposed using alpha==1.0
    forward = gutil.pairwise_dist(pred, gt) # for points in pred, get nearest point index in gt
    cd_dist = alpha*forward
    cd_dist = torch.mean(cd_dist, dim=1)
    cd_dist_norm = cd_dist / radius
    cd_loss = torch.mean(cd_dist_norm, dim=0) # batch dim

    return cd_loss