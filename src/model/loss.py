import torch
import model.grouping_util as gutil

def cd_loss(pred, gt, radius, alpha=1.0): # author proposed using alpha==1.0
    forward, _ = gutil.knn_point(1, gt, pred) # for points in pred, get nearest point index in gt
    backward, _ = gutil.knn_point(1, pred, gt)
    cd_dist = alpha*forward + (1.0-alpha)*backward
    cd_dist = torch.mean(cd_dist, dim=1)
    cd_dist_norm = cd_dist / radius
    cd_loss = torch.mean(cd_dist_norm, dim=0) # batch dim

    return cd_loss

if __name__ == '__main__':
    pred = torch.rand((1,3,1024))
    gt = torch.rand((1,3,1024))

    loss = cd_loss(pred, gt, 1.0)
    print(loss.item())