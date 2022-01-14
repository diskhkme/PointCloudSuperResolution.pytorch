import torch
import model._bak.grouping_util_bak as gutil

def cd_loss(pred, gt, radius, alpha=1.0): # author proposed using alpha==1.0
    forward, _ = gutil.knn_point_dist(1, gt, pred) # for points in pred, get nearest point index in gt
    backward, _ = gutil.knn_point_dist(1, pred, gt)
    cd_dist = alpha*forward + (1.0-alpha)*backward
    cd_dist = torch.mean(cd_dist, dim=1)
    cd_dist_norm = cd_dist / radius
    cd_loss = torch.mean(cd_dist_norm, dim=0) # batch dim

    return cd_loss

def finetune_gen_loss(d_fake, lambd, pred, gt, radius, alpha=1.0):
    cd_loss_val = cd_loss(pred, gt, radius, alpha)
    g_loss = torch.mean(d_fake**2, dim=2)

    return g_loss.mean() + lambd * cd_loss_val

def finetune_disc_loss(d_real, d_fake):
    d_loss_real = torch.mean((d_real - 1)**2, dim=2)
    d_loss_fake = torch.mean(d_fake**2, dim=2)
    d_loss = 0.5 * (d_loss_real + d_loss_fake)
    d_loss = torch.mean(d_loss.squeeze(1))
    return d_loss

if __name__ == '__main__':
    pred = torch.rand((4,3,1024))
    gt = torch.rand((4,3,1024))

    loss = cd_loss(pred, gt, 1.0)
    print(loss.item())