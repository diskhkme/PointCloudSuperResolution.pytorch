import torch
import model.grouping_util as gutil
# import pytorch3d.ops as ops

def get_cd_loss(pred, gt, radius, alpha=1.0): # author proposed using alpha==1.0
    forward = gutil.pairwise_dist(gt, pred) # for points in pred, get nearest point index in gt
    if alpha != 1.0:
        backward = gutil.pairwise_dist(pred, gt)
    else:
        backward = 0

    cd_dist = alpha*forward + (1.0-alpha)*backward
    cd_dist = torch.mean(cd_dist, dim=1)
    cd_dist_norm = cd_dist / radius
    cd_loss = torch.mean(cd_dist_norm, dim=0) # batch dim

    return cd_loss

def get_d_loss(d_real, d_fake):
    d_loss_real = torch.mean((d_real - 1)**2, dim=2)
    d_loss_fake = torch.mean(d_fake**2, dim=2)
    d_loss = 0.5 * (d_loss_real + d_loss_fake)
    d_loss = torch.mean(d_loss.squeeze(1))
    return d_loss

def get_g_loss(d_fake):
    g_loss = torch.mean(d_fake**2, dim=2)

    return g_loss.mean()


if __name__ == '__main__':
    gt = torch.tensor([[[1, 2,3], [4, 5,8], [3, 7,9]]], dtype=torch.float32) # 3차원 점 2개
    pred = torch.tensor([[[1.5, 2.3,2.5], [4.3, 5.2,7.2], [1.7, 2.3,10.5]]],dtype=torch.float32) # 3차원 점 2개
    # val = get_cd_loss(gt.transpose(1,2), pred.transpose(1,2), 1.0, 1.0)
    # print(val.item())
    print(pred[0,:,1])
    forward = gutil.pairwise_dist(gt, pred)
    print(forward)
    # val = get_cd_loss(pred, gt, 1.0, 1.0)
    # print(val.item())
    # val = get_cd_loss(pred.transpose(1,2), gt.transpose(1,2), 1.0, 1.0)
    # print(val.item())

    a = torch.tensor([2.0,5.0,7.0])
    b = torch.tensor([2.3,5.2,2.3])
    sub = (a-b)**2
    print(torch.sum(sub))
