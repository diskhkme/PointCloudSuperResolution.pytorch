import torch
import model.grouping_util as gutil
import pytorch3d.loss as p3_loss

def get_cd_loss(pred, gt, radius, alpha=1.0): # author proposed using alpha==1.0
    pred = pred.transpose(1,2)
    gt = gt.transpose(1,2)
    forward, _, _ = p3_loss.chamfer_distance(gt,pred) # default: mean/mean

    return forward


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
    gt = torch.rand(20, 3, 4096)
    pred = torch.rand(20, 3, 1024)
    val = get_cd_loss(pred, gt, 1.0, 1.0)

    print(val.shape)
