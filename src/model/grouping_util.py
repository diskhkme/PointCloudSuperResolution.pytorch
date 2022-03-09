import torch
import os
import pytorch3d.ops as ops

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def knn_point(k, xyz1, xyz2):
    xyz1 = xyz1.transpose(1, 2)
    xyz2 = xyz2.transpose(1, 2)
    dist, idx, _ = ops.knn_points(xyz1, xyz2, K=k)
    return dist.transpose(1,2), idx.transpose(1,2)

def group_point(x, idx):
    x = x.transpose(1,2)
    idx = idx.transpose(1,2)
    # print(x.shape, idx.shape)
    feature = ops.knn_gather(x, idx)
    return feature.transpose(1,3)

def group(xyz, points, k):
    _, idx = knn_point(k+1, xyz, xyz)
    idx = idx[:,1:,:] # exclude self matching
    grouped_xyz = group_point(xyz, idx) # (batch_size, num_dim, k, num_points)
    b,d,n = xyz.shape
    grouped_xyz -= xyz.unsqueeze(2).expand(-1,-1,k,-1) # translation normalization, (batch_size, num_points, k, num_dim(3))

    if points is not None:
        grouped_points = group_point(points, idx)
    else:
        grouped_points = grouped_xyz

    return grouped_xyz, grouped_points, idx


def farthest_point_sample(xyz, npoint):
    xyz = xyz.transpose(1, 2)
    points, inds = ops.sample_farthest_points(xyz,K=npoint)
    return points

def pool(xyz, points, k, npoint):

    new_xyz = farthest_point_sample(xyz, npoint)
    new_xyz = new_xyz.transpose(1,2)

    _, idx = knn_point(k, new_xyz, xyz)
    new_points, _ = torch.max(group_point(points, idx),dim=2)

    return new_xyz, new_points



