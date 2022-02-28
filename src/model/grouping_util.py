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

# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def pool(xyz, points, k, npoint):

    tr_xyz = xyz.transpose(1,2)
    fps_idx = farthest_point_sample(tr_xyz, npoint)
    new_xyz = index_points(tr_xyz, fps_idx).transpose(1,2)

    _, idx = knn_point(k, new_xyz, xyz)
    new_points, _ = torch.max(group_point(points, idx),dim=2)

    return new_xyz, new_points



