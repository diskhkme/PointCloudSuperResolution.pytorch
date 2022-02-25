import torch
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# TODO: consider https://github.com/krrish94/chamferdist

def pairwise_dist(gt, pred):
    gt = gt.transpose(1, 2)
    pred = pred.transpose(1, 2)

    r_xyz1 = torch.sum(gt * gt, dim=2, keepdim=True)  # (B,N,1)
    r_xyz2 = torch.sum(pred * pred, dim=2, keepdim=True)  # (B,M,1)
    mul = torch.matmul(pred, gt.permute(0, 2, 1))         # (B,M,N) (matmul (b,m,1)x(b,1,n)
    dist, _ = torch.min(r_xyz2 - 2 * mul + r_xyz1.permute(0,2,1),dim=1)       # (B,M)
    return dist # using squared dist, based on original impl

    # dist, _ = knn_point(1, gt, pred)
    # return dist

def knn_point(k, xyz1, xyz2):
    b1, d1, n1 = xyz1.shape
    xyz1 = xyz1.view(b1, d1, 1, n1)
    b2, d2, n2 = xyz2.shape
    xyz2 = xyz2.view(b2, d2, n2, 1)
    dist = torch.sum((xyz1 - xyz2) ** 2, 1)

    val, idx = torch.topk(-dist, k=k, dim=2)
    idx = idx.transpose(1, 2).contiguous()

    return -val, idx  # TODO: remove sqrt

def group_point(x, idx):
    k = idx.size(1)
    batch_size = x.size(0)
    num_input_points = x.size(2)
    num_query_points = idx.size(2)
    x = x.view(batch_size, -1, num_input_points)

    idx_base = torch.arange(0, batch_size, requires_grad=False).view(-1, 1, 1).to(x.device) * num_input_points
    idx_unpacked = idx + idx_base
    idx_unpacked = idx_unpacked.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_input_points, -1)[idx_unpacked, :]
    feature = feature.view(batch_size, num_query_points, k, num_dims)
    feature = feature.transpose(1,3) # (batch_size, num_dim, k, num_points)

    return feature

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

    _, idx = knn_point(k, xyz, new_xyz)
    new_points, _ = torch.max(group_point(points, idx),dim=2)

    return new_xyz, new_points

if __name__ == '__main__':
    # x = torch.rand((24, 3, 4096)).to(torch.device('cuda'))
    # y = torch.rand((24, 3, 4096)).to(torch.device('cuda'))
    #
    # new_xyz, new_points = pool(x,y,8,1024)
    # ind = knn_point(8,x,x)


    gt = torch.rand(20,3,4096)
    pred = torch.rand(20, 3, 1024)
    val = pairwise_dist(gt,pred)

    print(val.item())


