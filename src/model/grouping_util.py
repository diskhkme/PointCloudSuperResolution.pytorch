import torch
import os
from torch_geometric.nn import fps, knn

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def knn_point(k, xyz1, xyz2):
    b1, d1, n1 = xyz1.shape # input
    b2, d2, n2 = xyz2.shape # query

    xyz1_tr = xyz1.transpose(1,2).contiguous().view(b1*n1, d1)
    xyz2_tr = xyz2.transpose(1,2).contiguous().view(b2*n2, d2)
    batch1 = torch.arange(0, b1, requires_grad=False).repeat_interleave(n1).to(xyz1.device)
    batch2 = torch.arange(0, b2, requires_grad=False).repeat_interleave(n2).to(xyz2.device)

    ind = knn(xyz1_tr, xyz2_tr, k, batch_x=batch1, batch_y=batch2) # (2, b2*n2*k)
    ind_offset = torch.arange(0, b2, requires_grad=False).repeat_interleave(n2*k).to(xyz2.device) * n2
    ret_ind = (ind[1, :]-ind_offset).view(b2, k, n2).contiguous()

    return ret_ind

def knn_point_dist(k, xyz1, xyz2):
    b1, d1, n1 = xyz1.shape
    xyz1 = xyz1.view(b1, d1, 1, n1)
    b2, d2, n2 = xyz2.shape
    xyz2 = xyz2.view(b2, d2, n2, 1)
    dist = torch.sum((xyz1-xyz2) ** 2, 1)

    val, idx = torch.topk(-dist, k=k, dim=2)
    idx = idx.transpose(1,2).contiguous()

    return torch.sqrt(-val), idx # TODO: remove sqrt

def knn_old(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def group_point(x, idx, device):
    # TODO: Device argument can be omiited by match idx_base with (x.device)
    k = idx.size(1)
    batch_size = x.size(0)
    num_input_points = x.size(2)
    num_query_points = idx.size(2)
    x = x.view(batch_size, -1, num_input_points)

    idx_base = torch.arange(0, batch_size, requires_grad=False).view(-1, 1, 1).to(device) * num_input_points
    idx_unpacked = idx + idx_base
    idx_unpacked = idx_unpacked.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_input_points, -1)[idx_unpacked, :]
    feature = feature.view(batch_size, num_query_points, k, num_dims)
    feature = feature.transpose(1,3) # (batch_size, num_dim, k, num_points)

    return feature

def group(xyz, points, k, device):
    idx = knn_point(k, xyz, xyz)
    grouped_xyz = group_point(xyz, idx, device) # (batch_size, num_dim, k, num_points)
    b,d,n = xyz.shape
    grouped_xyz -= xyz.view(b, d, 1, n).repeat(1, 1, k, 1) # translation normalization, (batch_size, num_points, k, num_dim(3))

    if points is not None:
        grouped_points = group_point(points, idx, device)
    else:
        grouped_points = grouped_xyz

    return grouped_xyz, grouped_points, idx

def pool(xyz, points, k, npoint):
    b, d, n = xyz.shape
    tr_xyz = xyz.transpose(1,2).contiguous().view(b*n, d)
    batch = torch.arange(0, b).repeat_interleave(n).to(xyz.device)

    sample_ratio = torch.tensor([npoint / n]).to(xyz.device)
    index = fps(tr_xyz, batch=batch, ratio=sample_ratio)
    tr_xyz_points = tr_xyz[index,:] # (b*npoint, d)
    new_xyz = tr_xyz_points.view(b,npoint,d)
    new_xyz = new_xyz.transpose(1,2).contiguous() # pooled points

    _, idx = knn_point_dist(k, xyz, new_xyz)
    new_points, _ = torch.max(group_point(points, idx, xyz.device),dim=2)

    return new_xyz, new_points


if __name__ == '__main__':
    x = torch.rand((2, 3, 1024)).to(torch.device('cuda'))
    y = torch.rand((2, 128, 1024)).to(torch.device('cuda'))

    # knn(x,8)
    # pool_ind = pool(x,y,8,512)
    ind = knn_point(8,x,x)
    print(ind)
    ind = knn_point_dist(8,x,x)

    print(ind)