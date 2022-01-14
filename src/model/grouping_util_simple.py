import torch
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# TODO: consider https://github.com/krrish94/chamferdist

def pairwise_dist(xyz1, xyz2):
    xyz1 = xyz1.transpose(1,2)
    xyz2 = xyz2.transpose(1, 2)

    r_xyz1 = torch.sum(xyz1 * xyz1, dim=2, keepdim=True)  # (B,N,1)
    r_xyz2 = torch.sum(xyz2 * xyz2, dim=2, keepdim=True)  # (B,M,1)
    mul = torch.matmul(xyz2, xyz1.permute(0,2,1))         # (B,M,N) (matmul (b,m,1)x(b,1,n)
    dist, _ = torch.min(r_xyz2 - 2 * mul + r_xyz1.permute(0,2,1),dim=1)       # (B,M,N)
    return dist


def knn_point(k, xyz1, xyz2):
    b1, d1, n1 = xyz1.shape
    xyz1 = xyz1.view(b1, d1, 1, n1)
    b2, d2, n2 = xyz2.shape
    xyz2 = xyz2.view(b2, d2, n2, 1)
    dist = torch.sum((xyz1 - xyz2) ** 2, 1)

    val, idx = torch.topk(-dist, k=k, dim=2)
    idx = idx.transpose(1, 2).contiguous()

    return torch.sqrt(-val), idx  # TODO: remove sqrt

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
    _, idx = knn_point(k, xyz, xyz)
    grouped_xyz = group_point(xyz, idx) # (batch_size, num_dim, k, num_points)
    b,d,n = xyz.shape
    grouped_xyz -= xyz.unsqueeze(2).expand(-1,-1,k,-1) # translation normalization, (batch_size, num_points, k, num_dim(3))

    if points is not None:
        grouped_points = group_point(points, idx)
    else:
        grouped_points = grouped_xyz

    return grouped_xyz, grouped_points, idx


if __name__ == '__main__':
    x = torch.rand((24, 3, 4096)).to(torch.device('cuda'))
    y = torch.rand((24, 3, 4096)).to(torch.device('cuda'))

    # knn(x,8)
    # pool_ind = pool(x,y,8,512)
    # ind = knn_point(8,x,x)

    pairwise_dist(x,y)
