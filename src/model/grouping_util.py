import torch

# TODO: KNN requires too much memory ( num_points x num_points matrix )
def knn_point(k, xyz1, xyz2):
    b1, d1, n1 = xyz1.shape
    xyz1 = xyz1.view(b1, d1, 1, n1)
    b2, d2, n2 = xyz2.shape
    xyz2 = xyz2.view(b2, d2, n2, 1)
    dist = torch.sum((xyz1-xyz2) ** 2, 1)

    val, idx = torch.topk(-dist, k=k, dim=2)
    idx = idx.transpose(1,2).contiguous()

    return torch.sqrt(-val), idx # TODO: remove sqrt

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def group_point(x, idx, device):
    k = idx.size(1)
    batch_size = x.size(0)
    num_input_points = x.size(2)
    num_query_points = idx.size(2)
    x = x.view(batch_size, -1, num_input_points)

    idx_base = torch.arange(0, batch_size).view(-1, 1, 1).to(device) * num_input_points
    idx_unpacked = idx + idx_base
    idx_unpacked = idx_unpacked.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_input_points, -1)[idx_unpacked, :]
    feature = feature.view(batch_size, num_query_points, k, num_dims)
    feature = feature.transpose(1,3).contiguous() # (batch_size, num_dim, k, num_points)

    return feature

def group(xyz, points, k, device):
    _, idx = knn_point(k, xyz, xyz)
    grouped_xyz = group_point(xyz, idx, device) # (batch_size, num_dim, k, num_points)
    b,d,n = xyz.shape
    grouped_xyz -= xyz.view(b, d, 1, n).repeat(1, 1, k, 1) # translation normalization, (batch_size, num_points, k, num_dim(3))

    if points is not None:
        grouped_points = group_point(points, idx, device)
    else:
        grouped_points = grouped_xyz

    return grouped_xyz, grouped_points, idx


if __name__ == '__main__':
    x = torch.rand((1,3,1024))
    y = torch.rand((1, 3, 1024))

    knn(x,8)