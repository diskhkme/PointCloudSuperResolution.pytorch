import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import math
import model.grouping_util_simple as gutil

class FeatureNet(nn.Module):
    def __init__(self, k=8, dim=128):
        super(FeatureNet, self).__init__()
        self.k = k

        self.conv1 = nn.Conv2d(3, dim, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1, 1)
        self.conv3 = nn.Conv2d(dim, dim, 1, 1)

        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)
        self.bn3 = nn.BatchNorm2d(dim)

    def forward(self, x):
        _, out, _ = gutil.group(x, None, self.k) # (batch_size, num_dim(3), k, num_points)

        out = F.relu(self.bn1(self.conv1(out))) # (batch_size, num_dim, k, num_points)
        out = F.relu(self.bn2(self.conv2(out))) # (batch_size, num_dim, k, num_points)
        out = F.relu(self.bn3(self.conv3(out))) # (batch_size, num_dim, k, num_points)

        out, _ = torch.max(out, dim=2) # (batch_size, num_dim, num_points)

        return out

class ResGraphConvUnpool(nn.Module):
    def __init__(self, k=8, in_dim=128, dim=128):
        super(ResGraphConvUnpool, self).__init__()
        self.k = k
        self.num_blocks = 12

        self.layers = nn.ModuleList()
        for i in range(self.num_blocks):
            if i == 1:
                self.layers.append(nn.BatchNorm1d(in_dim))
            else:
                self.layers.append(nn.BatchNorm1d(dim))

            self.layers.append(nn.ReLU())

            self.layers.append(nn.Conv2d(in_dim, dim, 1, 1))
            self.layers.append(nn.Conv2d(dim, dim, 1, 1))

        self.unpool_center_conv = nn.Conv2d(dim, 6, 1, 1)
        self.unpool_neighbor_conv = nn.Conv2d(dim, 6, 1, 1)

    def forward(self, xyz, points):
        # xyz: (batch_size, num_dim(3), num_points)
        # points: (batch_size, num_dim(128), num_points)

        indices = None
        for idx in range(self.num_blocks): # 4 layers per iter
            shortcut = points # (batch_size, num_dim(128), num_points)

            points = self.layers[4 * idx](points) # Batch norm
            points = self.layers[4 * idx + 1](points) # ReLU

            if idx == 0 and indices is None:
                _, grouped_points, indices = gutil.group(xyz, points, self.k) # (batch_size, num_dim, k, num_points)
            else:
                grouped_points = gutil.group_point(points, indices)

            # Center Conv
            b,d,n = points.shape
            center_points = points.view(b, d, 1, n)
            points = self.layers[4 * idx + 2](center_points)  # (batch_size, num_dim(128), 1, num_points)
            # Neighbor Conv
            grouped_points_nn = self.layers[4 * idx + 3](grouped_points)
            # CNN
            points = torch.mean(torch.cat((points, grouped_points_nn), dim=2), dim=2) + shortcut

            if idx == self.num_blocks-1:
                num_points = xyz.shape[-1]
                # Center Conv
                points_xyz = self.unpool_center_conv(center_points) # (batch_size, 3*up_ratio, 1, num_points)
                # Neighbor Conv
                grouped_points_xyz = self.unpool_neighbor_conv(grouped_points) # (batch_size, 3*up_ratio, k, num_points)
                # CNN
                new_xyz = torch.mean(torch.cat((points_xyz, grouped_points_xyz), dim=2), dim=2) # (batch_size, 3*up_ratio, num_points)
                new_xyz = new_xyz.view(-1, 3, 2, num_points) # (batch_size, 3, up_ratio, num_points)

                b, d, n = xyz.shape
                new_xyz = new_xyz + xyz.view(b, d, 1, n).repeat(1, 1, 2, 1) # add delta x to original xyz to upsample
                new_xyz = new_xyz.view(-1, 3, 2*num_points)

                return new_xyz, points

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.featurenet = FeatureNet(k=8,dim=128)

        self.res_unpool_1 = ResGraphConvUnpool(8, 128, 128)
        self.res_unpool_2 = ResGraphConvUnpool(8, 128, 128)

    def forward(self, xyz):
        points = self.featurenet(xyz) # (batch_size, feat_dim, num_points)

        new_xyz, points = self.res_unpool_1(xyz, points)

        _, idx = gutil.knn_point(8, xyz, new_xyz)  # idx contains k nearest point of new_xyz in xyz
        grouped_points = gutil.group_point(points, idx)
        points = torch.mean(grouped_points, dim=2)

        new_xyz, points = self.res_unpool_2(new_xyz, points)

        return new_xyz

if __name__ == '__main__':

    # Profile forward
    model = Generator().cuda()
    xyz = torch.rand(24, 3, 1024).cuda()

    new_xyz = model(xyz)

    print(new_xyz.shape)
