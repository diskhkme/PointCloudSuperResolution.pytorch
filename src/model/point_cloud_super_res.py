import torch
import torch.nn as nn
import math
import model.grouping_util as gutil

class FeatureNet(nn.Module):
    def __init__(self, k=8, dim=128, num_blocks=3, device=torch.device('cuda')):
        super(FeatureNet, self).__init__()
        self.k = k
        self.dim = dim
        self.num_blocks = num_blocks
        self.device = device

        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(3, dim, kernel_size=1, stride=1))
        torch.nn.init.xavier_uniform_(self.layers[-1].weight)
        self.layers.append(nn.BatchNorm2d(dim))
        self.layers.append(nn.ReLU())
        for i in range(self.num_blocks-1):
            self.layers.append(nn.Conv2d(dim, dim, kernel_size=1, stride=1))
            torch.nn.init.xavier_uniform_(self.layers[-1].weight)
            self.layers.append(nn.BatchNorm2d(dim))
            self.layers.append(nn.ReLU())

    def forward(self, x):
        _, out, _ = gutil.group(x, None, self.k, self.device) # (batch_size, num_dim(3), k, num_points)

        for layer in self.layers:
            out = layer(out) # (batch_size, num_dim, k, num_points)

        out, _ = torch.max(out, dim=2) # (batch_size, num_dim, num_points)

        return out


class ResGraphConvUnpool(nn.Module):
    def __init__(self, k=8, feat_dim=128, dim=128, up_ratio=2, num_blocks=12, device=torch.device('cuda')):
        super(ResGraphConvUnpool, self).__init__()
        self.k = k
        self.feat_dim = feat_dim
        self.dim = dim
        self.up_ratio = up_ratio
        self.num_blocks = num_blocks
        self.device = device

        self.layers = nn.ModuleList()
        for i in range(self.num_blocks):
            if i == 1:
                self.layers.append(nn.BatchNorm1d(self.feat_dim))
            else:
                self.layers.append(nn.BatchNorm1d(self.dim))

            self.layers.append(nn.ReLU())

            self.layers.append(nn.Conv2d(self.feat_dim, dim, kernel_size=1, stride=1))
            self.layers.append(nn.Conv2d(dim,dim, kernel_size=1, stride=1))

        self.unpool_center_conv = nn.Conv2d(dim, 3*self.up_ratio, kernel_size=1, stride=1)
        self.unpool_neighbor_conv = nn.Conv2d(dim, 3*self.up_ratio, kernel_size=1, stride=1)

    def forward(self, xyz, points):
        # xyz: (batch_size, num_dim(3), num_points)
        # points: (batch_size, num_dim(128), num_points)

        indices = None
        for idx in range(self.num_blocks): # 4 layers per iter
            shortcut = points # (batch_size, num_dim(128), num_points)

            points = self.layers[4 * idx](points) # Batch norm
            points = self.layers[4 * idx + 1](points) # ReLU

            if idx == 0 and indices is None:
                _, grouped_points, indices = gutil.group(xyz, points, self.k, self.device) # (batch_size, num_dim, k, num_points)
            else:
                grouped_points = gutil.group_point(points, indices,self.device)

            # Center Conv
            b,d,n = points.shape
            center_points = points.view(b, d, 1, n)
            points = self.layers[4 * idx + 2](center_points)  # (batch_size, num_dim(128), 1, num_points)
            # Neighbor Conv
            grouped_points_nn = self.layers[4 * idx + 2](grouped_points)
            # CNN
            points = torch.mean(torch.cat((points, grouped_points_nn), dim=2), dim=2) + shortcut

            if idx == self.num_blocks-1:
                num_points = xyz.shape[-1]
                # Center Conv
                points_xyz = self.unpool_center_conv(center_points) # (batch_size, 3*up_ratio, 1, num_points)
                # Neighbor Conv
                grouped_points_xyz = self.unpool_neighbor_conv(grouped_points) # (batch_size, 3*up_ratio, 8, num_points)
                # CNN
                new_xyz = torch.mean(torch.cat((points_xyz, grouped_points_xyz), dim=2), dim=2) # (batch_size, 3*up_ratio, num_points)
                new_xyz = new_xyz.view(-1, 3, self.up_ratio, num_points) # (batch_size, 3, up_ratio, num_points)

                b, d, n = xyz.shape
                new_xyz = new_xyz + xyz.view(b, d, 1, n).repeat(1, 1, self.up_ratio, 1)
                new_xyz = new_xyz.view(-1, 3, self.up_ratio*num_points)

                return new_xyz, points

        return points

class Generator(nn.Module):
    def __init__(self, up_ratio=4, device=torch.device('cuda')):
        super(Generator, self).__init__()
        self.num_block = int(math.log2(up_ratio))
        self.featurenet = FeatureNet(k=8,dim=128,num_blocks=3)
        self.device = device

        self.layers = nn.ModuleList()
        for i in range(self.num_block):
            self.layers.append(ResGraphConvUnpool(k=8,feat_dim=128,dim=128,up_ratio=2,num_blocks=12,device=self.device))

    def forward(self, xyz):
        points = self.featurenet(xyz) # (batch_size, feat_dim, num_points)
        for i in range(self.num_block):
            new_xyz, points = self.layers[i](xyz, points)
            if i < self.num_block - 1:
                _, idx = gutil.knn_point(8, xyz, new_xyz) # idx contains k nearest point of new_xyz in xyz
                grouped_points = gutil.group_point(points, idx, self.device)
                points = torch.mean(grouped_points, dim=2)

            xyz = new_xyz

        return xyz

    # TODO: extend to AR-GCN with discriminator

if __name__ == '__main__':
    device = torch.device('cuda')
    xyz = torch.rand((1,3,1024)).to(device)

    # knn
    # ind = knn(xyz, 8)
    # val, ind2 = knn_point(8, xyz, xyz)

    # -- Module Test
    # k = 8
    # feat_dim = 128
    # num_feat_block = 3
    #
    # f = FeatureNet(k=k, dim=feat_dim, num_blocks=num_feat_block)
    # points = f(xyz) # 1 x 128 x 1024
    #
    # num_res_gcn_block = 12
    # res_gcn_dim = 128
    # up_ratio = 2
    #
    # res_gcn_unpool_1 = ResGraphConvUnpool(k=k, feat_dim=feat_dim, dim=res_gcn_dim, up_ratio=up_ratio, num_blocks=num_res_gcn_block)
    # new_xyz, points = res_gcn_unpool_1(xyz, points)
    #
    # res_gcn_unpool_2 = ResGraphConvUnpool(k=k, feat_dim=res_gcn_dim, dim=res_gcn_dim, up_ratio=up_ratio, num_blocks=num_res_gcn_block)
    # new_xyz, points = res_gcn_unpool_2(new_xyz, points)

    # -- Generator Test
    g = Generator(4).to(device)
    xyz = g(xyz)