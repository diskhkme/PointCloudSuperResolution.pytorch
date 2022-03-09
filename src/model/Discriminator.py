import torch
import torch.nn as nn
import torch.nn.functional as F
import model.grouping_util as gutil
from model.Generator import FeatureNet, init_weight_

class ResGraphConvPool(nn.Module):
    def __init__(self, k=8, feat_dim=64, dim=64):
        super(ResGraphConvPool, self).__init__()
        self.k = k
        self.num_blocks = 4

        self.layers = nn.ModuleList()
        for i in range(self.num_blocks):
            if i == 1:
                self.layers.append(nn.LeakyReLU())

                self.layers.append(nn.Conv2d(feat_dim, dim, 1, 1, bias=False))
                self.layers.append(nn.Conv2d(feat_dim, dim, 1, 1, bias=False))
            else:
                self.layers.append(nn.LeakyReLU())

                self.layers.append(nn.Conv2d(dim, dim, 1, 1, bias=False))
                self.layers.append(nn.Conv2d(dim, dim, 1, 1, bias=False))

        self.layers.apply(init_weight_)

    def forward(self, xyz, points):
        # xyz: (batch_size, num_dim(3), num_points)
        # points: (batch_size, num_dim(128), num_points)

        indices = None
        for idx in range(self.num_blocks):
            shortcut = points # (batch_size, num_dim(128), num_points)

            # 4 layers per iter
            points = self.layers[3 * idx](points) # LeakyReLU

            if idx == 0 and indices is None:
                _, grouped_points, indices = gutil.group(xyz, points, self.k) # (batch_size, num_dim, k, num_points)
            else:
                grouped_points = gutil.group_point(points, indices)

            # Center Conv
            b, d, n = points.shape
            center_points = points.view(b, d, 1, n)
            points = self.layers[3 * idx + 1](center_points)  # (batch_size, num_dim(128), 1, num_points)
            # Neighbor Conv
            grouped_points_nn = self.layers[3 * idx + 2](grouped_points)
            # CNN
            points = torch.mean(torch.cat((points, grouped_points_nn), dim=2), dim=2) + shortcut

        return points

class ResGraphConvPoolLast(nn.Module):
    def __init__(self, feat_dim=64, last_dim=1):
        super(ResGraphConvPoolLast, self).__init__()

        self.act = nn.LeakyReLU()
        self.conv = nn.Conv2d(feat_dim, last_dim, 1, 1, bias=False)

        self.conv.apply(init_weight_)

    def forward(self,x):
        points = self.act(x)
        b, d, n = points.shape
        center_points = points.view(b, d, 1, n)
        res = self.conv(center_points).squeeze(2)

        return res

class Discriminator(nn.Module):
    def __init__(self, cfg):
        super(Discriminator, self).__init__()
        self.k = cfg['k']
        self.feat_dim = cfg['feat_dim']
        self.res_conv_dim = cfg['res_conv_dim']

        self.featurenet = FeatureNet(k=self.k,dim=self.feat_dim,num_block=2)

        self.layers = nn.ModuleList()
        for i in range(3):
            self.layers.append(ResGraphConvPool(k=self.k, feat_dim=self.feat_dim, dim=self.res_conv_dim))

        self.last_layer = ResGraphConvPoolLast(feat_dim=self.res_conv_dim, last_dim=1)

    def forward(self,xyz):
        points = self.featurenet(xyz)
        for layer in self.layers:
            xyz, points = gutil.pool(xyz, points, k=self.k, npoint=points.size(2)//4)
            points = layer(xyz, points)

        points = self.last_layer(points)

        return points

if __name__ == '__main__':
    xyz = torch.rand(12,3,4096).cuda()
    cfg = {'k': 8, 'feat_dim': 64, 'res_conv_dim': 64}
    d = Discriminator(cfg).cuda()
    points = d(xyz)

    print(points.shape)