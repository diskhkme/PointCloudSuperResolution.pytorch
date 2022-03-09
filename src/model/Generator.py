import torch
import torch.nn as nn
import torch.nn.functional as F
import model.grouping_util as gutil\

def init_weight_(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)

class FeatureNet(nn.Module):
    def __init__(self, k=8, dim=128, num_block=3):
        super(FeatureNet, self).__init__()
        self.k = k
        self.num_block = num_block

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv2d(3, dim, 1, 1 ,bias=False))
        for i in range(self.num_block - 1):
            self.conv_layers.append(nn.Conv2d(dim, dim, 1, 1,bias=False))

        self.conv_layers.apply(init_weight_)

    def forward(self, x):
        _, out, _ = gutil.group(x, None, self.k) # (batch_size, num_dim(3), k, num_points)

        for conv_layer in self.conv_layers:
            out = F.relu(conv_layer(out))

        out, _ = torch.max(out, dim=2) # (batch_size, num_dim, num_points)

        return out

class ResGraphConvUnpool(nn.Module):
    def __init__(self, k=8, in_dim=128, dim=128):
        super(ResGraphConvUnpool, self).__init__()
        self.k = k
        self.num_blocks = 12

        self.bn_relu_layers = nn.ModuleList()
        for i in range(self.num_blocks):
            self.bn_relu_layers.append(nn.ReLU())

        self.conv_layers = nn.ModuleList()
        for i in range(self.num_blocks):
            self.conv_layers.append(nn.Conv2d(in_dim, dim, 1, 1,bias=False))
            self.conv_layers.append(nn.Conv2d(dim, dim, 1, 1,bias=False))

        self.unpool_center_conv = nn.Conv2d(dim, 6, 1, 1,bias=False)
        self.unpool_neighbor_conv = nn.Conv2d(dim, 6, 1, 1,bias=False)

        self.conv_layers.apply(init_weight_)
        self.unpool_center_conv.apply(init_weight_)
        self.unpool_neighbor_conv.apply(init_weight_)

    def forward(self, xyz, points):
        # xyz: (batch_size, num_dim(3), num_points)
        # points: (batch_size, num_dim(128), num_points)

        indices = None
        for idx in range(self.num_blocks): # 4 layers per iter
            shortcut = points # (batch_size, num_dim(128), num_points)

            points = self.bn_relu_layers[idx](points) # ReLU

            if idx == 0 and indices is None:
                _, grouped_points, indices = gutil.group(xyz, points, self.k) # (batch_size, num_dim, k, num_points)
            else:
                grouped_points = gutil.group_point(points, indices)

            # Center Conv
            b,d,n = points.shape
            center_points = points.view(b, d, 1, n)
            points = self.conv_layers[2 * idx](center_points)  # (batch_size, num_dim(128), 1, num_points)
            # Neighbor Conv
            grouped_points_nn = self.conv_layers[2 * idx + 1](grouped_points)
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
    def __init__(self, cfg):
        super(Generator, self).__init__()
        self.k = cfg['k']
        self.feat_dim = cfg['feat_dim']
        self.res_conv_dim = cfg['res_conv_dim']
        self.featurenet = FeatureNet(self.k, self.feat_dim,3)

        self.res_unpool_1 = ResGraphConvUnpool(self.k, self.feat_dim, self.res_conv_dim)
        self.res_unpool_2 = ResGraphConvUnpool(self.k, self.res_conv_dim, self.res_conv_dim)

    def forward(self, xyz):
        points = self.featurenet(xyz) # (batch_size, feat_dim, num_points)

        new_xyz, points = self.res_unpool_1(xyz, points)

        _, idx = gutil.knn_point(self.k, new_xyz, xyz)  # idx contains k nearest point of new_xyz in xyz
        grouped_points = gutil.group_point(points, idx)
        points = torch.mean(grouped_points, dim=2)

        new_xyz, points = self.res_unpool_2(new_xyz, points)

        return new_xyz

if __name__ == '__main__':
    # Profile forward
    cfg = {'k':8, 'feat_dim':128, 'res_conv_dim':128}

    model = Generator(cfg).cuda()
    xyz = torch.rand(24, 3, 1024).cuda()

    new_xyz = model(xyz)

    print(new_xyz.shape)
