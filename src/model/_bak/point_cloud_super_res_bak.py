import torch
import torch.nn as nn
import torch.autograd.profiler as profiler
import math
import model._bak.grouping_util_bak as gutil

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

            with profiler.record_function('Residual Graph Conv-Conv  {}'.format(idx)):
                points = self.layers[4 * idx](points) # Batch norm
                points = self.layers[4 * idx + 1](points) # ReLU

            with profiler.record_function('Residual Graph Conv-Group  {}'.format(idx)):
                if idx == 0 and indices is None:
                    _, grouped_points, indices = gutil.group(xyz, points, self.k, self.device) # (batch_size, num_dim, k, num_points)
                else:
                    grouped_points = gutil.group_point(points, indices, self.device)

            with profiler.record_function('Residual Graph Conv-Unpool {}'.format(idx)):
                # Center Conv
                b,d,n = points.shape
                center_points = points.view(b, d, 1, n)
                points = self.layers[4 * idx + 2](center_points)  # (batch_size, num_dim(128), 1, num_points)
                # Neighbor Conv
                grouped_points_nn = self.layers[4 * idx + 3](grouped_points)
                # CNN
                # points = torch.mean(torch.cat((points, grouped_points_nn), dim=2), dim=2) + shortcut
                mean_g = torch.mean(grouped_points_nn, dim=2).unsqueeze(2)
                points = (mean_g + (points-mean_g)/(grouped_points_nn.size(2) + 1)).squeeze(2) + shortcut

            if idx == self.num_blocks-1:
                num_points = xyz.shape[-1]
                # Center Conv
                points_xyz = self.unpool_center_conv(center_points) # (batch_size, 3*up_ratio, 1, num_points)
                # Neighbor Conv
                grouped_points_xyz = self.unpool_neighbor_conv(grouped_points) # (batch_size, 3*up_ratio, k, num_points)
                # CNN
                new_xyz = torch.mean(torch.cat((points_xyz, grouped_points_xyz), dim=2), dim=2) # (batch_size, 3*up_ratio, num_points)
                new_xyz = new_xyz.view(-1, 3, self.up_ratio, num_points) # (batch_size, 3, up_ratio, num_points)

                b, d, n = xyz.shape
                new_xyz = new_xyz + xyz.view(b, d, 1, n).repeat(1, 1, self.up_ratio, 1) # add delta x to original xyz to upsample
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
        with profiler.record_function('FeatureNet'):
            points = self.featurenet(xyz) # (batch_size, feat_dim, num_points)

        for i in range(self.num_block):
            with profiler.record_function('Residual Graph Conv {}'.format(i)):
                new_xyz, points = self.layers[i](xyz, points)

            if i < self.num_block - 1:
                with profiler.record_function('Unpooling'):
                    idx = gutil.knn_point(8, xyz, new_xyz) # idx contains k nearest point of new_xyz in xyz
                    grouped_points = gutil.group_point(points, idx, self.device)
                    points = torch.mean(grouped_points, dim=2)

            xyz = new_xyz

        return xyz

class ResGraphConvPool(nn.Module):
    def __init__(self, k=8, feat_dim=64, dim=64, num_blocks=12, device=torch.device('cuda')):
        super(ResGraphConvPool, self).__init__()
        self.k = k
        self.feat_dim = feat_dim
        self.dim = dim
        self.num_blocks = num_blocks
        self.device = device

        self.layers = nn.ModuleList()
        for i in range(self.num_blocks):
            if i == 1:
                self.layers.append(nn.BatchNorm1d(self.feat_dim))
                self.layers.append(nn.LeakyReLU())

                self.layers.append(nn.Conv2d(self.feat_dim, self.dim, kernel_size=1, stride=1))
                self.layers.append(nn.Conv2d(self.feat_dim, self.dim, kernel_size=1, stride=1))
            else:
                self.layers.append(nn.BatchNorm1d(self.dim))
                self.layers.append(nn.LeakyReLU())

                self.layers.append(nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1))
                self.layers.append(nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1))

    def forward(self, xyz, points):
        # xyz: (batch_size, num_dim(3), num_points)
        # points: (batch_size, num_dim(128), num_points)

        indices = None
        for idx in range(self.num_blocks): # 4 layers per iter
            shortcut = points # (batch_size, num_dim(128), num_points)

            points = self.layers[4 * idx](points) # Batch norm
            points = self.layers[4 * idx + 1](points) # LeakyReLU

            if idx == 0 and indices is None:
                _, grouped_points, indices = gutil.group(xyz, points, self.k, self.device) # (batch_size, num_dim, k, num_points)
            else:
                grouped_points = gutil.group_point(points, indices,self.device)

            # Center Conv
            b, d, n = points.shape
            center_points = points.view(b, d, 1, n)
            points = self.layers[4 * idx + 2](center_points)  # (batch_size, num_dim(128), 1, num_points)
            # Neighbor Conv
            grouped_points_nn = self.layers[4 * idx + 2](grouped_points)
            # CNN
            points = torch.mean(torch.cat((points, grouped_points_nn), dim=2), dim=2) + shortcut

        return points

class ResGraphConvPoolLast(nn.Module):
    def __init__(self, feat_dim=64, dim=1, device=torch.device('cuda')):
        super(ResGraphConvPoolLast, self).__init__()

        self.bn = nn.BatchNorm1d(feat_dim)
        self.act = nn.LeakyReLU()
        self.conv = nn.Conv2d(feat_dim, dim, kernel_size=1, stride=1)

    def forward(self,x):
        points = self.bn(x)
        points = self.act(points)
        b, d, n = points.shape
        center_points = points.view(b, d, 1, n)
        points = self.conv(center_points).squeeze(2)

        return points

class Discriminator(nn.Module):
    def __init__(self, num_hr_points=4096, device=torch.device('cuda')):
        super(Discriminator, self).__init__()
        self.device = device
        self.featurenet = FeatureNet(k=8,dim=64,num_blocks=2)
        self.block_num = int(math.log2(num_hr_points/64)/2)

        self.layers = nn.ModuleList()
        for i in range(self.block_num):
            self.layers.append(ResGraphConvPool(k=8, feat_dim=64, dim=64, num_blocks=12))

        self.last_layer = ResGraphConvPoolLast(feat_dim=64, dim=1)

    def forward(self,xyz):
        points = self.featurenet(xyz)
        for layer in self.layers:
            xyz, points = gutil.pool(xyz, points, k=8, npoint=points.size(2)//4)
            points = layer(xyz, points)

        points = self.last_layer(points)

        return points


if __name__ == '__main__':
    # model =Generator(4)
    # print(model)


    # Profile forward
    model = Generator(4).cuda()
    input = torch.rand(24, 3, 1024).cuda()

    model(input)

    with profiler.profile(use_cuda=True, record_shapes=True, with_stack=True, profile_memory=True) as prof:
        out = model(input)

    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_memory_usage'))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))


    # device = torch.device('cuda')
    # xyz = torch.rand((1,3,1024)).to(device)
    #
    # # knn
    # # ind = knn(xyz, 8)
    # # val, ind2 = knn_point(8, xyz, xyz)
    #
    # # -- Module Test
    # # k = 8
    # # feat_dim = 128
    # # num_feat_block = 3
    # #
    # # f = FeatureNet(k=k, dim=feat_dim, num_blocks=num_feat_block)
    # # points = f(xyz) # 1 x 128 x 1024
    # #
    # # num_res_gcn_block = 12
    # # res_gcn_dim = 128
    # # up_ratio = 2
    # #
    # # res_gcn_unpool_1 = ResGraphConvUnpool(k=k, feat_dim=feat_dim, dim=res_gcn_dim, up_ratio=up_ratio, num_blocks=num_res_gcn_block)
    # # new_xyz, points = res_gcn_unpool_1(xyz, points)
    # #
    # # res_gcn_unpool_2 = ResGraphConvUnpool(k=k, feat_dim=res_gcn_dim, dim=res_gcn_dim, up_ratio=up_ratio, num_blocks=num_res_gcn_block)
    # # new_xyz, points = res_gcn_unpool_2(new_xyz, points)
    #
    # # -- Generator Test
    # # g = Generator(4).to(device)
    # # xyz = g(xyz)
    #
    # # -- Discriminator Test
    # device = torch.device('cuda')
    # up_sampled = torch.rand((3, 3, 4096)).to(device)
    # d = Discriminator().to(device)
    # res = d(up_sampled)
