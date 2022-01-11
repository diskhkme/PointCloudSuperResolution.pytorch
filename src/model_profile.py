import torch
import numpy as np
import torch.nn as nn
import torch.autograd.profiler as profiler

class HomogeneousModel(nn.Module):
    def __init__(self):
        super(HomogeneousModel, self).__init__()

        self.featurenet = nn.Sequential(
            nn.Conv2d(3,   128, kernel_size=1, stride=1),
            nn.Conv2d(128, 128, kernel_size=1, stride=1),
            nn.Conv2d(128, 128, kernel_size=1, stride=1)
        )

        self.resgraph1_self_layers = nn.ModuleList()
        for i in range(12):
            self.resgraph1_self_layers.append(nn.Conv2d(128,128,kernel_size=1, stride=1))

        self.resgraph1_group_layers = nn.ModuleList()
        for i in range(12):
            self.resgraph1_group_layers.append(nn.Conv2d(128, 128, kernel_size=1, stride=1))

        self.resgraph2_self_layers = nn.ModuleList()
        for i in range(12):
            self.resgraph2_self_layers.append(nn.Conv2d(128, 128, kernel_size=1, stride=1))

        self.resgraph2_group_layers = nn.ModuleList()
        for i in range(12):
            self.resgraph2_group_layers.append(nn.Conv2d(128, 128, kernel_size=1, stride=1))

    def forward(self, xyz):
        points = self.featurenet(xyz)

        for i in range(12):
            points = self.resgraph1_self_layers[i](points)
            group_points = points.repeat(1,1,8,1)
            group_points = self.resgraph1_group_layers[i](group_points)

        points = points.repeat(1,1,1,2)

        for i in range(12):
            points = self.resgraph2_self_layers[i](points)
            group_points = points.repeat(1,1,8,1)
            group_points = self.resgraph2_group_layers[i](group_points)

if __name__ == '__main__':
    # Profile forward
    model = HomogeneousModel().cuda()
    input = torch.rand(26, 3, 1, 1024).cuda()

    model(input)

    with profiler.profile(use_cuda=True, record_shapes=True, with_stack=True, profile_memory=True) as prof:
        out = model(input)

    print(prof.key_averages().table(sort_by='self_cuda_memory_usage', row_limit=20))
    prof.export_chrome_trace('trace.json')
