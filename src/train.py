import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
from tqdm import tqdm

from config.config import get_train_config
from model.Generator import Generator
from model.Discriminator import Discriminator
from model.loss import get_cd_loss, get_d_loss, get_g_loss
from dataset.pu_net_hdf import PUNetDataset

class PointCloudSuperResolutionTrainer:
    def __init__(self, config_path):
        self.cfg = get_train_config(config_path)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.cfg['cuda_devices']

        self.generator = self.init_generator()
        self.discriminator = self.init_discriminator()
        self.pre_gen_optim, self.d_optim = self.init_optimizer()

    def init_generator(self):
        model = Generator(self.cfg['network']['generator'])
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.cuda()
        return model

    def init_discriminator(self):
        model = Discriminator(self.cfg['network']['discriminator'])
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.cuda()
        return model

    def init_optimizer(self):
        if self.cfg['optimizer'] == 'adam': # TODO: support other optimizer option
            pre_gen_optim = optim.Adam(self.generator.parameters(), lr=self.cfg['lr'], betas=(0.5, 0.999), weight_decay=self.cfg['weight_decay'])
            # TODO: split discriminator optimizer parameter
            d_optim = optim.Adam(self.discriminator.parameters(), lr=self.cfg['lr'], betas=(0.9, 0.999), weight_decay=self.cfg['weight_decay'])

        return pre_gen_optim, d_optim

    def init_dataloader(self):
        dataset = PUNetDataset(self.cfg['dataset']['data_path'], npoints=self.cfg['dataset']['in_point'], data_augmentation=self.cfg['dataset']['data_augmentation'])

        batch_size = self.cfg['batch_size']
        if torch.cuda.device_count() > 1:
            batch_size *= torch.cuda.device_count()

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   pin_memory=True,
                                                   shuffle=True)

        return loader

    def clip_gt_points(self, num_upsampled_point, gt_points):
        if num_upsampled_point != gt_points.shape[2]:
            gt_points = gt_points[:, :, :num_upsampled_point]

        return gt_points

    def do_train(self, train_dl, phase):
        num_batch = int(len(train_dl))
        pre_gen_loss_train = 0.0
        d_loss_train = 0.0
        g_loss_train = 0.0

        self.generator.train()
        if phase == 'gan':
            self.discriminator.train()
            for g in self.pre_gen_optim.param_groups:
                g['lr'] = self.cfg['finetune_lr']

        for i, (input, label, gt) in enumerate(tqdm(train_dl)):
            # torch.cuda.empty_cache()
            input_points = input.cuda()
            gt_points = gt.cuda()
            gt_points = self.clip_gt_points(input_points.size(2)*4, gt_points) # 4=upsample ratio

            torch.autograd.set_detect_anomaly(True)

            if phase == 'gan':
                d_real = self.discriminator(gt_points)
                # torch.cuda.empty_cache()

            pred_points = self.generator(input_points)
            # torch.cuda.empty_cache()

            if phase == 'gan':
                d_fake = self.discriminator(pred_points.detach())
                # torch.cuda.empty_cache()

            cd_loss = get_cd_loss(pred_points, gt_points, 1.0)

            if phase == 'gan':
                d_loss = get_d_loss(d_real, d_fake)
                self.d_optim.zero_grad()
                d_loss.backward()
                self.d_optim.step()

                d_loss_train += d_loss.item()

            pre_gen_loss = cd_loss
            if phase == 'gan':
                d_fake = self.discriminator(pred_points)
                # torch.cuda.empty_cache()
                g_loss = get_g_loss(d_fake)
                pre_gen_loss = g_loss + self.cfg['loss']['lambd'] * pre_gen_loss

                pre_gen_loss_train += g_loss.item()

            self.pre_gen_optim.zero_grad()
            pre_gen_loss.backward()
            self.pre_gen_optim.step()

            # print mini-batch loss for debug
            print('Batch loss {:.6f}'.format(pre_gen_loss.item()))

            pre_gen_loss_train += pre_gen_loss.item()

        return pre_gen_loss_train / num_batch, d_loss_train / num_batch, g_loss_train / num_batch

    def main(self):
        print('Starting {}, {}'.format(type(self).__name__, self.cfg))
        train_dl = self.init_dataloader()
        current_phase = self.cfg['phase']
        print(current_phase)

        if current_phase == 'gan':
            assert os.path.exists(self.cfg['pre_weight'])
            self.generator.load_state_dict(torch.load(self.cfg['pre_weight'], map_location=torch.device('cuda')))

        for epoch in range(1,self.cfg['max_epoch'] + 1):
            if current_phase == 'pre':
                train_loss, _, _ = self.do_train(train_dl, 'pre')
                print('{} Epoch {}, Pre-training loss {:.6f}'.format(datetime.datetime.now(), epoch, train_loss))
            elif current_phase == 'gan':
                train_loss, d_loss, g_loss = self.do_train(train_dl, phase='gan')
                print('{} Epoch {}, Training loss {:.6f}, G Loss {:.6f}, D Loss {:.6f}'.format(datetime.datetime.now(),
                                                                                               epoch, train_loss,
                                                                                               d_loss, g_loss))

            if epoch == 1 or epoch % self.cfg['save_steps'] == 0:
               torch.save(self.generator.state_dict(), os.path.join(self.cfg['ckpt_root'], '{}_result_{}_{:.6f}.pt'.format(self.cfg['phase'], epoch, train_loss)))

if __name__ == '__main__':
    PointCloudSuperResolutionTrainer('config/train_config_gen_only.yaml').main()
    # PointCloudSuperResolutionTrainer('config/train_config_gen_only.yaml').main()