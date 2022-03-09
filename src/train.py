import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
from tqdm import tqdm
import logging

from config.config import get_train_config
from model.Generator import Generator
from model.Discriminator import Discriminator
from model.loss import get_cd_loss, get_d_loss, get_g_loss
from dataset.pu_net_hdf import PUNetDataset

class PointCloudSuperResolutionTrainer:
    def __init__(self, config_path):
        self.config_path = config_path
        self.cfg = get_train_config(config_path)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.cfg['cuda_devices']

        self.generator = self.init_generator()
        self.discriminator = self.init_discriminator()
        self.pre_gen_optim, self.d_optim, self.pre_gen_scheduler, self.d_scheduler = self.init_optimizer()

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
            if self.cfg['phase'] == 'gan':
                pre_gen_optim = optim.Adam(self.generator.parameters(), lr=self.cfg['lr']/10, betas=(0.9, 0.999), weight_decay=self.cfg['weight_decay'])
                d_optim = optim.Adam(self.discriminator.parameters(), lr=self.cfg['lr'], betas=(0.5, 0.999), weight_decay=self.cfg['weight_decay'])
            else:
                pre_gen_optim = optim.Adam(self.generator.parameters(), lr=self.cfg['lr'], betas=(0.9, 0.999),
                                           weight_decay=self.cfg['weight_decay'])
                d_optim = None

        pre_gen_scheduler = optim.lr_scheduler.MultiStepLR(pre_gen_optim, milestones=[5, 10, 50], gamma=0.5)
        if self.cfg['phase'] == 'gan':
            d_scheduler = optim.lr_scheduler.MultiStepLR(d_optim, milestones=[5, 10, 20], gamma=0.5)
        else:
            d_scheduler = None

        return pre_gen_optim, d_optim, pre_gen_scheduler, d_scheduler

    def init_dataloader(self):
        dataset = PUNetDataset(self.cfg['dataset']['data_path'], npoints=self.cfg['dataset']['in_point'], data_augmentation=self.cfg['dataset']['data_augmentation'])

        batch_size = self.cfg['batch_size']
        if torch.cuda.device_count() > 1:
            batch_size *= torch.cuda.device_count()

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   pin_memory=True,
                                                   shuffle=True,
                                             num_workers = torch.cuda.device_count() * 4)

        return loader

    def clip_gt_points(self, num_upsampled_point, gt_points):
        if num_upsampled_point != gt_points.shape[2]:
            gt_points = gt_points[:, :, :num_upsampled_point]

        return gt_points

    def do_train(self, train_dl, phase):
        num_batch = int(len(train_dl))
        total_loss_train = 0.0
        cd_loss_train = 0.0
        d_loss_train = 0.0
        g_loss_train = 0.0

        self.generator.train()
        if phase == 'gan':
            self.discriminator.train()

        for i, (input, label, gt) in enumerate(tqdm(train_dl)):
            # torch.cuda.empty_cache()
            input_points = input.cuda()
            gt_points = gt.cuda()
            gt_points = self.clip_gt_points(input_points.size(2)*4, gt_points) # 4=upsample ratio

            # torch.autograd.set_detect_anomaly(True)

            if phase == 'gan':
                pred_points = self.generator(input_points)

                #--------Train discriminator--------#
                for p in self.discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

                d_real = self.discriminator(gt_points)
                d_fake = self.discriminator(pred_points.detach())

                d_loss_real = torch.mean((d_real - 1) ** 2)
                d_loss_fake = torch.mean(d_fake ** 2)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)

                d_loss_train += d_loss.item()

                self.d_optim.zero_grad()
                d_loss.backward()
                self.d_optim.step()

                # --------Train generator--------#
                d_gen_fake = self.discriminator(pred_points) # train only generator
                g_loss = torch.mean((d_gen_fake - 1) ** 2)
                cd_loss = get_cd_loss(gt_points, pred_points, 1.0)

                pre_gen_loss = g_loss + self.cfg['loss']['lambd'] * cd_loss

                self.pre_gen_optim.zero_grad()
                pre_gen_loss.backward()
                self.pre_gen_optim.step()

                cd_loss_train += cd_loss.item()
                total_loss_train += pre_gen_loss.item()
                g_loss_train += g_loss.item()


                # print mini-batch loss for debug
                logging.info('Batch loss total: {:.6f} (cd_loss:{:.6f} g_loss:{:.6f} d_loss:{:.6f} (d_loss_real:{:.6f}, d_loss_fake:{:.6f})'.format(
                    pre_gen_loss.item(), cd_loss.item(), g_loss.item(), d_loss.item(), d_loss_real.item(), d_loss_fake.item()))

            if phase != 'gan':
                pred_points = self.generator(input_points)

                cd_loss = get_cd_loss(gt_points, pred_points, 1.0)

                self.pre_gen_optim.zero_grad()
                cd_loss.backward()
                self.pre_gen_optim.step()

                cd_loss_train += cd_loss.item()

                pre_gen_loss = cd_loss
                total_loss_train += pre_gen_loss.item()

                # print mini-batch loss for debug
                logging.info('Batch loss total(=cd_loss): {:.6f}'.format(pre_gen_loss.item()))

        self.pre_gen_scheduler.step()
        if phase == 'gan':
            self.d_scheduler.step()

        return total_loss_train / num_batch, cd_loss_train / num_batch, g_loss_train / num_batch, d_loss_train / num_batch

    def main(self):
        # copy train cfg to ckpt folder
        os.system('cp {} {}'.format(self.config_path, self.cfg['ckpt_root']))




        train_dl = self.init_dataloader()
        current_phase = self.cfg['phase']


        if current_phase == 'gan':
            assert os.path.exists(self.cfg['pre_weight'])
            self.generator.load_state_dict(torch.load(self.cfg['pre_weight'], map_location=torch.device('cuda')))
            # logging
            log_file = os.path.join(self.cfg['ckpt_root'], 'log_train_ar_gan.txt')
        else:
            log_file = os.path.join(self.cfg['ckpt_root'], 'log_train_res_gcn.txt')

        logging.basicConfig(level=logging.INFO,
                            handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()])
        logging.info('Starting {}, {}'.format(type(self).__name__, self.cfg))
        logging.info(current_phase)

        for epoch in range(1,self.cfg['max_epoch'] + 1):
            if current_phase == 'pre':
                cd_loss, _, _, _ = self.do_train(train_dl, 'pre')
                logging.info('{} Epoch {}, Total loss {:.6f}'.format(datetime.datetime.now(), epoch, cd_loss))
            elif current_phase == 'gan':
                train_loss, cd_loss, g_loss, d_loss = self.do_train(train_dl, phase='gan')
                logging.info('{} Epoch {}, Total loss {:.6f}, CD loss: {:.6f}, G Loss {:.6f}, D Loss {:.6f}'.format(datetime.datetime.now(),
                                                                                               epoch, train_loss, cd_loss,
                                                                                               g_loss, d_loss))

            if epoch == 1 or epoch % self.cfg['save_steps'] == 0:
               torch.save(self.generator.state_dict(), os.path.join(self.cfg['ckpt_root'], '{}_result_{}_{:.6f}.pt'.format(self.cfg['phase'], epoch, cd_loss)))

if __name__ == '__main__':
    PointCloudSuperResolutionTrainer(sys.argv[1]).main()