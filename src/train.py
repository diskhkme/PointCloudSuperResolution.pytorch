import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
from tqdm import tqdm


from model._bak.point_cloud_super_res_bak import Generator, Discriminator
from model._bak.loss_bak import cd_loss, finetune_gen_loss, finetune_disc_loss
from dataset.pu_net_hdf import PUNetDataset

class PointCloudSuperResolutionTrainer:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers', type=int, default=2,
                                 help='dataloader threads. 0 for single-thread.')
        parser.add_argument('--dataset', type=str,
                            help='pu_net')
        parser.add_argument('--dataset-root', type=str,
                            help='pu_net(h5 path)')
        parser.add_argument('--output-root', type=str,
                            help='train output root')
        parser.add_argument('--lr-points', type=int, default=1024,
                            help='number of input points')
        parser.add_argument('--up-ratio', type=int, default=4,
                            help='up-sampling ratio')
        parser.add_argument('--batch-size', type=int, default=4)
        parser.add_argument('--resume-from', type=str,
                            help='resume train from the weight')
        parser.add_argument('--nepochs', type=int, default=100,
                            help='full epochs to train. 80% of epochs for pre-train, 20% for fine tune')
        parser.add_argument('--no-validate', action='store_true')

        self.args = parser.parse_args(sys_argv)

        if not os.path.exists(self.args.output_root):
            os.mkdir(self.args.output_root)

        #--
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.generator = self.init_generator()
        self.discriminator = self.init_discriminator()
        self.pre_optimizer, self.g_optimizer, self.d_optimizer = self.init_optimizer()
        self.pre_loss_fn = cd_loss
        self.fine_gen_loss_fn = finetune_gen_loss
        self.fine_disc_loss_fn = finetune_disc_loss

    def init_generator(self):
        model = Generator(self.args.up_ratio)
        if self.use_cuda:
            print("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
            return model

    def init_discriminator(self):
        model = Discriminator(self.args.lr_points * self.args.up_ratio)
        if self.use_cuda:
            print("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
            return model

    def init_optimizer(self):
        pre_optim = optim.Adam(self.generator.parameters(), lr=0.001, weight_decay=0.00001)
        g_optim = optim.Adam(self.generator.parameters(), lr=0.001, weight_decay=0.00001)
        d_optim = optim.Adam(self.discriminator.parameters(), lr=0.001, weight_decay=0.00001)

        return pre_optim, g_optim, d_optim

    def init_dataloader(self, split='train'):
        if self.args.dataset == 'pu_net':
            dataset = PUNetDataset(self.args.dataset_root, npoints=self.args.lr_points, split=split,
                                                            data_augmentation=True, no_validate=self.args.no_validate)
        else:
            print("Other dataset is not suppored")
            return

        batch_size = self.args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   num_workers=self.args.num_workers,
                                                   pin_memory=self.use_cuda,
                                                   shuffle=True)

        return loader

    def do_pre_train(self, train_dl):
        num_batch = int(len(train_dl) / self.args.batch_size)

        self.generator.train()
        loss_train = 0.0
        for i, (input, label, gt) in enumerate(tqdm(train_dl)):
            input_points = input.to(self.device)
            gt_points = gt.to(self.device)
            num_upsampled_points = input_points.shape[2] * self.args.up_ratio
            if num_upsampled_points != gt_points.shape[2]:
                gt_points = gt_points[:,:,:num_upsampled_points]

            pred_points = self.generator(input_points)
            assert pred_points.shape[2] == gt_points.shape[2]

            loss = self.pre_loss_fn(pred_points, gt_points, 1.0)

            self.pre_optimizer.zero_grad()
            loss.backward()
            self.pre_optimizer.step()

            loss_train += loss.item()
            # torch.cuda.empty_cache()

        return loss_train / num_batch

    def do_fine_train(self, train_dl):
        num_batch = int(len(train_dl) / self.args.batch_size)

        self.generator.train()
        self.discriminator.train()
        loss_g = 0.0
        loss_d = 0.0
        for i, (input, label, gt) in enumerate(tqdm(train_dl)):
            input_points = input.to(self.device)
            gt_points = gt.to(self.device)
            num_upsampled_points = input_points.shape[2] * self.args.up_ratio
            if num_upsampled_points != gt_points.shape[2]:
                gt_points = gt_points[:, :, :num_upsampled_points]

            # Train G & D (TODO: is not the most efficient approach, I guess)
            # -- Train discriminator
            with torch.no_grad():
                pred_points = self.generator(input_points)

            d_fake = self.discriminator(pred_points)
            d_real = self.discriminator(gt_points)

            d_loss = self.fine_disc_loss_fn(d_real,d_fake)

            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # -- Train generator
            pred_points = self.generator(input_points)

            with torch.no_grad():
                d_fake = self.discriminator(pred_points)

            g_loss = self.fine_gen_loss_fn(d_fake, 5000, pred_points, gt_points, 1.0)

            self.g_optimizer.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()

            loss_g += g_loss.item()
            loss_d += d_loss.item()

        return loss_g / num_batch, loss_d / num_batch

    def do_validate(self, val_dl):
        num_batch = int(len(val_dl) / self.args.batch_size)

        loss_val = 0
        with torch.no_grad():
            self.generator.eval()
            for i, (input, label, gt) in enumerate(tqdm(val_dl)):
                input_points = input.to(self.device)
                gt_points = gt.to(self.device)
                num_upsampled_points = input_points.shape[2] * self.args.up_ratio
                if num_upsampled_points != gt_points.shape[2]:
                    gt_points = gt_points[:, :, :num_upsampled_points]

                pred_points = self.generator(input_points)

                loss_val += cd_loss(pred_points, gt_points, 1.0).item()

        return loss_val / num_batch

    def main(self):
        print('Starting {}, {}'.format(type(self).__name__, self.args))
        train_dl = self.init_dataloader('train')
        if not self.args.no_validate:
            val_dl = self.init_dataloader('val')
        min_loss = 99999

        if self.args.resume_from is not None:
            self.generator.load_state_dict(torch.load(self.args.resume_from, map_location=self.device))
            start_epoch = int(os.path.basename(self.args.resume_from).split('_')[1]) + 1
        else:
            start_epoch = 1

        for epoch in range(start_epoch,self.args.nepochs + 1):
            # --- Pre training generator
            if epoch <= int(self.args.nepochs * 0.8):
                loss_train = self.do_pre_train(train_dl)
                print('{} Epoch {}, Pre-training loss {:.4f}'.format(datetime.datetime.now(), epoch, loss_train))

                if epoch == 1 or epoch % 10 == 0:
                   torch.save(self.generator.state_dict(), os.path.join(self.args.output_root, 'result_{}_{:.2f}.pt'.format(epoch, loss_train)))

            # --- Fine-tune training
            else:
                loss_g, loss_d = self.do_fine_train(train_dl)
                print('{} Epoch {}, Fine-training loss gen:{:.4f} disc:{:.4f}'.format(datetime.datetime.now(), epoch, loss_g, loss_d))
                if epoch % 10 == 0 or epoch == int(self.args.nepochs):
                   torch.save(self.generator.state_dict(), os.path.join(self.args.output_root, 'result_{}_{:.2f}.pt'.format(epoch, loss_g)))

if __name__ == '__main__':
    PointCloudSuperResolutionTrainer().main()
