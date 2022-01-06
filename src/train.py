import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
from tqdm import tqdm

from model.point_cloud_super_res import Generator
from model.loss import cd_loss
from dataset.pu_net_hdf import PUNetDataset

class PointCloudSuperResolutionTrainer:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers', type=int, default=4,
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
        parser.add_argument('--nepochs', type=int, default=80)
        parser.add_argument('--no-validate', action='store_true')

        self.args = parser.parse_args(sys_argv)

        if not os.path.exists(self.args.output_root):
            os.mkdir(self.args.output_root)

        #--
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = self.init_model()
        self.optimizer = self.init_optimizer()
        self.loss_fn = cd_loss

    def init_model(self):
        model = Generator(self.args.up_ratio)
        if self.use_cuda:
            print("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
            return model

    def init_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.00001)

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

    def do_train(self, train_dl):
        num_batch = int(len(train_dl) / self.args.batch_size)

        self.model.train()
        loss_train = 0.0
        for i, (input, label, gt) in enumerate(tqdm(train_dl)):
            input_points = input.to(self.device)
            gt_points = gt.to(self.device)
            num_upsampled_points = input_points.shape[2] * self.args.up_ratio
            if num_upsampled_points != gt_points.shape[2]:
                gt_points = gt_points[:,:,:num_upsampled_points]

            pred_points = self.model(input_points)
            assert pred_points.shape[2] == gt_points.shape[2]

            loss = self.loss_fn(pred_points, gt_points, 1.0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_train += loss.item()

        return loss_train / num_batch

    def do_validate(self, val_dl):
        num_batch = int(len(val_dl) / self.args.batch_size)

        loss_val = 0
        with torch.no_grad():
            self.model.eval()
            for i, (input, label, gt) in enumerate(tqdm(val_dl)):
                input_points = input.to(self.device)
                gt_points = gt.to(self.device)
                num_upsampled_points = input_points.shape[2] * self.args.up_ratio
                if num_upsampled_points != gt_points.shape[2]:
                    gt_points = gt_points[:, :, :num_upsampled_points]

                pred_points = self.model(input_points)

                loss_val += cd_loss(pred_points, gt_points, 1.0).item()

        return loss_val / num_batch

    def main(self):
        print('Starting {}, {}'.format(type(self).__name__, self.args))
        train_dl = self.init_dataloader('train')
        if not self.args.no_validate:
            val_dl = self.init_dataloader('val')
        min_loss = 99999
        for epoch in range(1,self.args.nepochs + 1):
            loss_train = self.do_train(train_dl)

            print('{} Epoch {}, Training loss {:.4f}'.format(datetime.datetime.now(), epoch, loss_train))

            if epoch == 1 or epoch % 10 == 0:

                if not self.args.no_validate:
                    val_loss = self.do_validate(val_dl)
                    if min_loss > val_loss:
                        min_loss = val_loss
                        torch.save(self.model.state_dict(), os.path.join(self.args.output_root,'result_{}_{:.2f}.pt'.format(epoch, val_loss)))

                    print('{} Epoch {}, Training loss {:.4f}, Val loss {:.2f} (Top performance :{:.2f})'.format(
                          datetime.datetime.now(), epoch, loss_train, val_loss, min_loss))
                else:
                    torch.save(self.model.state_dict(),
                               os.path.join(self.args.output_root, 'result_{}_{:.2f}.pt'.format(epoch, loss_train)))

if __name__ == '__main__':
    PointCloudSuperResolutionTrainer().main()
