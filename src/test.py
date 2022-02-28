import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
# from pyemd import emd_samples

from config.config import get_test_config
from model.Generator import Generator
from dataset.visualize.pc_visualization_util import point_cloud_three_views

class PointCloudSuperResolutionEvaluation:
    def __init__(self, config_path):
        self.cfg = get_test_config(config_path)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.cfg['cuda_devices']

        self.model = self.init_model()
        self.input_dir = self.cfg['dataset']['input_dir']
        self.gt_dir = self.cfg['dataset']['gt_dir']
        self.pred_dir = self.cfg['dataset']['pred_dir']

    def init_model(self):
        model = Generator(self.cfg['network']['generator'])
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.cuda()
        return model

    def get_file_list(self, dir):
        filename_list = os.listdir(dir)
        filename_list = [x for x in filename_list if x.find('xyz') != -1]
        filepath_list = [os.path.join(dir, x) for x in filename_list]

        return filepath_list, filename_list

    def prediction_whole_model(self, input_filepath_list, input_filename_list):
        if not os.path.exists(self.pred_dir):
            os.mkdir(self.pred_dir)

        self.model.load_state_dict(torch.load(self.cfg['load_model'], map_location=torch.device('cuda')))
        with torch.no_grad():
            self.model.eval()
            for filepath, filename in zip(input_filepath_list, input_filename_list):
                print('Predicting {}...'.format(filename))
                input = np.loadtxt(filepath)
                input = np.expand_dims(input, axis=0)
                input = input[:,:,:3]
                input = torch.from_numpy(input.astype(np.float32)).cuda().transpose(1,2).contiguous()

                output = self.model(input)
                self.save_xyz(os.path.join(self.pred_dir, filename), output.detach().cpu().numpy())


    # https://github.com/wuhuikai/PointCloudSuperResolution/blob/master/evaluation_code/evaluation_cd.py

    def evaluate_single(self, filename, gt_filename):
        # ABC Dataset adaptation
        assert filename.split('_')[0] == gt_filename.split('_')[0]

        gt_path = os.path.join(self.gt_dir, gt_filename)
        pre_path = os.path.join(self.pred_dir, filename) # pred always has the same filename with input
        assert os.path.exists(gt_path)
        assert os.path.exists(pre_path)

        gt_points = np.loadtxt(gt_path)  # (num_point, 3)
        pre_points = np.loadtxt(pre_path)  # (num_points, 3)

        gt2pre, _ = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(pre_points).kneighbors(gt_points)
        pre2gt, _ = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(gt_points).kneighbors(pre_points)

        return np.squeeze(gt2pre), np.squeeze(pre2gt) #, emd_samples(gt_points, pre_points)

    def evaluate(self, filename_list, gt_filename_list):
        filename_list = [x for x in filename_list if x.find('.xyz') != -1]
        gt_filename_list = [x for x in gt_filename_list if x.find('.xyz') != -1]
        distances = map(self.evaluate_single, filename_list, gt_filename_list)
        #gt2pre, pre2gt, emd = zip(*distances) # Something wrong with pyemd in pytorch3d env
        gt2pre, pre2gt = zip(*distances)
        gt2pre, pre2gt = np.hstack(gt2pre), np.hstack(pre2gt)
        print('GT  --> PRE')
        print('\tMean     : {}'.format(np.mean(gt2pre)))
        print('\tStd      : {}'.format(np.std(gt2pre)))
        print('\tRecall   : {}'.format(np.mean(gt2pre <= 1e-2)))
        print('\tRecall   : {}'.format(np.mean(gt2pre <= 2e-2)))
        print('PRE --> GT')
        print('\tMean     : {}'.format(np.mean(pre2gt)))
        print('\tStd      : {}'.format(np.std(pre2gt)))
        print('\tPrecision: {}'.format(np.mean(pre2gt <= 1e-2)))
        print('\tPrecision: {}'.format(np.mean(pre2gt <= 2e-2)))
        print('CD:')
        print('\t{}'.format(0.5 * (np.mean(gt2pre) + np.mean(pre2gt))))
        print('F-score:')
        print('\t{}'.format(2 / (1 / np.mean(gt2pre <= 1e-2) + 1 / np.mean(pre2gt <= 1e-2))))
        print('\t{}'.format(2 / (1 / np.mean(gt2pre <= 2e-2) + 1 / np.mean(pre2gt <= 2e-2))))
        print('EMD:')
        # print('\t{}'.format(np.mean(emd)))

    def save_xyz(self, path, points):
        if not os.path.exists(os.path.split(path)[0]):
            os.makedirs(os.path.split(path)[0])
        np.savetxt(path, points.squeeze(0).transpose(1,0))

    def write_pc_image(self, pc_path):
        points = np.loadtxt(pc_path)
        im_array = point_cloud_three_views(points)
        im_array = im_array * 255.0
        out_path = pc_path.replace('.xyz','.jpg')
        cv2.imwrite(out_path, im_array)

    def write_prdicted_images(self, filename_list, gt_filename_list):
        for filename, gt_filename in zip(filename_list, gt_filename_list):
            gt_path = os.path.join(self.gt_dir, gt_filename)
            pre_path = os.path.join(self.pred_dir, filename)
            input_path = os.path.join(self.input_dir, filename)
            self.write_pc_image(gt_path)
            self.write_pc_image(pre_path)
            self.write_pc_image(input_path)

    def main(self):
        input_filepath_list, input_filename_list = self.get_file_list(self.input_dir)
        gt_filepath_list, gt_filename_list = self.get_file_list(self.gt_dir)

        self.prediction_whole_model(input_filepath_list, input_filename_list)
        self.evaluate(input_filename_list, gt_filename_list)
        self.write_prdicted_images(input_filename_list, gt_filename_list)

if __name__ == '__main__':
    PointCloudSuperResolutionEvaluation(sys.argv[1]).main()

