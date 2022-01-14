import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
from pyemd import emd_samples

from model.Generator import Generator
from dataset.visualize.pc_visualization_util import point_cloud_three_views

class PointCloudSuperResolutionEvaluation:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('--weight-path', type=str,
                            help='trained weight path')
        parser.add_argument('--predict-in-dir', type=str,
                            help='predict directory where input points (xyz format, 5,000 points per model in paper) exists')
        parser.add_argument('--predict-out-dir', type=str,
                            help='directory to save prediction result. (xyz format, 20,000 points per model in paper)')
        parser.add_argument('--gt-dir', type=str,
                            help='directory where gt files(.xyz) exist')

        self.args = parser.parse_args(sys_argv)

        #--
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = self.init_model()

    def init_model(self):
        model = Generator()
        if self.use_cuda:
            print("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
            return model

    def prediction_whole_model(self):
        input_filename_list = os.listdir(self.args.predict_in_dir)
        input_filename_list = [x for x in input_filename_list if x.find('xyz') != -1]
        input_filepath_list = [os.path.join(self.args.predict_in_dir,x) for x in input_filename_list]

        if not os.path.exists(self.args.predict_out_dir):
            os.mkdir(self.args.predict_out_dir)

        self.model.load_state_dict(torch.load(self.args.weight_path, map_location=self.device))
        with torch.no_grad():
            self.model.eval()
            for filepath, filename in zip(input_filepath_list, input_filename_list):
                print('Predicting {}...'.format(filename))
                input = np.loadtxt(filepath)
                input = np.expand_dims(input, axis=0)
                input = input[:,:,:3]
                input = torch.from_numpy(input.astype(np.float32)).to(self.device).transpose(1,2).contiguous()

                output = self.model(input)
                self.save_xyz(os.path.join(self.args.predict_out_dir, filename), output.detach().cpu().numpy())

        return input_filename_list

    # https://github.com/wuhuikai/PointCloudSuperResolution/blob/master/evaluation_code/evaluation_cd.py

    def evaluate_single(self, filename):
        gt_path = os.path.join(self.args.gt_dir, filename)
        pre_path = os.path.join(self.args.predict_out_dir, filename)
        assert os.path.exists(gt_path)
        assert os.path.exists(pre_path)

        gt_points = np.loadtxt(gt_path)  # (num_point, 3)
        pre_points = np.loadtxt(pre_path)  # (num_points, 3)

        gt2pre, _ = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(pre_points).kneighbors(gt_points)
        pre2gt, _ = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(gt_points).kneighbors(pre_points)

        return np.squeeze(gt2pre), np.squeeze(pre2gt), emd_samples(gt_points, pre_points)

    def evaluate(self, filename_list):
        filename_list = [x for x in filename_list if x.find('.xyz') != -1]
        distances = map(self.evaluate_single, filename_list)
        gt2pre, pre2gt, emd = zip(*distances)
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
        print('\t{}'.format(np.mean(emd)))

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

    def visualize(self, filename_list):
        for filename in filename_list:
            gt_path = os.path.join(self.args.gt_dir, filename)
            pre_path = os.path.join(self.args.predict_out_dir, filename)
            input_path = os.path.join(self.args.predict_in_dir, filename)
            self.write_pc_image(gt_path)
            self.write_pc_image(pre_path)
            self.write_pc_image(input_path)

    def main(self):
        filename_list = self.prediction_whole_model()
        self.evaluate(filename_list)
        self.visualize(filename_list)

if __name__ == '__main__':
    PointCloudSuperResolutionEvaluation().main()

