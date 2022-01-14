import os
import torch
import csv
import numpy as np
import torch.utils.data as data
import h5py
from dataset.visualize.point_visualize import matplotlib_3d_ptcloud

class PUNetDataset(data.Dataset):
    def __init__(self,
                 path,
                 npoints=1024,
                 normalize=True,
                 data_augmentation=True):
        self.npoints = npoints
        self.path = path
        self.data_augmentation = data_augmentation

        f = h5py.File(self.path)
        self.gt = f['poisson_4096'][:]
        self.gt = self.gt[:,:,:3]
        self.input_data = f['poisson_4096'][:]
        self.input_data = self.input_data[:, :, :3]
        # data_1024 = f['montecarlo_1024'][:] # 1024 for test?
        self.names = f['name'][:]
        self.name_dict = self.generate_name_dict(self.names)

        if normalize == True:
            print('Normalize data')
            data_radius = np.ones(shape=(len(self.input_data)))
            centroid = np.mean(self.gt[:, :, 0:3], axis=1, keepdims=True)
            self.gt[:, :, 0:3] = self.gt[:, :, 0:3] - centroid
            furthest_distance = np.amax(np.sqrt(np.sum(self.gt[:, :, 0:3] ** 2, axis=-1)), axis=1, keepdims=True)
            self.gt[:, :, 0:3] = self.gt[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
            self.input_data[:, :, 0:3] = self.input_data[:, :, 0:3] - centroid
            self.input_data[:, :, 0:3] = self.input_data[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
        else:
            print("Do not normalize the data")
            centroid = np.mean(self.gt[:, :, 0:3], axis=1, keepdims=True)
            furthest_distance = np.amax(np.sqrt(np.sum((self.gt[:, :, 0:3] - centroid) ** 2, axis=-1)), axis=1,
                                        keepdims=True)
            data_radius = furthest_distance[0, :]

        self.radius = data_radius

    def get_name_from_name_path(self, name_path):
        return str(os.path.basename(name_path), 'utf-8').split('.')[0]

    def generate_name_dict(self, names):
        name_key = set()
        for name_path in names:
            name_key.add(self.get_name_from_name_path(name_path))

        name_dict = {}
        for i, key in enumerate(name_key):
            name_dict[key] = i

        return name_dict

    def __getitem__(self, idx):
        choice = np.random.choice(self.input_data.shape[1], self.npoints, replace=True)
        batch_input_data = np.expand_dims(self.input_data[idx, choice, :].copy(),0)
        batch_label_data = self.name_dict[self.get_name_from_name_path(self.names[idx])]
        batch_data_gt = np.expand_dims(self.gt[idx,:,:].copy(),0)
        radius = self.radius[idx]


        batch_input_data, batch_data_gt = self.rotate_point_cloud_and_gt(batch_input_data, batch_data_gt)
        batch_input_data, batch_data_gt, scales = self.random_scale_point_cloud_and_gt(batch_input_data,
                                                                                  batch_data_gt,
                                                                                  scale_low=0.9,
                                                                                  scale_high=1.1)
        radius = radius * scales
        batch_input_data, batch_data_gt = self.shift_point_cloud_and_gt(batch_input_data, batch_data_gt,
                                                                   shift_range=0.1)
        if np.random.rand() > 0.5:
            batch_input_data = self.jitter_perturbation_point_cloud(batch_input_data, sigma=0.025, clip=0.05)
        if np.random.rand() > 0.5:
            batch_input_data = self.rotate_perturbation_point_cloud(batch_input_data, angle_sigma=0.03,
                                                                   angle_clip=0.09)

        batch_input_data = np.squeeze(batch_input_data, 0)
        batch_data_gt = np.squeeze(batch_data_gt, 0)

        point_set = torch.from_numpy(batch_input_data.astype(np.float32)).transpose(0,1).contiguous()
        cls = torch.from_numpy(np.array([batch_label_data]).astype(np.int64))
        gt = torch.from_numpy(batch_data_gt.astype(np.float32)).transpose(0,1).contiguous()
        return point_set, cls, gt

    # Data augmentation is directly copied from official implementation
    # https://github.com/wuhuikai/PointCloudSuperResolution/blob/master/code/data_provider.py

    def rotate_point_cloud_and_gt(self, batch_data, batch_gt=None):
        """ Randomly rotate the point clouds to augument the dataset
            rotation is per shape based along up direction
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, rotated batch of point clouds
        """
        for k in range(batch_data.shape[0]):
            angles = np.random.uniform(size=3) * 2 * np.pi
            Rx = np.array([[1, 0, 0],
                           [0, np.cos(angles[0]), -np.sin(angles[0])],
                           [0, np.sin(angles[0]), np.cos(angles[0])]])
            Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                           [0, 1, 0],
                           [-np.sin(angles[1]), 0, np.cos(angles[1])]])
            Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                           [np.sin(angles[2]), np.cos(angles[2]), 0],
                           [0, 0, 1]])
            rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))

            batch_data[k, ..., 0:3] = np.dot(batch_data[k, ..., 0:3].reshape((-1, 3)), rotation_matrix)
            if batch_data.shape[-1] > 3:
                batch_data[k, ..., 3:] = np.dot(batch_data[k, ..., 3:].reshape((-1, 3)), rotation_matrix)

            if batch_gt is not None:
                batch_gt[k, ..., 0:3] = np.dot(batch_gt[k, ..., 0:3].reshape((-1, 3)), rotation_matrix)
                if batch_gt.shape[-1] > 3:
                    batch_gt[k, ..., 3:] = np.dot(batch_gt[k, ..., 3:].reshape((-1, 3)), rotation_matrix)

        return batch_data, batch_gt

    def shift_point_cloud_and_gt(self, batch_data, batch_gt=None, shift_range=0.3):
        """ Randomly shift point cloud. Shift is per point cloud.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, shifted batch of point clouds
        """
        B, N, C = batch_data.shape
        shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
        for batch_index in range(B):
            batch_data[batch_index, :, 0:3] += shifts[batch_index, 0:3]

        if batch_gt is not None:
            for batch_index in range(B):
                batch_gt[batch_index, :, 0:3] += shifts[batch_index, 0:3]

        return batch_data, batch_gt

    def random_scale_point_cloud_and_gt(self, batch_data, batch_gt=None, scale_low=0.5, scale_high=2.0):
        """ Randomly scale the point cloud. Scale is per point cloud.
            Input:
                BxNx3 array, original batch of point clouds
            Return:
                BxNx3 array, scaled batch of point clouds
        """
        B, N, C = batch_data.shape
        scales = np.random.uniform(scale_low, scale_high, B)
        for batch_index in range(B):
            batch_data[batch_index, :, 0:3] *= scales[batch_index]

        if batch_gt is not None:
            for batch_index in range(B):
                batch_gt[batch_index, :, 0:3] *= scales[batch_index]

        return batch_data, batch_gt, scales

    def rotate_perturbation_point_cloud(self, batch_data, angle_sigma=0.03, angle_clip=0.09):
        """ Randomly perturb the point clouds by small rotations
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, rotated batch of point clouds
        """
        for k in range(batch_data.shape[0]):
            angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
            Rx = np.array([[1, 0, 0],
                           [0, np.cos(angles[0]), -np.sin(angles[0])],
                           [0, np.sin(angles[0]), np.cos(angles[0])]])
            Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                           [0, 1, 0],
                           [-np.sin(angles[1]), 0, np.cos(angles[1])]])
            Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                           [np.sin(angles[2]), np.cos(angles[2]), 0],
                           [0, 0, 1]])
            R = np.dot(Rz, np.dot(Ry, Rx))
            batch_data[k, ..., 0:3] = np.dot(batch_data[k, ..., 0:3].reshape((-1, 3)), R)
            if batch_data.shape[-1] > 3:
                batch_data[k, ..., 3:] = np.dot(batch_data[k, ..., 3:].reshape((-1, 3)), R)

        return batch_data

    def jitter_perturbation_point_cloud(self, batch_data, sigma=0.005, clip=0.02):
        """ Randomly jitter points. jittering is per point.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, jittered batch of point clouds
        """
        B, N, C = batch_data.shape
        assert (clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
        jittered_data[:, :, 3:] = 0
        jittered_data += batch_data
        return jittered_data

    def __len__(self):
        return len(self.input_data)

if __name__ == '__main__':
    path = '../../data/Patches_noHole_and_collected.h5'
    d = PUNetDataset(path=path,split='train')
    print(len(d))

    pt, label, gt = d[0]

    matplotlib_3d_ptcloud(pt)