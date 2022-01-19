from pathlib import Path
from tqdm import tqdm
import numpy as np
from numpy.random import choice, rand
import meshio

np.random.seed(0)

def tune_negative(negate, other1, other2):
    neg_indices = negate < 0
    negate[neg_indices] = -negate[neg_indices]
    other1[neg_indices] -= negate[neg_indices]
    other2[neg_indices] -= negate[neg_indices]

    return negate, other1, other2

def heal_fault_weight(rand_u, rand_v, rand_w):
    while np.sum((rand_u < 0) | (rand_v < 0) | (rand_w < 0)) != 0:
        neg_num = [np.sum(rand_u < 0), np.sum(rand_v < 0), np.sum(rand_w < 0)]
        max_ind = neg_num.index(max(neg_num))

        if max_ind == 0:
            rand_u, rand_v, rand_w = tune_negative(rand_u, rand_v, rand_w)
        elif max_ind == 1:
            rand_v, rand_u, rand_w = tune_negative(rand_v, rand_u, rand_w)
        elif max_ind == 2:
            rand_w, rand_u, rand_v = tune_negative(rand_w, rand_u, rand_v)

    return rand_u, rand_v, rand_w

def sample_points_from_mesh(path, num_sample_point=4096):
    mesh = meshio.read(path)
    points = mesh.points
    assert len(mesh.cells) == 1
    triangles = mesh.cells[0].data
    n_triangle = triangles.shape[0]

    triangle_points = points[triangles] # triangle x point x dim
    triangle_n1s = triangle_points[:, 1, :] - triangle_points[:,0,:]
    triangle_n2s = triangle_points[:, 2, :] - triangle_points[:,0,:]

    triangle_doubled_area = np.linalg.norm(np.cross(triangle_n1s, triangle_n2s), axis=1)
    prob = triangle_doubled_area / np.sum(triangle_doubled_area)

    rand_indices = choice(n_triangle, (num_sample_point,), p=prob)
    rand_triangle_points = triangle_points[rand_indices,:,:]

    rand_u = rand(num_sample_point)
    rand_v = rand(num_sample_point)
    rand_w = 1.0 - (rand_u + rand_v)

    # process fault weight
    rand_u, rand_v, rand_w = heal_fault_weight(rand_u, rand_v, rand_w)

    rand_weight = np.vstack((rand_u.squeeze(), rand_v.squeeze(), rand_w.squeeze()))
    rand_weight = np.transpose(rand_weight)

    sampled_points = np.array([np.matmul(points.T, weights) for points, weights in zip(rand_triangle_points, rand_weight)])

    return sampled_points

def write_sampled_points(path, sampled_points):
    with open(path, 'w') as file:
        for point in sampled_points:
            file.write('{} {} {}\n'.format(point[0],point[1],point[2]))

if __name__ == '__main__':
    root = Path('D:/Test_Models/3D/ABC/220112_ABC_0000/abc_0000_obj')
    target = Path('D:/Test_Models/3D/ABC/220112_ABC_0000/abc_0000_point_subset_100_normalized')
    num_subset = 100
    num_point_for_low = 5000
    num_point_for_high = 20000
    center_normalize = True

    low_obj_list = sorted(list(root.glob('low/*.obj')))
    high_obj_list = sorted(list(root.glob('high/*.obj')))

    for shape, shape_h in tqdm(zip(low_obj_list[:num_subset], high_obj_list[:num_subset])):
        assert (shape.name.split('_')[0] == shape_h.name.split('_')[0])

        sampled_points_l = sample_points_from_mesh(str(shape), num_point_for_low)
        sampled_points_h = sample_points_from_mesh(str(shape_h), num_point_for_high)

        bbox_lower = np.min(sampled_points_l, axis=0)
        bbox_top = np.max(sampled_points_l, axis=0)

        if center_normalize:
            center = (bbox_lower + bbox_top) / 2
            sampled_points_l -= center
            sampled_points_h -= center

            max_len = np.sqrt(np.max(sampled_points_l[:, 0] ** 2 + sampled_points_l[:, 1] ** 2 + sampled_points_l[:, 2] ** 2))
            sampled_points_l /= max_len
            sampled_points_h /= max_len

        out_path_l = target / shape.relative_to(root).with_suffix('.xyz')
        out_path_l.parent.mkdir(parents=True, exist_ok=True)
        write_sampled_points(str(out_path_l), sampled_points_l)

        out_path_h = target / shape_h.relative_to(root).with_suffix('.xyz')
        out_path_h.parent.mkdir(parents=True, exist_ok=True)
        write_sampled_points(str(out_path_h), sampled_points_h)
