import matplotlib.pyplot as plt
import numpy as np

def matplotlib_3d_ptcloud(output_pcl):
    data = output_pcl.detach().cpu().numpy()
    xdata = data[0,:].squeeze()
    ydata = data[1,:].squeeze()
    zdata = data[2,:].squeeze()

    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection='3d')

    ax.scatter3D(xdata, ydata, zdata, marker='o')
    plt.show()

def matplotlib_3d_ptcloud_gt_in_out(gt, input, pred):
    gt_data = gt.detach().cpu().numpy()
    input_data = input.detach().cpu().numpy()
    pred_data = pred.detach().cpu().numpy()

    fig = plt.figure(figsize=plt.figaspect(0.33))

    ax = fig.add_subplot(1,3,1,projection='3d')
    xdata = gt_data[:,0].squeeze()
    ydata = gt_data[:,1].squeeze()
    zdata = gt_data[:,2].squeeze()
    ax.set_title('Ground Truth')
    ax.scatter3D(xdata, ydata, zdata, marker='o')

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    xdata = input_data[:, 0].squeeze()
    ydata = input_data[:, 1].squeeze()
    zdata = input_data[:, 2].squeeze()
    ax.set_title('Input Points')
    ax.scatter3D(xdata, ydata, zdata, marker='o')

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    xdata = pred_data[:, 0].squeeze()
    ydata = pred_data[:, 1].squeeze()
    zdata = pred_data[:, 2].squeeze()
    ax.set_title('Predicted Points')
    ax.scatter3D(xdata, ydata, zdata, marker='o')

    plt.show()