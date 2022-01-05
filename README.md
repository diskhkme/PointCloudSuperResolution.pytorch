# PointCloudSupterResolution.pytorch

This repository is implementation of AR-GCN(https://arxiv.org/abs/1908.02111) from "Point Cloud Super Resolution with Adversarial Residual Graph Networks" in Pytorch. You can find official Tensorflow implementation [here](https://github.com/wuhuikai/PointCloudSuperResolution).

The model is in `src/model/point_cloud_super_res.py` (Currently, only generator network implemented.)

## Note

You need tqdm, sklearn, h5py installed in your environment, other than standard Pytorch-related packages.
The code is tested under Pytorch 1.8.2(LTS) and Python 3.7 on Windows 10.

## Usage
### Training
You can download training patches in HDF5 format in [here](https://drive.google.com/file/d/1wMtNGvliK_pUTogfzMyrz57iDb_jSQR8/view?usp=sharing).
(Note that the link is not maintained by me. Please refere [PU-Net](https://github.com/yulequan/PU-Net) official repository for the dataset.)
```buildoutcfg
cd src
python train.py --dataset=pu_net --dataset-root=../data/Patches_noHole_and_collected.h5  --output-root=../trained
```

### Prediction and Evaluation
You can download evaluation data from author's [official repository](https://github.com/wuhuikai/PointCloudSuperResolution).
```buildoutcfg
cd src
python predict_eval.py --weight-path=../trained/result_1_0.23.pt --predict-in-dir=../data/input --predict-out-dir=../data/pred --gt-dir=../data/gt
```

## Performance

TBA

### Contact
hk.kim@jbnu.ac.kr