# PointCloudSuperResolution.pytorch

This repository is implementation of AR-GCN(https://arxiv.org/abs/1908.02111) from "Point Cloud Super Resolution with Adversarial Residual Graph Networks" in Pytorch. You can find official Tensorflow implementation [here](https://github.com/wuhuikai/PointCloudSuperResolution).

The model implementations are in `src/model`

## Note

The code is tested under Pytorch 1.9.0 and Python 3.8 on Ubuntu 18.04 LTS.

We used [pytorch3d](https://pytorch3d.org/) operation & loss functions for k-nn & cd_loss calculation.

Note that to enable non-symmetric chamfer distance calculation (only forward is used), 
we modified `pytorch3d/loss/chamfer.py` like below.

```python
    cham_dist = cham_x + cham_y
    cham_normals = cham_norm_x + cham_norm_y if return_normals else None

    forward = cham_x
    backward = cham_y
    return forward, backward, cham_normals
```


## Usage
### Training
You can download training patches in HDF5 format in [here](https://drive.google.com/file/d/1wMtNGvliK_pUTogfzMyrz57iDb_jSQR8/view?usp=sharing).
(Note that the link is not maintained by me. Please refere [PU-Net](https://github.com/yulequan/PU-Net) official repository for the dataset.)
```buildoutcfg
cd src
# ResGCN (80 epoch pre training. Train only generator)
python train.py config/train_config_res_gcn.yaml

# AR GCN (40 epoch GAN training. Must specify saved pre-trained weight from above)
python train.py config/train_config_ar_gcn.yaml
```

### Prediction and Evaluation
You can download evaluation data from author's [official repository](https://github.com/wuhuikai/PointCloudSuperResolution).
```buildoutcfg
cd src
python test.py # refer 'src/config/test_config.yaml' for test settings
```

## Performance

|         |   CD   | F-Score |
|--------:|:------:|:-------:|
| Res-GCN | 0.0092 | 0.6349  |
|  AR-GCN | 0.0090 | 0.6470  |

### Prediction example

- Ground truth
![GT](./img/camel_gt.jpg)
- Input
![Input](./img/camel_input.jpg)
- Prediction (Res-GCN)
![Pred](./img/camel_pred_res_gcn.jpg)
- Prediction (AR-GCN)
![Pred](./img/camel_pred_ar_gcn.jpg)

## Issues
- (fixed) ~~Training GAN takes too much time(~7s/it) compare to the author's implementation(~1s/it)~~
- (partially fixed) Performance is relatively low compared to the author's implementation. Especially during GAN training phase, discriminator does not trained well compare to the author's implementation.  

### Contact
hk.kim@jbnu.ac.kr