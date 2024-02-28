# Vehicle ReID 🚗🚘
The creation of this repository was motivated by the fact that there are not many ready-made solutions for a vehicle reid task that can be quickly obtained on the web. Here you can quickly train and test models of different sizes for different needs on an own dataset(see example folder). Training is done with [torchreid library](https://github.com/KaiyangZhou/deep-person-reid), model testing considers [CMC top-k accuracy](https://cysu.github.io/open-reid/notes/evaluation_metrics.html).

# TODO
* Add download links for the trained models

# Installation

Install via conda:
``` shell
git clone git@github.com:kasyanikita/vehicle_reid.git
cd ./vehicle_reid/deep-person-reid-master
pip install -r requirements.txt

# select the proper cuda version to suit your machine
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
python setup.py develop
```

# Custom dataset
An example of creating a custom dataset can be found in the file examples/vric_dataset.py. The dataset should be divided into three parts: train, query, gallery. The annotation for each part looks like a set of format strings (`filename` `object-id` `camera-id`):
```
MVI_40172_101_img02185.jpg 1249 58
MVI_40172_101_img01499.jpg 1249 58
MVI_40172_101_img02339.jpg 1249 58
MVI_40172_101_img02400.jpg 1249 58
MVI_40172_101_img00679.jpg 1249 57
MVI_40172_101_img01364.jpg 1249 58
MVI_40172_101_img01572.jpg 1249 58
...
```
Explore the VRIC dataset to better understand how annotations are formed.

# Training
Some train running example:
``` python
python train.py osnet_x1_0 --log_dir ./log/osnet_x1_0 --bs 16 --loss triplet --lr 0.0003 --max_epoch 60
```

# Testing
Test script calculates cmc ranks for the passed model and dataset.
``` python
python test.py <model_name> <model_pth_path> --query_data_root <path-to-query-images-dir> --query_annotation_path <path-to-query-annotation-file> --gallery_data_root <path-to-gallery-images-dir> --gallery_annotation_path <path-to-gallery-annotation-file>
```

# Trained models 💥
Models were taken from [torchreid model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html). Train and test dataset: [VRIC](https://qmul-vric.github.io/). Image size: *208x208*

| model            | Num. params (M) | GFLOPS | CMC Rank-1 (loss=softmax) | CMC Rank-1 (loss=triplet) | mAP (loss=softmax) | mAP (loss=triplet) |
|------------------|-----------------|--------|---------------------------|---------------------------|--------------------|-------------------|
| resnet50_fc512   | 24.56           | 5.35   | 0.649                     | -                         | 0.732              | -                 |
| resnet50         | 23.51           | 3.62   | 0.515                     | -                         | 0.629              | -                 |
| mlfn             | 32.47           | 3.75   | 0.582                     | -                         | 0.686              | -                 |
| shufflenet       | 0.9             | 0.12   | 0.482                     | -                         | 0.593              | -                 |
| mobilenetv2_x1_0 | 2.22            | 0.28   | 0.549                     | 0.539                     | 0.660              | 0.651             |
| mobilenetv2_x1_4 | 4.29            | 0.53   | 0.555                     | 0.548                     | 0.664              | 0.661             |
| osnet_x1_0       | 2.19            | 1.29   | 0.682                     | 0.695                     | 0.756              | 0.768             |
| osnet_x0_75      | 1.3             | 0.75   | 0.678                     | 0.681                     | 0.752              | 0.757             |
| osnet_x0_5       | 0.64            | 0.36   | 0.653                     | 0.662                     | 0.731              | 0.734             |
| osnet_x0_25      | 0.2             | 0.11   | 0.580                     | 0.575                     | 0.673              | 0.668             |
| osnet_ibn_x1_0   | 2.19            | 1.29   | 0.667                     | 0.666                     | 0.742              | 0.739             |
| osnet_ain_x1_0   | 2.19            | 1.29   | 0.687                     | 0.670                     | 0.757              | 0.745             |
| osnet_ain_x0_75  | 1.3             | 0.75   | 0.652                     | 0.665                     | 0.723              | 0.738             |
| osnet_ain_x0_5   | 0.64            | 0.36   | 0.639                     | 0.642                     | 0.719              | 0.722             |
| osnet_ain_x0_25  | 0.2             | 0.11   | 0.570                     | 0.566                     | 0.657              | 0.654             |

In order reproduce these trains run train.py from example folder. The VRIC dataset should be located at the path data/VRIC.

# Visualize train 📈
``` shell
tensorboard --logdir=<your_log_path>
```

# Citations
```
@inproceedings{2018gcpr-Kanaci,
  author={Aytac Kanaci and Xiatian Zhu and Shaogang Gong},
  title={Vehicle Re-Identification in Context},
  booktitle={Pattern Recognition - 40th German Conference, {GCPR} 2018, Stuttgart, Germany, September 10-12, 2018, Proceedings},
  year={2018}
}

@article{torchreid,
  title={Torchreid: A Library for Deep Learning Person Re-Identification in Pytorch},
  author={Zhou, Kaiyang and Xiang, Tao},
  journal={arXiv preprint arXiv:1910.10093},
  year={2019}
}

@inproceedings{zhou2019osnet,
  title={Omni-Scale Feature Learning for Person Re-Identification},
  author={Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
  booktitle={ICCV},
  year={2019}
}

@article{zhou2021osnet,
  title={Learning Generalisable Omni-Scale Representations for Person Re-Identification},
  author={Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
  journal={TPAMI},
  year={2021}
}
```


