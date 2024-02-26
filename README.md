# Vehicle ReID 🚗🚘

# Installation

Install via conda:
``` shell
git clone ...
cd ...
pip install -r requirements.txt
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
python setup.py develop
```

# Train example
``` python
python train.py osnet_x1_0 --log_dir ./log/osnet_x1_0 --bs 16 --loss triplet --lr 0.0003 --max_epoch 60
```

# Test example
``` python
```

# How to use custom dataset



# Trained models 💥
Models were taken from [torchreid model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html). Dataset: [VRIC](https://qmul-vric.github.io/).

**CMC Rank-1**
| model name       | loss=softmax | loss=triplet |
|------------------|--------------|--------------|
| resnet50_fc512   | 0.649        | -            |
| resnet50         | 0.515        | -            |
| mlfn             | 0.582        | -            |
| shufflenet       | 0.482        | -            |
| mobilenetv2_x1_0 | 0.549        | 0.539        |
| mobilenetv2_x1_4 | 0.555        | 0.548        |
| osnet_x1_0       | 0.682        | 0.695        |
| osnet_x0_75      | 0.678        | 0.681        |
| osnet_x0_5       | 0.653        | 0.662        |
| osnet_x0_25      | 0.580        | 0.575        |
| osnet_ibn_x1_0   | 0.667        | 0.666        |
| osnet_ain_x1_0   | 0.687        | 0.670        |
| osnet_ain_x0_75  | 0.652        | 0.665        |
| osnet_ain_x0_5   | 0.639        | 0.642        |
| osnet_ain_x0_25  | 0.570        | 0.566        |

**mAP**
| model name       | loss=softmax | loss=triplet |
|------------------|--------------|--------------|
| resnet50_fc512   | 0.732        | -            |
| resnet50         | 0.629        | -            |
| mlfn             | 0.686        | -            |
| shufflenet       | 0.593        | -            |
| mobilenetv2_x1_0 | 0.660        | 0.651        |
| mobilenetv2_x1_4 | 0.664        | 0.661        |
| osnet_x1_0       | 0.756        | 0.768        |
| osnet_x0_75      | 0.752        | 0.757        |
| osnet_x0_5       | 0.731        | 0.734        |
| osnet_x0_25      | 0.673        | 0.668        |
| osnet_ibn_x1_0   | 0.742        | 0.739        |
| osnet_ain_x1_0   | 0.757        | 0.745        |
| osnet_ain_x0_75  | 0.723        | 0.738        |
| osnet_ain_x0_5   | 0.719        | 0.722        |
| osnet_ain_x0_25  | 0.657        | 0.654        |

# Visualize train
``` shell
tensorboard --logdir=<your_log_path>
```


