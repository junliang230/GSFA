
# GUIDED SAMPLING BASED FEATURE AGGREGATION FOR VIDEO OBJECT DETECTION

Official code for GUIDED SAMPLING BASED FEATURE AGGREGATION FOR VIDEO OBJECT DETECTION.

## Installation
Our GSFA is based on [mmdetection](https://github.com/open-mmlab/mmdetection). Please refer to [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.

<br>

## Train

### Train with a single GPU

```shell
python tools/train.py ${CONFIG_FILE}
# python ./tools/train.py configs/faster_rcnn_r101_caffe_c4_1x_gsfa.py
```

### Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
# ./tools/dist_train.sh configs/faster_rcnn_r101_caffe_c4_1x_gsfa.py 4
```

<br>

## Test


```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]
# python ./tools/test.py configs/faster_rcnn_r101_caffe_c4_1x_gsfa.py ./work_dirs/faster_rcnn_r101_caffe_c4_1x_gsfa/epoch_12.pth --out result_12.pkl --eval bbox

```

<br>

## Acknowledgement
The implement of our GSFA is based on [mmdetection](https://github.com/open-mmlab/mmdetection), which is an open source object detection toolbox.

<br>

[![License](https://img.shields.io/badge/license-Apache-blue.svg)](LICENSE)
