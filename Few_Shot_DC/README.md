# [ICLR2021 Oral] Free Lunch for Few-Shot Learning: Distribution Calibration

- [Paper](https://openreview.net/forum?id=JWOiYxMG92s)

## Requirements

### Install libararies

```
$ pip3 install -r requirements.txt 
```

### Download the dataset and create base/val/novel splits

The original link has been changed.

1. Download dataset at [Google Drive](https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view)
2. Move the datasets at `./filelists/CUB`
3. `source ./parse_CUB.sh`

## Train feature extractor

```
$ python3 train.py --dataset CUB
```

## Extract and save features

```
$ python3 save_plk.py --dataset CUB 
```

## Evaluate our distribution calibration

```
$ python3 evaluate_DC.py
```