## Installation
安装

### Dependencies
依赖

The original library is build with

- python=3.8.8
- torch=1.10.2
- torchvision=0.11.3

while using decord module to read original videos (so that you don't need to make any transform on your original .mp4 input).

To get all the requirements, please run

```shell
pip install -r requirements.txt
```

## Usage
使用方法
### Train
训练
### Train your option score
```
python train.py -o [YOUR_OPTIONS]
```
### Such as train color score
```
python train.py -o options/train/train-color.yml
```
### Test
测试

### Test your option score
```
python test.py -o [YOUR_OPTIONS]
```
OR
```
python test_[YOUR_OPTIONS].py
```

### Such as test color score
```
python test.py -o options/test/test-color.yml
```
OR
```
python test_color.py
```
