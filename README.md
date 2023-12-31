# Multinomial likelihood with Bi-Weighting (Mult-BiW)

This is our Tensorflow implementation for the paper:

>Tianyu Zhu, Jiandong Ding, Yansong Shi, Guoqing Chen, Jian-Yun Nie. "Mitigating Popularity Bias in Recommendation with Global Listwise Learning and Progressive Bi-Weighting."

## Introduction
Mult-BiW is a framework for popularity debiasing in item recommendation.

![](https://github.com/zhuty16/Mult-BiW/blob/main/framework.jpg)

## Citation

## Environment Requirement
The code has been tested running under Python 3.8. The required packages are as follows:
* tensorflow == 2.8.0+
* numpy == 1.23.0+
* scipy == 1.8.0+
* pandas == 1.5.0+

## Example to Run the Codes
```
python MF.py --dataset amazon --lr 1e-4 --l2_reg 1e-6 --alpha -1.0 --eta 1.0
python LightGCN.py --dataset amazon --lr 1e-3 --l2_reg 1e-6 --alpha -1.0 --eta 1.0 --num_layer 2
```

