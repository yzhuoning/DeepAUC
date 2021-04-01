# Deep AUC Maximization  [![pdf](https://img.shields.io/badge/Arxiv-pdf-orange.svg?style=flat)](https://arxiv.org/abs/1908.10831)

This is the official implementation of the paper "**Stochastic AUC Maximization with Deep Neural Networks**" published on **ICLR2020**. 

## Installation
```
Python=3.5
Numpy=1.18.5 
Scipy=1.2.1
Scikit-Learn=0.20.3
Pillow=5.0.0
Tensorflow>=1.10.0
```

### Run
```
python PPD_SG.py/PPD_AdaGrad.py --dataset=10 --train_batch_size=128 --use_L2=False --split_index=4 --lr=0.01 --keep_index=0.1 --t0=200
```

### Hyperparameter tuning
```
gamma=[500, 1000, 2000, ...]
eta = [0.1, 0.01, ...]
T0=[1000, 2000, 3000, ...,]
```

## Bibtex 
If you use this repository in your work, please cite our paper:

```
@article{yuan2020robust,
title={Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification},
author={Yuan, Zhuoning and Yan, Yan and Sonka, Milan and Yang, Tianbao},
journal={arXiv preprint arXiv:2012.03173},
year={2020}
}
```

