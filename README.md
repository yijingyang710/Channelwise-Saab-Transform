# Channelwise-Saab-Transform
Feature extraction (Module 1) packages for PixelHop/PixelHop++.

### Introduction
This is an implementation by Yijing Yang for the feature extraction part in the paper by Chen et.al. [PixelHop++: A Small Successive-Subspace-Learning-Based (SSL-based) Model for Image Classification](https://arxiv.org/abs/2002.03141). 

It is modified based on Chengyao Wang's [implementation](https://github.com/ChengyaoWang/PixelHop-_c-wSaab) (**ObjectOriented / Numpy** version), with lower memory cost. 

Note that this is not the official implementation. 

### Installation
This code has been tested with Python 3.7 and Python 3.8. Other dependent packages include: numpy, scikit-image, numba and scikit-learn.

### Contents
* `saab.py`: Saab transform.
* `cwSaab.py`: Channel-wise Saab transform. Use energy threshold `TH1` and `TH2` to choose intermediate nodes and leaf nodes, respectively. Set `'cw'` to `'False'`  in order to turn off the channel-wise structure.
* `pixelhop.py`: Built upon `cwSaab.py` with additional functions of saving models, loading models, and concatenation operation across Hops.
* Example of usage can be found at the bottom of each file. 

  **Note**: All the images or data that are fed into these functions should be in the `channel last` format.
