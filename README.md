## MobileNet
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Shubhamai/pytorch-mobilenet/blob/main/LICENSE)

[`model.py`](/model.py)

This repo contains the implementation of the original MobileNet from the 2017 paper [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf) by Google. 

## Table of Contents
- [MobileNet](#mobilenet)
- [Table of Contents](#table-of-contents)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

The key takeways for this paper are -
1.  Using Depthwise Seperable Convolution with Pointwise Layer instead of standard convolution to substantially decrease the number of parameters making CNN viable for mobile & embedded devices.   



## Usage


## Acknowledgements

I found these resources helpful to understand MobileNet and Depthwise Seperable Convolution
- [MobileNet Research Paper Walkthrough](https://youtu.be/HD9FnjVwU8g) by Rahul Deora
- [depthwise separable convolution | Complete tensor operations for MobileNet architecture](https://youtu.be/vfCvmenkbZA) by When Maths Meet Coding
- [Depthwise Separable Convolution - A FASTER CONVOLUTION!](https://youtu.be/T7o3xvJLuHk) by CodeEmporium

## Citation
```
@misc{howard2017mobilenets,
    title={MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications},
    author={Andrew G. Howard and Menglong Zhu and Bo Chen and Dmitry Kalenichenko and Weijun Wang and Tobias Weyand and Marco Andreetto and Hartwig Adam},
    year={2017},
    eprint={1704.04861},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```