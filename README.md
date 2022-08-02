## MobileNet
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Shubhamai/pytorch-mobilenet/blob/main/LICENSE)

Standalone models files - [`MobileNetV1.py`](/MobileNetV1.py)
[`MobileNetV2.py`](/MobileNetV2.py)


This repo contains the implementation of MobileNetV1 and V2 from the 2017 paper [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf) and 2019 paper [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) respectively. 

## Table of Contents
- [MobileNet](#mobilenet)
- [Table of Contents](#table-of-contents)
- [Usage](#usage)
- [Notes](#notes)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

The key takeaways for this papers are -

1. MobileNetv1
   1.  Using Depthwise Seperable Convolution with Pointwise Convolution instead of standard convolution to substantially decrease the number of parameters making CNN more viable for mobile & embedded devices.   



## Usage


## Notes
- In `MobileNetV1` paper, there was no mention of using `ReLU6` as activation function, but since I found most blogs and resources using ReLU6,  I decided to use that by default. But in case, I added an extra parameter `use_relu6` ( defaults to `true` ) in allows the option to either use `ReLU` or `ReLU6`.  

## Acknowledgements

I found these resources helpful to understand MobileNet and Depthwise Seperable Convolution

- MobileNetV1
    - [MobileNet Research Paper Walkthrough](https://youtu.be/HD9FnjVwU8g) by Rahul Deora
    - [depthwise separable convolution | Complete tensor operations for MobileNet architecture](https://youtu.be/vfCvmenkbZA) by When Maths Meet Coding
    - [Depthwise Separable Convolution - A FASTER CONVOLUTION!](https://youtu.be/T7o3xvJLuHk) by CodeEmporium
- MobileNetV2
    - [MobileNetV2 and EfficientNet](https://youtu.be/IBndcd4UfTs) by Rahul Deora
    - [New mobile neural network architectures](https://machinethink.net/blog/mobile-architectures/) by Matthijs Hollemans

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


@misc{s2018mobilenetv2,
    title={MobileNetV2: Inverted Residuals and Linear Bottlenecks},
    author={Mark Sandler and Andrew Howard and Menglong Zhu and Andrey Zhmoginov and Liang-Chieh Chen},
    year={2018},
    eprint={1801.04381},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```