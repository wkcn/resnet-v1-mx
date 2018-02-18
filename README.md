# ResNet v1 on MXNet

Implemenation of [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) on MXNet.

The weights of the model is converted from [the original Caffe implemenation](https://github.com/KaimingHe/deep-residual-networks)

## Illustration 

- predict_resnet.py

An example to predict a picture for MXNet Model

- predict_resnet_caffe.py

An example to predict a picture for Caffe Model

- caffe2mx.py

Convert the weights file for Caffe(*.caffemodel) into the weights file for MXNet(*.params) 

- convert_mean.py  

Convert the mean-value file for Caffe (*.binaryproto)

- convertRGB.py

Convert the model to RGB-order input rather than BGR-order

## Note

The MXNet model uses BGR-order channels too, and we can subtract the channel mean value.

## Performance

1-crop validation error on ImageNet (center 224x224 crop from resized image with shorter side=256):

Model | top-1 | top-5
------|-------|------
ResNet-101 (MXNet, Channel Mean) | 25.37% | 8.02%
ResNet-101 (MXNet, Pixel Mean) | 25.29% | 8.00%
*ResNet-101 (Caffe, Pixel Mean) | 23.6% | 7.1% 

*The result of Caffe is provied from [it](https://github.com/KaimingHe/deep-residual-networks).*

Although the weights of MXNet and Caffe are the same, there is *some little difference* between them.

The reason is [the loss of precision in MXNet](https://github.com/apache/incubator-mxnet/issues/9216).


## Citation

```
@article{He2015,
	author = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
	title = {Deep Residual Learning for Image Recognition},
	journal = {arXiv preprint arXiv:1512.03385},
	year = {2015}
}
```
