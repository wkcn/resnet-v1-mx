import mxnet as mx
from mxnet import ndarray as nd

model_name = "resnet-v1-101"

data = nd.load('./%s.params' % model_name)

new_data = dict()
first_conv_weight = 'arg:conv1_weight'
first_conv_bias = 'arg:conv1_bias'
print (data[first_conv_weight].shape)
for k, v in data.items():
    if k == first_conv_weight:
        # change Channel
        v = mx.nd.array(v.asnumpy()[:, ::-1, :, :])
        print ("CHANGE")
    elif k == first_conv_bias:
        print (v.shape)
    new_data[k] = v

nd.save('%s-rgb-0000.params' % model_name, new_data)
