import mxnet as mx

class ResNetV1:
    def __init__(self):
        self.bn_eps = 1e-5
        self.bn_mom = 0.999
    def conv_bn(self, x, num_filter, kernel, stride, pad, name):
        if name == "conv1":
            bn_name = 'bn_conv1'
        else:
            bn_name = 'bn%s' % (name[3:] if name[:3] == 'res' else name)

        x = mx.sym.Convolution(data = x, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad, no_bias = True, name = name)
        x = mx.sym.BatchNorm(data = x, fix_gamma = False, use_global_stats = False, eps = self.bn_eps, momentum = self.bn_mom, name = bn_name) 
        return x

    def conv_bn_act(self, x, num_filter, kernel, stride, pad, name):
        relu_name = '%s_relu' % name 
        x = self.conv_bn(x, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad, name = name)
        x = mx.sym.Activation(data = x, act_type = 'relu', name = relu_name)
        return x

    def residual_block(self, x, num_filter, unit_id, block_id, same_shape, use_alpha, bottleneck):
        '''
            num_filter is the #filter of the last conv in the block
        '''

        
        if use_alpha:
            u = chr(ord('a') + block_id)
        else:
            u = 'b%d' % block_id
        prefix = ('res%d' % unit_id) + u

        a_stride = (2, 2) if unit_id >= 3 and block_id == 0 else (1, 1)

        if same_shape:
            bypass = x
        else:
            bypass = self.conv_bn(x, num_filter = num_filter, kernel = (1, 1), stride = a_stride, pad = (0, 0), name = '%s_branch1' % prefix)

        if bottleneck:
            # A
            x = self.conv_bn_act(x, num_filter = num_filter // 4, kernel = (1, 1), stride = a_stride, pad = (0, 0), name = '%s_branch2a' % prefix)
            # B
            x = self.conv_bn_act(x, num_filter = num_filter // 4, kernel = (3, 3), stride = (1, 1), pad = (1, 1), name = '%s_branch2b' % prefix)
            # C
            x = self.conv_bn(x, num_filter = num_filter, kernel = (1, 1), stride = (1, 1), pad = (0, 0), name = '%s_branch2c' % prefix)
        else:
            # A
            x = self.conv_bn_act(x, num_filter = num_filter, kernel = (3, 3), stride = a_stride, pad = (1, 1), name = '%s_branch2a' % prefix)
            # B
            x = self.conv_bn_act(x, num_filter = num_filter, kernel = (3, 3), stride = (1, 1), pad = (1, 1), name = '%s_branch2b' % prefix)

        x = mx.sym.Activation(data = x + bypass, act_type = 'relu', name = "%s_relu" % prefix)

        return x

    def residual_unit(self, x, num_filter, num_blocks, unit_id, bottleneck, use_all_alpha):
        use_alpha = num_blocks <= 3 or use_all_alpha
        x = self.residual_block(x, num_filter, unit_id, block_id = 0, same_shape = False, use_alpha = True, bottleneck = bottleneck)
        for i in range(1, num_blocks):
            x = self.residual_block(x, num_filter, unit_id, block_id = i, same_shape = True, use_alpha = use_alpha, bottleneck = bottleneck)
        return x

    def get_symbol(self, num_classes, units, bottleneck = True):
        '''
        Return ResNet symbol
        Parameters
        ----------
        num_classes: int
            The number of classes
        units : list
            The list of residual units (num_blocks, last channels)
        bottleneck: bool
            Whether to use bottleneck
        '''

        num_layers_each_block = 3 if bottleneck else 2
        num_layers = num_layers_each_block * sum(map(lambda x : x[0], units)) + 2 
        use_all_alpha = (num_layers <= 50)

        x = mx.sym.Variable(name = 'data')

        x = self.conv_bn_act(x, num_filter = 64, kernel = (7, 7), stride = (2, 2), pad = (3, 3), name = 'conv1')

        x = mx.sym.Pooling(data = x, kernel = (3, 3), pool_type = 'max', stride = (2, 2), pooling_convention = 'full', name = 'pool1') 

        for i, u in enumerate(units):
            x = self.residual_unit(x, num_filter = u[1], num_blocks = u[0], unit_id = i + 2, bottleneck = bottleneck, use_all_alpha = use_all_alpha)

        x = mx.sym.Pooling(data = x, global_pool = True, kernel = (7, 7), pool_type = 'avg', name = 'pool5') 
        x = mx.sym.FullyConnected(data = x, num_hidden = num_classes, flatten = True, name = 'fc%d' % num_classes, attr = {'__lr_mult__' : 1.0,
                '__initializer__' : "xavier"})
        x = mx.sym.SoftmaxOutput(data = x, name = 'softmax')
        return x

    def get_resnet18(self, num_classes = 1000):
        units = [
                (2, 64),
                (2, 128),
                (2, 256),
                (2, 512)
        ]
        return self.get_symbol(num_classes, units, bottleneck = False) 

    def get_resnet34(self, num_classes = 1000):
        units = [
                (3, 64),
                (4, 128),
                (6, 256),
                (3, 512)
        ]
        return self.get_symbol(num_classes, units, bottleneck = False) 

    def get_resnet50(self, num_classes = 1000):
        units = [
                (3, 256),
                (4, 512),
                (6, 1024),
                (3, 2048)
        ]
        return self.get_symbol(num_classes, units, bottleneck = True) 

    def get_resnet101(self, num_classes = 1000):
        units = [
                (3, 256),
                (4, 512),
                (23, 1024),
                (3, 2048)
        ]
        return self.get_symbol(num_classes, units, bottleneck = True) 

    def get_resnet152(self, num_classes = 1000):
        units = [
                (3, 256),
                (8, 512),
                (36, 1024),
                (3, 2048)
        ]
        return self.get_symbol(num_classes, units, bottleneck = True) 

