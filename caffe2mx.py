import caffe
import mxnet as mx

def get_params(caffe_net, caffe_model):
    nn = caffe.Net(caffe_net, caffe_model, caffe.TEST)
    arg_params = dict()
    aux_params = dict()
    for name in nn.params.keys():
        p = nn.params[name] 
        if name.startswith('bn'):
            factor = 0 if p[2].data[0] == 0 else 1.0 / p[2].data[0]
            aux_params["%s_moving_mean" % (name)] = mx.nd.array(p[0].data) * factor
            aux_params["%s_moving_var" % (name)] = mx.nd.array(p[1].data) * factor
        elif name.startswith('scale'):
            bn_name = name.replace("scale", "bn")
            arg_params["%s_gamma" % (bn_name)] = mx.nd.array(p[0].data)
            arg_params["%s_beta" % (bn_name)] = mx.nd.array(p[1].data)
        else:
            arg_params["%s_weight" % (name)] = mx.nd.array(p[0].data) 
            if len(p) > 1:
                # bias
                arg_params["%s_bias" % (name)] = mx.nd.array(p[1].data) 
    return arg_params, aux_params

def save_params(filename, arg_params, aux_params):
    data = {}
    for name, value in arg_params.items():
        data["arg:%s" % name] = value
    for name, value in aux_params.items():
        data["aux:%s" % name] = value
    mx.nd.save(filename, data)

if __name__ == "__main__":
    CAFFE_NET = './ResNet-101-deploy.prototxt'
    CAFFE_MODEL = './ResNet-101-model.caffemodel'
    arg_params, aux_params = get_params(CAFFE_NET, CAFFE_MODEL)
    save_params("resnet-v1-101.params", arg_params, aux_params)
    print ("Save OK")
