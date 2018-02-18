import mxnet as mx
import numpy as np
import logging
import argparse
import time
from symbol_resnetv1 import ResNetV1 
from moduleEXT import ModuleEXT
import os
try:
    import Queue
except:
    import queue as Queue

MEAN_COLOR = [103.06, 115.90, 123.15][::-1] # RGB  

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def load_checkpoint(prefix, epoch):
    """
    Load model checkpoint from file.
    :param prefix: Prefix of model name.
    :param epoch: Epoch number of model we would like to load.
    :return: (arg_params, aux_params)
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    """
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params


def multi_factor_scheduler(begin_epoch, epoch_size, step=[60, 75, 90], factor=0.1):
    step_ = [epoch_size * (x-begin_epoch) for x in step if x-begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None
    
def main():
    begin_epoch = args.model_load_epoch if args.model_load_epoch else 0
    kv = mx.kvstore.create("device")
    epoch_size = max(int(args.num_examples / args.batch_size / kv.num_workers), 1)
    
    resnetv1 = ResNetV1()
    sym = resnetv1.get_resnet50(num_classes = args.num_classes)


    devs = mx.cpu() if len(args.gpus) == 0 else [mx.gpu(int(i)) for i in args.gpus.split(",")]
    print ("Using Device {}".format(devs))

    train_iter = mx.io.ImageRecordIter(
        path_imgrec =   os.path.join(args.data_path, "train256.rec"),
        path_imglist = os.path.join(args.data_path, "train.lst"),
        data_name   =   'data',
        label_name  =   'softmax_label',
        data_shape  =   (3, 224, 224),
        batch_size  =   args.batch_size, 
        rand_crop   =   True,
        max_random_scale    =   288.0 / 256.0,
        min_random_scale    =   224.0 / 256.0, 
        max_aspect_ratio    =   0.25,
        mean_r      =   MEAN_COLOR[0],
        mean_g      =   MEAN_COLOR[1],
        mean_b      =   MEAN_COLOR[2],
        random_h    =   20,
        random_s    =   40,
        random_l    =   50,
        max_rotate_angle    =   10,
        max_shear_ratio     =   0.1,
        rand_mirror =   True,
        shuffle     =   True
    )
    
    val_iter = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(args.data_path, "val256.rec"),
        path_imglist = os.path.join(args.data_path, "val.lst"),
        data_name   =   'data',
        label_name  =   'softmax_label',
        data_shape  =   (3, 224, 224),
        batch_size  =   args.batch_size, 
        rand_crop   =   False,
        mean_r      =   MEAN_COLOR[0],
        mean_g      =   MEAN_COLOR[1],
        mean_b      =   MEAN_COLOR[2],
        rand_mirror =   False,
        shuffle     =   False
    )
    
    model = ModuleEXT(
            context = devs,
            symbol = sym,
            data_names = ("data", ),
            label_names = ("softmax_label", ),
    )

    if begin_epoch == 0:
        arg_params, aux_params = load_checkpoint(args.pretrain_prefix, args.pretrain_epoch)
    else:
        # Load Params
        _, arg_params, aux_params = mx.model.load_checkpoint(args.model_prefix, begin_epoch)

        
    initializer = mx.init.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2)
    
    optimizer = mx.optimizer.SGD(learning_rate = args.lr, 
                                    momentum = args.mom, 
                                    wd = args.wd, 
                                    #lr_scheduler = multi_factor_scheduler(begin_epoch, epoch_size, step = [30, 60, 90], factor = 0.1),
                                    rescale_grad = 1.0 / args.batch_size,
                                    sym = sym)

    checkpoint = mx.callback.module_checkpoint(model, args.model_prefix, save_optimizer_states = True)

    print ("Start to fit the model")
    model.fit(
            train_data = train_iter,
            eval_data = val_iter, 
            eval_metric = [mx.metric.CrossEntropy(), mx.metric.Accuracy()], 

            begin_epoch = begin_epoch,
            num_epoch = args.num_epoch,
            
            optimizer = optimizer,
            
            arg_params = arg_params,
            aux_params = aux_params,
            
            initializer = initializer, 
            allow_missing = False,
            
            batch_end_callback = mx.callback.Speedometer(args.batch_size, args.frequent),
            epoch_end_callback = checkpoint
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "ResNet")
    parser.add_argument("--gpus", type = str, default = "0,1", help = "the gpus will be used, e.g '0,1'")
    parser.add_argument("--lr", type = float, default = 1e-4, help = "learning rate")
    parser.add_argument("--batch-size", type = int, default = 128*2, help = "batch size")
    parser.add_argument("--num-epoch", type = int, default = 120)
    parser.add_argument("--num-classes", type = int, default = 1000)
    parser.add_argument("--frequent", type = int, default = 50, help = "frequency of logging")
    parser.add_argument("--mom", type = float, default = 0.9, help = "momentum for optimizer")
    parser.add_argument("--wd", type = float, default = 0.0001, help = "weight decay for optimizer")
    parser.add_argument("--data-path", type = str, default = "./", help = "the path of the data")
    parser.add_argument("--num-examples", type = int, default = 80000, help = "the number of training examples")
    parser.add_argument("--pretrain-prefix", type = str, default = "./resnet-v1-50-rgb", help = "the prefix name of pretrain model")
    parser.add_argument("--pretrain-epoch", type = int, default = 0, help = 'load the pretrain model on an epoch using the pretrain-epoch')
    parser.add_argument("--model-prefix", type = str, default = "./models/resnet", help = "the prefix name of the model")
    parser.add_argument("--model-load-epoch", type = int, default = 0, help = 'load the model on an epoch using the model-load-prefix')
    parser.add_argument("--log-path", type = str, default = "./logs", help = "the path of the logs")
    args = parser.parse_args()
    
    ts = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    hdlr = logging.FileHandler(os.path.join(args.log_path, "log-%s.log") % (ts))
    logger.addHandler(hdlr)

    print (args)
    logging.info(args)
    main()
