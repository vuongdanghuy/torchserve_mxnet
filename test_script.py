import logging
import os
# Uncomment following line will cause error
# import torch

import mxnet as mx


if __name__ == "__main__":

    ctx_id = 0
    prefix = "src/model/resnet-50"
    epoch = 0

    # load model
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    if ctx_id >= 0:
        ctx = mx.gpu(ctx_id)
    else:
        ctx = mx.cpu()

    model = mx.mod.Module(
        symbol=sym,
        context=ctx,
        label_names=None
    )
    model.bind(
        data_shapes=[('data', (1, 3, 640, 640))],
        for_training=False
    )
    model.set_params(arg_params, aux_params)
    logging.info("="*16 + " SUCCESS " + "="*16)
