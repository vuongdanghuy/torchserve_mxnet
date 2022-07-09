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
    logging.info("="*16 + " SUCCESS " + "="*16)
