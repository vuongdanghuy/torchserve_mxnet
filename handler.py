import logging
import os
import torch

import mxnet as mx


class FaceDetectHandler:
    def __init__(self) -> None:
        self.model = None
        self.initialized = False
        self.context = None
        self.manifest = None

    def initialize(self, context):
        properties = context.system_properties
        if (torch.cuda.is_available() and
                properties.get("gpu_id") is not None):
            ctx_id = properties.get("gpu_id")
        else:
            ctx_id = -1

        self.manifest = context.manifest
        model_dir = properties.get("model_dir")
        prefix = os.path.join(model_dir, "model/resnet-50")

        # load model
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, 0)
        if ctx_id >= 0:
            self.ctx = mx.gpu(ctx_id)
        else:
            self.ctx = mx.cpu()
        self.model = mx.mod.Module(symbol=sym,
                                   context=self.ctx,
                                   label_names=None)
        self.model.bind(
            data_shapes=[('data', (1, 3, 640, 640))],
            for_training=False
        )
        self.model.set_params(arg_params, aux_params)
        self.initialized = True
        logging.info("Loading ArcFace model done")

    def preprocess(self, requests):
        pass

    def inference(self, data):
        pass

    def postprocess(self, data):
        pass

    def handle(self, requests, context):
        output = []
        logging.info("-"*16 + " SUCCESS " + "-"*16)

        return output
