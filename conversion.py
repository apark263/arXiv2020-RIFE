import os
import torch
import numpy as np
from torch.nn import functional as F
import warnings
from model.IFNet_HDv3 import IFNet
from absl import app
from absl import flags
import coremltools as ct

warnings.filterwarnings("ignore")

FLAGS = flags.FLAGS
flags.DEFINE_string('output_dir', None, 'Directory to write images to.')
flags.DEFINE_string('model_dir', 'train_log', 'Model Directory.')


class WrappedFlowNet(torch.nn.Module):

    def __init__(self, model_dir):
        super(WrappedFlowNet, self).__init__()
        params = torch.load(f'{model_dir}/flownet.pkl', 'cpu')
        params = {k.replace('module.', ''): v for k, v in params.items()}
        self.model = IFNet()
        self.model.load_state_dict(params)
        self.model.eval()

    def forward(self, x):
        _, _, (_, _, y) = self.model(y)
        return y


def main(argv):
    del argv    # Unused
    params=torch.load(f'{FLAGS.model_dir}/flownet.pkl', 'cpu')
    params={
                    k.replace("module.", ""): v
                    for k, v in params.items()
                    if "module." in k
                }
    torch_model = WrappedFlowNet()
    # Trace with random data
    example_input = torch.rand(1, 3, 224, 224)
    # after test, will get 'size mismatch' error message with size 256x256
    traced_model = torch.jit.trace(torch_model, example_input)
    # Convert to Core ML using the Unified Conversion API
    model = ct.convert(
        traced_model,
        # name "input_1" is used in 'quickstart'
        inputs=[ct.ImageType(name="input_1", shape=example_input.shape)],
    )
    model.save(f'{FLAGS.output_dir}/flownet.mlmodel')


if __name__ == '__main__':
    app.run(main)
