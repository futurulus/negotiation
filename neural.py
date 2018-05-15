import argparse
import datetime
import numpy as np
import numbers
import torch as th

from stanza.monitoring import summary
from stanza.research import config

from thutils import to_numpy, to_native, maybe_cuda as cu

parser = config.get_options_parser()
parser.add_argument('--monitor_activations', type=config.boolean, default=False,
                    help='If True, output scalar or histogram summaries for tensors stored '
                         'in module.monitor_activations.')
parser.add_argument('--monitor_params', type=config.boolean, default=False,
                    help='If True, output scalar or histogram summaries for all parameters of '
                         'the module.')
parser.add_argument('--monitor_grads', type=config.boolean, default=False,
                    help='If True, output scalar or histogram summaries for gradients with '
                         'respect to all parameters of the module.')
parser.add_argument('--grad_clip', type=float, default=5.0,
                    help='The maximum norm of the gradients to use in training, to prevent '
                         'exploding gradient problems.')


class TorchModel():
    def __init__(self, module, loss, optimizer, optimizer_params, vectorizer):
        self.get_options()
        self.module = cu(module)
        self.loss = cu(loss)
        self.optimizer_class = optimizer
        self.optimizer_params = optimizer_params
        self.build_optimizer()
        self.vectorizer = vectorizer
        summary_path = config.get_file_path('monitoring.tfevents')
        if summary_path:
            self.summary_writer = summary.SummaryWriter(summary_path)
        else:
            self.summary_writer = None
        self.step = 0
        self.last_timestamp = datetime.datetime.now()

    def build_optimizer(self):
        if not hasattr(self, 'optimizer_class') or not hasattr(self, 'optimizer_params'):
            import warnings
            warnings.warn('Cannot recreate optimizer: old pickle file does not have '
                          'saved optimizer parameters.')
            self.optimizer = None
            return
        self.optimizer = self.optimizer_class(self.module.parameters(), **self.optimizer_params)

    def get_options(self):
        if not hasattr(self, 'options'):
            options = config.options()
            self.options = argparse.Namespace(**options.__dict__)

    def train(self, tuples):
        tuples = list(tuples)
        arrays = self.vectorizer.vectorize_all(tuples)

        self.module.train()
        self.module.zero_grad()

        before = self.transform_and_predict(arrays)

        loss = self.loss(*before)
        loss.backward()
        self.monitor(loss, len(tuples))

        th.nn.utils.clip_grad_norm(self.module.parameters(), self.options.grad_clip)
        self.optimizer.step()

        return self.unvectorize(*before)

    def transform_and_predict(self, arrays):
        return self.module(*(th.autograd.Variable(cu(th.from_numpy(a))) for a in arrays))

    def monitor(self, loss, num_examples):
        new_timestamp = datetime.datetime.now()
        examples_per_sec = num_examples / (new_timestamp - self.last_timestamp).total_seconds()

        if self.summary_writer:
            self.summary_writer.log_scalar(self.step, 'loss', to_numpy(loss))
            self.summary_writer.log_scalar(self.step, 'examples_per_sec', examples_per_sec)

            if self.options.monitor_activations and hasattr(self.module, 'activations'):
                for k, v in self.module.activations.__dict__.items():
                    self.log_scalar_or_histogram('activations/{}'.format(k), v)

            if self.options.monitor_params:
                for k, v in self.module.named_parameters():
                    self.log_scalar_or_histogram('params/{}'.format(k), v)

            if self.options.monitor_grads:
                for k, p in self.module.named_parameters():
                    self.log_scalar_or_histogram('grads/{}'.format(k), p.grad)

        self.last_timestamp = new_timestamp
        self.step += 1

    def log_scalar_or_histogram(self, k, v):
        if v is None:
            import warnings
            warnings.warn('monitored value "{}" is None'.format(k))
            return
        elif isinstance(v, numbers.Number):
            log = self.summary_writer.log_scalar
        elif np.prod(v.size()) == 1:
            log = self.summary_writer.log_scalar
            v = float(to_numpy(v).squeeze())
        else:
            log = self.summary_writer.log_histogram
            v = to_numpy(v)
        log(self.step, k, v)

    def unvectorize(self, predict, score):
        return {
            k: self.vectorizer.unvectorize_all(*to_numpy(v))
            for k, v in predict.items()
        }, to_native(score)

    def eval(self, pairs, split='eval'):
        arrays = self.vectorizer.vectorize_all(pairs)
        self.module.eval()
        self.module.apply(self.split(split))
        predict, score = self.transform_and_predict(arrays)
        return self.unvectorize(predict, score)

    def split(self, split):
        def apply_split(module):
            if hasattr(module, 'split') and hasattr(module.split, '__call__'):
                module.split(split)

        return apply_split


class Activations():
    '''
    A do-nothing class for storing neural net activations to be logged as Tensorboard
    summaries by TorchModel. Store as self.activations (where self is your module),
    and register an activation tensor for logging in the forward method:

        self.activations.encoder_repr = self.encoder(input)

    This will cause the output of self.encoder to be logged as a histogram (assuming
    the encoder produces more than one number as output) as "activations/encoder_repr".

    Activations can be Torch Tensors, Variables, or NumPy arrays.

    Note: holding onto lots of activations increases memory use. Use sparingly.
    '''
    pass
