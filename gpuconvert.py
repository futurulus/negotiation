import pickle

from stanza.research import config
import thutils

parser = config.get_options_parser()
parser.add_argument('--from_device', default='',
                    help='The original device of the model.')
parser.add_argument('--to_device', default='',
                    help='The new device of the saved model.')
parser.add_argument('--load', metavar='MODEL_FILE', default='',
                    help='Name of the pickle file to load the model from.')
parser.add_argument('--save', metavar='MODEL_FILE', default='',
                    help='Name of the pickle file to save the model to.')


def convert_gpu():
    options = config.options()

    with open(options.load, 'rb') as infile:
        with thutils.device_context(options.from_device):
            learner = pickle.load(infile)
    model = learner.model
    with thutils.device_context(options.to_device):
        model.module = thutils.maybe_cuda(model.module)
        model.loss = thutils.maybe_cuda(model.loss)
        model.build_optimizer()
    learner.options.device = options.to_device
    with open(options.save, 'wb') as outfile:
        learner.dump(outfile)


if __name__ == '__main__':
    convert_gpu()
