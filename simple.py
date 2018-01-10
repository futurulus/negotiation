#!/usr/bin/env python
r'''
python simple.py -R runs/simple_unicode \
                 --train_file data/unicode_train.json \
                 --validation_file data/unicode_val.json \
                 --eval_file data/unicode_dev.json \
                 --tokenizer character
'''
from stanza.research import config
config.redirect_output()

import datetime
import gzip
from itertools import islice
import json

from stanza.monitoring import progress
from stanza.research import evaluate, output, iterators
from stanza.research.instance import Instance

import metrics
import thutils
import tokenizers

from seq2seq import SimpleSeq2SeqLearner

parser = config.get_options_parser()
parser.add_argument('--device', default='',
                    help='The device to use in PyTorch ("cpu" or "gpu[0-n]"). If None, '
                         'pick a free-ish device automatically.')
parser.add_argument('--load', metavar='MODEL_FILE', default='',
                    help='If provided, skip training and instead load a pretrained model '
                         'from the specified path. If None or an empty string, train a '
                         'new model.')
parser.add_argument('--train_file', help='JSON file giving training sequences.')
parser.add_argument('--validation_file', default='',
                    help='JSON file giving validation sequences.')
parser.add_argument('--eval_file', help='JSON file giving evaluation sequences.')
parser.add_argument('--train_size', type=int, default=-1,
                    help='The number of examples to use in training. If negative, use the '
                         'whole training set.')
parser.add_argument('--validation_size', type=int, default=-1,
                    help="The number of examples to use in validation. If negative, use the "
                         "whole validation set. If 0 (or if the data_source doesn't have "
                         "a validation set), validation will be skipped.")
parser.add_argument('--eval_size', type=int, default=-1,
                    help='The number of examples to use in evaluation. '
                         'If negative, use the whole evaluation set.')
parser.add_argument('--train_epochs', type=int, default=15,
                    help='The number of epochs (passes through the dataset) for training.')
parser.add_argument('--tokenizer', choices=tokenizers.TOKENIZERS, default='unigram',
                    help='(De)tokenizer to split strings in dataset.')
parser.add_argument('--metrics', default=['accuracy', 'perplexity', 'log_likelihood_bits',
                                          'token_perplexity_micro'],
                    choices=metrics.METRICS.keys(),
                    help='The evaluation metrics to report for the experiment.')
parser.add_argument('--output_train_data', type=config.boolean, default=False,
                    help='If True, write out the training dataset (after cutting down to '
                         '`train_size`) as a JSON-lines file in the output directory.')
parser.add_argument('--output_eval_data', type=config.boolean, default=False,
                    help='If True, write out the evaluation dataset (after cutting down to '
                         '`eval_size`) as a JSON-lines file in the output directory.')
parser.add_argument('--progress_tick', type=int, default=10,
                    help='The number of seconds between logging progress updates.')


def dataset(filename):
    if not filename:
        return
    openfunc = gzip.open if filename.endswith('.gz') else open
    with openfunc(filename, 'r') as infile:
        for line in infile:
            yield Instance(**json.loads(line.strip()))


def nin(x):
    '''"None if negative"'''
    return None if x < 0 else x


def main():
    options = config.options()

    with thutils.device_context(options.device):
        progress.set_resolution(datetime.timedelta(seconds=options.progress_tick))

        SG = iterators.SizedGenerator

        if not hasattr(options, 'verbosity') or options.verbosity >= 2:
            print('Pre-calculating dataset sizes')
        train_data = SG(lambda: islice(dataset(options.train_file), 0, nin(options.train_size)),
                        length=None)
        if not hasattr(options, 'verbosity') or options.verbosity >= 4:
            print('Training set size: {}'.format(len(train_data)))

        validation_data = None
        if options.validation_file:
            validation_data = SG(lambda: islice(dataset(options.validation_file),
                                                0, nin(options.validation_size)),
                                 length=None)
            if not hasattr(options, 'verbosity') or options.verbosity >= 4:
                print('Validation set size: {}'.format(len(validation_data)))

        eval_data = SG(lambda: islice(dataset(options.eval_file), 0, nin(options.eval_size)),
                       length=None)
        if not hasattr(options, 'verbosity') or options.verbosity >= 4:
            print('Eval set size: {}'.format(len(eval_data)))

        learner = SimpleSeq2SeqLearner()

        m = [metrics.METRICS[m] for m in options.metrics]

        if options.load:
            with open(options.load, 'rb') as infile:
                learner.load(infile)
        else:
            learner.train(train_data, validation_data, metrics=m)
            model_path = config.get_file_path('model.pkl')
            if model_path:
                with open(model_path, 'wb') as outfile:
                    learner.dump(outfile)

            train_results = evaluate.evaluate(learner, train_data, metrics=m, split_id='train',
                                              write_data=options.output_train_data)
            output.output_results(train_results, 'train')

        eval_results = evaluate.evaluate(learner, eval_data, metrics=m, split_id='eval',
                                         write_data=options.output_eval_data)
        output.output_results(eval_results, 'eval')


if __name__ == '__main__':
    main()
