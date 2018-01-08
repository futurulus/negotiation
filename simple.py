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

import argparse
import datetime
import gzip
from itertools import islice
import json

import torch as th

from stanza.monitoring import progress
from stanza.research import evaluate, output, iterators, learner
from stanza.research.instance import Instance

import metrics
import neural
import seq2seq
import thutils
import vectorizers
import tokenizers

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


class SimpleSeq2SeqLearner(learner.Learner):
    def __init__(self):
        super(SimpleSeq2SeqLearner, self).__init__()
        self.get_options()

    @property
    def num_params(self):
        # TODO
        return 0

    def get_options(self):
        if not hasattr(self, 'options'):
            options = config.options()
            self.options = argparse.Namespace(**options.__dict__)

    def train(self, training_instances, validation_instances=None, metrics=None):
        if not hasattr(self, 'model'):
            self.model = self.build_model(self.init_vectorizer(training_instances))

        minibatches = iterators.gen_batches(training_instances, self.options.batch_size)
        progress.start_task('Epoch', self.options.train_epochs)
        for epoch in range(self.options.train_epochs):
            progress.progress(epoch)

            progress.start_task('Minibatch', len(minibatches))
            for b, batch in enumerate(minibatches):
                progress.progress(b)
                self.model.train([self.instance_to_pair(inst) for inst in batch])
            progress.end_task()

            self.validate_and_log(validation_instances, metrics,
                                  self.model.summary_writer, epoch=epoch)
        progress.end_task()

    def init_vectorizer(self, training_instances):
        vec = vectorizers.Seq2SeqVectorizer()
        vec.add((['<s>', '</s>'], ['<s>', '</s>']))

        progress.start_task('Vectorizer instance', len(training_instances))
        for i, inst in enumerate(training_instances):
            progress.progress(i)
            vec.add(self.instance_to_pair(inst))
        progress.end_task()

        return vec

    def instance_to_pair(self, inst):
        wrap = lambda seq: ['<s>'] + seq + ['</s>']
        tokenize, _ = tokenizers.TOKENIZERS[self.options.tokenizer]
        return (wrap(tokenize(inst.input)), wrap(tokenize(inst.output)))

    def build_model(self, vectorizer):
        delimiters = tuple(int(i) for i in vectorizer.tgt_vec.vectorize(['<s>', '</s>'])[0][:2])
        module = seq2seq.Seq2Seq(src_vocab=vectorizer.vocab_size()[0],
                                 tgt_vocab=vectorizer.vocab_size()[1],
                                 cell_size=self.options.cell_size,
                                 num_layers=self.options.num_layers,
                                 beam_size=self.options.beam_size,
                                 max_len=self.options.max_length,
                                 embed_size=self.options.embed_size,
                                 dropout=self.options.dropout,
                                 delimiters=delimiters,
                                 monitor_activations=self.options.monitor_activations)
        model = neural.TorchModel(
            module=module,
            loss=seq2seq.MeanScoreLoss(),
            optimizer=th.optim.Adagrad,
            optimizer_params={'lr': self.options.learning_rate},
            vectorizer=vectorizer,
        )
        return model

    def validate_and_log(self, validation_instances, metrics, writer, epoch):
        validation_results = self.validate(validation_instances, metrics, iteration=epoch)
        if writer is not None:
            for key, value in validation_results.items():
                tag = 'val/' + key.split('.', 1)[1].replace('.', '/')
                writer.log_scalar(epoch, tag, value)

    def predict_and_score(self, eval_instances, random=False, verbosity=0):
        predictions = []
        scores = []

        minibatches = iterators.gen_batches(eval_instances, self.options.batch_size)
        tokenize, detokenize = tokenizers.TOKENIZERS[self.options.tokenizer]

        progress.start_task('Eval minibatch', len(minibatches))
        for b, batch in enumerate(minibatches):
            progress.progress(b)
            outputs_batch, scores_batch = self.model.eval([self.instance_to_pair(inst)
                                                           for inst in batch])
            preds_batch = outputs_batch['sample' if random else 'beam']
            detokenized = [detokenize(s) for s in preds_batch]
            predictions.extend(detokenized)
            scores.extend(scores_batch)
        progress.end_task()
        return predictions, scores


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

        eval_data = SG(lambda: islice(dataset(options.eval_file), 0, nin(options.train_size)),
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
