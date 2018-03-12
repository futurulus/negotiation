'''
Load data, learn model and evaluate
'''

import argparse
import random
import os
import time
import sys
from itertools import chain
from torch import optim
from torch import cuda
from torch.nn.utils import clip_grad_norm
from torch.nn import NLLLoss, parameter

from cocoa.core.util import read_json, write_json, read_pickle, write_pickle
from cocoa.core.schema import Schema
from cocoa.lib import logstats

from neural_model.encdec import add_model_arguments, GRU_Encoder, Attn_Decoder
from neural_model.learner import add_learner_arguments, Learner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    # parser.add_argument('--stats-file', help='Path to save json statistics (dataset, training etc.) file')
    parser.add_argument('--test', default=False, action='store_true', help='Test mode')
    parser.add_argument('--eval', default=False, action='store_true', help='Eval mode')
    parser.add_argument('--save-results', default=False, action='store_true', help='If true, store model checkpoints')
    parser.add_argument('--eval-output', default=None, help='JSON file to save evaluation results')
    parser.add_argument('--best', default=False, action='store_true', help='Test using the best model on dev set')
    parser.add_argument('--verbose', default=False, action='store_true', help='More prints')
    # add_data_generator_arguments(parser)
    add_model_arguments(parser)
    add_learner_arguments(parser)
    args = parser.parse_args()

    random.seed(args.random_seed)
    # if not os.path.isdir(os.path.dirname(args.stats_file)):
    #     os.makedirs(os.path.dirname(args.stats_file))
    # logstats.init(args.stats_file)
    # logstats.add_args('config', args)

    # Save or load models
    if args.init_from:
        start = time.time()
        # config_path = os.path.join(args.init_from, 'config.json')
        # saved_config = read_json(config_path)

        # NOTE: args below can be overwritten
        # TODO: separate temperature from decoding arg
        # saved_config['decoding'] = args.decoding
        # saved_config['temperature'] = args.temperature
        # saved_config['batch_size'] = args.batch_size
        # saved_config['pretrained_wordvec'] = args.pretrained_wordvec
        # saved_config['ranker'] = args.ranker
        # model_args = argparse.Namespace(**saved_config)

        # Load Model
        if args.test and args.best:
            print("Loading model from checkpoint ...")
            encoder = torch.load(args.init_from+args.checkpoint+'encoder.pt')
            decoder = torch.load(args.init_from+args.checkpoint+'decoder.pt')
        else:
            print("Creating new model...")
            encoder = GRU_Encoder(args.word_embed_size, args.num_layers)
            decoder = Attn_Decoder(args.word_embed_size, args.attn_method, args.dropout)
    else:
        # Save config
        if not os.path.isdir(args.checkpoint):
            os.makedirs(args.checkpoint)
        config_path = os.path.join(args.checkpoint, 'config.json')
        write_json(vars(args), config_path)
        model_args = args
        ckpt = None

    # Load vocab
    vocab_path = 'data/persona_vocab.pkl'
    if not os.path.exists(vocab_path):
        print 'Vocab not found at', vocab_path
        vocab = None
        args.ignore_cache = True
    else:
        print 'Loading vocab from', vocab_path
        vocab = read_pickle(vocab_path)

    # schema = Schema(model_args.schema_path, None)
    # train_batches = DialogueBatcher(vocab, "train")
    # val_batches = DialogueBatcher(vocab, "valid")

    if cuda.is_available():
        print 'Using GPU'
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    else:
        print 'GPU is disabled'

    # for d, n in data_generator.num_examples.iteritems():
    #     logstats.add('data', d, 'num_dialogues', n)

    # if args.test:
    #     evaluator = get_evaluator(data_generator, model, splits=('test',), batch_size=args.batch_size, verbose=args.verbose)
    #     learner = Learner(data_generator, model, evaluator, batch_size=batch_size, verbose=verbose, summary_dir=args.summary_dir)

    #     if args.model != 'ir' and args.init_from:
    #         print 'Load PT model'
    #         start = time.time()
    #         encoder = torch.load(learner.encoder_checkpoint_path)
    #         decoder = torch.load(learner.decoder_checkpoint_path)
    #         print 'Done [%fs]' % (time.time() - start)
    #     for split, test_data, num_batches in evaluator.dataset():
    #         results = learner.eval(sess, split, test_data, num_batches, output=args.eval_output, modes=args.eval_modes)
    #         learner.log_results(split, results)

    # else:
    # evaluator = get_evaluator(data_generator, model, splits=('dev',), batch_size=args.batch_size, verbose=args.verbose)
    learner = Learner(args, encoder, decoder, vocab)
    results = learner.learn(args)
    if args.save_results:
        learner.save_model(args)