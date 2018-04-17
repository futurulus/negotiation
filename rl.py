import pickle
import torch as th
import numpy as np

from stanza.monitoring import progress
from stanza.research import config

import neural
import seq2seq
import vectorizers
import tokenizers
from agents import RLNegotiator, Negotiator, SupervisedLoss, RLLoss


parser = config.get_options_parser()

parser.add_argument('--load_init_a', default='',
                    help='Pretrained model file to load for initializing module A in '
                         'reinforcement learning.')
parser.add_argument('--load_init_b', default='',
                    help='Pretrained model file to load for initializing module B in '
                         'reinforcement learning.')
parser.add_argument('--selection_alpha', type=float, default=0.5,
                    help='Relative weight of the selection compared to the responses (which '
                         'always get weight 1) in the supervised loss function.')
parser.add_argument('--rl_epsilon', type=float, default=0.1,
                    help='Fraction of the time to use a random sample from the policy '
                         'for epsilon-greedy exploration.')
parser.add_argument('--rl_gamma', type=float, default=0.99,
                    help='Amount reward is discounted for each dialogue step.')


class NegotiationLearner(seq2seq.SimpleSeq2SeqLearner):
    vectorizer_class = vectorizers.NegotiationVectorizer

    def __init__(self):
        super(NegotiationLearner, self).__init__()

    def build_model(self, vectorizer):
        def negotiator_opts(vec):
            goal_delims = tuple(int(i) for i in
                                vec.goal_vec.vectorize(['<input>', '</input>'])[0][:2])
            resp_delims = tuple(int(i) for i in
                                vec.resp_vec.vectorize(['<dialogue>', '</dialogue>'])[0][:2])

            return {
                'options': self.options,
                'goal_vocab': vec.goal_vec.vocab_size(),
                'resp_vocab': vec.resp_vec.vocab_size(),
                'sel_vocab': vec.sel_vec.vocab_size(),
                'delimiters': (goal_delims, resp_delims),
            }

        if self.options.load_init_a:
            with open(self.options.load_init_a, 'rb') as infile:
                model_a = pickle.load(infile)
                if isinstance(model_a, NegotiationLearner):
                    model_a = model_a.model
                assert isinstance(model_a, neural.TorchModel)
                module_a = model_a.module
                vectorizer.inherit(model_a.vectorizer)
        else:
            module_a = Negotiator(**negotiator_opts(vectorizer))

        if self.options.load_init_b:
            with open(self.options.load_init_a, 'rb') as infile:
                model_b = pickle.load(infile)
                if isinstance(model_b, NegotiationLearner):
                    model_a = model_b.model
                assert isinstance(self.module_b, neural.TorchModel)
                self.module_b = self.model_b.module
                assert model_b.vectorizer.resp_vec.tokens == vectorizer.resp_vec.tokens, \
                    "Vectorizers aren't compatible. Were models a and b trained on the same data?"
        else:
            self.module_b = Negotiator(**negotiator_opts(vectorizer))
            self.module_b.load_state_dict(module_a.state_dict())

        return self.wrap_negotiator(module_a, vectorizer)

    def wrap_negotiator(self, module, vectorizer):
        raise NotImplementedError

    def init_vectorizer(self, training_instances):
        vec = self.vectorizer_class()
        vec.add((['<input>', '</input>'],
                 ['<dialogue>', '</dialogue>', '<eos>', 'YOU:', 'THEM:'],
                 ['<output>', '</output>']))

        progress.start_task('Vectorizer instance', len(training_instances))
        for i, inst in enumerate(training_instances):
            progress.progress(i)
            vec.add(self.instance_to_tuple(inst))
        progress.end_task()

        return vec

    def instance_to_tuple(self, inst):
        tokenize, _ = tokenizers.TOKENIZERS[self.options.tokenizer]
        return (tokenize(inst.input),
                ['<dialogue>'] + tokenize(inst.output[0]) + ['</dialogue>'],
                tokenize(inst.output[1]),
                tokenize(inst.output[2]))

    def train(self, training_instances, validation_instances=None, metrics=None):
        super(NegotiationLearner, self).train(training_instances,
                                              validation_instances=validation_instances,
                                              metrics=metrics)

        # Don't write activations to pickle file
        self.model.module.apply(clear_activations)
        with config.open('model_a.pkl', 'wb') as outfile:
            pickle.dump(self.model, outfile)

    def collate_preds(self, preds, detokenize):
        raise NotImplementedError

    def collate_scores(self, scores):
        raise NotImplementedError


def clear_activations(module):
    if hasattr(module, 'activations'):
        module.activations.__dict__.clear()


class SupervisedLearner(NegotiationLearner):
    def wrap_negotiator(self, module, vectorizer):
        return neural.TorchModel(
            module=module,
            loss=SupervisedLoss(self.options),
            optimizer=th.optim.SGD,
            optimizer_params={'lr': self.options.learning_rate},
            vectorizer=vectorizer,
        )

    def collate_preds(self, preds, detokenize):
        return [(detokenize(r), detokenize(s)) for r, s in zip(*preds)]

    def collate_scores(self, scores):
        return list(zip(*(s['target'] for s in scores)))


class RLLearner(NegotiationLearner):
    vectorizer_class = vectorizers.SelfPlayVectorizer

    def train_batch(self, batch):
        for inst in batch:
            t = self.instance_to_tuple(inst)
            self.model.train([t])

    def wrap_negotiator(self, module, vectorizer):
        if self.options.batch_size != 1:
            raise ValueError('RLNegotiator and StaticSelfPlayLearner currently only support a '
                             'batch size of 1. Pass --batch_size 1 for RL/self-play evaluation.')

        return neural.TorchModel(
            module=RLNegotiator(module, self.module_b, vectorizer, self.options),
            loss=RLLoss(self.options),
            optimizer=th.optim.SGD,
            optimizer_params={'lr': self.options.learning_rate},
            vectorizer=vectorizer,
        )

    def collate_preds(self, preds, detokenize):
        return [([detokenize(t) for t in d], detokenize(sa), detokenize(sb), ra, rb)
                for d, sa, sb, ra, rb in zip(*preds)]

    def collate_scores(self, scores):
        dialogue_scores, selection_scores = scores
        return list(zip([np.sum(dialogue_scores)], selection_scores['target']))


class StaticSelfPlayLearner(RLLearner):
    def train_batch(self, batch):
        pass
