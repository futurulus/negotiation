import pickle
import torch as th

from stanza.monitoring import progress
from stanza.research import config

import neural
import seq2seq
import vectorizers
import tokenizers
from agent import RLAgent, Negotiator, SupervisedLoss, RLLoss


parser = config.get_options_parser()

parser.add_argument('--load_init_a', default='',
                    help='Pretrained module file to load for initializing module A in '
                         'reinforcement learning.')
parser.add_argument('--load_init_b', default='',
                    help='Pretrained module file to load for initializing module B in '
                         'reinforcement learning.')
parser.add_argument('--max_dialogue_len', type=int, default=20,
                    help='Maximum number of turns in a reinforcement learning dialogue rollout.')
parser.add_argument('--selection_alpha', type=float, default=0.5,
                    help='Pretrained module file to load for initializing module B in '
                         'reinforcement learning.')
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
        goal_delims = tuple(int(i) for i in
                            vectorizer.goal_vec.vectorize(['<input>', '</input>'])[0][:2])
        resp_delims = tuple(int(i) for i in
                            vectorizer.resp_vec.vectorize(['<dialogue>', '</dialogue>'])[0][:2])
        negotiator_opts = {
            'options': self.options,
            'goal_vocab': vectorizer.goal_vec.vocab_size(),
            'resp_vocab': vectorizer.resp_vec.vocab_size(),
            'sel_vocab': vectorizer.sel_vec.vocab_size(),
            'delimiters': (goal_delims, resp_delims),
        }
        if self.options.load_init_a:
            with open(self.options.load_init_a, 'rb') as infile:
                module_a = pickle.load(infile)
                if isinstance(module_a, neural.TorchModel):
                    module_a = module_a.module
        else:
            module_a = Negotiator(**negotiator_opts)

        if self.options.load_init_b:
            with open(self.options.load_init_a, 'rb') as infile:
                self.module_b = pickle.load(infile)
                if isinstance(self.module_b, neural.TorchModel):
                    self.module_b = self.module_b.module
        else:
            self.module_b = Negotiator(**negotiator_opts)
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

        with config.open('module_a.pkl', 'wb') as outfile:
            pickle.dump(self.model.module, outfile)

    def collate_preds(self, preds, detokenize):
        raise NotImplementedError

    def collate_scores(self, scores):
        raise NotImplementedError


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
        return list(zip(*scores))


class RLLearner(NegotiationLearner):
    vectorizer_class = vectorizers.SelfPlayVectorizer

    def train_batch(self, batch):
        for inst in batch:
            t = self.instance_to_tuple(inst)
            self.model.train([t])

    def wrap_negotiator(self, module, vectorizer):
        if self.options.batch_size != 1:
            raise ValueError('RLAgent and StaticSelfPlayLearner currently only support a batch '
                             'size of 1. Pass --batch_size 1 for RL/self-play evaluation.')

        return neural.TorchModel(
            module=RLAgent(module, self.module_b, vectorizer, self.options),
            loss=RLLoss(self.options),
            optimizer=th.optim.SGD,
            optimizer_params={'lr': self.options.learning_rate},
            vectorizer=vectorizer,
        )

    def collate_preds(self, preds, detokenize):
        return [([detokenize(t) for t in d], detokenize(sa), detokenize(sb), ra, rb)
                for d, sa, sb, ra, rb in zip(*preds)]

    def collate_scores(self, scores):
        return list(zip(*scores))


class StaticSelfPlayLearner(RLLearner):
    def train_batch(self, batch):
        pass
