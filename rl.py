import pickle
import torch as th

from stanza.monitoring import progress
from stanza.research import config

import neural
import seq2seq
import vectorizers
import tokenizers
from agent import Negotiator, SupervisedLoss, RLLoss


parser = config.get_options_parser()

parser.add_argument('--load_init_a', default='',
                    help='Pretrained model file to load for initializing model A in '
                         'reinforcement learning.')
parser.add_argument('--load_init_b', default='',
                    help='Pretrained model file to load for initializing model B in '
                         'reinforcement learning.')
parser.add_argument('--selection_alpha', type=float, default=0.5,
                    help='Pretrained model file to load for initializing model B in '
                         'reinforcement learning.')


class NegotiationLearner(seq2seq.SimpleSeq2SeqLearner):
    loss_class = NotImplemented

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
                model_a = pickle.load(infile)
        else:
            model_a = Negotiator(**negotiator_opts)

        if self.options.load_init_b:
            with open(self.options.load_init_a, 'rb') as infile:
                self.model_b = pickle.load(infile)
        else:
            self.model_b = Negotiator(**negotiator_opts)
            self.model_b.load_state_dict(model_a.state_dict())

        return neural.TorchModel(
            module=model_a,
            loss=self.loss_class(self.options),
            optimizer=th.optim.SGD,
            optimizer_params={'lr': self.options.learning_rate},
            vectorizer=vectorizer,
        )

    def init_vectorizer(self, training_instances):
        vec = vectorizers.NegotiationVectorizer()
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
                tokenize(inst.output[1]))

    def train(self, training_instances, validation_instances=None, metrics=None):
        super(NegotiationLearner, self).train(training_instances,
                                              validation_instances=validation_instances,
                                              metrics=metrics)

        with config.open('model_a.pkl', 'wb') as outfile:
            pickle.dump(self.model, outfile)

    def collate_preds(self, preds, detokenize):
        return [(detokenize(r), detokenize(s)) for r, s in zip(*preds)]

    def collate_scores(self, scores):
        return list(zip(*scores))


class SupervisedLearner(NegotiationLearner):
    loss_class = SupervisedLoss


class RLLearner(NegotiationLearner):
    loss_class = RLLoss

    def train(self, training_instances, validation_instances=None, metrics=None):
        raise NotImplementedError
