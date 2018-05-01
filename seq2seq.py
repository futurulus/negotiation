import argparse
import itertools
import torch as th
import numpy as np

from stanza.monitoring import progress
from stanza.research import config, iterators, learner
from stanza.research.rng import get_rng

import neural
import vectorizers
import tokenizers
from thutils import lrange, index_sequence, varlen_rnn, maybe_cuda as cu

rng = get_rng()

CELLS = {
    name: getattr(th.nn, name)
    for name in ['RNN', 'LSTM', 'GRU']
}

parser = config.get_options_parser()
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size for training neural models.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Batch size for training neural models.')
parser.add_argument('--cell_size', type=int, default=100,
                    help='Recurrent cell size for the encoder and decoder.')
parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='Batch size for training neural models.')
parser.add_argument('--rnn_cell', choices=CELLS, default='LSTM',
                    help='Type of recurrent cell to use for the encoder and decoder.')
parser.add_argument('--num_layers', type=int, default=1,
                    help='Number of recurrent layers for the encoder and decoder.')
parser.add_argument('--embed_size', type=int, default=50,
                    help='Size of input embeddings for the encoder.')
parser.add_argument('--beam_size', type=int, default=5,
                    help='Number of candidates to keep at each step of beam decoding.')
parser.add_argument('--max_length', type=int, default=100,
                    help='Maximum length of predicted output in decoding and sampling.')
parser.add_argument('--bidirectional', type=config.boolean, default=False,
                    help='If True, use a bidirectional recurrent layer for encoding.')
parser.add_argument('--attention', type=config.boolean, default=False,
                    help='If True, use attention over the encoding (RNN only).')


class SimpleSeq2SeqLearner(learner.Learner):
    def __init__(self):
        super(SimpleSeq2SeqLearner, self).__init__()
        self.get_options()

    @property
    def num_params(self):
        total = 0
        for p in self.model.module.parameters():
            total += th.numel(p.data)
        return total

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
                self.train_batch(batch)
            progress.end_task()

            self.validate_and_log(validation_instances, metrics,
                                  self.model.summary_writer, epoch=epoch)
        progress.end_task()

    def train_batch(self, batch):
        self.model.train([self.instance_to_tuple(inst) for inst in batch])

    def init_vectorizer(self, training_instances):
        vec = vectorizers.Seq2SeqVectorizer()
        vec.add((['<s>', '</s>'], ['<s>', '</s>']))

        progress.start_task('Vectorizer instance', len(training_instances))
        for i, inst in enumerate(training_instances):
            progress.progress(i)
            vec.add(self.instance_to_tuple(inst))
        progress.end_task()

        return vec

    def instance_to_tuple(self, inst):
        def wrap(seq):
            return ['<s>'] + seq + ['</s>']
        tokenize, _ = tokenizers.TOKENIZERS[self.options.tokenizer]
        return (wrap(tokenize(inst.input)), wrap(tokenize(inst.output)))

    def build_model(self, vectorizer):
        delimiters = tuple(int(i) for i in vectorizer.tgt_vec.vectorize(['<s>', '</s>'])[0][:2])
        module = RNN2RNN(src_vocab=vectorizer.vocab_size()[0],
                         tgt_vocab=vectorizer.vocab_size()[1],
                         cell_size=self.options.cell_size,
                         num_layers=self.options.num_layers,
                         beam_size=self.options.beam_size,
                         max_len=self.options.max_length,
                         embed_size=self.options.embed_size,
                         dropout=self.options.dropout,
                         rnn_cell=self.options.rnn_cell,
                         bidirectional=self.options.bidirectional,
                         attention=self.options.attention,
                         delimiters=delimiters,
                         monitor_activations=self.options.monitor_activations)
        model = neural.TorchModel(
            module=module,
            loss=MeanScoreLoss(),
            optimizer=th.optim.Adagrad,
            optimizer_params={'lr': self.options.learning_rate},
            vectorizer=vectorizer,
        )
        return model

    def validate_and_log(self, validation_instances, metrics, writer, epoch):
        validation_results = self.validate(validation_instances, metrics,
                                           iteration=epoch, pass_split=True)
        if writer is not None:
            for key, value in validation_results.items():
                tag = 'val/' + key.split('.', 1)[1].replace('.', '/')
                writer.log_scalar(epoch, tag, value)

    def predict_and_score(self, eval_instances, random=False, split='default', verbosity=4):
        predictions = []
        scores = []

        minibatches = iterators.gen_batches(eval_instances, self.options.batch_size)
        tokenize, detokenize = tokenizers.TOKENIZERS[self.options.tokenizer]

        if verbosity > 2:
            progress.start_task('Eval minibatch', len(minibatches))
        for b, batch in enumerate(minibatches):
            if verbosity > 2:
                progress.progress(b)
            outputs_batch, scores_batch = self.model.eval([self.instance_to_tuple(inst)
                                                           for inst in batch], split=split)
            preds_batch = outputs_batch['sample' if random else 'beam']
            detokenized = self.collate_preds(preds_batch, detokenize)
            predictions.extend(detokenized)
            scores.extend(self.collate_scores(scores_batch))
        if verbosity > 2:
            progress.end_task()
        return predictions, scores

    def collate_preds(self, preds, detokenize):
        return [detokenize(s) for s in preds]

    def collate_scores(self, scores):
        return scores['target']


class Seq2Seq(th.nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        if not hasattr(self, 'activations'):
            self.activations = neural.Activations()

    def forward(self, src_indices, src_lengths, tgt_indices, tgt_lengths,
                extra_delimiter=None, output_beam=None, output_sample=None):
        enc_out, enc_state = self.encoder(src_indices, src_lengths)
        predict, score, _ = self.decoder(enc_state, tgt_indices, tgt_lengths,
                                         output_beam=output_beam, output_sample=output_sample)
        return predict, score


class RNN2RNN(Seq2Seq):
    def __init__(self,
                 src_vocab, tgt_vocab,
                 cell_size,
                 embed_size,
                 dropout,
                 delimiters,
                 rnn_cell='LSTM',
                 num_layers=1,
                 beam_size=1,
                 bidirectional=False,
                 attention=False,
                 max_len=None,
                 monitor_activations=True):
        self.activations = neural.Activations()
        if monitor_activations:
            child_activations = self.activations
        else:
            child_activations = None

        encoder = RNNEncoder(src_vocab=src_vocab,
                             cell_size=cell_size, embed_size=embed_size,
                             dropout=dropout, num_layers=num_layers,
                             rnn_cell=rnn_cell,
                             bidirectional=bidirectional,
                             attention=attention,
                             delimiters=delimiters,
                             activations=child_activations)
        decoder = RNNDecoder(tgt_vocab=tgt_vocab,
                             cell_size=cell_size, embed_size=embed_size,
                             dropout=dropout, num_layers=num_layers,
                             rnn_cell=rnn_cell,
                             delimiters=delimiters,
                             beam_size=beam_size, max_len=max_len,
                             activations=child_activations)
        super(RNN2RNN, self).__init__(encoder, decoder)


class Conv2RNN(Seq2Seq):
    def __init__(self,
                 src_vocab, tgt_vocab,
                 cell_size,
                 embed_size,
                 dropout,
                 delimiters,
                 rnn_cell='LSTM',
                 num_layers=1,
                 beam_size=1,
                 bidirectional='ignored',
                 max_len=None,
                 monitor_activations=True):
        self.activations = neural.Activations()
        if monitor_activations:
            child_activations = self.activations
        else:
            child_activations = None

        encoder = ConvEncoder(src_vocab=src_vocab,
                              cell_size=cell_size, embed_size=embed_size,
                              dropout=dropout, num_layers=num_layers,
                              rnn_cell=rnn_cell,
                              delimiters=delimiters,
                              activations=child_activations)
        decoder = RNNDecoder(tgt_vocab=tgt_vocab,
                             cell_size=cell_size, embed_size=embed_size,
                             dropout=dropout, num_layers=num_layers,
                             rnn_cell=rnn_cell,
                             delimiters=delimiters,
                             beam_size=beam_size, max_len=max_len,
                             activations=child_activations)
        super(RNN2RNN, self).__init__(encoder, decoder)


class RNNEncoder(th.nn.Module):
    def __init__(self,
                 src_vocab,
                 cell_size,
                 embed_size,
                 dropout,
                 delimiters,
                 rnn_cell='LSTM',
                 num_layers=1,
                 bidirectional=False,
                 attention=False,
                 activations=None):
        super(RNNEncoder, self).__init__()

        self.monitor_activations = (activations is not None)
        if isinstance(activations, neural.Activations):
            self.activations = activations
        else:
            self.activations = neural.Activations()

        self.enc_embedding = th.nn.Embedding(src_vocab, embed_size)
        cell = CELLS[rnn_cell]
        self.use_c = (rnn_cell == 'LSTM')
        if cell_size % 2 != 0:
            raise ValueError('cell_size must be even for bidirectional encoder '
                             '(instead got {})'.format(cell_size))
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.cell_size = cell_size
        self.cell = cell(input_size=embed_size,
                         hidden_size=cell_size // self.num_directions,
                         num_layers=num_layers,
                         dropout=dropout,
                         batch_first=True,
                         bidirectional=bidirectional)
        self.h_init = th.nn.Linear(1, cell_size * num_layers, bias=False)
        self.c_init = th.nn.Linear(1, cell_size * num_layers, bias=False)
        self.use_attention = attention
        if attention:
            self.attention = Attention(cell_size, activations=activations, save_weights=True)

    def forward(self, src_indices, src_lengths):
        a = self.activations

        # TODO: PackedSequence?
        max_len = src_lengths.data.max()
        in_embed = self.enc_embedding(src_indices[:, :max_len])
        batch_size = src_indices.size()[0]
        init = generate_rnn_state(self, self.h_init, self.c_init, batch_size)
        a.enc_out, enc_state = varlen_rnn(self.cell, in_embed, src_lengths.data, init)
        if self.use_c:
            (a.enc_h_out, a.enc_c_out) = enc_state
            result = (a.enc_out,
                      (self.concat_directions(a.enc_h_out), self.concat_directions(a.enc_c_out)))
        else:
            a.enc_h_out = enc_state
            result = (a.enc_out,
                      (self.concat_directions(a.enc_h_out),))
        if hasattr(self, 'use_attention') and self.use_attention:
            h_size = result[1][0].size()
            attn_out, attn_weights = self.attention(result[0], src_lengths)
            assert attn_out.size() == h_size[1:], (attn_out.size(), h_size[1:])
            attn_out_layers = attn_out[None, :, :].expand(*h_size).contiguous()
            result = (result[0], (attn_out_layers,) + result[1][1:])
            assert result[1][0].size() == h_size, (result[1][0].size(), h_size)

        if not self.monitor_activations:
            # Free up memory
            a.__dict__.clear()

        return result

    def concat_directions(self, out):
        if self.num_directions == 1:
            return out
        else:
            out = (out.view(out.size()[0] // self.num_directions,
                            self.num_directions, out.size()[1], out.size()[2])
                      .transpose(1, 2).contiguous())
            assert out.size()[2] == self.num_directions, out.size()
            return out.view(out.size()[0], out.size()[1], out.size()[2] * out.size()[3])


class Attention(th.nn.Module):
    def __init__(self, repr_size, activations=None, save_weights=False):
        super(Attention, self).__init__()

        self.repr_size = repr_size
        self.save_weights = save_weights

        self.monitor_activations = (activations is not None)
        if isinstance(activations, neural.Activations):
            self.activations = activations
        else:
            self.activations = neural.Activations()

        self.hidden1 = th.nn.Linear(repr_size, repr_size)
        self.hidden2 = th.nn.Linear(repr_size, repr_size)
        self.target = th.nn.Linear(1, repr_size)
        self.output = th.nn.Linear(repr_size, repr_size)

        self.current_split = 'default'

        self.dump_file = None
        self.close_generator = None

    def __del__(self):
        self.close_dump_file()

    def close_dump_file(self):
        if not hasattr(self, 'save_weights') or not self.save_weights:
            return
        try:
            if hasattr(self, 'close_generator') and self.close_generator is not None:
                next(self.close_generator)
                self.close_generator = None
        except IOError as e:
            print("Couldn't close attention weights file: {}".format(e))

    def get_close_generator(self):
        with config.open('attn_weights.{}.jsons'.format(self.current_split), 'w') as dump_file:
            self.dump_file = dump_file
            yield

        self.dump_file = None
        yield

    def open_dump_file(self):
        if not self.save_weights or self.dump_file is not None:
            return
        try:
            self.close_generator = self.get_close_generator()
            next(self.close_generator)
        except IOError as e:
            print("Couldn't open attention weights file: {}".format(e))

    def split(self, split):
        if split == self.current_split:
            return

        self.close_dump_file()
        self.current_split = split

    def dump_weights(self, weights):
        if self.training or not self.save_weights:
            return
        if self.dump_file is None:
            self.open_dump_file()
        weights_list = weights.tolist()
        try:
            for seq in weights_list:
                print('[{}]'.format(', '.join('{:.3f}'.format(e) for e in seq)),
                      file=self.dump_file)
        except IOError as e:
            print("Couldn't write to attention weights file: {}".format(e))

    def forward(self, outputs, src_lengths):
        a = self.activations

        assert outputs.dim() == 3, outputs.size()
        assert outputs.size()[2] == self.repr_size, (outputs.size(), self.repr_size)
        batch_size, max_len, repr_size = outputs.size()

        a.attn_h1 = th.nn.Tanh()(self.hidden1(outputs))
        a.attn_h2 = self.hidden2(outputs)
        assert a.attn_h2.size() == (batch_size, max_len, repr_size), \
            (a.attn_h2.size(), (batch_size, max_len, repr_size))
        init_var = th.autograd.Variable(cu(th.FloatTensor([1.0])))
        a.target = self.target(init_var)
        assert a.target.size() == (repr_size,), (a.target.size(), repr_size)
        a.attn_scores = th.matmul(a.attn_h2, a.target)
        assert a.attn_scores.size() == (batch_size, max_len), \
            (a.attn_scores.size(), (batch_size, max_len))
        attn_mask = th.autograd.Variable(cu(
            th.log((lrange(max_len)[None, :] < src_lengths.data[:, None]).float())
        ))
        a.attn_weights = th.exp(th.nn.LogSoftmax(dim=1)(a.attn_scores + attn_mask))
        assert a.attn_weights.size() == (batch_size, max_len), \
            (a.attn_weights.size(), (batch_size, max_len))
        a.attn_out = th.matmul(a.attn_weights[:, None, :], outputs)[:, 0, :]
        assert a.attn_out.size() == (batch_size, repr_size), \
            (a.attn_out.size(), (batch_size, repr_size))

        self.dump_weights(a.attn_weights.data)

        result = a.attn_out, a.attn_weights

        if not self.monitor_activations:
            # Free up memory
            a.__dict__.clear()

        return result

    def __getstate__(self):
        d = self.__dict__.copy()
        d['dump_file'] = None
        d['close_generator'] = None
        d['current_split'] = 'default'
        return d


class ConvEncoder(th.nn.Module):
    def __init__(self,
                 src_vocab,
                 cell_size,
                 embed_size,
                 dropout,
                 delimiters,
                 rnn_cell='LSTM',
                 num_layers=1,
                 bidirectional='ignored',
                 activations=None):
        super(ConvEncoder, self).__init__()

        self.monitor_activations = (activations is not None)
        if isinstance(activations, neural.Activations):
            self.activations = activations
        else:
            self.activations = neural.Activations()

        self.enc_embedding = th.nn.Embedding(src_vocab, cell_size)
        self.conv = th.nn.Conv1d(in_channels=cell_size, out_channels=cell_size, kernel_size=2)
        self.c_init = th.nn.Linear(1, cell_size * num_layers, bias=False)
        self.nonlinearity = th.nn.Tanh()

    def forward(self, src_indices, src_lengths):
        a = self.activations

        # TODO: PackedSequence?
        batch_size = src_indices.size()[0]
        max_len = src_lengths.data.max()
        a.in_embed = self.enc_embedding(src_indices[:, :max_len])
        conv_stack = [a.in_embed.transpose(1, 2)]
        for i in range(max_len - 1):
            conv_stack.append(self.conv(self.nonlinearity(conv_stack[-1])))
        a.conv_repr = (th.stack([conv_stack[n - 1][j, :, 0]
                                 for j, n in enumerate(src_lengths.data)], 0)
                         .view(1, batch_size, self.cell_size)
                         .repeat(self.num_layers, 1, 1))
        init_var = th.autograd.Variable(cu(th.FloatTensor([1.0])))
        c_init = (self.c_init(init_var)
                      .view(self.num_layers, 1, self.cell_size)
                      .repeat(1, batch_size, 1))

        result = a.conv_repr, c_init

        if not self.monitor_activations:
            # Free up memory
            a.__dict__.clear()

        return result


class RNNDecoder(th.nn.Module):
    def __init__(self,
                 tgt_vocab,
                 cell_size,
                 embed_size,
                 dropout,
                 delimiters,
                 rnn_cell='LSTM',
                 num_layers=1,
                 beam_size=1,
                 extra_input_size=0,
                 max_len=None,
                 activations=None):
        super(RNNDecoder, self).__init__()

        self.monitor_activations = (activations is not None)
        if isinstance(activations, neural.Activations):
            self.activations = activations
        else:
            self.activations = neural.Activations()

        self.dec_embedding = th.nn.Embedding(tgt_vocab, embed_size)
        cell = CELLS[rnn_cell]
        self.decoder = cell(input_size=embed_size + extra_input_size,
                            hidden_size=cell_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.output = th.nn.Linear(cell_size, tgt_vocab)
        self.beam_predictor = BeamPredictor(self.decode,
                                            beam_size=beam_size,
                                            max_len=max_len,
                                            delimiters=delimiters)
        self.sampler = Sampler(self.decode, max_len=max_len, delimiters=delimiters)

    def forward(self, enc_state, tgt_indices, tgt_lengths, extra_inputs=None,
                extra_delimiter=None, output_beam=None, output_sample=None):
        if output_beam is None:
            output_beam = not self.training
        if output_sample is None:
            output_sample = not self.training

        a = self.activations

        if output_beam:
            (beam, beam_lengths,
             beam_scores, beam_outputs) = self.beam_predictor(enc_state, extra_inputs=extra_inputs,
                                                              extra_delimiter=extra_delimiter)
        if output_sample:
            (sample, sample_lengths,
             sample_scores, sample_outputs) = self.sampler(enc_state, extra_inputs=extra_inputs,
                                                           extra_delimiter=extra_delimiter)

            '''
            if not hasattr(neural.TorchModel, 'debug'):
                if not hasattr(neural.TorchModel, 'debug_counts'):
                    from collections import Counter
                    neural.TorchModel.debug_counts = Counter()
                print('  <COUNTS>')
                neural.TorchModel.debug_counts.update(sample[:, 0, 1].data.tolist())
            '''

        if extra_inputs is None:
            extra_inputs = []
        else:
            extra_inputs = [
                inp[:, None, ...].expand((inp.size()[0], tgt_indices.size()[1] - 1) +
                                         tuple(inp.size()[1:]))
                for inp in extra_inputs
            ]
        a.log_softmax, (dec_out, dec_state) = self.decode(tgt_indices[:, :-1], enc_state,
                                                          extra_inputs=extra_inputs,
                                                          monitor=True)

        a.log_prob_token = index_sequence(a.log_softmax, tgt_indices.data[:, 1:])
        a.mask = (lrange(a.log_prob_token.size()[1])[None, :] < tgt_lengths.data[:, None]).float()
        a.log_prob_masked = a.log_prob_token * th.autograd.Variable(a.mask)
        a.log_prob_seq = a.log_prob_masked.sum(1)

        predict = {}
        score = {
            'target': a.log_prob_seq
        }
        output = {
            'target': (dec_out, dec_state)
        }
        if output_beam:
            predict['beam'] = (beam[:, 0, 1:], beam_lengths[:, 0])
            score['beam'] = beam_scores[:, 0]
            output['beam'] = beam_outputs
        if output_sample:
            predict['sample'] = (sample[:, 0, 1:], sample_lengths[:, 0])
            score['sample'] = sample_scores[:, 0]
            output['sample'] = sample_outputs

        if not self.monitor_activations:
            # Free up memory
            a.__dict__.clear()

        return predict, score, output

    def decode(self, tgt_indices, enc_state, extra_inputs=None, monitor=False):
        if monitor:
            a = self.activations
        else:
            a = neural.Activations()

        prev_embed = self.dec_embedding(tgt_indices)
        if len(enc_state) == 1:
            enc_state = enc_state[0]

        if extra_inputs:
            input_embed = th.cat([prev_embed] + extra_inputs, 2)
        else:
            input_embed = prev_embed

        a.dec_out, dec_state = self.decoder(input_embed, enc_state)
        a.out = self.output(a.dec_out)
        log_softmax = th.nn.LogSoftmax(dim=2)(a.out)
        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        return log_softmax, (a.dec_out, dec_state)


class BeamPredictor(th.nn.Module):
    def __init__(self, decode_fn, delimiters, beam_size=1, max_len=None):
        super(BeamPredictor, self).__init__()

        self.beam_size = beam_size
        self.decode_fn = decode_fn
        self.max_len = max_len
        self.delimiters = delimiters

    def forward(self, enc_state, extra_inputs=None, extra_delimiter=None):
        if not isinstance(enc_state, tuple):
            enc_state = (enc_state,)
        assert len(enc_state[0].size()) == 3, enc_state[0].size()
        num_layers, batch_size, h_size = enc_state[0].size()
        state_sizes = []
        state = []
        for enc_c in enc_state:
            assert len(enc_c.size()) == 3, enc_c.size()
            assert enc_c.size()[:2] == (num_layers, batch_size), enc_c.size()
            c_size = enc_c.size()[2]
            state_sizes.append(c_size)
            state.append(enc_c[:, :, None, :].expand(num_layers, batch_size,
                                                     self.beam_size, c_size))
        if extra_inputs is None:
            extra_inputs = []
        else:
            extra_inputs = [
                inp[:, None, ...].expand((inp.size()[0], self.beam_size) + tuple(inp.size()[1:]))
                                 .contiguous()
                                 .view((inp.size()[0] * self.beam_size, 1) + tuple(inp.size()[1:]))
                for inp in extra_inputs
            ]

        def ravel(x):
            return x.contiguous().view(*tuple(x.size()[:-2]) +
                                       (batch_size, self.beam_size, x.size()[-1]))

        def unravel(x):
            return x.contiguous().view(*tuple(x.size()[:-3]) +
                                       (batch_size * self.beam_size, x.size()[-1]))

        beam = th.autograd.Variable(cu(th.LongTensor(batch_size, self.beam_size, 1)
                                         .fill_(self.delimiters[0])))
        beam_scores = th.autograd.Variable(cu(th.zeros(batch_size, self.beam_size)))
        beam_lengths = th.autograd.Variable(cu(th.LongTensor(batch_size, self.beam_size).zero_()))
        outputs = []
        states = []

        for length in itertools.count(1):
            last_tokens = beam[:, :, -1:]
            assert last_tokens.size() == (batch_size, self.beam_size, 1), last_tokens.size()
            word_scores, (dec_out, state) = self.decode_fn(unravel(last_tokens),
                                                           tuple(unravel(c) for c in state),
                                                           extra_inputs=extra_inputs)
            word_scores = ravel(word_scores[:, 0, :])
            state = tuple(ravel(c) for c in state)
            states.append(state)
            outputs.append(dec_out)
            assert word_scores.size()[:2] == (batch_size, self.beam_size), word_scores.size()
            beam, beam_lengths, beam_scores = self.step(word_scores, length,
                                                        beam, beam_scores, beam_lengths,
                                                        extra_delimiter=extra_delimiter)
            if (beam_lengths.data != length).prod() or \
                    (self.max_len is not None and length == self.max_len):
                break

        all_states_collated = [th.stack(s, dim=3) for s in zip(*states)]
        final_indices = th.clamp(beam_lengths.data, max=self.max_len - 1)
        final_states = [s[:, lrange(batch_size)[:, None],
                          lrange(self.beam_size)[None, :], final_indices, :]
                        for s in all_states_collated]
        all_outputs = th.stack(outputs, dim=1)
        return (beam,
                th.clamp(beam_lengths, max=self.max_len),
                beam_scores,
                (all_outputs, final_states))

    def step(self, word_scores, length, beam, beam_scores, beam_lengths, extra_delimiter):
        assert len(word_scores.size()) == 3, word_scores.size()
        batch_size, beam_size, vocab_size = word_scores.size()
        assert beam_size == self.beam_size, word_scores.size()
        assert len(beam.size()) == 3, beam.size()
        assert beam.size()[:2] == (batch_size, beam_size), \
            '%s != (%s, %s, *)' % (beam.size(), batch_size, beam_size)
        assert beam_scores.size() == (batch_size, beam_size), \
            '%s != %s' % (beam_scores.size(), (batch_size, beam_size))
        assert beam_lengths.size() == (batch_size, beam_size), \
            '%s != %s' % (beam_lengths.size(), (batch_size, beam_size))

        # Compute updated scores
        done_mask = (beam_lengths == length - 1).type_as(word_scores)[:, :, None]
        new_scores = (word_scores * done_mask +
                      beam_scores[:, :, np.newaxis]).view(batch_size, beam_size * vocab_size)
        # Get top k scores and their indices
        new_beam_scores, topk_indices = new_scores.topk(beam_size, dim=1)
        # Transform into previous beam indices and new token indices
        rows, new_indices = unravel_index(topk_indices, (beam_size, vocab_size))
        assert rows.size() == (batch_size, beam_size), \
            '%s != %s' % (rows.size(), (batch_size, beam_size))
        assert new_indices.size() == (batch_size, beam_size), \
            '%s != %s' % (new_indices.size(), (batch_size, beam_size))

        # Extract best pre-existing rows
        beam = beam[lrange(batch_size)[:, None], rows.data, :]
        assert beam.size()[:2] == (batch_size, beam_size), (beam.size(), (batch_size, beam_size))
        # Get previous done status and update it with
        # which rows have newly reached </s>
        new_beam_lengths = beam_lengths[lrange(batch_size)[:, None], rows.data].clone()
        # Pad already-finished sequences with </s>
        pad_delimiter = extra_delimiter if extra_delimiter is not None else self.delimiters[1]
        new_indices[(new_beam_lengths != length - 1)] = pad_delimiter
        # Add one to the beam lengths that are not done
        continue_mask = (new_indices != self.delimiters[1]) * (new_beam_lengths == length - 1)
        if extra_delimiter is not None:
            continue_mask = continue_mask * (new_indices != extra_delimiter)
        new_beam_lengths += continue_mask.type_as(beam_lengths)
        # Append new token indices
        new_beam = th.cat([beam, new_indices[:, :, None]], dim=2)

        return new_beam, new_beam_lengths, new_beam_scores


def output_debug_probs(vec):
    for i, (w, cum_prob) in enumerate(zip(vec.tokens,
                                          neural.TorchModel.debug_probs.data.tolist())):
        actual_count = neural.TorchModel.debug_counts[i]
        if abs(actual_count - cum_prob) > 0.5:
            print('{}: {:.4f} [{}]'.format(w, cum_prob, actual_count))


class Sampler(BeamPredictor):
    def __init__(self, decode_fn, delimiters, num_samples=1, max_len=None):
        super(Sampler, self).__init__(decode_fn, delimiters=delimiters, max_len=max_len,
                                      beam_size=num_samples)

    def step(self, word_scores, length, beam, beam_scores, beam_lengths, extra_delimiter):
        assert len(word_scores.size()) == 3, word_scores.size()
        batch_size, beam_size, vocab_size = word_scores.size()
        assert beam_size == self.beam_size, word_scores.size()
        assert len(beam.size()) == 3, beam.size()
        assert beam.size()[:2] == (batch_size, beam_size), \
            '%s != (%s, %s, *)' % (beam.size(), batch_size, beam_size)
        assert beam_scores.size() == (batch_size, 1), \
            '%s != %s' % (beam_scores.size(), (batch_size, beam_size))
        assert beam_lengths.size() == (batch_size, 1), \
            '%s != %s' % (beam_lengths.size(), (batch_size, beam_size))

        '''
        if not hasattr(neural.TorchModel, 'debug') and length == 1:
            if not hasattr(neural.TorchModel, 'debug_probs'):
                neural.TorchModel.debug_probs = cu(th.zeros(vocab_size))
            print('  <PROBS>')
            neural.TorchModel.debug_probs += th.exp(word_scores.data)[:, 0, :].sum(dim=0)
        '''

        # Sample new words
        def ravel(x):
            return x.contiguous().view(*tuple(x.size()[:-2]) +
                                       (batch_size, self.beam_size, x.size()[-1]))

        def unravel(x):
            return x.contiguous().view(*tuple(x.size()[:-3]) +
                                       (batch_size * self.beam_size, x.size()[-1]))

        new_indices = ravel(
            th.multinomial(unravel(th.exp(word_scores)), 1, replacement=True)
        )[:, :, 0]
        # Compute updated scores
        new_word_scores = index_sequence(word_scores, new_indices.data)
        done_mask = (beam_lengths == length - 1).type_as(new_word_scores)[:, :]
        new_beam_scores = beam_scores + new_word_scores * done_mask

        # Get previous done status and update it with
        # which rows have newly reached </s>
        new_beam_lengths = beam_lengths.clone()
        # Pad already-finished sequences with </s>
        pad_delimiter = extra_delimiter if extra_delimiter is not None else self.delimiters[1]
        new_indices[(new_beam_lengths != length - 1)] = pad_delimiter
        # Add one to the beam lengths that are not done
        continue_mask = (new_indices != self.delimiters[1]) * (new_beam_lengths == length - 1)
        if extra_delimiter is not None:
            continue_mask = continue_mask * (new_indices != extra_delimiter)
        new_beam_lengths += continue_mask.type_as(beam_lengths)
        # Append new token indices
        new_beam = th.cat([beam, new_indices[:, :, None]], dim=2)

        return new_beam, new_beam_lengths, new_beam_scores


class MeanScoreLoss(th.nn.Module):
    def forward(self, predict, score):
        return -score['target'].mean()


def unravel_index(indices, size):
    '''
    Convert a tensor of indices into an "unraveled" tensor (a 1-dimensional tensor of length
    equal to the product of the elements of size) into a tuple of tensors of indices into the
    "raveled" tensor of size `size`. The return value will be a tuple of length equal to the
    number of elements in size, and each tensor in the tuple will have a size that is the same
    as the size of `indices`.

    >>> unravel_index(th.IntTensor([8, 2, 3, 6]), (4, 5))
    (
     1
     0
     0
     1
    [torch.IntTensor of size 4]
    , 
     3
     2
     3
     1
    [torch.IntTensor of size 4]
    )
    '''  # NOQA: doctest whitespace
    result = []
    for s in size[::-1]:
        indices, q = (indices / s, th.remainder(indices, s))
        result.append(q)
    return tuple(result[::-1])


def generate_rnn_state(encoder, h_init_mod, c_init_mod, batch_size):
    init_var = th.autograd.Variable(cu(th.FloatTensor([1.0])))

    h_init = (h_init_mod(init_var).view(encoder.num_layers * encoder.num_directions, 1,
                                        encoder.cell_size // encoder.num_directions)
                                  .repeat(1, batch_size, 1))
    if encoder.use_c:
        c_init = (c_init_mod(init_var).view(encoder.num_layers * encoder.num_directions, 1,
                                            encoder.cell_size // encoder.num_directions)
                                      .repeat(1, batch_size, 1))
        return (h_init, c_init)
    else:
        return h_init
