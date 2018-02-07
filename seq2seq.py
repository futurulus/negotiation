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
from thutils import lrange, index_sequence, maybe_cuda as cu

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
        module = Seq2Seq(src_vocab=vectorizer.vocab_size()[0],
                         tgt_vocab=vectorizer.vocab_size()[1],
                         cell_size=self.options.cell_size,
                         num_layers=self.options.num_layers,
                         beam_size=self.options.beam_size,
                         max_len=self.options.max_length,
                         embed_size=self.options.embed_size,
                         dropout=self.options.dropout,
                         rnn_cell=self.options.rnn_cell,
                         bidirectional=self.options.bidirectional,
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


class MeanScoreLoss(th.nn.Module):
    def forward(self, predict, score):
        return -score.mean()


class SeqDecoder(th.nn.Module):
    def __init__(self,
                 tgt_vocab,
                 cell_size,
                 embed_size,
                 dropout,
                 delimiters,
                 rnn_cell='LSTM',
                 num_layers=1,
                 beam_size=1,
                 max_len=None,
                 monitor_activations=True):
        super(SeqDecoder, self).__init__()

        self.cell_size = cell_size
        self.num_layers = num_layers
        self.monitor_activations = monitor_activations

        self.activations = neural.Activations()

        self.dec_embedding = th.nn.Embedding(tgt_vocab, embed_size)
        cell = CELLS[rnn_cell]
        self.decoder = cell(input_size=embed_size,
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

    def forward(self, src_indices, src_lengths, tgt_indices, tgt_lengths):
        a = self.activations

        enc_state = self.encode(src_indices, src_lengths)

        # predict = (a.out.max(2)[1], tgt_lengths)
        beam, beam_lengths = self.beam_predictor(enc_state)
        sample, sample_lengths = self.sampler(enc_state)

        a.log_softmax, _ = self.decode(tgt_indices[:, :-1], enc_state, monitor=True)

        a.log_prob_token = index_sequence(a.log_softmax, tgt_indices.data[:, 1:])
        a.mask = (lrange(a.log_prob_token.size()[1])[None, :] < tgt_lengths.data[:, None]).float()
        a.log_prob_masked = a.log_prob_token * th.autograd.Variable(a.mask)
        a.log_prob_seq = a.log_prob_masked.sum(1)
        score = a.log_prob_seq

        if not self.monitor_activations:
            # Free up memory
            a.__dict__.clear()

        return {
            'beam': (beam[:, 0, :], beam_lengths[:, 0]),
            'sample': (sample[:, 0, :], sample_lengths[:, 0]),
        }, score

    def decode(self, tgt_indices, enc_state, monitor=False):
        if monitor:
            a = self.activations
        else:
            a = neural.Activations()

        prev_embed = self.dec_embedding(tgt_indices)
        if len(enc_state) == 1:
            enc_state = enc_state[0]
        a.dec_out, dec_state = self.decoder(prev_embed, enc_state)
        a.out = self.output(a.dec_out)
        log_softmax = th.nn.LogSoftmax()(a.out.transpose(2, 0)).transpose(2, 0)
        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        return log_softmax, dec_state


class Seq2Seq(SeqDecoder):
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
                 max_len=None,
                 monitor_activations=True):
        super(Seq2Seq, self).__init__(tgt_vocab=tgt_vocab,
                                      cell_size=cell_size, embed_size=embed_size,
                                      dropout=dropout, num_layers=num_layers,
                                      rnn_cell=rnn_cell,
                                      delimiters=delimiters,
                                      beam_size=beam_size, max_len=max_len,
                                      monitor_activations=monitor_activations)
        self.enc_embedding = th.nn.Embedding(src_vocab, embed_size)
        cell = CELLS[rnn_cell]
        self.use_c = (rnn_cell == 'LSTM')
        if cell_size % 2 != 0:
            raise ValueError('cell_size must be even for bidirectional encoder '
                             '(instead got {})'.format(cell_size))
        self.num_directions = 2 if bidirectional else 1
        self.encoder = cell(input_size=embed_size,
                            hidden_size=cell_size // self.num_directions,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.h_init = th.nn.Linear(1, cell_size * num_layers, bias=False)
        self.c_init = th.nn.Linear(1, cell_size * num_layers, bias=False)

    def encode(self, src_indices, src_lengths):
        a = self.activations

        # TODO: PackedSequence?
        max_len = src_lengths.data.max()
        in_embed = self.enc_embedding(src_indices[:, :max_len])
        init_var = th.autograd.Variable(cu(th.FloatTensor([1.0])))
        batch_size = src_indices.size()[0]
        h_init = (self.h_init(init_var)
                      .view(self.num_layers * self.num_directions, 1,
                            self.cell_size // self.num_directions)
                      .repeat(1, batch_size, 1))
        if self.use_c:
            c_init = (self.c_init(init_var)
                          .view(self.num_layers * self.num_directions, 1,
                                self.cell_size // self.num_directions)
                          .repeat(1, batch_size, 1))
            init = (h_init, c_init)
        else:
            init = h_init
        a.enc_out, enc_state = self.encoder(in_embed, init)
        if self.use_c:
            (a.enc_h_out, a.enc_c_out) = enc_state
            return (self.concat_directions(a.enc_h_out), self.concat_directions(a.enc_c_out))
        else:
            a.enc_h_out = enc_state
            return (self.concat_directions(a.enc_h_out),)

    def concat_directions(self, out):
        if self.num_directions == 1:
            return out
        else:
            out = (out.view(out.size()[0] // self.num_directions,
                            self.num_directions, out.size()[1], out.size()[2])
                      .transpose(1, 2).contiguous())
            assert out.size()[2] == self.num_directions, out.size()
            return out.view(out.size()[0], out.size()[1], out.size()[2] * out.size()[3])


class Conv2Seq(SeqDecoder):
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
        super(Conv2Seq, self).__init__(tgt_vocab=tgt_vocab,
                                       cell_size=cell_size, embed_size=embed_size,
                                       dropout=dropout, num_layers=num_layers,
                                       rnn_cell=rnn_cell,
                                       delimiters=delimiters,
                                       beam_size=beam_size, max_len=max_len,
                                       monitor_activations=monitor_activations)
        self.enc_embedding = th.nn.Embedding(src_vocab, cell_size)
        self.conv = th.nn.Conv1d(in_channels=cell_size, out_channels=cell_size, kernel_size=2)
        self.c_init = th.nn.Linear(1, cell_size * num_layers, bias=False)
        self.nonlinearity = th.nn.Tanh()

    def encode(self, src_indices, src_lengths):
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

        return a.conv_repr, c_init


class BeamPredictor(th.nn.Module):
    def __init__(self, decode_fn, delimiters, beam_size=1, max_len=None):
        super(BeamPredictor, self).__init__()

        self.beam_size = beam_size
        self.decode_fn = decode_fn
        self.max_len = max_len
        self.delimiters = delimiters

    def forward(self, enc_state):
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

        ravel = lambda x: x.contiguous().view(*tuple(x.size()[:-2]) +
                                              (batch_size, self.beam_size, x.size()[-1]))
        unravel = lambda x: x.contiguous().view(*tuple(x.size()[:-3]) +
                                                (batch_size * self.beam_size, x.size()[-1]))

        beam = th.autograd.Variable(cu(th.LongTensor(batch_size, self.beam_size, 1)
                                         .fill_(self.delimiters[0])))
        beam_scores = th.autograd.Variable(cu(th.zeros(batch_size, self.beam_size)))
        beam_lengths = th.autograd.Variable(cu(th.LongTensor(batch_size, self.beam_size).zero_()))

        for length in itertools.count(1):
            last_tokens = beam[:, :, -1:]
            assert last_tokens.size() == (batch_size, self.beam_size, 1), last_tokens.size()
            word_scores, state = self.decode_fn(unravel(last_tokens),
                                                tuple(unravel(c) for c in state))
            word_scores, state = ravel(word_scores[:, 0, :]), tuple(ravel(c) for c in state)
            assert word_scores.size()[:2] == (batch_size, self.beam_size), word_scores.size()
            beam, beam_scores, beam_lengths = self.step(word_scores, length,
                                                        beam, beam_scores, beam_lengths)
            if (beam_lengths.data != length).prod() or \
                    (self.max_len is not None and length == self.max_len):
                break

        return beam[:, :, 1:], th.clamp(beam_lengths, max=self.max_len)

    def step(self, word_scores, length, beam, beam_scores, beam_lengths):
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
        new_indices[(new_beam_lengths != length - 1)] = self.delimiters[1]
        # Add one to the beam lengths that are not done
        new_beam_lengths += ((new_indices != self.delimiters[1]) *
                             (new_beam_lengths == length - 1)).type_as(beam_lengths)
        # Append new token indices
        new_beam = th.cat([beam, new_indices[:, :, None]], dim=2)

        return new_beam, new_beam_scores, new_beam_lengths


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


class Sampler(BeamPredictor):
    def __init__(self, decode_fn, delimiters, num_samples=1, max_len=None):
        super(Sampler, self).__init__(decode_fn, delimiters=delimiters, max_len=max_len,
                                      beam_size=num_samples)

    def step(self, word_scores, length, beam, beam_scores, beam_lengths):
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

        # Sample new words
        ravel = lambda x: x.contiguous().view(*tuple(x.size()[:-2]) +
                                              (batch_size, self.beam_size, x.size()[-1]))
        unravel = lambda x: x.contiguous().view(*tuple(x.size()[:-3]) +
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
        new_indices[(new_beam_lengths != length - 1)] = self.delimiters[1]
        # Add one to the beam lengths that are not done
        new_beam_lengths += ((new_indices != self.delimiters[1]) *
                             (new_beam_lengths == length - 1)).type_as(beam_lengths)
        # Append new token indices
        new_beam = th.cat([beam, new_indices[:, :, None]], dim=2)

        return new_beam, new_beam_scores, new_beam_lengths
