import torch as th

import neural
import seq2seq
from vectorizers import MAX_FEASIBLE, NUM_ITEMS, GOAL_SIZE

from thutils import index_sequence, lrange, log_softmax, maybe_cuda as cu


class Negotiator(th.nn.Module):
    def __init__(self, options,
                 goal_vocab, resp_vocab, sel_vocab,
                 delimiters,
                 monitor_activations=True):
        super(Negotiator, self).__init__()
        self.monitor_activations = monitor_activations
        self.activations = neural.Activations()
        if monitor_activations:
            child_activations = self.activations
        else:
            child_activations = None

        self.h_init = th.nn.Linear(1, options.cell_size * options.num_layers, bias=False)
        self.c_init = th.nn.Linear(1, options.cell_size * options.num_layers, bias=False)

        self.context_encoder = seq2seq.RNNEncoder(src_vocab=goal_vocab,
                                                  cell_size=options.cell_size,
                                                  embed_size=options.embed_size,
                                                  dropout=options.dropout,
                                                  delimiters=delimiters[0],
                                                  rnn_cell=options.rnn_cell,
                                                  num_layers=options.num_layers,
                                                  bidirectional=False,
                                                  activations=child_activations)
        self.response_decoder = seq2seq.RNNDecoder(tgt_vocab=resp_vocab,
                                                   cell_size=options.cell_size,
                                                   embed_size=options.embed_size,
                                                   dropout=options.dropout,
                                                   delimiters=delimiters[1],
                                                   rnn_cell=options.rnn_cell,
                                                   num_layers=options.num_layers,
                                                   beam_size=options.beam_size,
                                                   extra_input_size=options.cell_size,
                                                   max_len=options.max_length,
                                                   activations=child_activations)
        self.response_encoder = seq2seq.RNNEncoder(src_vocab=resp_vocab,
                                                   cell_size=options.cell_size,
                                                   embed_size=options.embed_size,
                                                   dropout=options.dropout,
                                                   delimiters=delimiters[1],
                                                   rnn_cell=options.rnn_cell,
                                                   num_layers=options.num_layers,
                                                   bidirectional=options.bidirectional,
                                                   activations=child_activations)
        self.combined_layer = th.nn.Linear(options.cell_size * 2, options.cell_size, bias=False)
        self.selection_layer = th.nn.Linear(options.cell_size, sel_vocab, bias=False)

    def forward(self,
                goal_indices,
                resp_indices, resp_len,
                sel_indices, feasible_sels, num_feasible_sels):
        a = self.activations

        # "GRU_g": encode goals (values of items)
        batch_size, goal_size = goal_indices.size()
        assert goal_size == GOAL_SIZE, goal_indices.size()

        goal_len = th.autograd.Variable(cu(
            (th.ones(batch_size) * goal_size).int()
        ))
        assert goal_len.size() == (batch_size,), goal_len.size()

        a.context_repr, _ = self.context_encoder(goal_indices, goal_len)
        assert a.context_repr.dim() == 3, a.context_repr.size()
        assert a.context_repr.size()[:2] == (batch_size, goal_size), a.context_repr.size()
        context_repr_size = a.context_repr.size()[2]

        a.last_context_repr = a.context_repr[:, -1, :]
        assert a.last_context_repr.size() == (batch_size, context_repr_size), \
            a.last_context_repr.size()

        # "GRU_w": encode and produce dialogue
        assert resp_indices.dim() == 2, resp_indices.size()
        assert resp_indices.size()[0] == batch_size, resp_indices.size()
        max_resp_len = resp_indices.size()[1]

        response_predict, response_score, dec_out = self.response_decoder(
            seq2seq.generate_rnn_state(self.response_encoder, self.h_init, self.c_init, batch_size),
            resp_indices, resp_len,
            extra_inputs=[a.last_context_repr]
        )
        (a.output_dialogue_repr, _) = dec_out
        assert a.output_dialogue_repr.dim() == 3, a.output_dialogue_repr.size()
        assert a.output_dialogue_repr.size()[:2] == (batch_size, max_resp_len - 1), \
            (a.output_dialogue_repr.size(), (batch_size, max_resp_len - 1))
        dialogue_repr_size = a.output_dialogue_repr.size()[2]

        # "GRU_o": encode dialogue for selection
        a.last_dialogue_repr = index_sequence(a.output_dialogue_repr.transpose(1, 2),
                                              (resp_len.data - 2)[:, None])
        assert a.last_dialogue_repr.dim() == 2, a.last_dialogue_repr.size()
        assert a.last_dialogue_repr.size() == (batch_size, dialogue_repr_size), \
            (a.last_dialogue_repr.size(), (batch_size, dialogue_repr_size))

        a.combined_repr = self.combined_layer(th.cat([a.last_context_repr, a.last_dialogue_repr],
                                                     dim=1))
        assert a.combined_repr.dim() == 2, a.combined_repr.size()
        assert a.combined_repr.size()[0] == batch_size, (a.combined_repr.size(), batch_size)

        a.all_item_scores = log_softmax(self.selection_layer(a.combined_repr))
        assert a.all_item_scores.size() == (batch_size, self.selection_layer.out_features), \
            (a.all_item_scores.size(), (batch_size, self.selection_layer.out_features))

        a.feasible_item_scores = a.all_item_scores[
            lrange(a.all_item_scores.size()[0])[:, None, None],
            feasible_sels.data
        ]
        assert a.feasible_item_scores.size() == (batch_size, MAX_FEASIBLE + 3, NUM_ITEMS), \
            (a.feasible_item_scores.size(), batch_size)

        num_feasible_mask = th.autograd.Variable(cu(
            (lrange(a.feasible_item_scores.size()[1])[None, :, None] >
             num_feasible_sels.data[:, None, None]).float()
        ))
        a.feasible_masked = a.feasible_item_scores + th.log(num_feasible_mask)
        a.full_selection_scores = log_softmax(a.feasible_item_scores.sum(dim=2))
        assert a.full_selection_scores.size() == (batch_size, MAX_FEASIBLE + 3), \
            (a.full_selection_scores.size(), batch_size)

        selection_beam = a.full_selection_scores.max(dim=1)[1]
        assert selection_beam.size() == (batch_size,), (selection_beam.size(), batch_size)
        selection_sample = th.multinomial(th.exp(a.full_selection_scores),
                                          1, replacement=True)[:, 0]
        assert selection_sample.size() == (batch_size,), (selection_sample.size(), batch_size)
        selection_predict = {
            'beam': self.sel_indices_to_selection(feasible_sels, selection_beam),
            'sample': self.sel_indices_to_selection(feasible_sels, selection_sample),
        }
        assert selection_predict['beam'].size() == (batch_size, NUM_ITEMS), \
            (selection_predict['beam'].size(), batch_size)
        assert selection_predict['sample'].size() == (batch_size, NUM_ITEMS), \
            (selection_predict['sample'].size(), batch_size)
        selection_score = a.full_selection_scores[
            lrange(a.full_selection_scores.size()[0]),
            sel_indices.data
        ]
        assert selection_score.size() == (batch_size,), (selection_score.size(), batch_size)

        predict = {
            k: response_predict[k] + (selection_predict[k],)
            for k in response_predict
        }
        score = (response_score, selection_score)
        return predict, score

    def sel_indices_to_selection(self, feasible_sels, sel_indices):
        return feasible_sels[lrange(feasible_sels.size()[0]), sel_indices.data, :]


class SupervisedLoss(th.nn.Module):
    def __init__(self, options):
        super(SupervisedLoss, self).__init__()
        self.alpha = options.selection_alpha

    def forward(self, predict, score):
        score_response, score_selection = score
        return -score_response.mean() - self.alpha * score_selection.mean()


RLLoss = SupervisedLoss  # TODO
