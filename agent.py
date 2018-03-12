import torch as th
import numpy as np

from stanza.research.rng import get_rng

import neural
import seq2seq
from vectorizers import MAX_FEASIBLE, NUM_ITEMS, GOAL_SIZE
from thutils import index_sequence, lrange, log_softmax, maybe_cuda as cu

rng = get_rng()


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
                goal_indices, partner_goal_indices_,
                resp_indices, resp_len,
                sel_indices, feasible_sels, num_feasible_sels):
        a = self.activations

        batch_size, goal_size = goal_indices.size()

        self.context(goal_indices)

        assert resp_indices.size()[0] == batch_size, resp_indices.size()
        response_predict, response_score = self.dialogue(resp_indices, resp_len)

        assert a.dialogue_repr.size()[0] == batch_size, (a.dialogue_repr.size(), batch_size)
        selection_predict, selection_score = self.selection(sel_indices, feasible_sels,
                                                            num_feasible_sels)

        predict = {
            k: response_predict[k] + (selection_predict[k],)
            for k in response_predict
        }
        score = (response_score, selection_score)
        return predict, score

    def context(self, goal_indices):
        # "GRU_g": encode goals (values of items)
        a = self.activations

        batch_size, goal_size = goal_indices.size()
        assert goal_size == GOAL_SIZE, goal_indices.size()

        goal_len = th.autograd.Variable(cu(
            (th.ones(batch_size) * goal_size).int()
        ))
        assert goal_len.size() == (batch_size,), goal_len.size()

        a.context_repr_seq, _ = self.context_encoder(goal_indices, goal_len)
        assert a.context_repr_seq.dim() == 3, a.context_repr_seq.size()
        assert a.context_repr_seq.size()[:2] == (batch_size, goal_size), a.context_repr_seq.size()

        a.context_repr = a.context_repr_seq[:, -1, :]
        context_repr_size = a.context_repr_seq.size()[2]
        assert a.context_repr.size() == (batch_size, context_repr_size), a.context_repr.size()

        a.dec_state = seq2seq.generate_rnn_state(self.response_encoder,
                                                 self.h_init, self.c_init, batch_size)
        if not isinstance(a.dec_state, tuple):
            a.dec_state = (a.dec_state,)

    def dialogue(self, resp_indices, resp_len, persist=True, predict=True, eos_token=None):
        # "GRU_w": encode and produce dialogue
        a = self.activations

        assert resp_indices.dim() == 2, resp_indices.size()
        batch_size, max_resp_len = resp_indices.size()

        dec_state_concat = tuple(self.response_encoder.concat_directions(c) for c in a.dec_state)
        response_predict, response_score, response_output = self.response_decoder(
            dec_state_concat,
            resp_indices, resp_len,
            extra_inputs=[a.context_repr],
            extra_delimiter=eos_token,
            output_beam=predict, output_sample=predict
        )
        (dialogue_repr_seq, dec_state) = response_output['target']
        if persist:
            a.dialogue_repr_seq, a.dec_state = dialogue_repr_seq, dec_state
        assert dialogue_repr_seq.dim() == 3, dialogue_repr_seq.size()
        assert dialogue_repr_seq.size()[:2] == (batch_size, max_resp_len - 1), \
            (dialogue_repr_seq.size(), (batch_size, max_resp_len - 1))
        dialogue_repr_size = dialogue_repr_seq.size()[2]

        dialogue_repr = index_sequence(dialogue_repr_seq.transpose(1, 2),
                                       th.clamp(resp_len.data, max=max_resp_len - 2)[:, None])
        if persist:
            a.dialogue_repr = dialogue_repr
        assert dialogue_repr.dim() == 2, dialogue_repr.size()
        assert dialogue_repr.size() == (batch_size, dialogue_repr_size), \
            (dialogue_repr.size(), (batch_size, dialogue_repr_size))

        return response_predict, response_score

    def selection(self, sel_indices, feasible_sels, num_feasible_sels):
        # "GRU_o": encode dialogue for selection
        a = self.activations

        assert sel_indices.dim() == 1, sel_indices.size()
        batch_size = sel_indices.size()[0]

        a.combined_repr = self.combined_layer(th.cat([a.context_repr, a.dialogue_repr],
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

        a.selection_beam_score, selection_beam = a.full_selection_scores.max(dim=1)
        assert selection_beam.size() == (batch_size,), (selection_beam.size(), batch_size)
        selection_sample = th.multinomial(th.exp(a.full_selection_scores),
                                          1, replacement=True)[:, 0]
        a.selection_sample_score = th.exp(a.full_selection_scores)[
            lrange(a.full_selection_scores.size()[0]),
            selection_sample.data
        ]
        assert selection_sample.size() == (batch_size,), (selection_sample.size(), batch_size)
        selection_predict = {
            'beam': self.sel_indices_to_selection(feasible_sels, selection_beam),
            'sample': self.sel_indices_to_selection(feasible_sels, selection_sample),
        }
        assert selection_predict['beam'].size() == (batch_size, NUM_ITEMS), \
            (selection_predict['beam'].size(), batch_size)
        assert selection_predict['sample'].size() == (batch_size, NUM_ITEMS), \
            (selection_predict['sample'].size(), batch_size)
        a.selection_target_score = a.full_selection_scores[
            lrange(a.full_selection_scores.size()[0]),
            sel_indices.data
        ]
        assert a.selection_target_score.size() == (batch_size,), (a.selection_score.size(),
                                                                  batch_size)
        selection_score = {
            'target': a.selection_target_score,
            'beam': a.selection_beam_score,
            'sample': a.selection_sample_score,
        }

        return selection_predict, selection_score

    def sel_indices_to_selection(self, feasible_sels, sel_indices):
        return feasible_sels[lrange(feasible_sels.size()[0]), sel_indices.data, :]

    def speak(self, you_token, eos_token=None):
        empty_resp_indices = th.autograd.Variable(cu(th.LongTensor([[0, 1]])))
        empty_resp_len = th.autograd.Variable(cu(th.LongTensor([2])))
        response_predict, response_score = self.dialogue(empty_resp_indices, empty_resp_len,
                                                         persist=False, eos_token=eos_token)
        del response_score['target']
        return response_predict, response_score

    def listen(self, resp_indices, resp_len):
        self.dialogue(resp_indices, resp_len, predict=False)


class SupervisedLoss(th.nn.Module):
    def __init__(self, options):
        super(SupervisedLoss, self).__init__()
        self.alpha = options.selection_alpha

    def forward(self, predict, score):
        response_score, selection_score = score
        return -response_score['target'].mean() - self.alpha * selection_score['target'].mean()


class RLLoss(th.nn.Module):
    def __init__(self, options):
        super(RLLoss, self).__init__()
        self.reward_history = []
        self.gamma = options.rl_gamma

    def forward(self, predict, score):
        dialogue, sel_a, sel_b, reward, partner_reward = predict
        response_scores, selection_score = score

        reward_transformed = self.transform_reward(reward)
        step_rewards = []
        discount = th.Variable(cu(th.FloatTensor([1.0])))
        for i in range(len(response_scores)):
            step_rewards.append(discount * reward_transformed)
            discount = discount * self.gamma

        loss = th.Variable(cu(th.FloatTensor([0.0])))
        for score, step_reward in zip(response_scores, step_rewards):
            loss -= score * step_reward

        return loss

    def transform_reward(self, reward):
        self.reward_history.append(reward)
        mu = np.mean(self.reward_history)
        sigma = max(1.0, np.std(self.reward_history))
        return (reward - mu) / sigma


class RLAgent(th.nn.Module):
    def __init__(self, negotiator, partner, vectorizer, options):
        super(RLAgent, self).__init__()
        self.negotiator = negotiator
        self.partner = partner
        self.vectorizer = vectorizer
        self.eos = th.LongTensor(self.vectorizer.resp_vec.vectorize(['<eos>'])[0])[0]
        self.you = th.LongTensor(self.vectorizer.resp_vec.vectorize(['YOU:'])[0])[0]

        self.epsilon = options.rl_epsilon
        self.max_dialogue_len = options.max_dialogue_len

    def forward(self,
                goal_indices, partner_goal_indices,
                resp_indices_, resp_len_,
                sel_indices_, feasible_sels, num_feasible_sels):
        num_feasible_sels = th.autograd.Variable(cu(th.LongTensor(
            [feasible_sels.size()[1]]
        )))

        self.negotiator.context(goal_indices)
        self.partner.context(goal_indices)

        my_turn = rng.choice([True, False])
        dialogue = []
        policy_scores = []
        for _ in range(self.max_dialogue_len):
            agent = self.negotiator if my_turn else self.partner
            other = self.partner if my_turn else self.negotiator

            output_predict, output_score = agent.speak(self.you, self.eos)
            (agent_resp_indices, resp_len), policy_score = self.policy(output_predict, output_score)
            start_with_you = th.autograd.Variable(cu(th.LongTensor([[self.you]])))
            agent_resp_indices = th.cat([start_with_you.expand(resp_len.size()[0], 1),
                                         agent_resp_indices], 1)
            agent.listen(agent_resp_indices, resp_len + 1)

            other_resp_indices = self.transform_dialogue(agent_resp_indices)
            other.listen(other_resp_indices, resp_len + 1)

            dialogue.append(((agent_resp_indices if my_turn else other_resp_indices), resp_len))
            policy_scores.append(policy_score)
            if self.is_selection(agent_resp_indices, resp_len):
                break

            my_turn = not my_turn

        empty_sel_indices = th.autograd.Variable(cu(th.LongTensor([0])))
        # TODO: epsilon-greedy here too?
        selection_predict, selection_score = self.negotiator.selection(empty_sel_indices,
                                                                       feasible_sels,
                                                                       num_feasible_sels)
        sel_a = selection_predict['beam']
        sel_b = self.partner.selection(empty_sel_indices,
                                       feasible_sels, num_feasible_sels)[0]['beam']

        reward = compute_reward(sel_a, sel_b, goal_indices)
        partner_reward = compute_reward(sel_b, sel_a, partner_goal_indices)

        result = (dialogue, sel_a, sel_b, reward, partner_reward)
        return {'sample': result, 'beam': result}, (th.stack(policy_scores, 0)[:, 0],
                                                    selection_score)

    def policy(self, output_predict, output_score):
        if rng.random_sample() <= self.epsilon:
            return output_predict['sample'], output_score['sample']
        else:
            return output_predict['beam'], th.autograd.Variable(cu(th.FloatTensor([1.0])))
            # output_score['beam']

    def transform_dialogue(self, resp_indices):
        you, them = th.LongTensor(self.vectorizer.resp_vec.vectorize(['YOU:', 'THEM:'])[0][:2])
        you_mask = (resp_indices == you)
        them_mask = (resp_indices == them)
        transformed = resp_indices.clone()
        transformed[you_mask.data] = them
        transformed[them_mask.data] = you
        return transformed

    def is_selection(self, resp_indices, resp_len):
        selection = th.LongTensor(self.vectorizer.resp_vec.vectorize(['<selection>'])[0])[0]
        return resp_indices.data[0, 0] == selection and resp_len.data[0] == 1


def compute_reward(sel, other_sel, goal_indices):
    assert goal_indices.size()[1] == NUM_ITEMS * 2, goal_indices.size()
    counts = goal_indices[:, cu(th.LongTensor(range(0, NUM_ITEMS * 2, 2)))]
    values = goal_indices[:, cu(th.LongTensor(range(1, NUM_ITEMS * 2, 2)))]
    total_claimed = sel + other_sel
    # feasible = (total_claimed >= 0).prod() * (total_claimed <= counts).prod()
    feasible = (total_claimed == counts).prod().long()

    return ((values * sel).sum(1) * feasible).float()
