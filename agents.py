import torch as th
import numpy as np
from scipy.misc import logsumexp

from stanza.research import config
from stanza.research.rng import get_rng
from stanza.research.instance import Instance

import neural
import seq2seq
import tokenizers
from agent import Agent, random_agent_name
from baselines import RuleBasedAgent  # NOQA: prevent cyclic import
from vectorizers import MAX_FEASIBLE, NUM_ITEMS, GOAL_SIZE, all_possible_subcounts
import thutils
from thutils import index_sequence, lrange, log_softmax, maybe_cuda as cu

rng = get_rng()

parser = config.get_options_parser()
parser.add_argument('--max_dialogue_len', type=int, default=20,
                    help='Maximum number of turns in a reinforcement learning dialogue rollout.')
parser.add_argument('--goal_candidates', default=3, type=int,
                    help='Number of candidates to sample for goal-directed decoding.')
parser.add_argument('--goal_rollouts', default=3, type=int,
                    help='Number of rollouts per candidate for goal-directed decoding.')


ITEMS = ('📕 ', '🎩 ', '⚽ ')
NAMES = ('book', 'hat', 'ball')

AGREE = +1
DISAGREE = -1
NO_AGREEMENT = 0


class HumanAgent(Agent):
    def start(self):
        print('===Negotiation REPL===')
        print('')
        print('Type dialogue responses normally. Selection commands start with a slash:')
        print('  /s 1 2 0 : select a final deal (ask for 1 book, 2 hats, 0 balls)')
        print("  /y , /a : agree with partner's choice")
        print("  /n , /d : indicate no agreement or disagree with partner's choice")
        print('')

    def new_game(self, game):
        counts, your_values, _ = game
        print('NEW GAME')
        print('')
        for i in range(3):
            print(f'    {ITEMS[i] * counts[i]:8s} {NAMES[i]:4s} x{counts[i]}'
                  f' worth {your_values[i]:d} each')
        print('')
        self.game = game
        self.selection = None

    def act(self, goal_directed='ignored', both_sides='ignored'):
        while True:
            line = input('YOU: ').lower()
            if self.selection is not None:
                if not line[:2] in ('/a', '/d', '/y', '/n', '/s'):
                    print('  [partner has made proposal, choose agree (/a, /y) or '
                          'disagree (/d, /n, /s)]')
                    continue
                elif line[:2] in ('/a', '/y'):
                    return self.selection
                elif line[:2] in ('/d', '/n'):
                    return []
                elif line.startswith('/s'):
                    try:
                        return self.parse_selection(line, self.game[0])
                    except ValueError:
                        continue
                else:
                    continue
            elif line.startswith('/'):
                if line[:2] == '/s':
                    try:
                        return self.parse_selection(line, self.game[0])
                    except ValueError:
                        continue
                elif line[:2] in ('/d', '/n'):
                    return []
                elif line[:2] in ('/a', '/y'):
                    print('  [no proposal to agree to]')
                else:
                    print('  [unknown command: {}]'.format())
            else:
                return ' '.join(tokenizers.basic_unigram_tokenizer(line.strip()))

    def observe(self, result):
        if isinstance(result, list):
            self.print_selection('Your partner', result)
            if self.selection is None:
                self.selection = invert_proposal(result, self.game)
                print('')
            else:
                return True
        else:
            assert isinstance(result, str)
            print(f'THEM: {result}')

        return False

    def parse_selection(self, line, counts):
        try:
            elems = line.split()
            selection = [int(e) for e in elems[1:4]]
            for s, c in zip(selection, counts):
                if s < 0:
                    print("  [number of items can't be negative]")
                    raise ValueError
                elif s > c:
                    print(f"  [selection ({s}) greater than number of items ({c})]")
                    raise ValueError

            self.print_selection('You', selection)
            return selection
        except (IndexError, ValueError):
            print('  [/s must be followed by three integers (books, hats, balls)]')
            raise ValueError

    def print_selection(self, agent, result):
        print('')
        if result:
            print(f'  {agent} requested:')
            for i in range(3):
                print(f'    {ITEMS[i] * result[i]:8s} {NAMES[i]:4s} x{result[i]}')
        else:
            print(f'  {agent} indicated no agreement.')

    def outcome(self, outcome):
        agreement, my_value, their_value = outcome
        print('')
        if agreement == DISAGREE:
            print('  RESULT: Disagreement (0 points each).')
        elif agreement == NO_AGREEMENT:
            print('  RESULT: No agreement (0 points each).')
        else:
            print(f'  RESULT: Agreement, you got {my_value} points. (Partner got {their_value}.)')
        print('')


class TwoModelAgent(Agent):
    def new_game(self, game):
        if not hasattr(self, 'agent_id'):
            self.agent_id = random_agent_name()
        self.game = game
        self.dialogue = []
        self.sel_singleton = [None]

    def sample_action(self):
        return self.act(dialogue=self.dialogue)

    def dialogue_rollout(self, candidate, both_sides):
        rollout = list(self.dialogue)
        sel_singleton = list(self.sel_singleton)
        self.commit(candidate, dialogue=rollout, sel_singleton=sel_singleton)
        invert = True
        end = False
        while True:
            action = self.act(both_sides=both_sides, invert=invert,
                              dialogue=rollout, sel_singleton=sel_singleton)
            if invert:
                if self.observe(action, dialogue=rollout, sel_singleton=sel_singleton):
                    break
            else:
                if sel_singleton[0] is not None:
                    end = True
                self.commit(action, dialogue=rollout, sel_singleton=sel_singleton)
                if end:
                    break
            invert = not invert
        return compute_outcome(self.game, sel_singleton[0], action)

    def act(self, goal_directed=False, both_sides=False,
            invert=False, dialogue=None, sel_singleton=None):
        if goal_directed:
            return self.goal_directed_action(self.options.goal_candidates,
                                             self.options.goal_rollouts,
                                             both_sides=both_sides)

        if dialogue is None:
            dialogue = self.dialogue
            indent = ''
            inner_verbosity = 0
        else:
            indent = '    '
            inner_verbosity = -1
        if sel_singleton is None:
            sel_singleton = self.sel_singleton
        resp_model, sel_model = self.models[:2]

        if sel_singleton[0] is not None:
            inst = self.get_input_instance(self.game, dialogue, invert=invert)
            with thutils.device_context(sel_model.options.device):
                output = sel_model.predict([inst], random=True, verbosity=0)[0]
            if self.options.verbosity + inner_verbosity >= 5:
                print(f'      {indent}--OUTPUT [{self.agent_id}]: {repr(output)}')
            return parse_selection(output, self.game[0])
        else:
            inst = self.get_input_instance(self.game, dialogue, invert=(invert and not both_sides))
            if len(dialogue) >= self.options.max_dialogue_len:
                response = '<selection>'
            else:
                with thutils.device_context(resp_model.options.device):
                    response = resp_model.predict([inst], random=True, verbosity=0)[0]
            if self.options.verbosity + inner_verbosity >= 5:
                print(f'      {indent}--RESPONSE [{self.agent_id}]: {repr(response)}')
            if response == '<selection>':
                inst = self.get_input_instance(self.game, dialogue + ['YOU: <selection>'],
                                               invert=invert)
                with thutils.device_context(sel_model.options.device):
                    output = sel_model.predict([inst], random=True, verbosity=0)[0]
                if self.options.verbosity + inner_verbosity >= 5:
                    print(f'      {indent}--OUTPUT [{self.agent_id}]: {repr(output)}')
                return parse_selection(output, self.game[0])
            else:
                return response

    def commit(self, action, dialogue=None, sel_singleton=None):
        if dialogue is None:
            dialogue = self.dialogue
        if sel_singleton is None:
            sel_singleton = self.sel_singleton

        if isinstance(action, list):
            dialogue.append('YOU: <selection>')
            if sel_singleton[0] is not None:
                sel_singleton[0] = action
        else:
            dialogue.append(f'YOU: {action}')

    def observe(self, result, dialogue=None, sel_singleton=None):
        if dialogue is None:
            dialogue = self.dialogue
        if sel_singleton is None:
            sel_singleton = self.sel_singleton

        if isinstance(result, list):
            dialogue.append(f'THEM: <selection>')
            if sel_singleton[0] is None:
                sel_singleton[0] = result
            else:
                return True
        else:
            assert isinstance(result, str)
            dialogue.append('THEM: ' + result)

        return False

    def get_input_instance(self, game, dialogue, invert=False):
        if invert:
            rewards = self.infer_their_rewards(game, self.dialogue)
        else:
            rewards = game[1]
        pieces = [f'{game[0][0]} {rewards[0]} {game[0][1]} {rewards[1]} {game[0][2]} {rewards[2]}']
        for entry in dialogue:
            if invert:
                entry = entry.replace('YOU:', 'XYOU:')
                entry = entry.replace('THEM:', 'YOU:')
                entry = entry.replace('XYOU:', 'THEM:')
            pieces.append(f'{entry} <eos>')
        input = ' '.join(pieces)
        if dialogue:
            input = input[:-len(' <eos>')]
        result = Instance(input, '')
        if self.options.verbosity >= 6:
            print(result.__dict__)
        return result

    def goal_directed_action(self, num_candidates, num_rollouts, both_sides):
        candidates = [self.sample_action() for _ in range(num_candidates)]
        if self.options.verbosity >= 5:
            for candidate in candidates:
                print(f'        --CANDIDATE [{self.agent_id}]: {repr(candidate)}')

        best_candidates = []
        best_ave_reward = 0.0
        for candidate in candidates:
            outcomes = [self.dialogue_rollout(candidate, both_sides=both_sides)
                        for _ in range(num_rollouts)]
            ave_reward = np.mean([our_outcome[1] for our_outcome, _ in outcomes])
            if self.options.verbosity >= 5:
                print(f'        --AVE_REWARD [{self.agent_id}]: {ave_reward} <= '
                      f'{repr(candidate)}')
            if ave_reward > best_ave_reward:
                best_candidates = [candidate]
                best_ave_reward = ave_reward
            else:
                best_candidates.append(candidate)
        choice = best_candidates[rng.randint(len(best_candidates))]
        if self.options.verbosity >= 5:
            print(f'      --CHOICE [{self.agent_id}]: {repr(choice)}')
        return choice

    def infer_their_rewards(self, game, dialogue):
        # Pick something feasible at random.
        possible = [r for r in all_possible_rewards(game[0])
                    if not has_double_zeros(r, game[1])]
        return possible[rng.randint(len(possible))]

    def outcome(self, outcome):
        if self.options.verbosity >= 5:
            print(f"  --GAME [{self.agent_id}]: {self.game}")


class RSAAgent(TwoModelAgent):
    def infer_their_rewards(self, game, dialogue):
        assert len(self.models) >= 3, \
            'Not enough models for RSA agent (need 3, got {})'.format(len(self.models))
        # Use model to sample possible other rewards
        inst = self.get_input_instance(game, dialogue)
        possible = [r for r in all_possible_rewards(game[0])
                    if not has_double_zeros(r, game[1])]
        score_insts = [self.fill_score_instance(inst, r, game[0])
                       for r in possible]
        pred_model = self.models[2]
        with thutils.device_context(pred_model.options.device):
            scores = pred_model.score(score_insts)
        probs = np.exp(np.array(scores) - logsumexp(scores))
        if self.options.verbosity >= 6:
            print([i.output for i in score_insts])
            print(scores)
            print(probs)
        return possible[rng.choice(np.arange(len(possible)),
                                   p=probs)]

    def fill_score_instance(self, inst, rewards, counts):
        inst_dict = inst.__dict__.copy()
        inst_dict['output'] = \
            f'{counts[0]} {rewards[0]} {counts[1]} {rewards[1]} {counts[2]} {rewards[2]}'
        return Instance(**inst_dict)


class FBReproAgent(Agent):
    def start(self):
        self.negotiator = self.models[0].model.module
        self.vectorizer = self.models[0].model.vectorizer
        self.tokenize, self.detokenize = tokenizers.TOKENIZERS[self.models[0].options.tokenizer]
        self.eos = th.LongTensor(self.vectorizer.resp_vec.vectorize(['<eos>'])[0])[0]
        self.you = th.LongTensor(self.vectorizer.resp_vec.vectorize(['YOU:'])[0])[0]
        self.them = th.LongTensor(self.vectorizer.resp_vec.vectorize(['THEM:'])[0])[0]

    def new_game(self, game):
        if not hasattr(self, 'agent_id'):
            self.agent_id = random_agent_name()

        with self.use_device():
            goal_indices, self.feasible_sels, self.num_feasible_sels = self.vectorize_game(game)
            self.negotiator.context(goal_indices)
        self.game = game
        self.sel_singleton = [None]

    def vectorize_game(self, game):
        input_tokens = [str(e) for pair in zip(game[0], game[1]) for e in pair]
        partner_tokens = [str(e) for pair in zip(game[0], game[2]) for e in pair]
        (goal_indices, partner_,
         resp_, resp_len_,
         sel_, feasible_sels,
         num_feasible_sels) = self.vectorizer.vectorize((input_tokens,
                                                        ['<dialogue>', '</dialogue>'],
                                                        ['<no_agreement>'] * 3,
                                                        partner_tokens))
        return (thutils.to_torch(goal_indices)[None, :],
                thutils.to_torch(feasible_sels)[None, :],
                thutils.to_torch(num_feasible_sels)[None, :])

    def act(self, goal_directed=False, both_sides=False,
            invert=False, dialogue=None, sel_singleton=None):
        if goal_directed or both_sides:
            raise NotImplementedError
        if sel_singleton is None:
            sel_singleton = self.sel_singleton

        with self.use_device():
            if sel_singleton[0] is not None:
                action = self.make_selection()
            else:
                output_predict, output_score = self.negotiator.speak(self.you, self.eos)
                (resp_indices, resp_len) = output_predict['sample']

                if is_selection(self.vectorizer, resp_indices, resp_len):
                    action = self.make_selection()
                else:
                    action = self.vectorizer.resp_vec.unvectorize(thutils.to_numpy(resp_indices)[0],
                                                                  thutils.to_numpy(resp_len)[0])
                    action = self.detokenize(action[1:])

        if self.options.verbosity >= 5:
            print(f'      --ACT [{self.agent_id}]: {repr(action)}')
        return action

    def make_selection(self):
        empty_sel_indices = th.autograd.Variable(cu(th.LongTensor([0])))
        sel_predict, sel_score = self.negotiator.selection(empty_sel_indices,
                                                           self.feasible_sels,
                                                           self.num_feasible_sels)
        return parse_selection(' '.join(self.vectorizer.sel_vec.unvectorize(
            thutils.to_numpy(sel_predict['sample'])[0]
        )), self.game[0])

    def commit(self, action, dialogue=None, sel_singleton=None):
        if sel_singleton is None:
            sel_singleton = self.sel_singleton

        if isinstance(action, list):
            if sel_singleton[0] is not None:
                sel_singleton[0] = action
            return

        with self.use_device():
            resp_indices, resp_len = self.vectorize_response(action, self.you)
            self.negotiator.listen(resp_indices, resp_len)

    def observe(self, result, dialogue=None, sel_singleton=None):
        if sel_singleton is None:
            sel_singleton = self.sel_singleton

        if isinstance(result, list):
            if sel_singleton[0] is None:
                sel_singleton[0] = result
                result = '<selection>'
            else:
                return True

        if self.options.verbosity >= 5:
            print(f'      --OBSERVE [{self.agent_id}]: {repr(result)}')

        with self.use_device():
            resp_indices, resp_len = self.vectorize_response(result, self.them)
            self.negotiator.listen(resp_indices, resp_len)

        return False

    def vectorize_response(self, response, you_them):
        tag = th.autograd.Variable(cu(th.LongTensor([[self.you]])))
        resp_indices, resp_len = self.vectorizer.resp_vec.vectorize(self.tokenize(response))
        tagged_resp_indices = th.cat([tag.expand(1, 1),
                                      thutils.to_torch(resp_indices)[None, :]], 1)
        return (tagged_resp_indices, thutils.to_torch(resp_len + 1))

    def use_device(self):
        return thutils.device_context(self.models[0].options.device)


def invert_proposal(response, game):
    return [c - s for c, s in zip(game[0], response)]


def parse_selection(line, counts):
    if line.startswith('<'):
        return []

    import re
    match = re.search(r'item0=(\d+) item1=(\d+) item2=(\d+)', line)
    if not match:
        return []
    else:
        return [max(0, min(c, int(s))) for c, s in zip(counts, match.groups())]


def has_double_zeros(their_rewards, our_rewards):
    assert len(their_rewards) == len(our_rewards) == 3, (their_rewards, our_rewards)
    for t, o in zip(their_rewards, our_rewards):
        if t == 0 == o:
            return True
    return False


def compute_outcome(game, proposal_a, response_a):
    if response_a != proposal_a:
        return (DISAGREE, 0, 0), (DISAGREE, 0, 0)
    elif proposal_a == []:
        return (NO_AGREEMENT, 0, 0), (NO_AGREEMENT, 0, 0)
    else:
        value_a = sum([s * v for s, v in zip(proposal_a, game[1])])
        value_b = sum([(c - s) * v for c, s, v in zip(game[0], proposal_a, game[2])])
        return (AGREE, value_a, value_b), (AGREE, value_b, value_a)


AGENTS = {
    c.__name__: c
    for c in [HumanAgent, TwoModelAgent, RSAAgent, RuleBasedAgent,
              FBReproAgent]
}


REWARDS_CACHE = {}


def all_possible_rewards(counts):
    counts = tuple(counts)
    if counts not in REWARDS_CACHE:
        possible = []
        for r1, r2, r3 in all_possible_subcounts([10, 10, 10]):
            if r1 * counts[0] + r2 * counts[1] + r3 * counts[2] == 10:
                possible.append((r1, r2, r3))
        REWARDS_CACHE[counts] = possible
    return REWARDS_CACHE[counts]


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

        self.dec_state = seq2seq.generate_rnn_state(self.response_encoder,
                                                    self.h_init, self.c_init, batch_size)
        if not isinstance(self.dec_state, tuple):
            self.dec_state = (self.dec_state,)

    def dialogue(self, resp_indices, resp_len, persist=True, predict=True, eos_token=None):
        # "GRU_w": encode and produce dialogue
        a = self.activations

        assert resp_indices.dim() == 2, resp_indices.size()
        batch_size, max_resp_len = resp_indices.size()

        dec_state_concat = tuple(self.response_encoder.concat_directions(c) for c in self.dec_state)
        response_predict, response_score, response_output = self.response_decoder(
            dec_state_concat,
            resp_indices, resp_len,
            extra_inputs=[a.context_repr],
            extra_delimiter=eos_token,
            output_beam=predict, output_sample=predict
        )
        (dialogue_repr_seq, dec_state) = response_output['target']
        if persist:
            a.dialogue_repr_seq, self.dec_state = dialogue_repr_seq, dec_state
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


class RLNegotiator(th.nn.Module):
    def __init__(self, negotiator, partner, vectorizer, options):
        super(RLNegotiator, self).__init__()
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
            me = self.negotiator if my_turn else self.partner
            other = self.partner if my_turn else self.negotiator

            output_predict, output_score = me.speak(self.you, self.eos)
            (me_resp_indices, resp_len), policy_score = self.policy(output_predict, output_score)
            start_with_you = th.autograd.Variable(cu(th.LongTensor([[self.you]])))
            me_resp_indices = th.cat([start_with_you.expand(resp_len.size()[0], 1),
                                      me_resp_indices], 1)
            me.listen(me_resp_indices, resp_len + 1)

            other_resp_indices = self.transform_dialogue(me_resp_indices)
            other.listen(other_resp_indices, resp_len + 1)

            dialogue.append(((me_resp_indices if my_turn else other_resp_indices), resp_len))
            policy_scores.append(policy_score)
            if is_selection(self.vectorizer, me_resp_indices, resp_len):
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
            return output_predict['beam'], th.autograd.Variable(cu(th.FloatTensor([0.0])))
            # output_score['beam']

    def transform_dialogue(self, resp_indices):
        you, them = th.LongTensor(self.vectorizer.resp_vec.vectorize(['YOU:', 'THEM:'])[0][:2])
        you_mask = (resp_indices == you)
        them_mask = (resp_indices == them)
        transformed = resp_indices.clone()
        transformed[you_mask.data] = them
        transformed[them_mask.data] = you
        return transformed


def is_selection(vectorizer, resp_indices, resp_len):
    selection = th.LongTensor(vectorizer.resp_vec.vectorize(['<selection>'])[0])[0]
    return resp_indices.data[0, 0] == selection and resp_len.data[0] == 1


def compute_reward(sel, other_sel, goal_indices):
    assert goal_indices.size()[1] == NUM_ITEMS * 2, goal_indices.size()
    counts = goal_indices[:, cu(th.LongTensor(range(0, NUM_ITEMS * 2, 2)))]
    values = goal_indices[:, cu(th.LongTensor(range(1, NUM_ITEMS * 2, 2)))]
    total_claimed = sel + other_sel
    # feasible = (total_claimed >= 0).prod() * (total_claimed <= counts).prod()
    feasible = (total_claimed == counts).prod().long()

    return ((values * sel).sum(1) * feasible).float()
