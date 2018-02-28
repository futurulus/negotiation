import numpy as np
from collections import Counter

import thutils

from stanza.research.rng import get_rng

rng = get_rng()


class Dimension():
    pass


class Seq2SeqVectorizer(object):
    def __init__(self, unk_threshold=0):
        self.src_vec = SequenceVectorizer(unk_threshold=unk_threshold)
        self.tgt_vec = SequenceVectorizer(unk_threshold=unk_threshold)

    def vocab_size(self):
        return self.src_vec.vocab_size(), self.tgt_vec.vocab_size()

    def add(self, pair):
        self.add_all([pair])

    def add_all(self, pairs):
        pairs = list(pairs)
        self.src_vec.add_all([p[0] for p in pairs])
        self.tgt_vec.add_all([p[1] for p in pairs])

    def output_types(self):
        return (int, int, int, int)

    def output_shapes(self):
        return ((self.src_vec.max_len,),
                (),
                (self.tgt_vec.max_len,),
                ())

    def vectorize(self, pair):
        return tuple(v[0] for v in self.vectorize_all([pair]))

    def vectorize_all(self, pairs):
        pairs = list(pairs)
        src_vec = self.src_vec.vectorize_all([p[0] for p in pairs])
        tgt_vec = self.tgt_vec.vectorize_all([p[1] for p in pairs])
        return src_vec + tgt_vec

    def unvectorize(self, indices, length):
        return self.tgt_vec.unvectorize(indices, length)

    def unvectorize_all(self, indices, lengths):
        return self.tgt_vec.unvectorize_all(indices, lengths)


class SymbolVectorizer(object):
    '''
    Maps symbols from an alphabet/vocabulary of indefinite size to and from
    sequential integer ids.

    >>> vec = SymbolVectorizer()
    >>> vec.add_all(['larry', 'moe', 'larry', 'curly', 'moe'])
    >>> vec.vectorize_all(['curly', 'larry', 'moe', 'pikachu'])
    (array([3, 1, 2, 0]),)
    >>> vec.unvectorize_all([3, 3, 2])
    ['curly', 'curly', 'moe']
    '''
    def __init__(self, use_unk=True):
        self.tokens = []
        self.token_indices = {}
        self.indices_token = {}
        if use_unk:
            self.add('<unk>')

    def vocab_size(self):
        return len(self.token_indices)

    def output_types(self):
        return (int,)

    def output_shapes(self):
        return ((),)

    def add_all(self, symbols):
        for sym in symbols:
            self.add(sym)

    def add(self, symbol):
        if symbol not in self.token_indices:
            self.token_indices[symbol] = len(self.tokens)
            self.indices_token[len(self.tokens)] = symbol
            self.tokens.append(symbol)

    def vectorize(self, symbol):
        return (self.vectorize_all([symbol])[0],)

    def vectorize_all(self, symbols):
        return (np.array([self.token_indices[sym] if sym in self.token_indices
                          else self.token_indices['<unk>']
                          for sym in symbols], dtype=np.int64),)

    def unvectorize(self, index):
        return self.indices_token[index]

    def unvectorize_all(self, array):
        if hasattr(array, 'tolist'):
            array = array.tolist()
        return [self.unvectorize(elem) for elem in array]


class SequenceVectorizer(object):
    '''
    Maps sequences of symbols from an alphabet/vocabulary of indefinite size
    to and from sequential integer ids.

    >>> vec = SequenceVectorizer()
    >>> vec.add_all([['the', 'flat', 'cat', '</s>', '</s>'], ['the', 'cat', 'in', 'the', 'hat']])
    >>> vec.vectorize_all([['in', 'the', 'cat', 'flat', '</s>'],
    ...                    ['the', 'cat', 'sat', '</s>']])
    (array([[5, 1, 3, 2, 4],
           [1, 3, 0, 4, 0]]), array([5, 4]))
    >>> vec.unvectorize_all([[1, 3, 0, 5, 1], [1, 2, 3, 6, 4]], [5, 5])
    [['the', 'cat', '<unk>', 'in', 'the'], ['the', 'flat', 'cat', 'hat', '</s>']]
    '''
    def __init__(self, unk_threshold=0):
        self.tokens = []
        self.token_indices = {}
        self.indices_token = {}
        self.counts = Counter()
        self.max_len = 0
        self.unk_threshold = unk_threshold
        self.add(['<unk>'] * (unk_threshold + 1))

    def vocab_size(self):
        return len(self.token_indices)

    def output_types(self):
        return (int, int)

    def output_shapes(self):
        return ((self.max_len,), ())

    def add_all(self, sequences):
        for seq in sequences:
            self.add(seq)

    def add(self, sequence):
        self.max_len = max(self.max_len, len(sequence))
        self.counts.update(sequence)
        for token in sequence:
            if token not in self.token_indices and self.counts[token] > self.unk_threshold:
                self.token_indices[token] = len(self.tokens)
                self.indices_token[len(self.tokens)] = token
                self.tokens.append(token)

    def unk_replace(self, sequence):
        return [(token if token in self.token_indices else '<unk>')
                for token in sequence]

    def unk_replace_all(self, sequences):
        return [self.unk_replace(s) for s in sequences]

    def vectorize(self, sequence):
        return tuple(v[0] for v in self.vectorize_all([sequence]))

    def vectorize_all(self, sequences):
        padded, lengths = zip(*(self.pad(s) for s in sequences))
        return (
            np.array([[(self.token_indices[token] if token in self.token_indices
                        else self.token_indices['<unk>'])
                       for token in sequence]
                      for sequence in padded], dtype=np.int64),
            np.array(lengths, dtype=np.int64)
        )

    def unvectorize_all(self, indices, lengths):
        return [self.unvectorize(idx_seq, length)
                for idx_seq, length in zip(indices, lengths)]

    def unvectorize(self, idx_seq, length):
        return [self.indices_token[idx] for idx in list(idx_seq)[:length]]

    def pad(self, sequence):
        if len(sequence) >= self.max_len:
            return sequence[:self.max_len], self.max_len
        else:
            return list(sequence) + ['<unk>'] * (self.max_len - len(sequence)), len(sequence)


GOAL_SIZE = 6
NUM_ITEMS = 3
MAX_COUNT = 4
MIN_FEASIBLE = 16
MAX_FEASIBLE = 36


class NegotiationVectorizer(object):
    def __init__(self, unk_threshold=0):
        self.goal_vec = SequenceVectorizer(unk_threshold=0)
        self.resp_vec = SequenceVectorizer(unk_threshold=unk_threshold)
        self.sel_vec = SelectionVectorizer()

    def vocab_size(self):
        return self.goal_vec.vocab_size(), self.resp_vec.vocab_size(), self.sel_vec.vocab_size()

    def add(self, triple):
        self.add_all([triple])

    def add_all(self, triples):
        triples = list(triples)
        self.goal_vec.add_all([t[0] for t in triples])
        self.resp_vec.add_all([t[1] for t in triples])

    def output_types(self):
        return (int, int, int, int) + self.sel_vec.output_types()

    def output_shapes(self):
        return ((GOAL_SIZE,),
                (self.resp_vec.max_len,),
                ()) + self.sel_vec.output_shapes()

    def vectorize(self, t):
        return tuple(v[0] for v in self.vectorize_all([t]))

    def vectorize_all(self, tuples):
        tuples = list(tuples)
        goal_vec = self.goal_vec.vectorize_all([t[0] for t in tuples])
        partner_goal_vec = self.goal_vec.vectorize_all([t[3] for t in tuples])
        assert goal_vec[0].shape[1] == GOAL_SIZE
        resp_vec = self.resp_vec.vectorize_all([t[1] for t in tuples])
        sel_vec = self.sel_vec.vectorize_all([(t[0], t[2]) for t in tuples])
        return (goal_vec[0], partner_goal_vec[0]) + resp_vec + sel_vec

    def unvectorize(self, resp_indices, resp_len, selections):
        return (self.resp_vec.unvectorize(resp_indices, resp_len),
                self.sel_vec.unvectorize(selections))

    def unvectorize_all(self, resp_indices, resp_len, selections):
        return (self.resp_vec.unvectorize_all(resp_indices, resp_len),
                self.sel_vec.unvectorize_all(selections))


class SelectionVectorizer():
    def __init__(self):
        self.tokens = ['<no_agreement>', '<disagree>', '<disconnect>'] + [
            f'item{i}={sel}'
            for i in range(3)
            for sel in range(5)
        ]
        pad_idx = len(self.tokens)
        self.tokens.append('<pad>')

        self.cache = {}
        for count1 in range(MAX_COUNT + 1):
            for count2 in range(MAX_COUNT + 1):
                for count3 in range(MAX_COUNT + 1):
                    if not (MIN_FEASIBLE <=
                            (count1 + 1) * (count2 + 1) * (count3 + 1) <=
                            MAX_FEASIBLE):
                        continue

                    assert self.tokens[0] == '<no_agreement>'
                    assert self.tokens[1] == '<disagree>'
                    assert self.tokens[2] == '<disconnect>'
                    feasible = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

                    for sel1 in range(count1 + 1):
                        for sel2 in range(count2 + 1):
                            for sel3 in range(count3 + 1):
                                feasible.append([
                                    self.tokens.index(f'item{i}={sel}')
                                    for i, sel in enumerate([sel1, sel2, sel3])
                                ])

                    num_feasible = len(feasible)
                    while len(feasible) < MAX_FEASIBLE + 3:
                        feasible.append([pad_idx, pad_idx, pad_idx])
                    feasible_array = np.array(feasible)
                    assert len(feasible_array.shape) == 2, feasible_array.shape
                    assert feasible_array.shape[0] <= MAX_FEASIBLE + 3, feasible_array.shape
                    assert 'int' in feasible_array.dtype.name, feasible_array.dtype

                    for j, feas in enumerate(feasible[:num_feasible]):
                        cache_key = (count1, count2, count3) + tuple(self.tokens[t] for t in feas)
                        self.cache[cache_key] = (j, feasible_array, num_feasible)

    def vocab_size(self):
        return len(self.tokens)

    def output_types(self):
        return (int, int, int)

    def output_shapes(self):
        return ((),
                (MAX_FEASIBLE + 3, NUM_ITEMS),
                ())

    def vectorize(self, t):
        assert len(t) == 2, t
        input_tokens, sel_tokens = t
        assert len(input_tokens) == GOAL_SIZE, input_tokens
        counts = (int(input_tokens[0]), int(input_tokens[2]), int(input_tokens[4]))
        assert len(sel_tokens) == NUM_ITEMS, sel_tokens

        return self.cache[counts + tuple(sel_tokens)]

    def vectorize_all(self, tuples):
        tuples = list(tuples)
        rows = [self.vectorize(t) for t in tuples]
        result = (np.array([r[0] for r in rows]),
                  np.array([r[1] for r in rows]),
                  np.array([r[2] for r in rows]))
        assert 'int' in result[0].dtype.name
        assert 'int' in result[1].dtype.name
        assert 'int' in result[2].dtype.name
        return result

    def unvectorize(self, idx_seq):
        return [self.tokens[idx] for idx in idx_seq]

    def unvectorize_all(self, indices):
        return [self.unvectorize(idx_seq)
                for idx_seq in indices]


class SelfPlayVectorizer(NegotiationVectorizer):
    def unvectorize(self, dialogue, sel_a, sel_b, reward, partner_reward):
        return self.unvectorize_all([dialogue], [sel_a], [sel_b], [reward], [partner_reward])

    def unvectorize_all(self, dialogue, sel_a, sel_b, reward, partner_reward):
        dialogue_tokens = [self.resp_vec.unvectorize_all(inds, lens) for inds, lens in dialogue]
        dialogue_transpose = [[dialogue_tokens[t][b] for t in range(len(dialogue_tokens))]
                              for b in range(len(dialogue_tokens[0]))]

        sel_a_tokens = self.sel_vec.unvectorize_all(sel_a)
        sel_b_tokens = self.sel_vec.unvectorize_all(sel_b)
        return (dialogue_transpose, sel_a_tokens, sel_b_tokens,
                thutils.to_native(reward), thutils.to_native(partner_reward))
