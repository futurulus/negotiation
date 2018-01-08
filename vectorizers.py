import numpy as np
from collections import Counter

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
