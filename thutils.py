import contextlib
import numbers
import torch as th
import torch.nn.utils.rnn as rnn


def lrange(start, stop=None, step=None):
    if step is None:
        if stop is None:
            r = th.arange(0, start)
        else:
            r = th.arange(start, stop)
    else:
        r = th.arange(start, stop, step)
    return maybe_cuda(r).long()


def index_sequence(seq, idx):
    '''
    >>> from torch import FloatTensor as FT
    >>> s = [[[0.0, 0.1, 0.2],
    ...       [1.0, 1.1, 1.2],
    ...       [2.0, 2.1, 2.2]],
    ...      [[10.0, 10.1, 10.2],
    ...       [11.0, 11.1, 11.2],
    ...       [12.0, 12.1, 12.2]],
    ...      [[20.0, 20.1, 20.2],
    ...       [21.0, 21.1, 21.2],
    ...       [22.0, 22.1, 22.2]]]
    >>> i = [[2, 0, 1], [1, 1, 0], [0, 1, 2]]
    >>> index_sequence(FT(s), i)
    <BLANKLINE>
      0.2000   1.0000   2.1000
     10.1000  11.1000  12.0000
     20.0000  21.1000  22.2000
    [torch.FloatTensor of size 3x3]
    <BLANKLINE>
    '''
    return seq[lrange(seq.size()[0])[:, None],
               lrange(seq.size()[1])[None, :],
               idx]


def varlen_rnn(cell, input, lengths, hidden):
    '''
    Handles running a variable-length input through an RNN using PyTorch's PackedSequence
    functionality. Simply replace

        rnn_cell(input, hidden)

    with

        varlen_rnn(rnn_cell, input, lengths, hidden)

    In particular, you do not need to sort the sequences in decreasing length.

    Examples with an RNN cell that is constructed to compute tanh(1/2 * h + x) at each step:
    >>> cell = th.nn.RNN(3, 3, 1, batch_first=True)
    >>> cell.weight_ih_l0.data = th.eye(3)
    >>> cell.weight_hh_l0.data = 0.5 * th.eye(3)
    >>> cell.bias_ih_l0.data = th.zeros(3)
    >>> cell.bias_hh_l0.data = th.zeros(3)

    >>> from torch import FloatTensor as FT
    >>> from torch.autograd import Variable as Var
    >>> s = th.zeros((3, 3, 3))
    >>> s[0, :, :] = 16.
    >>> s[2, :, :] = 32.
    >>> h0 = [[[8., 8., 8.],
    ...        [16., 16., 16.],
    ...        [24., 24., 24.]]]

    Running the cell without PackedSequence:
    >>> cell(Var(FT(s) * 1e-5), Var(FT(h0) * 1e-5))
    (Variable containing:
    (0 ,.,.) = 
    1.00000e-04 *
       2.0000  2.0000  2.0000
       2.6000  2.6000  2.6000
       2.9000  2.9000  2.9000
    <BLANKLINE>
    (1 ,.,.) = 
    1.00000e-04 *
       0.8000  0.8000  0.8000
       0.4000  0.4000  0.4000
       0.2000  0.2000  0.2000
    <BLANKLINE>
    (2 ,.,.) = 
    1.00000e-04 *
       4.4000  4.4000  4.4000
       5.4000  5.4000  5.4000
       5.9000  5.9000  5.9000
    [torch.FloatTensor of size 3x3x3]
    , Variable containing:
    (0 ,.,.) = 
    1.00000e-04 *
       2.9000  2.9000  2.9000
       0.2000  0.2000  0.2000
       5.9000  5.9000  5.9000
    [torch.FloatTensor of size 1x3x3]
    )

    Running the cell with PackedSequence:
    >>> lens = [1, 3, 2]
    >>> varlen_rnn(cell, Var(FT(s) * 1e-5), lens, Var(FT(h0) * 1e-5))
    (Variable containing:
    (0 ,.,.) = 
    1.00000e-04 *
       2.0000  2.0000  2.0000
       0.0000  0.0000  0.0000
       0.0000  0.0000  0.0000
    <BLANKLINE>
    (1 ,.,.) = 
    1.00000e-04 *
       0.8000  0.8000  0.8000
       0.4000  0.4000  0.4000
       0.2000  0.2000  0.2000
    <BLANKLINE>
    (2 ,.,.) = 
    1.00000e-04 *
       4.4000  4.4000  4.4000
       5.4000  5.4000  5.4000
       0.0000  0.0000  0.0000
    [torch.FloatTensor of size 3x3x3]
    , Variable containing:
    (0 ,.,.) = 
    1.00000e-04 *
       2.0000  2.0000  2.0000
       0.2000  0.2000  0.2000
       5.4000  5.4000  5.4000
    [torch.FloatTensor of size 1x3x3]
    )
    '''  # NOQA: verbatim trailing whitespace
    batch_first = cell.batch_first

    input_packed, indices = sort_and_pack(input, lengths, batch_first=batch_first)
    hidden_sorted = index_sorted(hidden, indices, batch_first=False)
    out_packed, state_sorted = cell(input_packed, hidden_sorted)
    out_sorted, _ = rnn.pad_packed_sequence(out_packed, batch_first=batch_first)

    out = unsort(out_sorted, indices, batch_first=batch_first)
    state = unsort(state_sorted, indices, batch_first=False)
    return out, state


def sort_and_pack(input, lengths, batch_first=False):
    '''
    >>> from torch import FloatTensor as FT
    >>> x = [[0.0, 0.1, 0.2],
    ...      [1.0, 1.1, 1.2],
    ...      [2.0, 2.1, 2.2]]
    >>> lens = [3, 1, 2]
    >>> sort_and_pack(FT(x), lens)
    (PackedSequence(data=
     0.0000
     0.2000
     0.1000
     1.0000
     1.2000
     2.0000
    [torch.FloatTensor of size 6]
    , batch_sizes=[3, 2, 1]), 
     0
     2
     1
    [torch.LongTensor of size 3]
    )

    >>> sort_and_pack(FT(x), lens, batch_first=True)
    (PackedSequence(data=
     0.0000
     2.0000
     1.0000
     0.1000
     2.1000
     0.2000
    [torch.FloatTensor of size 6]
    , batch_sizes=[3, 2, 1]), 
     0
     2
     1
    [torch.LongTensor of size 3]
    )
    '''  # NOQA: verbatim trailing whitespace
    if isinstance(lengths, list):
        lengths = maybe_cuda(th.FloatTensor(lengths))
    elif isinstance(lengths, th.LongTensor):
        lengths = lengths.float()

    lengths_sorted, sort_indices = th.sort(maybe_cuda(th.FloatTensor(lengths)), 0, descending=True)
    input_sorted = index_sorted(input, sort_indices, batch_first=batch_first)

    packed = rnn.pack_padded_sequence(input_sorted, lengths_sorted.tolist(),
                                      batch_first=batch_first)
    return packed, sort_indices


def index_sorted(x, indices, batch_first=False):
    '''
    >>> from torch import FloatTensor as FT
    >>> x = [[0.0, 0.1, 0.2],
    ...      [1.0, 1.1, 1.2],
    ...      [2.0, 2.1, 2.2]]
    >>> i = [2, 0, 1]
    >>> index_sorted(FT(x), i)
    <BLANKLINE>
     0.2000  0.0000  0.1000
     1.2000  1.0000  1.1000
     2.2000  2.0000  2.1000
    [torch.FloatTensor of size 3x3]
    <BLANKLINE>

    >>> index_sorted(FT(x), i, batch_first=True)
    <BLANKLINE>
     2.0000  2.1000  2.2000
     0.0000  0.1000  0.2000
     1.0000  1.1000  1.2000
    [torch.FloatTensor of size 3x3]
    <BLANKLINE>

    >>> x = [[[0.0, 0.1, 0.2],
    ...       [1.0, 1.1, 1.2],
    ...       [2.0, 2.1, 2.2]],
    ...      [[10.0, 10.1, 10.2],
    ...       [11.0, 11.1, 11.2],
    ...       [12.0, 12.1, 12.2]],
    ...      [[20.0, 20.1, 20.2],
    ...       [21.0, 21.1, 21.2],
    ...       [22.0, 22.1, 22.2]]]
    >>> index_sorted(FT(x), i)
    <BLANKLINE>
    (0 ,.,.) = 
       2.0000   2.1000   2.2000
       0.0000   0.1000   0.2000
       1.0000   1.1000   1.2000
    <BLANKLINE>
    (1 ,.,.) = 
      12.0000  12.1000  12.2000
      10.0000  10.1000  10.2000
      11.0000  11.1000  11.2000
    <BLANKLINE>
    (2 ,.,.) = 
      22.0000  22.1000  22.2000
      20.0000  20.1000  20.2000
      21.0000  21.1000  21.2000
    [torch.FloatTensor of size 3x3x3]
    <BLANKLINE>

    >>> index_sorted(FT(x), i, batch_first=True)
    <BLANKLINE>
    (0 ,.,.) = 
      20.0000  20.1000  20.2000
      21.0000  21.1000  21.2000
      22.0000  22.1000  22.2000
    <BLANKLINE>
    (1 ,.,.) = 
       0.0000   0.1000   0.2000
       1.0000   1.1000   1.2000
       2.0000   2.1000   2.2000
    <BLANKLINE>
    (2 ,.,.) = 
      10.0000  10.1000  10.2000
      11.0000  11.1000  11.2000
      12.0000  12.1000  12.2000
    [torch.FloatTensor of size 3x3x3]
    <BLANKLINE>
    '''  # NOQA: verbatim trailing whitespace
    if isinstance(indices, (th.LongTensor, th.ByteTensor)):
        indices = indices.tolist()
    assert len(indices) == 0 or not isinstance(indices[0], list), \
        'indices is not 1-dimensional [{} x {} x ...] (varlen_rnn and sort_and_pack do not ' \
        'currently support multidimensional batch sizes)'.format(len(indices), len(indices[0]))

    if isinstance(x, tuple):
        return tuple(index_sorted(e, indices, batch_first=batch_first) for e in x)
    elif batch_first:
        return x[indices, :]
    else:
        return x[:, indices]


def unsort(sorted, indices, batch_first=False):
    '''
    >>> from torch import FloatTensor as FT
    >>> x = [[0.0, 0.1, 0.2],
    ...      [1.0, 1.1, 1.2],
    ...      [2.0, 2.1, 2.2]]
    >>> i = [2, 0, 1]
    >>> unsort(FT(x), i)
    <BLANKLINE>
     0.1000  0.2000  0.0000
     1.1000  1.2000  1.0000
     2.1000  2.2000  2.0000
    [torch.FloatTensor of size 3x3]
    <BLANKLINE>
    >>> unsort(FT(x), i, batch_first=True)
    <BLANKLINE>
     1.0000  1.1000  1.2000
     2.0000  2.1000  2.2000
     0.0000  0.1000  0.2000
    [torch.FloatTensor of size 3x3]
    <BLANKLINE>
    '''
    if isinstance(indices, list):
        indices = maybe_cuda(th.FloatTensor(indices))
    elif isinstance(indices, th.LongTensor):
        indices = indices.float()

    _, indices_inverse = th.sort(indices, 0)
    return index_sorted(sorted, indices_inverse, batch_first=batch_first)


def to_numpy(obj):
    import numpy as np
    if isinstance(obj, (numbers.Number, np.ndarray)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_numpy(e) for e in obj)
    elif isinstance(obj, dict):
        return {k: to_numpy(v) for k, v in obj.items()}

    if isinstance(obj, th.autograd.Variable):
        obj = obj.data

    return obj.cpu().numpy()


def to_native(obj):
    import numpy as np
    if isinstance(obj, numbers.Number):
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_native(e) for e in obj)
    elif isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}

    if isinstance(obj, th.autograd.Variable):
        obj = obj.data

    return obj.cpu().tolist()


def to_torch(obj):
    import numpy as np
    if isinstance(obj, numbers.Number):
        obj = np.array([obj])

    if isinstance(obj, np.ndarray):
        result = th.from_numpy(obj)
    elif isinstance(obj, (list, tuple)):
        result = type(obj)(to_torch(e) for e in obj)
    elif isinstance(obj, dict):
        result = {k: to_torch(v) for k, v in obj.items()}

    return th.autograd.Variable(maybe_cuda(result))


def log_softmax(x, dim=-1):
    return th.nn.LogSoftmax(dim=dim)(x)


_device = 'cpu'


@contextlib.contextmanager
def device_context(device):
    global _device
    from stanza.cluster import pick_gpu
    with pick_gpu.torch_context(device) as dev:
        old_device = _device
        _device = dev
        yield
        _device = old_device


def maybe_cuda(tensor_or_module):
    if th.cuda.is_available() and _device != 'cpu':
        return tensor_or_module.cuda()
    else:
        return tensor_or_module
