import contextlib
import numbers
import torch as th


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
    return th.nn.LogSoftmax()(x.transpose(0, dim)).transpose(0, dim)


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
