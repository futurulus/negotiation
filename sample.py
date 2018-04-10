r'''
python sample.py --load runs/deal_response/model.pkl \
                 --eval_file runs/response_train.jsons \
                 --device gpu1
'''
import pickle
import sys
import json

from stanza.research import config
from stanza.research.instance import Instance

import thutils
import run_experiment
from seq2seq import SimpleSeq2SeqLearner


def sample(model_pkl_file, device, insts_file):
    dev_insts = []
    with open(insts_file, 'r') as infile:
        for line in infile:
            line = line.strip()
            if line:
                dev_insts.append(Instance(**json.loads(line)))

    with thutils.device_context(device):
        with open(model_pkl_file, 'rb') as infile:
            model = pickle.load(infile)

        import pdb; pdb.set_trace()
        samples = model.predict(dev_insts, random=True, verbosity=0)

    for inst, sample in zip(dev_insts, samples):
        print(json.dumps(sample))


def test_selection(model, inst, verbose=False, target='<disagree> <disagree> <disagree>'):
    samps = []
    for _ in range(1000):
        s = model.predict([inst], random=True, verbosity=0)[0]
        samps.append(s)
        if verbose:
            print(s)
    return samps.count(target)


def test_response(model, inst, verbose=False, target='<selection>'):
    return test_selection(model, inst, verbose=verbose, target=target)


if __name__ == '__main__':
    options = config.options()
    sample(options.load, options.device, options.eval_file)
