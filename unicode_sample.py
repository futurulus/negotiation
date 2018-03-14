'''
python unicode_sample.py --load runs/unicode_simple/model.pkl --device gpu1
'''
import pickle
import sys
import json

from stanza.research import config
from stanza.research.instance import Instance

import thutils
from seq2seq import SimpleSeq2SeqLearner


def sample_unicode(model_pkl_file, device):
    dev_insts = []
    with open('data/unicode_dev.json', 'r') as infile:
        for line in infile:
            line = line.strip()
            if line:
                dev_insts.append(Instance(**json.loads(line)))
    dev_insts = dev_insts[:256]

    with thutils.device_context(device):
        with open(model_pkl_file, 'rb') as infile:
            model = pickle.load(infile)

        samples = model.predict(dev_insts, random=True)

    for inst, sample in zip(dev_insts, samples):
        char = chr(int(inst.input, 16))
        print(f'{char} U+{inst.input} {sample}')


if __name__ == '__main__':
    options = config.options()
    sample_unicode(options.load, options.device)
