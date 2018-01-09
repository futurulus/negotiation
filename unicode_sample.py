import pickle
import sys
import json

from stanza.research.instance import Instance

from simple import SimpleSeq2SeqLearner


def sample_unicode(model_pkl_file):
    with open(model_pkl_file, 'rb') as infile:
        model = pickle.load(infile)

    dev_insts = []
    with open('data/unicode_dev.json', 'r') as infile:
        for line in infile:
            line = line.strip()
            if line:
                dev_insts.append(Instance(**json.loads(line)))
    dev_insts = dev_insts[:256]

    for inst, sample in zip(dev_insts, model.predict(dev_insts, random=True)):
        char = chr(int(inst.input, 16))
        print(f'{char} U+{inst.input} {sample}')


if __name__ == '__main__':
    model_pkl_file = sys.argv[1]
    sample_unicode(model_pkl_file)
