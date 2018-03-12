'''
Preprocess examples in a dataset and generate data for models.
'''

import random
import re
import time
import os
import pdb # set_trace()
import copy
import numpy as np
from itertools import izip, izip_longest
from collections import namedtuple, defaultdict

from cocoa.core.util import read_pickle, write_pickle, read_json
from cocoa.core.entity import Entity, CanonicalEntity, is_entity
from cocoa.lib.bleu import compute_bleu
from cocoa.model.vocab import Vocabulary
from cocoa.core.tokenizer import tokenize

# def add_preprocess_arguments(parser):
#     parser.add_argument('--entity-encoding-form', choices=['type', 'canonical'], default='canonical', help='Input entity form to the encoder')
#     parser.add_argument('--entity-decoding-form', choices=['canonical', 'type'], default='canonical', help='Input entity form to the decoder')
#     parser.add_argument('--entity-target-form', choices=['canonical', 'type'], default='canonical', help='Output entity form to the decoder')
#     parser.add_argument('--candidates-path', nargs='*', default=[], help='Path to json file containing retrieved candidates for dialogues')
#     parser.add_argument('--slot-filling', action='store_true', help='Where to do slot filling')
#     parser.add_argument('--cache', default='.cache', help='Path to cache for preprocessed batches')
#     parser.add_argument('--ignore-cache', action='store_true', help='Ignore existing cache')

SpecialSymbols = namedtuple('SpecialSymbols', ['EOS', 'SOS', 'PERSONA_1', 'PERSONA_2', 'PERSONA_3', 'PERSONA_4', 'PERSONA_5'])
markers = SpecialSymbols(EOS='</s>', SOS='<s>', PERSONA_1='<p1>', PERSONA_2='<p2>', PERSONA_3='<p3>', PERSONA_4='<p4>', PERSONA_5='<p5>')

def build_vocab(dialogues, special_symbols=[], entity_forms=[]):
    vocab = Vocabulary(offset=0, unk=True)

    for dialogue in dialogues:
        for token in tokenize(dialogue):
            vocab.add_word(token)

    # Add special symbols
    vocab.add_words(special_symbols, special=True)
    vocab.finish(size_threshold=10000)
    print 'Utterance vocabulary size:', vocab.size
    return vocab

def generator(self, name, shuffle=True):
    dialogue_batches = self.batches[name]
    yield len(dialogue_batches)
    inds = range(len(dialogue_batches))
    while True:
        if shuffle:
            random.shuffle(inds)
        for ind in inds:
            yield dialogue_batches[ind]

def create_batches(personas, events, agent_id):
    markers = ["<p1>", "<p2>", "<p3>", "<p4>", "<p5>"]
    persona_tokens = []
    for p_idx, persona in enumerate(personas):
        persona_tokens.append(markers[p_idx])
        persona_tokens.extend(tokenize(persona))

    batch = []
    for turn in range(len(events)):
        if turn % 2 == agent_id:
            continue
        source = copy.copy(persona_tokens)
        if turn > 0:
            source.append("<s>")
            source.extend(tokenize(events[turn - 1]))
        target = tokenize(events[turn])
        batch.append((source, target))
    return batch

def generate_data():
    versions = ["revised", "original"]
    splits = ["test", "train", "valid"]

    all_sentences = []
    for split in splits:
        all_batches = []
        out_filename = "data/{}_batches.pkl".format(split)
        for version in versions:

            in_filename = "data/{0}_{1}_examples.json".format(split, version)
            examples = read_json(in_filename)
            examples = examples[100:104]
            for example in examples:
                personas_0 = example["scenario"]["kbs"][0]
                personas_1 = example["scenario"]["kbs"][1]
                events = [event["data"] for event in example["events"]]

                sentences = personas_0 + personas_1 + events
                all_sentences.extend([s for s in sentences if s])  # removes None

                all_batches.extend(create_batches(personas_0, events, 0))
                all_batches.extend(create_batches(personas_1, events, 1))
            print("Finished pre-processing {0}_{1}".format(split, version))

        write_pickle(all_batches, out_filename)
    vocabulary = build_vocab(all_sentences, markers)
    write_pickle(vocabulary, "data/persona_vocab.pkl")

if __name__ == "__main__":
    generate_data()
