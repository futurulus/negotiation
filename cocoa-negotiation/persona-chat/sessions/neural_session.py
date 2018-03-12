from session import Session

from torch import cuda

from cocoa.core.util import read_json, write_json, read_pickle, write_pickle
from cocoa.core.schema import Schema
from cocoa.lib import logstats

from neural_model.encdec import add_model_arguments, GRU_Encoder, Attn_Decoder
from neural_model.learner import add_learner_arguments, Learner

class NeuralSession(Session):
    """A wrapper for LstmRolloutAgent.
    """
    def __init__(self, agent, kb, encoder, decoder, args):
        super(NeuralSession, self).__init__(agent)
        self.kb = kb
        self.model = # AttentionAgent(encoder, decoder, args)
        self.encoder = encoder
        self.decoder = decoder
        context = self.kb_to_context(self.kb)
        self.model.feed_context(context)
        self.state = {'done': False}

    def receive(self, event):
        if event.action == 'done':
            self.state['done'] = True
        elif event.action == 'message':
            # add self.personas for conditioning
            tokens = event.data.lower().strip()
            self.model.read(tokens)

    def send(self):
        if self.state['done']:
            return self.done()

        tokens = self.model.write()
        # Omit the last <eos> symbol
        return self.message(' '.join(tokens[:-1]))