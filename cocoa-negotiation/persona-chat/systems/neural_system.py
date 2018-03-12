from collections import namedtuple

from cocoa.systems.system import System
from cocoa.sessions.timed_session import TimedSessionWrapper
from sessions.neural_session import NeuralSession

import torch
from fb_model.agent import LstmRolloutAgent

def add_neural_system_arguments(parser):
    parser.add_argument('--model-file', type=str,
        help='model file')
    parser.add_argument('--temperature', type=float, default=1.0,
        help='temperature')
    parser.add_argument('--gpu', action='store_true',
        help='Use GPU or not')

# `args` for LstmRolloutAgent
Args = namedtuple('Args', ['temperature', 'domain'])

class NeuralSystem(System):
    def __init__(self, model_file, temperature, timed_session=False, gpu=False):
        super(NeuralSystem, self).__init__()
        self.timed_session = timed_session
        self.encoder = torch.load(model_file+"_encoder.pt")
        self.decoder = torch.load(model_file+"_decoder.pt")
        # self.args = Args(temperature=temperature, domain='object_division')

    @classmethod
    def name(cls):
        return 'neural'

    def new_session(self, agent, kb):
        session = NeuralSession(agent, kb, self.encoder, self.decoder, self.args)
        if self.timed_session:
            session = TimedSessionWrapper(session)
        return session

