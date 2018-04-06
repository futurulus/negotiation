import argparse
import base64

from stanza.research import config, learner
from stanza.monitoring import progress

from systems import get_system
from cocoa.core.schema import Schema
from cocoa.core.event import Event
from core.scenario import Scenario

import agent


class Memorization(learner.Learner):
    def __init__(self):
        super(Memorization, self).__init__()
        self.get_options()
        self.lookup = {}

    @property
    def num_params(self):
        return 0

    def get_options(self):
        if not hasattr(self, 'options'):
            options = config.options()
            self.options = argparse.Namespace(**options.__dict__)

    def train(self, training_instances, validation_instances=None, metrics=None):
        progress.start_task('Instance', len(training_instances))
        for i, inst in enumerate(training_instances):
            progress.progress(i)
            self.train_inst(inst)
        progress.end_task()

    def train_inst(self, inst):
        game = get_game(inst.input)
        num_turns = count_dialogue_turns(inst.input)
        self.lookup[game] = inst.output
        self.lookup[(game, num_turns)] = inst.output

    def predict_and_score(self, eval_instances, random=False, split='default', verbosity=4):
        predictions = []
        scores = []

        if verbosity > 2:
            progress.start_task('Eval instances', len(eval_instances))
        for i, inst in enumerate(eval_instances):
            if verbosity > 2:
                progress.progress(i)
            game = get_game(inst.input)
            num_turns = count_dialogue_turns(inst.input)
            if (game, num_turns) in self.lookup:
                pred = self.lookup[(game, num_turns)]
            elif game in self.lookup:
                pred = self.lookup[game]
            else:
                pred = 'NEVER BEEN HERE BEFORE'
            predictions.append(pred)
            scores.append(0.0)
        if verbosity > 2:
            progress.end_task()
        return predictions, scores


def get_game(input):
    return ' '.join(input.split(' ')[:6])


def count_dialogue_turns(input):
    tokens = input.split(' ')
    return tokens.count('THEM:') + tokens.count('YOU:')


class RuleBasedAgent(agent.Agent):
    def __init__(self, options, models=None):
        super(RuleBasedAgent, self).__init__(options, models=models)
        args = argparse.Namespace(
            random_seed=hash(self.options.random_seed),
            schema_path='cocoa-negotiation/fb-negotiation/data/bookhatball-schema.json',
            scenarios_path='cocoa-negotiation/fb-negotiation/data/toy-scenarios.json',
            train_examples_paths='cocoa-negotiation/fb-negotiation/data/rulebased-transcripts.json',
            train_max_examples=1,
            test_max_examples=0,
            max_turns=20,
            agents=['rulebased', 'rulebased'],
            templates='cocoa-negotiation/fb-negotiation/templates.pkl',
            policy='cocoa-negotiation/fb-negotiation/model.pkl',
        )
        self.schema = Schema(args.schema_path)
        self.agent = get_system('rulebased', args, self.schema,
                                model_path='cocoa-negotiation/model.pkl')

    def new_game(self, game):
        if not hasattr(self, 'agent_id'):
            self.agent_id = agent.random_agent_name()
        self.scenario = Scenario.from_dict(self.schema, self.kb_dict(game))
        self.session = None
        self.time = 0
        self.outcomes = [None, None]

    def kb_dict(self, game):
        return {
            'uuid': base64.b64encode(str(hash(repr(game))).encode('latin-1')),
            'kbs': [
                {
                    'Role': 'first',
                    'Item_counts': {'book': game[0][0], 'hat': game[0][1], 'ball': game[0][2]},
                    'Item_values': {'book': game[1][0], 'hat': game[1][1], 'ball': game[1][2]}
                },
                {
                    'Role': 'second',
                    'Item_counts': {'book': game[0][0], 'hat': game[0][1], 'ball': game[0][2]},
                    'Item_values': {'book': game[2][0], 'hat': game[2][1], 'ball': game[2][2]}
                },
            ],
        }

    def act(self, goal_directed=False, invert=False, dialogue=None, sel_singleton=None):
        if goal_directed:
            raise NotImplementedError

        if self.session is None:
            self.session = self.agent.new_session(0, self.scenario.kbs[0])
            self.agent_num = 0

        event = self.session.send()
        self.time += 1
        if event.action == 'message':
            return event.data
        elif event.action == 'select':
            selection = [event.data['book'], event.data['hat'], event.data['ball']]
            self.outcomes[0] = selection
            return selection
        elif event.action == 'reject':
            self.outcomes[0] = []
            return []
        else:
            assert False, (event.action, event.data)

    def observe(self, result, dialogue=None, sel_singleton=None):
        if self.session is None:
            self.session = self.agent.new_session(1, self.scenario.kbs[0])
            self.agent_num = 1

        self.time += 1
        event = self.build_event(result)
        self.session.receive(event)
        if isinstance(result, list):
            self.outcomes[1] = result
        return all(o is not None for o in self.outcomes)

    def build_event(self, result):
        if isinstance(result, list):
            if len(result) == 0:
                action = 'reject'
                data = {}
            else:
                action = 'select'
                data = {'book': result[0], 'hat': result[1], 'ball': result[2]}
        else:
            action = 'message'
            data = result

        return Event(
            agent=1 - self.agent_num,
            time=self.time,
            action=action,
            data=data,
        )
