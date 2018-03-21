import argparse

from stanza.research import config, learner
from stanza.monitoring import progress


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
