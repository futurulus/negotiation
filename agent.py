from stanza.research.rng import get_rng
rng = get_rng()


class Agent():
    def __init__(self, options, models=None):
        if models is None:
            models = []
        self.models = models
        self.options = options

    def start(self):
        pass

    def new_game(self, game):
        pass

    def act(self, goal_directed=False, both_sides=False):
        raise NotImplementedError

    def commit(self, action):
        pass

    def observe(self, result):
        raise NotImplementedError

    def outcome(self, outcome):
        pass


def random_agent_name():
    CONSONANTS = 'bcdfghjklmnpqrstvwxyz'
    VOWELS = 'aeoiu'
    return (CONSONANTS[rng.randint(len(CONSONANTS))] +
            VOWELS[rng.randint(len(VOWELS))] +
            CONSONANTS[rng.randint(len(CONSONANTS))]).title()
