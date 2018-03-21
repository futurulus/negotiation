#!/usr/bin/env python
r'''
python repl.py --response runs/response/model.pkl \
               --selection runs/selection/model.pkl \
'''
from stanza.research import config
config.redirect_output()  # NOQA: imports not at top of file

import random
import pickle

from stanza.research import output
from stanza.monitoring import progress

import agent
from vectorizers import all_possible_subcounts
import thutils

parser = config.get_options_parser()
parser.add_argument('--agent_a', default='HumanAgent', choices=agent.AGENTS,
                    help='Class name for agent A in dialogue simulation.')
parser.add_argument('--agent_b', default='TwoModelAgent', choices=agent.AGENTS,
                    help='Class name for agent B in dialogue simulation.')
parser.add_argument('--device_a', default=[], nargs='*',
                    help='The devices to use in PyTorch for model A ("cpu" or "gpu[0-n]"). If None, '
                         'pick a free-ish device automatically. Should be the same number and '
                         'appear in the same order as in --load_a.')
parser.add_argument('--device_b', default=[], nargs='*',
                    help='The devices to use in PyTorch for model B ("cpu" or "gpu[0-n]"). If None, '
                         'pick a free-ish device automatically. Should be the same number and '
                         'appear in the same order as in --load_b.')
parser.add_argument('--load_a', metavar='MODEL_FILE', default=[], nargs='*',
                    help='Model pickle file to load for agent A (two files as different arguments ='
                         ' response.pkl selection.pkl in the case of TwoModelAgent).')
parser.add_argument('--load_b', metavar='MODEL_FILE', default=[], nargs='*',
                    help='Model pickle file to load for agent B (two files as different arguments ='
                         ' response.pkl selection.pkl in the case of TwoModelAgent).')
parser.add_argument('--goal_directed_a', type=config.boolean, default=False,
                    help='If `True`, use goal-directed decoding (maximum reward over rollouts) '
                         'in model A.')
parser.add_argument('--goal_directed_b', type=config.boolean, default=False,
                    help='If `True`, use goal-directed decoding (maximum reward over rollouts) '
                         'in model B.')
parser.add_argument('--contexts', metavar='CONTEXT_FILE', default='data/selfplay.txt',
                    help='Text file giving game contexts, each consisting of two lines in the '
                         'format "na va nb vb nc vc".')
parser.add_argument('--verbosity', type=int, default=0,
                    help='Amount of debugging output to print: >=5 = show all model outputs, '
                         "<5 = don't.")


def repl():
    options = config.options()

    models_a = []
    models_b = []
    assert len(options.load_a) == len(options.device_a), (options.load_a, options.device_a)
    assert len(options.load_b) == len(options.device_b), (options.load_b, options.device_b)
    for a_filename, device in zip(options.load_a, options.device_a):
        if options.verbosity >= 2:
            print(f'Loading {a_filename}...')
        with open(a_filename, 'rb') as infile:
            with thutils.device_context(device):
                models_a.append(pickle.load(infile))
                assert models_a[-1].options.device == device, models_a[-1].options.device
    for b_filename, device in zip(options.load_b, options.device_b):
        if options.verbosity >= 2:
            print(f'Loading {b_filename}...')
        with open(b_filename, 'rb') as infile:
            with thutils.device_context(device):
                models_b.append(pickle.load(infile))
                assert models_b[-1].options.device == device, models_b[-1].options.device
    agent_a = agent.AGENTS[options.agent_a](options, models_a)
    agent_b = agent.AGENTS[options.agent_b](options, models_b)

    agent_a.start()
    agent_b.start()

    games = []
    outcomes_a = []
    dialogues = []
    deals_a = []

    generate_games(options.contexts)

    try:
        if options.verbosity >= 1:
            progress.start_task('Game', len(GAMES))
        for i, game in enumerate(generate_games(options.contexts)):
            if options.verbosity >= 1:
                progress.progress(i)

            games.append(game)
            agent_a.new_game(game)
            agent_b.new_game([game[0], game[2], game[1]])

            dialogue = []
            proposal_a = None

            a_goes = (random.randint(0, 1) == 1)
            while True:
                current_agent = agent_a if a_goes else agent_b
                other_agent = agent_b if a_goes else agent_a
                goal_dir = options.goal_directed_a if a_goes else options.goal_directed_b
                response = current_agent.act(goal_directed=goal_dir)
                current_agent.commit(response)
                other_agent.observe(response)
                if isinstance(response, list):
                    response = response if a_goes else agent.invert_proposal(response, game)
                    if proposal_a is None:
                        proposal_a = response
                    else:
                        break
                else:
                    dialogue.append(('A: ' if a_goes else 'B: ') + response)

                a_goes = not a_goes

            outcome_a, outcome_b = agent.compute_outcome(game, proposal_a, response)
            agent_a.outcome(outcome_a)
            agent_b.outcome(outcome_b)
            outcomes_a.append(outcome_a)
            dialogues.append(dialogue)
            if outcome_a[0] == agent.AGREE:
                deals_a.append(proposal_a)
            else:
                deals_a.append(None)
    except KeyboardInterrupt:
        pass
    finally:
        if options.verbosity >= 1:
            progress.end_task()

    analyze_games(games, outcomes_a, dialogues, deals_a)


GAMES = []


def generate_games(context_filename):
    global GAMES
    if not GAMES:
        with open(context_filename, 'r') as infile:
            lines = [[int(e) for e in line.split()]
                     for line in infile
                     if line.strip()]
        for i in range(0, len(lines), 2):
            pair = lines[i:i+2]
            if len(pair) != 2:
                continue
            a, b = pair
            assert [a[0], a[2], a[4]] == [b[0], b[2], b[4]], (a, b)
            GAMES.append([[a[0], a[2], a[4]],
                          [a[1], a[3], a[5]],
                          [b[1], b[3], b[5]]])
        random.shuffle(GAMES)

    return iter(GAMES)


def analyze_games(games, outcomes, dialogues, deals_a):
    import numpy as np
    results = {}

    results['sim.num_games'] = len(games)
    agreement = [int(outcome[0] == agent.AGREE) for outcome in outcomes]
    results['sim.agreement.mean'] = np.mean(agreement)
    results['sim.agreement.std'] = np.std(agreement)
    results['sim.agreement.sum'] = np.sum(agreement)
    rewards_a = [outcome[1] for outcome in outcomes]
    results['sim.rewards_a.mean'] = np.mean(rewards_a)
    results['sim.rewards_a.std'] = np.std(rewards_a)
    results['sim.rewards_a.sum'] = np.sum(rewards_a)
    rewards_a_agree = [outcome[1] for outcome in outcomes if outcome[0] == agent.AGREE]
    results['sim.rewards_a_agree.mean'] = np.mean(rewards_a_agree)
    results['sim.rewards_a_agree.std'] = np.std(rewards_a_agree)
    results['sim.rewards_a_agree.sum'] = np.sum(rewards_a_agree)
    rewards_b = [outcome[2] for outcome in outcomes]
    results['sim.rewards_b.mean'] = np.mean(rewards_b)
    results['sim.rewards_b.std'] = np.std(rewards_b)
    results['sim.rewards_b.sum'] = np.sum(rewards_b)
    rewards_b_agree = [outcome[2] for outcome in outcomes if outcome[0] == agent.AGREE]
    results['sim.rewards_b_agree.mean'] = np.mean(rewards_b_agree)
    results['sim.rewards_b_agree.std'] = np.std(rewards_b_agree)
    results['sim.rewards_b_agree.sum'] = np.sum(rewards_b_agree)

    pareto_optimal = [int(is_pareto_optimal(game, deal_a))
                      for game, deal_a, outcome_a in zip(games, deals_a, outcomes)
                      if outcome_a[0] == agent.AGREE]
    results['sim.pareto_optimal.mean'] = np.mean(pareto_optimal)
    results['sim.pareto_optimal.std'] = np.std(pareto_optimal)
    results['sim.pareto_optimal.sum'] = np.sum(pareto_optimal)

    output.output_results(results, 'sim')


def is_pareto_optimal(game, deal_a):
    '''
    >>> is_pareto_optimal([[2, 2, 2], [3, 2, 0], [0, 3, 2]], [2, 2, 0])
    True
    >>> is_pareto_optimal([[2, 2, 2], [3, 2, 0], [0, 3, 2]], [0, 2, 0])
    False
    >>> is_pareto_optimal([[4, 1, 1], [1, 0, 6], [0, 6, 4]], [4, 0, 1])
    True
    >>> is_pareto_optimal([[4, 1, 1], [1, 0, 6], [0, 6, 4]], [4, 0, 0])
    True
    >>> is_pareto_optimal([[4, 1, 1], [1, 0, 6], [0, 6, 4]], [1, 0, 1])
    False
    '''
    counts, values_a, values_b = game
    _, reward_a, reward_b = agent.compute_outcome(game, deal_a, deal_a)[0]
    for alt_deal in all_possible_subcounts(game[0]):
        _, alt_reward_a, alt_reward_b = agent.compute_outcome(game, alt_deal, alt_deal)[0]
        if alt_reward_a > reward_a and alt_reward_b >= reward_b:
            return False
        if alt_reward_a >= reward_a and alt_reward_b > reward_b:
            return False
    return True


if __name__ == '__main__':
    try:
        repl()
    except KeyboardInterrupt:
        print('')
        pass
