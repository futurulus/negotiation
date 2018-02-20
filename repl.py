#!/usr/bin/env python
r'''
python repl.py --response runs/response/model.pkl \
               --selection runs/selection/model.pkl \
'''
import random
import pickle

from stanza.research import config

import tokenizers

parser = config.get_options_parser()
parser.add_argument('--response', metavar='RESP_MODEL_FILE', default='',
                    help='Model pickle file for the response (dialogue) model.')
parser.add_argument('--selection', metavar='SEL_MODEL_FILE', default='',
                    help='Model pickle file for the selection (final deal) model.')
parser.add_argument('--contexts', metavar='CONTEXT_FILE', default='data/selfplay.txt',
                    help='Text file giving game contexts, each consisting of two lines in the '
                         'format "na va nb vb nc vc".')


def repl():
    options = config.options()
    print('===Negotiation REPL===')
    print('')
    print('Type dialogue responses normally. Selection commands start with a slash:')
    print('  /s 1 2 0 : select a final deal (ask for 1 book, 2 hats, 0 balls)')
    print("  /y , /a : agree with partner's choice")
    print("  /n , /d : indicate no agreement or disagree with partner's choice")
    print('')

    with open(options.response, 'rb') as infile:
        resp_model = pickle.load(infile)
    with open(options.selection, 'rb') as infile:
        sel_model = pickle.load(infile)

    while True:
        game = generate_game(options.contexts)
        print_game(game)
        dialogue = []
        selection = None

        human_goes = (random.randint(0, 1) == 1)
        while True:
            turn = human_turn if human_goes else bot_turn
            prefix = ('YOU: ' if human_goes else 'THEM: ')
            result = turn(resp_model, sel_model, game, dialogue, selection)
            if isinstance(result, list):
                if human_goes:
                    agent = 'You'
                else:
                    agent = 'Your partner'
                print_selection(agent, result)
                dialogue.append(f'{prefix}<selection>')
                if selection is None:
                    if human_goes:
                        selection = result
                    else:
                        selection = [c - s for c, s in zip(game[0], result)]
                        print('')
                else:
                    if not human_goes:
                        result = [c - s for c, s in zip(game[0], result)]
                    break
            else:
                assert isinstance(result, str)
                if not human_goes:
                    print(f'THEM: {result}')
                dialogue.append(prefix + result)
            human_goes = not human_goes

        print_outcome(game, dialogue, selection, result)


GAMES = []


def generate_game(context_filename):
    global GAMES
    if not GAMES:
        with open('data/selfplay.txt', 'r') as infile:
            lines = [[int(e) for e in line.split()]
                     for line in infile
                     if line.strip()]
        for i in range(0, len(lines), 2):
            a, b = lines[i:i+2]
            assert [a[0], a[2], a[4]] == [b[0], b[2], b[4]], (a, b)
            GAMES.append([[a[0], a[2], a[4]],
                          [a[1], a[3], a[5]],
                          [b[1], b[3], b[5]]])

    return random.choice(GAMES)


def human_turn(resp_model, sel_model, game, dialogue, selection):
    while True:
        line = input('YOU: ').lower()
        if selection is not None:
            if not line[:2] in ('/a', '/d', '/y', '/n', '/s'):
                print('  [partner has made proposal, choose agree (/a, /y) or '
                      'disagree (/d, /n, /s)]')
                continue
            elif line[:2] in ('/a', '/y'):
                return selection
            elif line[:2] in ('/d', '/n'):
                return []
            elif line.startswith('/s'):
                try:
                    return parse_human_selection(line, game[0])
                except ValueError:
                    continue
            else:
                continue
        elif line.startswith('/'):
            if line[:2] == '/s':
                try:
                    return parse_human_selection(line, game[0])
                except ValueError:
                    continue
            elif line[:2] in ('/d', '/n'):
                '''
                if len(dialogue) < 9:
                    print('  [wait {} more turn{} before agreeing to '
                          'disagree]'.format(9 - len(dialogue), ('s' if len(dialogue) < 8 else '')))
                else:
                '''
                return []
            elif line[:2] in ('/a', '/y'):
                print('  [no proposal to agree to]')
            else:
                print('  [unknown command: {}]'.format())
        else:
            return ' '.join(tokenizers.basic_unigram_tokenizer(line.strip()))


'''
def bot_turn(resp_model, sel_model, game, dialogue, selection):
    if selection:
        return (random.randint(0, 1) == 1)
    elif len(dialogue) < 6:
        return 'give me all the books'
    else:
        return [random.randint(0, c) for c in game[0]]
'''


def bot_turn(resp_model, sel_model, game, dialogue, selection):
    inst = get_input_instance(game, dialogue)
    if selection:
        output = sel_model.predict([inst], random=True, verbosity=0)[0]
        # print(f'      --OUTPUT: {repr(output)}')
        return parse_bot_selection(output, game[0])
    else:
        response = resp_model.predict([inst], random=True, verbosity=0)[0]
        # print(f'      --RESPONSE: {repr(response)}')
        if response == '<selection>':
            output = sel_model.predict([inst], random=True, verbosity=0)[0]
            # print(f'      --OUTPUT: {repr(output)}')
            return parse_bot_selection(output, game[0])
        else:
            return response


def get_input_instance(game, dialogue):
    pieces = [f'{game[0][0]} {game[2][0]} {game[0][1]} {game[2][1]} {game[0][2]} {game[2][2]}']
    for entry in dialogue:
        pieces.append(f'{entry} <eos>')
    input = ' '.join(pieces)[:-len('<eos>')]
    from stanza.research.instance import Instance
    return Instance(input, '')


ITEMS = ('📕 ', '🎩 ', '⚽ ')
NAMES = ('book', 'hat', 'ball')


def print_game(game):
    counts, your_values, _ = game
    print('NEW GAME')
    print('')
    for i in range(3):
        print(f'    {ITEMS[i] * counts[i]:8s} {NAMES[i]:4s} x{counts[i]}'
              f' worth {your_values[i]:d} each')
    print('')


def print_selection(agent, result):
    print('')
    if result:
        print(f'  {agent} requested:')
        for i in range(3):
            print(f'    {ITEMS[i] * result[i]:8s} {NAMES[i]:4s} x{result[i]}')
    else:
        print(f'  {agent} indicated no agreement.')


def print_outcome(game, dialogue, selection, result):
    print('')
    if result != selection:
        print('  RESULT: Disagreement (0 points each).')
    elif selection == []:
        print('  RESULT: No agreement (0 points each).')
    else:
        your_value = sum([s * v for s, v in zip(selection, game[1])])
        their_value = sum([(c - s) * v for c, s, v in zip(game[0], selection, game[2])])
        print(f'  RESULT: Agreement, you got {your_value} points. (Partner got {their_value}.)')
        # print(f'  --GAME: {game}')
    print('')


def parse_human_selection(line, counts):
    try:
        elems = line.split()
        selection = [int(e) for e in elems[1:4]]
        for s, c in zip(selection, counts):
            if s < 0:
                print("  [number of items can't be negative]")
                raise ValueError
            elif s > c:
                print(f"  [selection ({s}) greater than number of items ({c})]")
                raise ValueError
        return selection
    except (IndexError, ValueError):
        print('  [/s must be followed by three integers (books, hats, balls)]')
        raise ValueError


def parse_bot_selection(line, counts):
    if line.startswith('<'):
        return []

    import re
    match = re.search(r'item0=(\d+) item1=(\d+) item2=(\d+)', line)
    if not match:
        return []
    else:
        return [max(0, min(c, int(s))) for c, s in zip(counts, match.groups())]


if __name__ == '__main__':
    try:
        repl()
    except KeyboardInterrupt:
        print('')
        pass
