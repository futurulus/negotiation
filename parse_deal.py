import json


def parse_deal(mode, split):
    input_filename = f'data/deal_{split}.txt'
    response_filename = f'data/response{mode}_{split}.jsons'
    selection_filename = f'data/selection{mode}_{split}.jsons'
    partner_filename = f'data/partner{mode}_{split}.jsons'

    with open(input_filename, 'r') as infile, \
            open(response_filename, 'w') as response_file, \
            open(selection_filename, 'w') as selection_file, \
            open(partner_filename, 'w') as partner_file:
        for line in infile:
            if line.strip():
                tokens = line.split()
                dialogue_i = tokens.index('<dialogue>')
                output_i = tokens.index('<output>')
                partner_input_i = tokens.index('<partner_input>')

                input = tokens[1:dialogue_i - 1]
                dialogue = tokens[dialogue_i + 1:output_i - 1]
                output = tokens[output_i + 1:partner_input_i - 1]
                partner_input = tokens[partner_input_i + 1:-1]
                dump_responses(mode, input, dialogue, output, partner_input, response_file)
                dump_selection(mode, input, dialogue, output, partner_input, selection_file)
                dump_partner(mode, input, dialogue, output, partner_input, partner_file)
                selection_file.write('\n')


def dump_responses(mode, input, dialogue, output, partner_input, response_file):
    if mode == '_repro':
        return

    start = -1
    while True:
        you_start = next_or_end(dialogue, start, 'YOU:')

        if mode == '_bothsides':
            them_start = next_or_end(dialogue, start, 'THEM:')
            start = min(you_start, them_start)
        else:
            start = you_start

        if start == len(dialogue):
            return

        you_end = next_or_end(dialogue, start, 'THEM:')

        if mode == '_bothsides':
            them_end = next_or_end(dialogue, start, 'YOU:')
            end = min(you_end, them_end)
        else:
            end = you_end

        if dialogue[end - 1] == '<selection>':
            end += 1

        input_end = max(0, start - 1)
        json.dump({
            'input': ' '.join(input + dialogue[:input_end]),
            'output': ' '.join(dialogue[start + 1:end - 1])
        }, response_file)
        response_file.write('\n')


def next_or_end(seq, start, target):
    try:
        increment = seq[start + 1:].index(target)
    except ValueError:
        return len(seq)
    else:
        return start + 1 + increment


def dump_selection(mode, input, dialogue, output, partner_input, selection_file):
    if mode == '_repro':
        inst_dict = {
            'input': ' '.join(input),
            'output': [
                ' '.join(dialogue),
                ' '.join(output[:3]),
                ' '.join(partner_input),
            ]
        }
    else:
        inst_dict = {
            'input': ' '.join(input + dialogue),
            'output': ' '.join(output[:3]),
        }

    json.dump(inst_dict, selection_file)


def dump_partner(mode, input, dialogue, output, partner_input, partner_file):
    if mode == '_repro':
        return

    start = 0
    while True:
        try:
            start = start + 1 + dialogue[start + 1:].index('YOU:')
        except ValueError:
            return

        try:
            end = start + 1 + dialogue[start + 1:].index('THEM:')
        except ValueError:
            end = len(dialogue)
        if dialogue[end - 1] == '<selection>':
            end += 1

        json.dump({
            'input': ' '.join(input + dialogue[:start - 1]),
            'output': ' '.join(partner_input),
        }, partner_file)
        partner_file.write('\n')


if __name__ == '__main__':
    for mode_ in '', '_repro', '_bothsides':
        for split_ in 'train', 'val', 'test':
            parse_deal(mode_, split_)
