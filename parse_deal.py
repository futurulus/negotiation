import json


def parse_deal(mode, split):
    input_filename = f'data/deal_{split}.txt'
    response_filename = f'data/response{mode}_{split}.jsons'
    selection_filename = f'data/selection{mode}_{split}.jsons'

    with open(input_filename, 'r') as infile, \
            open(response_filename, 'w') as response_file, \
            open(selection_filename, 'w') as selection_file:
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
                selection_file.write('\n')


def dump_responses(mode, input, dialogue, output, partner_input, response_file):
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
            'output': ' '.join(dialogue[start + 1:end - 1])
        }, response_file)
        response_file.write('\n')


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


if __name__ == '__main__':
    for mode_ in '', '_repro':
        for split_ in 'train', 'val', 'test':
            parse_deal(mode_, split_)
