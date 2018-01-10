import json


def parse_deal(split):
    input_filename = f'data/deal_{split}.txt'
    response_filename = f'data/response_{split}.jsons'
    selection_filename = f'data/selection_{split}.jsons'

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
                dump_responses(input, dialogue, output, response_file)
                json.dump({
                    'input': ' '.join(input + dialogue),
                    'output': ' '.join(output[:3]),
                }, selection_file)
                selection_file.write('\n')


def dump_responses(input, dialogue, output, response_file):
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


if __name__ == '__main__':
    for split in 'train', 'val', 'test':
        parse_deal(split)
