from stanza.research.metrics import *
from stanza.research.instance import Instance


def agreement(eval_data, predictions, scores, learner='ignored'):
    return [float(_is_agreement(inst.input, sel_a, sel_b))
            for inst, (dialogue, sel_a, sel_b,
                       reward, partner_reward) in zip(eval_data, predictions)]


def _is_agreement(input, sel_a, sel_b):
    if sel_a.startswith('<') or sel_b.startswith('<'):
        return False

    total_counts = [input[0], input[2], input[4]]
    a_counts = _extract_counts(sel_a)
    b_counts = _extract_counts(sel_b)
    return all(a + b == c for a, b, c in zip(a_counts, b_counts, total_counts))


def _extract_counts(sel):
    counts = [0, 0, 0]
    for token in sel.split(' '):
        try:
            if not token.startswith('item'):
                raise ValueError('no "item" found')

            idx_str, count_str = token[len('item'):].split('=')
            idx, count = int(idx_str), int(count_str)
            counts[idx] = count
        except (ValueError, IndexError):
            raise ValueError('invalid token {} in {}'.format(repr(token), repr(sel)))

    return counts


def average_score(eval_data, predictions, scores, learner='ignored'):
    return [float(reward) for dialogue, sel_a, sel_b, reward, partner_reward in predictions]


def average_partner_score(eval_data, predictions, scores, learner='ignored'):
    return [float(partner_reward) for dialogue, sel_a, sel_b, reward, partner_reward in predictions]


def response_log_likelihood(eval_data, predictions, scores, learner='ignored'):
    return log_likelihood(_responses(eval_data), 'ignored', [s[0] for s in scores])


def response_log_likelihood_bits(eval_data, predictions, scores, learner='ignored'):
    return log_likelihood_bits(_responses(eval_data), 'ignored', [s[0] for s in scores])


def response_perplexity(eval_data, predictions, scores, learner='ignored'):
    return perplexity(_responses(eval_data), 'ignored', [s[0] for s in scores])


def response_token_perplexity_macro(eval_data, predictions, scores, learner='ignored'):
    return token_perplexity_macro(_responses(eval_data), 'ignored', [s[0] for s in scores])


def response_token_perplexity_micro(eval_data, predictions, scores, learner='ignored'):
    return token_perplexity_micro(_responses(eval_data), 'ignored', [s[0] for s in scores])


def _responses(insts):
    return [
        Instance(inst.input, inst.output[0], source=inst.source)
        for inst in insts
    ]


def selection_log_likelihood(eval_data, predictions, scores, learner='ignored'):
    return log_likelihood(_selections(eval_data), 'ignored', [s[1] for s in scores])


def selection_log_likelihood_bits(eval_data, predictions, scores, learner='ignored'):
    return log_likelihood_bits(_selections(eval_data), 'ignored', [s[1] for s in scores])


def selection_accuracy(eval_data, predictions, scores='ignored', learner='ignored'):
    return accuracy(_selections(eval_data), [p[1] for p in predictions])


def selection_perplexity(eval_data, predictions, scores, learner='ignored'):
    return perplexity(_selections(eval_data), 'ignored', [s[1] for s in scores])


def _selections(insts):
    return [
        Instance(inst.input, inst.output[1], source=inst.source)
        for inst in insts
    ]


METRICS = {
    name: globals()[name]
    for name in dir()
    if (name not in ['np']
        and not name.startswith('_'))
}
