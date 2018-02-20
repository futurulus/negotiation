from stanza.research.metrics import *
from stanza.research.instance import Instance


def agreement(eval_data, predictions, scores='ignored', learner='ignored'):
    return [float(sel_a == sel_b and not sel_a[1].startswith('<'))
            for dialogue, sel_a, sel_b, reward in predictions]


def average_score(eval_data, predictions, scores='ignored', learner='ignored'):
    return [float(reward) for dialogue, sel_a, sel_b, reward in predictions]


def response_log_likelihood(eval_data, predictions, scores, learner='ignored'):
    return log_likelihood(_responses(eval_data), 'ignored', [s[0] for s in scores])


def response_log_likelihood_bits(eval_data, predictions, scores, learner='ignored'):
    return log_likelihood_bits(_responses(eval_data), 'ignored', [s[0] for s in scores])


def response_accuracy(eval_data, predictions, scores='ignored', learner='ignored'):
    return accuracy(_responses(eval_data), [p[0] for p in predictions])


def response_bleu(eval_data, predictions, scores='ignored', learner='ignored'):
    return bleu(_responses(eval_data), [p[0] for p in predictions])


def response_wer(eval_data, predictions, scores='ignored', learner='ignored'):
    return wer(_responses(eval_data), [p[0] for p in predictions])


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
