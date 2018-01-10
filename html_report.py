import colorsys
import glob
import io
import json
import numpy as np
import os
import warnings
from cgi import escape as html_escape
from collections import namedtuple, defaultdict
from numbers import Number

from stanza.research import config, instance  # NOQA (for doctest)
from tokenizers import basic_unigram_tokenizer as tokenize


parser = config.get_options_parser()
parser.add_argument('--compare_dir', type=str, default=None,
                    help='The directory containing other results providing a point for '
                         'comparison to the run_dir, if not None. These results will also '
                         'be included in the report.')
parser.add_argument('--per_token_prob', type=config.boolean, default=False,
                    help='If True, normalize probabilities by dividing scores by the number '
                         'of tokens in the reference. This makes high-perplexity examples '
                         'sort to the top, as opposed to simply long examples.')
parser.add_argument('--only_differing_preds', type=config.boolean, default=False,
                    help='If True, only include examples that changed prediction in the '
                         '"biggest improvement/decline" tables.')
parser.add_argument('--show_all', type=config.boolean, default=False,
                    help='If True, replace the Head section with a section that shows '
                         'all examples.')
parser.add_argument('--show_tokens', type=config.boolean, default=True,
                    help='If True, add tables to the output showing token-by-token '
                         'probabilities. Speaker only, does nothing if tokens.*.txt is not '
                         'present in the run directory.')

Output = namedtuple('Output', 'config,results,data,scores,predictions,tokens')

MAX_ALTS = 10


class NotPresent(object):
    def __repr__(self):
        return '&nbsp;'


def html_report(output, compare=None, per_token=False, only_differing=False,
                show_all=False, show_tokens=True):
    '''
    >>> config_dict = {'run_dir': 'runs/test', 'listener': True}
    >>> results_dict = {'dev.perplexity.gmean': 14.0}
    >>> data = [instance.Instance([0.0, 100.0, 100.0], 'red').__dict__]
    >>> scores = [-2.639057329615259]
    >>> predictions = ['bright red']
    >>> print(html_report(Output(config_dict, results_dict, data, scores, predictions, None)))
    <html>
    <head>
    <link rel="stylesheet" href="http://web.stanford.edu/~wmonroe4/css/style.css" type="text/css">
    <meta charset="UTF-8">
    <title>runs/test - Output report</title>
    </head>
    <body>
        <h1>runs/test</h1>
        <p>Compared to: (no comparison set)</p>
        <h2>Configuration options</h2>
        <table>
            <tr><th>Option</th><th>Value</th></tr>
            <tr><td>listener</td><td>True</td></tr>
            <tr><td>run_dir</td><td>'runs/test'</td></tr>
        </table>
        <h2>Results</h2>
        <h3>dev</h3>
        <table>
            <tr><th>Metric</th><th>gmean</th></tr>
            <tr><td>perplexity</td><td align="right">14.000</td></tr>
        </table>
        <h2>Error analysis</h2>
        <h3>Worst</h3>
        <table>
            <tr><th>input</th><th>gold</th><th>prediction</th><th>prob</th></tr>
            <tr><td bgcolor="#ff0000">[0, 100, 100]</td><td bgcolor="#fff">'red'</td><td bgcolor="#fff">'bright red'</td><td>0.071</td></tr>
        </table>
        <h3>Best</h3>
        <table>
            <tr><th>input</th><th>gold</th><th>prediction</th><th>prob</th></tr>
            <tr><td bgcolor="#ff0000">[0, 100, 100]</td><td bgcolor="#fff">'red'</td><td bgcolor="#fff">'bright red'</td><td>0.071</td></tr>
        </table>
        <h3>Head</h3>
        <table>
            <tr><th>input</th><th>gold</th><th>prediction</th><th>prob</th></tr>
            <tr><td bgcolor="#ff0000">[0, 100, 100]</td><td bgcolor="#fff">'red'</td><td bgcolor="#fff">'bright red'</td><td>0.071</td></tr>
        </table>
    </body>
    </html>
    '''  # NOQA

    main_template = u'''<html>
<head>
<link rel="stylesheet" href="http://web.stanford.edu/~wmonroe4/css/style.css" type="text/css">
<meta charset="UTF-8">
<title>{run_dir} - Output report</title>
</head>
<body>
    <h1>{run_dir}</h1>
    <p>Compared to: {compare_dir}</p>
    <h2>Configuration options</h2>
    <table>
        <tr><th>Option</th><th>Value</th>{compare_header}</tr>
{config_opts}
    </table>
    <h2>Results</h2>
{results}
    <h2>Error analysis</h2>
{error_analysis}
</body>
</html>'''

    return main_template.format(
        run_dir=output.config['run_dir'],
        compare_dir=compare.config['run_dir'] if compare else '(no comparison set)',
        compare_header='<th>Comparison</th>' if compare else '',
        config_opts=format_config_dict(output.config, compare.config if compare else None),
        results=format_results(output.results, compare.results if compare else None),
        error_analysis=format_error_analysis(output, compare, per_token=per_token,
                                             only_differing=only_differing,
                                             show_all=show_all,
                                             show_tokens=show_tokens)
    )


def format_config_dict(this_config, compare_config):
    config_opt_template = u'        <tr><td>{key}</td>{values}</tr>'
    config_value_template = u'<td>{!r}</td>'
    all_keys = set(this_config.keys())
    dicts = [this_config]
    if compare_config:
        all_keys.update(compare_config.keys())
        dicts.append(compare_config)
    all_keys = sorted(all_keys)
    return '\n'.join(
        config_opt_template.format(
            key=k,
            values=''.join(
                config_value_template.format(safe_lookup(d, k))
                for d in dicts
            )
        )
        for k in all_keys
    )


def safe_lookup(d, key):
    if not d:
        return NotPresent
    if key not in d:
        return NotPresent
    return d[key]


def format_results(results, compare=None):
    # TODO: compare
    results_table_template = u'''    <h3>{split}</h3>
    <table>
{header}
{rows}
    </table>'''
    header_template = u'        <tr><th>Metric</th>{aggregates}</tr>'
    row_template = u'        <tr><td>{metric}</td>{values}</tr>'

    splits = sorted(set(metric.split('.')[0] for metric in results.keys()))
    tables = []
    for split in splits:
        items = [i for i in results.items() if i[0].startswith(split + '.')]
        metrics = sorted(set(u''.join(m.split('.')[1]) for m, v in items))
        aggregates = sorted(set(u''.join(m.split('.')[2:]) for m, v in items))
        header = header_template.format(aggregates=u''.join('<th>{}</th>'.format(a)
                                                            for a in aggregates))
        values_table = [
            [
                get_formatted_result(results, split, m, a)
                for a in aggregates
            ]
            for m in metrics
        ]
        rows = u'\n'.join(
            row_template.format(metric=m, values=u''.join(u'<td align="right">{}</td>'.format(v)
                                                          for v in row))
            for m, row in zip(metrics, values_table)
        )
        tables.append(results_table_template.format(split=split, header=header, rows=rows))
    return u'\n'.join(tables)


def get_formatted_result(results, split, m, a):
    key = u'.'.join((split, m, a) if a else (split, m))
    if key in results:
        return format_number(results[key])
    else:
        return u''


def format_number(value, exp_sig_figs=6):
    if not isinstance(value, Number):
        return repr(value)
    elif isinstance(value, int):
        return u'{:,d}'.format(value)
    elif value > 1e8 or abs(value) < 1e-3:
        return u'{0:.{1}e}'.format(value, exp_sig_figs)
    else:
        return u'{:,.3f}'.format(value)


def format_error_analysis(output, compare=None, per_token=False,
                          only_differing=False, show_all=False, show_tokens=True):
    examples_table_template = u'''    <h3>{cond}</h3>
    <table>
        <tr><th>input</th>{alt_inputs_header}{alt_outputs_header}<th>gold</th><th>prediction</th><th>{prob_header}</th>{compare_header}</tr>
{examples}
    </table>'''

    example_template = u'        <tr>{input}{alt_inputs}{alt_outputs}{output}' \
                       u'{prediction}{pprob}{comparison}{cprob}</tr>'
    score_template = u'<td>{}</td>'
    show_alt_inputs = max_len(output.data, 'alt_inputs')
    show_alt_outputs = max_len(output.data, 'alt_outputs')

    if compare and 'input' not in compare.data[0]:
        # Results when there's an error loading the comparison file;
        # no need to print a second warning.
        compare = None
    if compare and len(compare.data) != len(output.data):
        warnings.warn("Skipping comparison--mismatch between number of output examples (%s) "
                      "and number of comparison examples (%s)" %
                      (len(output.data), len(compare.data)))
        compare = None

    collated = []
    for i, (inst, score, pred) in enumerate(zip(output.data, output.scores, output.predictions)):
        example = {}
        example['input'] = format_value(inst['input'])
        example['alt_inputs'] = format_alts(inst['alt_inputs'], show_alt_inputs)
        if show_tokens and output.tokens:
            example['output'] = format_tokens(inst['output'], output.tokens[i])
        else:
            example['output'] = format_value(inst['output'])
        example['alt_outputs'] = format_alts(inst['alt_outputs'], show_alt_outputs)
        example['prediction'] = format_value(pred)
        if isinstance(score, Number):
            if per_token:
                num_tokens = len(tokenize(inst['output'])) + 1
            else:
                num_tokens = 1
            pprob = np.exp(score / num_tokens)
        else:
            pprob = score
        example['pprob'] = score_template.format(format_number(pprob))
        example['pprob_val'] = pprob if isinstance(pprob, Number) else 0
        if compare:
            if compare.data[i]['input'] != inst['input']:
                warnings.warn((u"Comparison input doesn't match this input: %s != %s" %
                               (compare.data[i]['input'], inst['input'])).encode('utf_8'))
            example['comparison'] = format_value(compare.predictions[i])
            cscore = compare.scores[i]
            if isinstance(cscore, Number):
                cprob = np.exp(cscore / num_tokens)
            else:
                cprob = cscore
            example['cprob'] = score_template.format(format_number(cprob))
            example['cprob_val'] = cprob if isinstance(cprob, Number) else 0
        else:
            example['comparison'] = ''
            example['cprob'] = ''
            example['cprob_val'] = 0.0
        collated.append(example)

    score_order = sorted(collated, key=lambda e: e['pprob_val'])
    tables = [
        ('Worst', score_order[:100]),
        ('Best', reversed(score_order[-100:])),
        (('All', collated)
         if show_all else
         ('Head', collated[:100])),
    ]
    if compare:
        if only_differing:
            differing = [e for e in collated if e['prediction'] != e['comparison']]
        else:
            differing = collated
        diff_order = sorted(differing, key=lambda e: e['pprob_val'] - e['cprob_val'])
        tables.extend([
            ('Biggest decline', diff_order[:100]),
            ('Biggest improvement', reversed(diff_order[-100:])),
        ])

    prob_header = 'prob (per token)' if per_token else 'prob'
    compare_header = (u'<th>comparison</th><th>{prob_header}</th>'.format(prob_header=prob_header)
                      if compare else '')
    return u'\n'.join(examples_table_template.format(
        cond=cond,
        alt_inputs_header=((u'<th>alt inputs</th>' if show_alt_inputs else u'') +
                           u'<th></th>' * (show_alt_inputs - 1)),
        alt_outputs_header=((u'<th>alt outputs</th>' if show_alt_outputs else u'') +
                            u'<th></th>' * (show_alt_outputs - 1)),
        compare_header=compare_header,
        prob_header=prob_header,
        examples=u'\n'.join(
            example_template.format(**inst) for inst in examples
        )
    ) for cond, examples in tables)


def format_value(value, suppress_colors=False):
    if isinstance(value, (list, tuple)) and len(value) == 3 and isinstance(value[0], Number):
        color = web_color(value)
        if suppress_colors:
            value = NotPresent()
        else:
            value = [int(c) for c in value]
    else:
        color = '#fff'
    value_repr = html_escape(repr(value).encode('latin_1')
                                        .decode('unicode_escape'), quote=True)
    return u'<td bgcolor="{color}">{value}</td>'.format(color=color, value=value_repr)


def format_alts(alts, num_alts):
    if not alts:
        alts = []
    alts = alts[:num_alts]
    alts = alts + [NotPresent()] * (num_alts - len(alts))
    return ''.join(format_value(v) for v in alts)


def format_tokens(output, tokens):
    output_cell_template = (u'<td>{output_repr}<br>'
                            u'<table>{rows}</table></td>')
    output_repr = repr(output).encode('latin_1').decode('unicode_escape')
    row_template = u'<tr>{gold_row}</tr><tr>{pred_row}</tr>'
    rows = []
    for i in range(0, len(output), 16):
        tokens_part = tokens[i:i + 16]
        gold_row = format_tokens_row(tokens_part, 0)
        pred_row = format_tokens_row(tokens_part, 2)
        rows.append(row_template.format(gold_row=gold_row, pred_row=pred_row))
    return output_cell_template.format(output_repr=output_repr, rows=''.join(rows))


def format_tokens_row(tokens, tuple_offset):
    token_cell_template = u'<td bgcolor="{color}">{token}<br><small>{prob}</small></td>'
    cells = []
    for t in tokens:
        log_prob = -t[tuple_offset + 1]
        cells.append(token_cell_template.format(token=html_escape(t[tuple_offset], quote=True),
                                                prob=format_number(np.exp(log_prob), 0),
                                                color=web_color(color_log_prob(log_prob))))
    return ''.join(cells)


def color_log_prob(log_prob):
    if np.isinf(log_prob):
        # gray: -inf
        return (0.0, 0.0, 50.0)
    elif not np.isfinite(log_prob):
        # neon green: nan
        return (120.0, 100.0, 100.0)
    else:
        # shades of red/pink: negative log probabilities
        # shades of blue: positive "log probabilities"
        hue = 0.0 if log_prob <= 0 else 240.0
        sat = max(0.0, min(abs(log_prob) * 10.0 / 2.0, 100.0))
        return (hue, sat, 100.0)


def web_color(hsv):
    '''
    >>> web_color((0.0, 100.0, 100.0))
    '#ff0000'
    >>> web_color((120.0, 50.0, 50.0))
    '#408040'
    '''
    hue, sat, val = hsv
    hsv_0_1 = (hue / 360., sat / 100., val / 100.)
    rgb = colorsys.hsv_to_rgb(*hsv_0_1)
    rgb_int = tuple(min(int(c * 256.0), 255) for c in rgb)
    return '#%02x%02x%02x' % rgb_int


def max_len(insts, field):
    result = 0
    for inst in insts:
        result = max(result, len(inst[field]) if inst[field] else 0)
    return min(MAX_ALTS, result)


def generate_html_reports(run_dir=None, compare_dir=None):
    options = config.options(read=True)
    run_dir = run_dir or options.run_dir
    compare_dir = compare_dir or options.compare_dir

    for output, compare, out_path in get_all_outputs(run_dir, options.compare_dir):
        with io.open(out_path, 'w', encoding='utf_8') as outfile:
            outfile.write(html_report(output, compare, per_token=options.per_token_prob,
                                      only_differing=options.only_differing_preds,
                                      show_all=options.show_all,
                                      show_tokens=options.show_tokens))


def get_all_outputs(run_dir, compare_dir):
    num_outputs = 0
    for filename in glob.glob(os.path.join(run_dir, 'data.*.jsons')):
        split = os.path.basename(filename).split('.')[-2]
        this_output = get_output(run_dir, split)
        if compare_dir:
            compare = get_output(compare_dir, split)
        else:
            compare = None

        out_path = os.path.join(run_dir, 'report.%s.html' % split)
        yield this_output, compare, out_path

        num_outputs += 1

    if num_outputs == 0:
        print('No data files found! Re-run with --output_test_data.')


def get_output(run_dir, split):
    config_dict = load_dict(os.path.join(run_dir, 'config.json'))

    results = {}
    max_val_iter = -1
    for filename in glob.glob(os.path.join(run_dir, 'results.*.json')):
        results.update(load_dict(filename))
        results_split = os.path.basename(filename)[len('results.'):-len('.json')]
        if results_split.startswith('val') and results_split[len('val'):].isdigit():
            val_iter = int(results_split[len('val'):])
            max_val_iter = max(val_iter, max_val_iter)

    max_tokens_idx = -1
    for filename in glob.glob(os.path.join(run_dir, 'tokens.*.txt')):
        tokens_idx = os.path.basename(filename)[len('tokens.'):-len('.txt')]
        if tokens_idx.isdigit():
            max_tokens_idx = max(int(tokens_idx), max_tokens_idx)

    data = load_dataset(os.path.join(run_dir, 'data.%s.jsons' % split))
    scores = load_dataset(os.path.join(run_dir, 'scores.%s.jsons' % split))
    predictions = load_dataset(os.path.join(run_dir, 'predictions.%s.jsons' % split))
    tokens = load_tokens(run_dir, split, max_val_iter, max_tokens_idx)
    return Output(config_dict, results, data, scores, predictions, tokens)


def load_dict(filename):
    try:
        with open(filename) as infile:
            return json.load(infile)
    except IOError as e:
        warnings.warn(str(e))
        return {'error.message.value': str(e)}


def load_dataset(filename, transform_func=(lambda x: x)):
    try:
        dataset = []
        with open(filename) as infile:
            for line in infile:
                js = json.loads(line.strip())
                if isinstance(js, dict):
                    js = defaultdict(lambda: None, js)
                dataset.append(transform_func(js))
        return dataset
    except IOError as e:
        warnings.warn(str(e))
        return [{'error': str(e)}]


def load_tokens(run_dir, split, max_val_iter, max_tokens_idx):
    if split.startswith('val') and split[len('val'):].isdigit():
        tokens_idx = int(split[len('val'):])
    elif split == 'train':
        tokens_idx = max_tokens_idx - 1
    elif split in ('dev', 'test', 'eval'):
        tokens_idx = max_tokens_idx
    else:
        return None
    filename = os.path.join(run_dir, 'tokens.%d.txt' % tokens_idx)
    try:
        with open(filename, 'r') as infile:
            sents = []
            for line in infile:
                line = line.decode('utf-8')
                tokens_str = line[:-1].split(u'\t')
                tuples = []
                for token_str in tokens_str:
                    if token_str[:2] == u'  ':
                        token_str = u'_ ' + token_str[2:]
                    token_str = token_str.replace(u'   ', u' _ ')
                    t = token_str.split(u' ')
                    if len(t) == 4:
                        tuples.append((t[0], float(t[1]),
                                       t[2], float(t[3])))
                    else:
                        warnings.warn(u'invalid token format: "{}"\n'
                                       '(should be "goldword 4.0 predword 5.2", where "4.0" is '
                                       'negative log probability)'
                                      .format(token_str).encode('utf-8'))
                        tuples.append(('ERROR', 0.0, 'ERROR', 0.0))
                sents.append(tuples)
            return sents
    except IOError:
        return None


if __name__ == '__main__':
    generate_html_reports()
