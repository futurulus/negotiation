import baselines
import seq2seq
import rl

LEARNERS = {c.__name__: c for c in [
    baselines.Memorization,
    seq2seq.SimpleSeq2SeqLearner,
    rl.SupervisedLearner,
    rl.RLLearner,
    rl.StaticSelfPlayLearner,
]}


def new(classname):
    return LEARNERS[classname]()
