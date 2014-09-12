import logging

from nose.tools import assert_equal, assert_true
import numpy as np
from numpy.testing import assert_allclose

import evaluate
import glove


# Mock corpus (shamelessly stolen from Gensim word2vec tests)
test_corpus = ("""human interface computer
survey user computer system response time
eps user interface system
system human system eps
user response time
trees
graph trees
graph minors trees
graph minors survey
I like graph and stuff
I like trees and stuff
Sometimes I build a graph
Sometimes I build trees""").split("\n")

glove.logger.setLevel(logging.ERROR)
vocab = glove.build_vocab(test_corpus)
cooccur = glove.build_cooccur(vocab, test_corpus, window_size=10)
id2word = evaluate.make_id2word(vocab)

W = glove.train_glove(vocab, cooccur, vector_size=10, iterations=500)

# Merge and normalize word vectors
W = evaluate.merge_main_context(W)


def test_similarity():
    similar = evaluate.most_similar(W, vocab, id2word, 'graph')
    logging.debug(similar)

    assert_equal('trees', similar[0])
