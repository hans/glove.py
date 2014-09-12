#!/usr/bin/env python

from argparse import ArgumentParser
import codecs
from collections import Counter
import itertools
from functools import partial
import logging
from math import log
import os.path
import pickle
from random import shuffle

import numpy as np
from scipy import sparse


logging.basicConfig()
logger = logging.getLogger('glove')


def parse_args():
    parser = ArgumentParser(
        description=('Build a GloVe vector-space model from the '
                     'provided corpus'))

    parser.add_argument('corpus', metavar='corpus_path',
                        type=partial(codecs.open, encoding='utf-8'))

    g_vocab = parser.add_argument_group('Vocabulary options')
    g_vocab.add_argument('--vocab-path',
                         help=('Path to vocabulary file. If this path '
                               'exists, the vocabulary will be loaded '
                               'from the file. If it does not exist, '
                               'the vocabulary will be written to this '
                               'file.'))

    g_cooccur = parser.add_argument_group('Cooccurrence tracking options')
    g_cooccur.add_argument('--cooccur-path',
                           help=('Path to cooccurrence matrix file. If '
                                 'this path exists, the matrix will be '
                                 'loaded from the file. If it does not '
                                 'exist, the matrix will be written to '
                                 'this file.'))
    g_cooccur.add_argument('-w', '--window-size', type=int, default=10,
                           help=('Number of context words to track to '
                                 'left and right of each word'))

    g_glove = parser.add_argument_group('GloVe options')
    g_glove.add_argument('--vector-path',
                         help=('Path to which to save computed word '
                               'vectors'))
    g_glove.add_argument('-s', '--vector-size', type=int, default=100,
                         help=('Dimensionality of output word vectors'))
    g_glove.add_argument('--iterations', type=int, default=25,
                         help='Number of training iterations')
    g_glove.add_argument('--learning-rate', type=float, default=0.05,
                         help='Initial learning rate')
    g_glove.add_argument('--min-count', type=int, default=10,
                         help=('Discard cooccurrence pairs where at '
                               'least one of the words occurs fewer '
                               'than this many times in the training '
                               'corpus'))

    return parser.parse_args()


def get_or_build(path, build_fn, *args, **kwargs):
    """
    Load from serialized form or build an object, saving the built
    object.

    Remaining arguments are provided to `build_fn`.
    """

    save = False
    obj = None

    if path is not None and os.path.isfile(path):
        with open(path, 'rb') as obj_f:
            obj = pickle.load(obj_f)
    else:
        save = True

    if obj is None:
        obj = build_fn(*args, **kwargs)

        if save and path is not None:
            with open(path, 'wb') as obj_f:
                pickle.dump(obj, obj_f)

    return obj


def build_vocab(corpus):
    """
    Build a vocabulary with word frequencies for an entire corpus.
    """

    logger.info("Building vocab from corpus")

    vocab = Counter()
    for line in corpus:
        tokens = line.strip().split()
        vocab.update(tokens)

    logger.info("Done building vocab from corpus.")

    return vocab


def build_cooccur(vocab, corpus, window_size=10):
    """
    Build a word co-occurrence matrix for the given corpus.

    Returns a pair `(word_ids, cooccurrences)`, where `word_ids` is a
    dictionary mapping from word string to unique integer ID, and
    `cooccurrences` is the computed co-occurrence matrix.
    """

    vocab_size = len(vocab)
    word_ids = {word: id for id, word in enumerate(vocab.iterkeys())}
    cooccurrences = sparse.lil_matrix((vocab_size, vocab_size),
                                      dtype=np.float64)

    for i, line in enumerate(corpus):
        if i % 1000 == 0:
            logger.info("Building cooccurrence matrix: on line %i", i)

        tokens = line.strip().split()
        token_ids = [word_ids[word] for word in tokens]

        for center_i, center_id in enumerate(token_ids):
            # Collect all word IDs in left window of center word
            context_ids = token_ids[max(0, center_i - window_size) : center_i]
            contexts_len = len(context_ids)

            for left_i, left_id in enumerate(context_ids):
                # Distance from center word
                distance = contexts_len - left_i

                # Weight by inverse of distance between words
                increment = 1.0 / float(distance)

                # Build co-occurrence matrix symmetrically (pretend we
                # are calculating right contexts as well)
                cooccurrences[center_id, left_id] += increment
                cooccurrences[left_id, center_id] += increment

    return word_ids, cooccurrences


def iter_cooccurrences(lil_matrix, vocab, id2word, min_count=None):
    """
    Yield `(w1, w2, x)` pairs from a LiL sparse matrix as produced by
    `build_cooccur`, where `w1` is a row index (word ID), `w2` is a
    column index (context word ID), and `x` is a cell value
    ($X_{w1, w2}$).

    Only yield cooccurrence pairs where each associated word occurs at
    least `min_count` times in the training corpus. If `min_count` is
    `None`, all words are yielded.
    """

    # This function is built for LiL-format sparse matrices only
    assert isinstance(lil_matrix, sparse.lil_matrix)

    for i, (row, data) in enumerate(itertools.izip(lil_matrix.rows,
                                                   lil_matrix.data)):
        if vocab[id2word[i]] < min_count:
            continue

        for data_idx, j in enumerate(row):
            if vocab[id2word[j]] < min_count:
                continue

            yield i, j, data[data_idx]


def run_iter(word_ids, data,
             learning_rate=0.05, x_max=100, alpha=0.75):
    """
    Run a single iteration of GloVe training using the given
    cooccurrence data and the previously computed weight vectors /
    biases and accompanying gradient histories.

    `word_ids` should be provided as returned by `build_cooccur`.

    `data` is a pre-fetched data / weights list where each element is of
    the form

        (v_main, v_context,
         b_main, b_context,
         gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context,
         cooccurrence)

    as produced by the `train_glove` function. Each element in this
    tuple is an `ndarray` view into the data structure which contains
    it.

    See the `train_glove` function for information on the shapes of `W`,
    `biases`, `gradient_squared`, `gradient_squared_biases` and how they
    should be initialized.

    The parameters `x_max`, `alpha` define our weighting function when
    computing the cost for two word pairs; see the GloVe paper for more
    details.

    Returns the cost associated with the given weight assignments and
    updates the weights by online AdaGrad in place.
    """

    global_cost = 0

    # We want to iterate over data randomly so as not to unintentionally
    # bias the word vector contents
    shuffle(data)

    for (v_main, v_context, b_main, b_context, gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context, cooccurrence) in data:

        weight = (cooccurrence / x_max) ** alpha if cooccurrence < x_max else 1

        # Compute cost
        #
        #   $$ J' = f(X_{ij}) (w_i^Tw_j + b_i + b_j - log(X_{ij}))^2 $$
        cost = weight * ((v_main.dot(v_context) + b_main[0] + b_context[0]
                          - log(cooccurrence)) ** 2)

        # Add weighted cost to the global cost tracker
        global_cost += 0.5 * cost

        # Compute gradients for word vector elements.
        #
        # NB: `main_word` is only a view into `W` (not a copy), so our
        # modifications here will affect the global weight matrix;
        # likewise for context_word
        grad_main = cost * v_context
        grad_context = cost * v_main

        # Now perform adaptive updates
        v_main -= (learning_rate * grad_main / np.sqrt(gradsq_W_main))
        v_context -= (learning_rate * grad_context / np.sqrt(gradsq_W_context))

        # Update squared gradient sums
        gradsq_W_main += np.square(grad_main)
        gradsq_W_context += np.square(grad_context)

        # Compute gradients for bias terms
        grad_bias_main = cost
        grad_bias_context = cost

        b_main -= (learning_rate * grad_bias_main / np.sqrt(gradsq_b_main))
        b_context -= (learning_rate * grad_bias_context / np.sqrt(
                gradsq_b_context))

        # Update squared gradient sums
        gradsq_b_main += grad_bias_main ** 2
        gradsq_b_context += grad_bias_context ** 2

    return global_cost


def train_glove(word_ids, cooccurrences, vector_size=100, iterations=25,
                **kwargs):
    """
    Train GloVe vectors on the given generator `cooccurrences`, where
    each element is of the form

        (word_i_id, word_j_id, x_ij)

    where `x_ij` is a cooccurrence value $X_{ij}$ as presented in the
    matrix defined by `build_cooccur` and the Pennington et al. (2014)
    paper itself.

    Keyword arguments are passed on to the iteration step function
    `run_iter`.

    Returns the computed word vector matrix `W`.
    """

    vocab_size = len(word_ids)

    # Word vector matrix. This matrix is (2V) * d, where N is the size
    # of the corpus vocabulary and d is the dimensionality of the word
    # vectors. All elements are initialized randomly in the range (-0.5,
    # 0.5]. We build two word vectors for each word: one for the word as
    # the main (center) word and one for the word as a context word.
    #
    # It is up to the client to decide what to do with the resulting two
    # vectors. Pennington et al. (2014) suggest adding or averaging the
    # two for each word, or discarding the context vectors.
    W = (np.random.rand(vocab_size * 2, vector_size) - 0.5) / float(vector_size + 1)

    # Bias terms, each associated with a single vector. An array of size
    # $2V$, initialized randomly in the range (-0.5, 0.5].
    biases = (np.random.rand(vocab_size * 2) - 0.5) / float(vector_size + 1)

    # Training is done via adaptive gradient descent (AdaGrad). To make
    # this work we need to store the sum of squares of all previous
    # gradients.
    #
    # Like `W`, this matrix is (2V) * d.
    #
    # Initialize all squared gradient sums to 1 so that our initial
    # adaptive learning rate is simply the global learning rate.
    gradient_squared = np.ones((vocab_size * 2, vector_size),
                               dtype=np.float64)

    # Sum of squared gradients for the bias terms.
    gradient_squared_biases = np.ones(vocab_size * 2, dtype=np.float64)

    # Build a reusable list from the given cooccurrence generator,
    # pre-fetching all necessary data.
    #
    # NB: These are all views into the actual data matrices, so updates
    # to them will pass on to the real data structures
    #
    # (We even extract the single-element biases as slices so that we
    # can use them as views)
    data = [(W[i_main], W[i_context + vocab_size],
             biases[i_main : i_main + 1],
             biases[i_context + vocab_size : i_context + vocab_size + 1],
             gradient_squared[i_main], gradient_squared[i_context + vocab_size],
             gradient_squared_biases[i_main : i_main + 1],
             gradient_squared_biases[i_context + vocab_size
                                     : i_context + vocab_size + 1],
             cooccurrence)
            for i_main, i_context, cooccurrence in cooccurrences]

    for i in range(iterations):
        logger.info("\tBeginning iteration %i..", i)

        cost = run_iter(word_ids, data, **kwargs)

        logger.info("\t\tDone (cost %f)", cost)

    return W


def main(arguments):
    corpus = arguments.corpus

    logger.info("Fetching vocab..")
    vocab = get_or_build(arguments.vocab_path, build_vocab, corpus)
    logger.info("Vocab has %i elements.\n", len(vocab))

    logger.info("Fetching cooccurrence matrix..")
    corpus.seek(0)
    word_ids, cooccurrences = get_or_build(arguments.cooccur_path,
                                           build_cooccur, vocab, corpus,
                                           arguments.window_size)
    logger.info("Cooccurrence matrix fetch complete; %i nonzero values.\n",
                cooccurrences.getnnz())

    id2word = dict((y, x) for x, y in word_ids.iteritems())
    cooccurrences = iter_cooccurrences(cooccurrences, vocab, id2word,
                                       min_count=arguments.min_count)

    logger.info("Beginning GloVe training..")
    W = train_glove(word_ids, cooccurrences,
                    vector_size=arguments.vector_size,
                    iterations=arguments.iterations,
                    learning_rate=arguments.learning_rate)

    # Model data to be saved
    model = (word_ids, W)

    # TODO shave off bias values, do something with context vectors
    with open(arguments.vector_path, 'wb') as vector_f:
        pickle.dump(model, vector_f)

    logger.info("Saved vectors to %s", arguments.vector_path)


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    main(parse_args())
