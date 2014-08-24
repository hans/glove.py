from argparse import ArgumentParser
import codecs
from collections import Counter
import itertools
from functools import partial
import logging
from math import log
import os.path
import pickle
from random import sample

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
                           help=('Path to coocurrence matrix file. If '
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

        for center_i, center in enumerate(tokens):
            center_id = word_ids[center]

            # Collect all words in left window of center word
            contexts = tokens[max(0, center_i - window_size) : center_i]
            contexts_len = len(contexts)

            for left_i, left in enumerate(contexts):
                left_id = word_ids[left]

                # Distance from center word
                distance = contexts_len - left_i

                # Weight by inverse of distance between words
                increment = 1.0 / float(distance)

                # Build co-occurrence matrix symmetrically (pretend we
                # are calculating right contexts as well)
                cooccurrences[center_id, left_id] += increment
                cooccurrences[left_id, center_id] += increment

    return word_ids, cooccurrences


def run_iter(word_ids, cooccurrences, W, biases, gradient_squared,
             gradient_squared_biases,
             learning_rate=0.05, x_max=100, alpha=0.75):
    """
    Run a single iteration of GloVe training using the given
    cooccurrence data and the previously computed weight vectors /
    biases and accompanying gradient histories.

    `word_ids` and `cooccurrences` should be provided as returned by
    `build_cooccur`.

    See the `train_glove` function for information on the shapes of `W`,
    `biases`, `gradient_squared`, `gradient_squared_biases` and how they
    should be initialized.

    The parameters `x_max`, `alpha` define our weighting function when
    computing the cost for two word pairs; see the GloVe paper for more
    details.

    Returns a tuple of the form

        (cost, W, biases, gradient_squared, gradient_squared_biases)
    """

    vocab_size = len(word_ids)
    global_cost = 0

    # We want to iterate over co-occurrence pairs randomly so as not to
    # unintentionally bias the word vector contents. We'll simply work
    # on the shuffled Cartesian product of the word IDs.
    ids = word_ids.values()
    paired_ids = itertools.product(sample(ids, vocab_size),
                                   sample(ids, vocab_size))

    for main_word_id, context_word_id in paired_ids:
        cooccurrence = cooccurrences[main_word_id, context_word_id]

        # TODO is this correct? C code seems to ignore the case where we
        # don't have any cooccurrence data
        if cooccurrence == 0:
            continue

        # Shift context word ID so that we fetch a different vector when
        # we examine a given word as main and that same word as context
        context_word_shifted = context_word_id + vocab_size

        # Fetch main word vector
        main_word = W[main_word_id]

        # Fetch context word vector (stored separately from its own
        # main-word vector)
        context_word = W[context_word_shifted]

        # Compute unweighted cost
        #
        #   $$ J' = w_i^Tw_j + b_i + b_j - log(X_{ij}) $$
        cost = (main_word.dot(context_word) + biases[main_word_id]
                + biases[context_word_shifted] - log(cooccurrence))

        weight = (cooccurrence / x_max) ** alpha if cooccurrence < x_max else 1

        # Add weighted cost to the global cost tracker
        global_cost += 0.5 * weight * (cost ** 2)

        # Compute gradients for word vector elements.
        #
        # NB: `main_word` is only a view into `W` (not a copy), so our
        # modifications here will affect the global weight matrix;
        # likewise for context_word
        grad_main = weight * cost * context_word
        grad_context = weight * cost * main_word

        # Now perform adaptive updates
        main_word -= (learning_rate * grad_main / np.sqrt(
            gradient_squared[main_word_id]))
        context_word -= (learning_rate * grad_context / np.sqrt(
            gradient_squared[context_word_shifted]))

        # Update squared gradient sums
        gradient_squared[main_word_id] += np.square(grad_main)
        gradient_squared[context_word_shifted] += np.square(grad_context)

        # Compute gradients for bias terms
        grad_bias_main = weight * cost
        grad_bias_context = weight * cost

        biases[main_word_id] -= (learning_rate * grad_bias_main / np.sqrt(
            gradient_squared_biases[main_word_id]))

        biases[context_word_shifted] -= (
            learning_rate * grad_bias_context / np.sqrt(
                gradient_squared_biases[context_word_shifted]))

        # Update squared gradient sums
        gradient_squared_biases[main_word_id] += grad_bias_main ** 2
        gradient_squared_biases[context_word_shifted] += grad_bias_context ** 2

    return global_cost, W, biases, gradient_squared, gradient_squared_biases


def train_glove(word_ids, cooccurrences, vector_size=100, iterations=25,
                **kwargs):
    """
    Train GloVe vectors on the given cooccurrence data. Keyword
    arguments are passed on to the iteration step function `run_iter`.

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
    W = np.random.randn(vocab_size * 2, vector_size) - 0.5

    # Bias terms, each associated with a single vector. An array of size
    # $2V$, initialized randomly in the range (-0.5, 0.5].
    biases = np.random.randn(vocab_size * 2) - 0.5

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

    for i in range(iterations):
        logger.info("\tBeginning iteration %i..", i)

        cost, W, biases, gradient_squared, gradient_squared_biases = (
            run_iter(word_ids, cooccurrences, W, biases, gradient_squared,
                     gradient_squared_biases))

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
    logger.info("Cooccurrence matrix fetch complete.\n")

    # Convert cooccurrence matrix to CSR form for faster reads
    cooccurrences = cooccurrences.asformat('csr')

    logger.info("Beginning GloVe training..")
    W = train_glove(word_ids, cooccurrences, vector_size=arguments.vector_size,
                    iterations=arguments.iterations,
                    learning_rate=arguments.learning_rate)

    # TODO shave off bias values, do something with context vectors
    with open(arguments.vector_path, 'wb') as vector_f:
        pickle.dump(W, vector_f)

    logger.info("Saved vectors to %s", arguments.vector_path)


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    main(parse_args())
