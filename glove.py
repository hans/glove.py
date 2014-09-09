from argparse import ArgumentParser
import codecs
from collections import Counter
import cPickle as pickle
import itertools
from functools import partial
import logging
from math import log
import os.path
from random import shuffle

import numpy as np
from scipy import sparse
import theano
import theano.tensor as T


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


def iter_cooccurrences(lil_matrix):
    """
    Yield `(w1, w2, x)` pairs from a LiL sparse matrix as produced by
    `build_cooccur`, where `w1` is a row index (word ID), `w2` is a
    column index (context word ID), and `x` is a cell value
    ($X_{w1, w2}$).
    """

    # This function is built for LiL-format sparse matrices only
    assert isinstance(lil_matrix, sparse.lil_matrix)

    for i, (row, data) in enumerate(itertools.izip(lil_matrix.rows,
                                                   lil_matrix.data)):
        for data_idx, j in enumerate(row):
            yield i, j, data[data_idx]


def gradient_updates(cost, v_main, v_context, b_main, b_context,
                     gradsq_W_main, gradsq_W_context,
                     gradsq_b_main, gradsq_b_context,
                     learning_rate):
    """
    Compute gradient updates for adaptive gradient descent on a single
    example.

    Returns a list of a form suitable for use with Theano function
    updates.
    """

    updates = []

    # TODO make sure this works, then compress into a `params` array
    # with a for loop!

    # Compute gradients for word vector elements.
    grad_main = T.grad(cost, wrt=v_main)
    grad_context = T.grad(cost, wrt=v_context)

    # Now perform adaptive updates
    W_ = T.inc_subtensor(v_main,
                         -learning_rate * grad_main / T.sqrt(gradsq_W_main))
    W_ = T.inc_subtensor(v_context,
                         -learning_rate * grad_context / T.sqrt(gradsq_W_context))
    updates.append((W, W_))
    # updates.append((v_main,
    #                 -learning_rate * grad_main / T.sqrt(gradsq_W_main)))
    # updates.append((v_context,
    #                 -learning_rate * grad_context / T.sqrt(gradsq_W_context)))

    # Update squared gradient sums
    gradsq_W_ = T.inc_subtensor(gradsq_W_main, grad_main ** 2)
    gradsq_W_ = T.inc_subtensor(gradsq_W_context, grad_context ** 2)
    updates.append((gradsq_W, gradsq_W_))
    # updates.append((gradsq_main, grad_main ** 2))
    # updates.append((gradsq_context, grad_context ** 2))

    # Compute gradients for bias terms
    grad_b_main = T.grad(cost, b_main)
    grad_b_context = T.grad(cost, b_context)

    b_ = T.inc_subtensor(b_main,
                         -learning_rate * grad_b_main / T.sqrt(gradsq_b_main))
    b_ = T.inc_subtensor(b_context,
                         -learning_rate * grad_b_context / T.sqrt(gradsq_b_context))
    updates.append((b, b_))
    # updates.append((b_main,
    #                 -learning_rate * grad_b_main / T.sqrt(gradsq_b_main)))
    # updates.append((b_context,
    #                 -learning_rate * grad_b_context / T.sqrt(gradsq_b_context)))

    # Update squared gradient sums
    gradsq_b_ = T.inc_subtensor(gradsq_b_main, grad_b_main ** 2)
    gradsq_b_ = T.inc_subtensor(gradsq_b_context, grad_b_context ** 2)
    updates.append((gradsq_b, gradsq_b_))
    # updates.append((gradsq_b_main, grad_b_main ** 2))
    # updates.append((gradsq_b_context, grad_b_context ** 2))

    return updates


def run_iter(word_ids, cooccurrence_list,
             learning_rate=0.05, x_max=100, alpha=0.75):
    """
    Run a single iteration of GloVe training using the given
    cooccurrence data and the previously computed weight vectors /
    biases and accompanying gradient histories.

    `word_ids` should be provided as returned by `build_cooccur`.

    `cooccurrence_list` is a list where each element is of the form

        (word_i_id, word_j_id, x_ij)

    where `x_ij` is a cooccurrence value $X_{ij}$ as presented in the
    matrix defined by `build_cooccur` and the Pennington et al. (2014)
    paper itself.

    See the `train_glove` function for information on the shapes of `W`,
    `biases`, `gradient_squared`, `gradient_squared_biases` and how they
    should be initialized.

    The parameters `x_max`, `alpha` define our weighting function when
    computing the cost for two word pairs; see the GloVe paper for more
    details.

    Returns the feedforward cost associated with the given inputs and updates
    the weights by SGD in place.
    """

    vocab_size = len(word_ids)
    global_cost = 0

    # We want to iterate over co-occurrence pairs randomly so as not to
    # unintentionally bias the word vector contents. We'll simply work
    # on the shuffled Cartesian product of the word IDs.
    #
    # TODO do we need to shuffle every time?
    shuffle(cooccurrence_list)

    i_main = T.lscalar('i_main')
    i_context = T.lscalar('i_context')

    v_main, v_context = W[i_main], W[i_context]
    b_main, b_context = b[i_main], b[i_context]
    gradsq_W_main, gradsq_W_context = gradsq_W[i_main], gradsq_W[i_context]
    gradsq_b_main, gradsq_b_context = gradsq_b[i_main], gradsq_b[i_context]
    cooccurrence = T.dscalar('cooccurrence')

    fetch_params = theano.function([i_main, i_context],
                                   [v_main, v_context, b_main, b_context,
                                    gradsq_W_main, gradsq_W_context,
                                    gradsq_b_main, gradsq_b_context])

    # Cost function
    cost = T.dot(v_main, v_context) + b_main + b_context - T.log(cooccurrence)

    train = theano.function(
        [i_main, i_context, v_main, v_context, b_main, b_context,
         gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context,
         cooccurrence],
        cost,
        updates=gradient_updates(cost, v_main, v_context, b_main, b_context,
                                 gradsq_W_main, gradsq_W_context,
                                 gradsq_b_main, gradsq_b_context,
                                 learning_rate),
        on_unused_input='warn')

    global_cost = 0.

    # Run online AdaGrad learning
    for i_main, i_context, cooccurrence in cooccurrence_list:
        # Shift context word ID so that we fetch a different vector when
        # we examine a given word as main and that same word as context
        i_context += vocab_size

        (v_main, v_context, b_main, b_context,
         gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context) = fetch_params(i_main, i_context)

        global_cost += train(i_main, i_context,
                             v_main, v_context, b_main, b_context,
                             gradsq_W_main, gradsq_W_context,
                             gradsq_b_main, gradsq_b_context,
                             cooccurrence)

    return global_cost


W, b, gradsq_W, gradsq_b = None, None, None, None

def train_glove(word_ids, cooccurrence_list, vector_size=100, iterations=25,
                **kwargs):
    """
    Train GloVe vectors on the given `cooccurrence_list`, where each
    element is of the form

        (word_i_id, word_j_id, x_ij)

    where `x_ij` is a cooccurrence value $X_{ij}$ as presented in the
    matrix defined by `build_cooccur` and the Pennington et al. (2014)
    paper itself.

    Keyword arguments are passed on to the iteration step function
    `run_iter`.

    Returns the computed word vector matrix `W`.
    """

    vocab_size = len(word_ids)

    global W, b, gradsq_W, gradsq_b
    # Word vector matrix. This matrix is (2V) * d, where N is the size
    # of the corpus vocabulary and d is the dimensionality of the word
    # vectors. All elements are initialized randomly in the range (-0.5,
    # 0.5]. We build two word vectors for each word: one for the word as
    # the main (center) word and one for the word as a context word.
    #
    # It is up to the client to decide what to do with the resulting two
    # vectors. Pennington et al. (2014) suggest adding or averaging the
    # two for each word, or discarding the context vectors.
    W_ = np.random.randn(vocab_size * 2, vector_size) - 0.5
    W = theano.shared(W_.astype(theano.config.floatX), name='W')

    # Bias terms, each associated with a single vector. An array of size
    # $2V$, initialized randomly in the range (-0.5, 0.5].
    b_ = np.random.randn(vocab_size * 2) - 0.5
    b = theano.shared(b_.astype(theano.config.floatX), name = 'b')

    # Training is done via adaptive gradient descent (AdaGrad). To make
    # this work we need to store the sum of squares of all previous
    # gradients.
    #
    # Like `W`, this matrix is (2V) * d.
    #
    # Initialize all squared gradient sums to 1 so that our initial
    # adaptive learning rate is simply the global learning rate.
    gradsq_W_ = np.ones((vocab_size * 2, vector_size))
    gradsq_W = theano.shared(gradsq_W_.astype(theano.config.floatX),
                             name='gradsq_W')

    # Sum of squared gradients for the bias terms.
    gradsq_b_ = np.ones(vocab_size * 2)
    gradsq_b = theano.shared(gradsq_b_.astype(theano.config.floatX),
                             name='gradsq_b')

    for i in range(iterations):
        logger.info("\tBeginning iteration %i..", i)

        cost = run_iter(word_ids, cooccurrence_list)

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

    cooccurrence_list = list(iter_cooccurrences(cooccurrences))

    logger.info("Beginning GloVe training..")
    W = train_glove(word_ids, cooccurrence_list,
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
