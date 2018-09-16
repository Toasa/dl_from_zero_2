import numpy as np
import matplotlib.pyplot as plt

def preprocess(text):
    text = text.lower()
    text = text.replace(".", " .")
    words = text.split(" ")

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word

# vocab_sizeは語彙数、つまり "We will see the sun" なら 5
def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - 1
            right_idx = idx + 1

            if 0 <= left_idx:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix

def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print("%s is not found" % query)
        return

    print("\n[query] " + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(query_vec, word_matrix[i])

    count = 0
    # np.array([40, 30, 50]).argsort() => array([1, 0, 2])
    # と昇順にソート時のindexの配列が帰る
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return

def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)

    # a = np.array([[1, 2, 3], [3, 4, 5]])
    # a0 = np.sum(a, axis=0) => array([4, 6, 8])
    # a1 = np.sum(a, axis=1) => array([6, 12])
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / ((S[i] * S[j]) + eps))
            M[i, j] = max(pmi, 0)

            if verbose:
                cnt += 1
                if cnt % (total//100) == 0:
                    print("%.1f%% done" % (100*cnt/total))

    return M

def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size : -window_size]
    contexts = []
    for i in range(len(target)):
        a = []
        a.append(corpus[i])
        a.append(corpus[i + 2])
        contexts.append(a)
    return np.array(contexts), np.array(target)

def convert_one_hot(corpus, vocab_size):
    '''one-hot表現への変換

    :param corpus: 単語IDのリスト（1次元もしくは2次元のNumPy配列）
    :param vocab_size: 語彙数
    :return: one-hot表現（2次元もしくは3次元のNumPy配列）
    '''
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


# text = "You say goodbye and I say hello."
# text = "You see the moon and I see the sun."
# corpus, word_to_id, id_to_word = preprocess(text)
# contexts, target = create_contexts_target(corpus)
# vocab_size = len(word_to_id)
#
# target = convert_one_hot(target, vocab_size)
# contexts = convert_one_hot(contexts, vocab_size)

# C = create_co_matrix(corpus, vocab_size)
# # most_similar("moon", word_to_id, id_to_word, C, 5)
# W = ppmi(C)

# # 有効数字3桁でprint
# np.set_printoptions(precision = 3)
# # singular value decomposition
# U, S, V = np.linalg.svd(W)
#
# for word, word_id in word_to_id.items():
#     plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
#
# plt.scatter(U[:,0], U[:,1], alpha=0.5)
# plt.show()
