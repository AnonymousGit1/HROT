import numpy as np
import ot
import lda

from sklearn.metrics.pairwise import euclidean_distances
from scipy.io import loadmat
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

#######################
## SPARSE WEIGHTS DIST ##
#######################
def sparseOT(weights1, weights2, M):
    if weights1.sum()==0:
        weights1=weights2
    if weights2.sum()==0:
        weights2=weights1
    weights1 = weights1/weights1.sum() #Renormalization
    weights2 = weights2/weights2.sum()
    
    active1 = np.where(weights1)[0] # idx where weights is not 0
    active2 = np.where(weights2)[0]
    
    weights_1_active = weights1[active1]
    weights_2_active = weights2[active2]
    M_reduced = np.ascontiguousarray(M[active1][:,active2]) 
    return weights_1_active,weights_2_active,M_reduced


#######################
#### LOAD DATA ########
#######################
def load_wmd_data(path):
    """Load data used in the WMD paper.
    """
    mat_data = loadmat(path, squeeze_me=True, chars_as_strings=True)

    try:
        y = mat_data['Y'].astype(int)
    except KeyError:
        y = np.concatenate((mat_data['ytr'].astype(int),
                            mat_data['yte'].astype(int)))
    try:
        embeddings_of_doc_words = mat_data['X']
    except KeyError:
        embeddings_of_doc_words = np.concatenate((mat_data['xtr'],
                                                  mat_data['xte']))
    try:
        doc_word_counts = mat_data['BOW_X']
    except KeyError:
        doc_word_counts = np.concatenate((mat_data['BOW_xtr'], mat_data['BOW_xte']))
    try:
        doc_words = mat_data['words']
    except KeyError:
        doc_words = np.concatenate((mat_data['words_tr'], mat_data['words_te']))

    vocab = []
    embed_vocab = {}
    for d_w, d_e in zip(doc_words, embeddings_of_doc_words):
        if type(d_w) == str:
            d_w = [d_w]
        words = [w for w in d_w if type(w) == str]
        if len(words) == 1:
            d_e = d_e.reshape((-1, 1))
        for i, w in enumerate(words):
            if w not in vocab:
                vocab.append(w)
                embed_vocab[w] = d_e[:, i]
            else:
                if not np.allclose(embed_vocab[w], d_e[:, i]):
                    print('Problem with embeddings')
                    break

    bow_data = np.zeros((len(doc_word_counts), len(vocab)),
                        dtype=int)
    for doc_idx, (d_w, d_c) in enumerate(zip(doc_words, doc_word_counts)):
        if type(d_w) == str:
            d_w = [d_w]
        words = [w for w in d_w if type(w) == str]
        if len(words) == 1:
            d_c = np.array([d_c])
        words_idx = np.array([vocab.index(w) for w in words])
        bow_data[doc_idx, words_idx] = d_c.astype(int)

    return vocab, embed_vocab, bow_data, y


def reduce_vocab(bow_data, vocab, embed_vocab, embed_aggregate='mean'):
    """Reduce vocabulary size by stemming and removing stop words.
    """
    vocab = np.array(vocab)
    short = np.array([len(w) > 2 for w in vocab])
    stop_words = set(stopwords.words('english'))
    stop = np.array([w not in stop_words for w in vocab])
    reduced_vocab = vocab[np.logical_and(short, stop)]
    reduced_bow_data = bow_data[:, np.logical_and(short, stop)]
    stemmer = SnowballStemmer("english")
    stemmed_dict = {}
    stemmed_idx_mapping = {}
    stemmed_vocab = []
    for i, w in enumerate(reduced_vocab):
        stem_w = stemmer.stem(w)
        if stem_w in stemmed_vocab:
            stemmed_dict[stem_w].append(w)
            stemmed_idx_mapping[stemmed_vocab.index(stem_w)].append(i)
        else:
            stemmed_dict[stem_w] = [w]
            stemmed_vocab.append(stem_w)
            stemmed_idx_mapping[stemmed_vocab.index(stem_w)] = [i]

    stemmed_bow_data = np.zeros((bow_data.shape[0], len(stemmed_vocab)),
                                dtype=int)
    for i in range(len(stemmed_vocab)):
        stemmed_bow_data[:, i] = reduced_bow_data[:, stemmed_idx_mapping[i]].sum(axis=1).flatten()

    word_counts = stemmed_bow_data.sum(axis=0)
    stemmed_reduced_vocab = np.array(stemmed_vocab)[word_counts > 2].tolist()
    stemmed_reduced_bow_data = stemmed_bow_data[:, word_counts > 2]

    stemmed_reduced_embed_vocab = {}
    for w in stemmed_reduced_vocab:
        old_w_embed = [embed_vocab[w_old] for w_old in stemmed_dict[w]]
        if embed_aggregate == 'mean':
            new_w_embed = np.mean(old_w_embed, axis=0)
        elif embed_aggregate == 'first':
            new_w_embed = old_w_embed[0]
        else:
            print('Unknown embedding aggregation')
            break
        stemmed_reduced_embed_vocab[w] = new_w_embed

    return (stemmed_reduced_vocab,
            stemmed_reduced_embed_vocab,
            stemmed_reduced_bow_data)


def get_embedded_data(bow_data, embed_vocab, vocab):
    """Map bag-of-words data to embedded representation."""
    M, V = bow_data.shape
    embed_data = [[] for _ in range(M)]
    for i in range(V):
        for d in range(M):
            if bow_data[d, i] > 0:
                for _ in range(bow_data[d, i]):
                    embed_data[d].append(embed_vocab[vocab[i]])
    return [np.array(embed_doc) for embed_doc in embed_data]


def change_embeddings(vocab, bow_data, embed_path):
    """Change embedding data if vocabulary has been reduced."""
    all_embed_vocab = {}
    with open(embed_path, 'r') as file:
        for line in file.readlines():
            word = line.split(' ')[0]
            embedding = [float(x) for x in line.split(' ')[1:]]
            all_embed_vocab[word] = embedding

    data_embed_vocab = {}
    new_vocab_idx = []
    new_vocab = []
    for i, w in enumerate(vocab):
        if w in all_embed_vocab:
            data_embed_vocab[w] = all_embed_vocab[w]
            new_vocab_idx.append(i)
            new_vocab.append(w)
    bow_data = bow_data[:, new_vocab_idx]
    return new_vocab, data_embed_vocab, bow_data


def fit_topics(data, embeddings, vocab, K):
    """Fit a topic model to bag-of-words data."""
    model = lda.LDA(n_topics=K, n_iter=100, random_state=1)
    model.fit(data)
    topics = model.topic_word_
    lda_centers = np.matmul(topics, embeddings)
    print('LDA Gibbs topics')
    n_top_words = 20
    for i, topic_dist in enumerate(topics):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    print('\n')
    topic_proportions = model.doc_topic_

    return topics, lda_centers, topic_proportions

def loader(data_path,
           embeddings_path,
           p=1,
           K_lda=70,
           glove_embeddings=True,
           stemming=True,
           n_words_keep = 20):
    """ Load dataset and embeddings from data path."""
    # Load dataset from data_path
    vocab, embed_vocab, bow_data, y = load_wmd_data(data_path)
    y = y - 1

    # Use GLOVE word embeddings
    if glove_embeddings:
        vocab, embed_vocab, bow_data = change_embeddings(
            vocab, bow_data, embeddings_path)
    # Reduce vocabulary by removing short words, stop words, and stemming
    if stemming:
        vocab, embed_vocab, bow_data = reduce_vocab(
            bow_data, vocab, embed_vocab, embed_aggregate='mean')

    # Matrix of word embeddings
    embeddings = np.array([embed_vocab[w] for w in vocab])

    topics, lda_centers, topic_proportions = fit_topics(
        bow_data, embeddings, vocab, K_lda)

    cost_embeddings = euclidean_distances(embeddings, embeddings) ** p
    cost_topics = np.zeros((topics.shape[0], topics.shape[0]))
    
    ## Reduce topics to top-20 words
    if n_words_keep is not None:
        for k in range(K_lda):
            to_0_idx = np.argsort(-topics[k])[n_words_keep:]
            topics[k][to_0_idx] = 0

        
    for i in range(cost_topics.shape[0]):
        for j in range(i + 1, cost_topics.shape[1]):
            cost_topics[i, j] = sparse_ot(topics[i], topics[j], cost_embeddings)
    cost_topics = cost_topics + cost_topics.T

    out = {'X': bow_data, 'y': y,
           'embeddings': embeddings,
           'topics': topics, 'proportions': topic_proportions,
           'cost_E': cost_embeddings, 'cost_T': cost_topics}

    return out
    
#######################
### k-NN Classifier ###
#######################

def predict(neighbor_classes, C):
    # Make sure all classes are considered
    labels = np.concatenate((neighbor_classes, list(range(C))))
    # Find class frequency among neighbors
    weights = np.unique(labels, return_counts=True)[1]
    # Find most popular class
    prediction = np.argmax(weights)

    # If most popular class is ambiguous try with fewer neighbors; else return
    if sum(weights[prediction] == weights) > 1:
        return predict(neighbor_classes[:-2], C)
        
    else:
        return prediction


def knn(X_train, X_test, y_train, y_test, method, C, n_neighbors=7):
    # Number of classes
    n_classes = len(np.unique(y_train))

    prediction = []
    for doc in X_test:
        doc_to_train = [method(doc, x, C) for x in X_train]
        # Find indices of n_neighbors closest documents
        rank = np.argsort(doc_to_train)[:n_neighbors]

        # Make prediction based on most popular class among neighbors
        prediction.append(predict(y_train[rank], n_classes))

    # Print and return test error
    test_error = 1 - (prediction == y_test).mean()
    return test_error
