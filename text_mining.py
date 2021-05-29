import math
import string
from collections import Counter

import numpy as np
from nltk import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity


def calculate_df(vocab=None, corpus=None):
    """
    :param vocab: words in corpus
    :param corpus: all documents
    :return: how many times specific word occours given a corpus
    """
    DF = {}
    for term in vocab:
        for document in corpus:
            if term in document:
                if term in DF.keys():
                    DF[term] += 1
                else:
                    DF[term] = 1
    return DF


def calculate_idf(VOCAB, DF, corpus_size):
    """
    :param VOCAB: words in corpus
    :param DF: ow many times specific word occours given a corpus
    :param corpus_size:
    :return: inverse document frequency
    """
    IDF = {}
    for word in VOCAB:
        IDF[word] = math.log10(corpus_size / float(DF[word]))
    return IDF


def calculate_tf(corpus, VOCAB):
    def tf_calc(term):
        if term > 0:
            term = math.log10(term) + 1
        return term

    TF = []
    for index, document in enumerate(corpus):
        document_tf = {}
        word_count = Counter(document)
        for term in VOCAB:
            item = word_count.get(term, 0)
            document_tf[term] = tf_calc(item)
        TF.append({"movie": index, "TF": document_tf, index: document_tf})
    return TF


def calculate_tf_idf(TF, IDF):
    TF_IDF = {}
    for document in TF:
        TF_IDF[document['movie']] = {}
        for item, value in document['TF'].items():
            idf = IDF.get(item, 0)
            TF_IDF[document['movie']][item] = value * idf

    return TF_IDF


def calc_cosine_similarity(vector1, vector2):
    return cosine_similarity(vector1, vector2)


def get_similarity_items(n=1, similarity=None):
    return sorted(similarity, key=lambda x: x['distance'], reverse=True)[:n]


def pipeline_cleaning(text):
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    return words


def get_tf_idf_vectors(vocab, tokenized_documents):
    DF = calculate_df(vocab=vocab, corpus=tokenized_documents)
    IDF = calculate_idf(VOCAB=vocab, DF=DF, corpus_size=len(tokenized_documents))
    TF = calculate_tf(corpus=tokenized_documents, VOCAB=vocab)
    TF_IDF = calculate_tf_idf(TF=TF, IDF=IDF)

    vectors = []
    for key in TF_IDF:
        vectors.append(np.array(list(TF_IDF[key].values())))
    return vectors
