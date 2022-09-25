import os
import math
from directory import Directory
from document import Document
from similarity import SimilarityMeasure
from parsivar import Normalizer
from parsivar import Tokenizer
from parsivar import FindStems

stop_words = [" "]

vocabulary = 500
spam_training_samples_number = 300
ham_training_samples_number = 300
spam_testing_samples_number = 200
ham_testing_samples_number = 200

spam_training_documents = []
ham_training_documents = []
spam_testing_documents = []
ham_testing_documents = []

spam_words_documents_frequency = {}  # word: number of documents containing the word
ham_words_documents_frequency = {}  # word: number of documents containing the word


# Preprocessing Document
def get_words_from_file(file_name):  # reads a file and returns a list of its words
    normalizer = Normalizer()
    tokenizer = Tokenizer()
    with open(file_name, 'r', encoding='utf-8') as infile:
        text = infile.read()
        normal_text = normalizer.normalize(text)
        return tokenizer.tokenize_words(normal_text)


def email_preprocess(file_name):  # gives a file name, preprocess it for analyzing, returns a list of its useful words
    raw_words_list = get_words_from_file(file_name)  # list of all of words in document
    stemmer = FindStems()
    final_words_list = []  # list of wanted words
    for word in raw_words_list:
        # removing english words, punctuations ans stop words
        if is_valid_persian_word(word) and not is_stop_word(word):
            final_word = stemmer.convert_to_stem(word)  # stemming
            final_words_list.append(final_word)
    return words_frequency_in_document(final_words_list)


def is_valid_persian_word(word):
    for letter in word:
        if is_english(letter) or is_punctuation(letter):
            return False
    return True


def is_english(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def is_punctuation(s):
    return s == '،' or s == '.' or s == '؟' or s == '»' or s == '«'


def is_stop_word(word):
    return word in stop_words


# Getting Word Vectors
def get_documents_vectors(is_training, directory, is_spam):
    files = os.listdir(str(directory))
    set_samples_number(is_training, is_spam, len(files))
    return convert_documents_to_vectors(files, str(directory))


def set_samples_number(is_training, is_spam, num):
    if is_training:
        if is_spam:
            global spam_training_samples_number
            spam_training_samples_number = num
        else:
            global ham_training_samples_number
            ham_training_samples_number = num
    else:
        if is_spam:
            global spam_testing_samples_number
            spam_testing_samples_number = num
        else:
            global ham_testing_samples_number
            ham_testing_samples_number = num


def convert_documents_to_vectors(files, prefix):
    documents_set = {}
    for f_name in files:
        if f_name[0] != '.':
            email = email_preprocess(prefix + '/' + f_name)
            documents_set[f_name] = email
    return documents_set


def get_feature_vector(features, document_vector):
    vector = {}
    for word in features:
        if word in document_vector:
            vector[word] = document_vector[word]
        else:
            vector[word] = 0
    return vector


def add_to_documents(is_training, document):
    if is_training:
        if document.is_spam:
            global spam_training_documents
            spam_training_documents.append(document)
        else:
            global ham_training_documents
            ham_training_documents.append(document)
    else:
        if document.is_spam:
            global spam_testing_documents
            spam_testing_documents.append(document)
        else:
            global ham_testing_documents
            ham_testing_documents.append(document)


def set_document(is_training, features, raw_samples, is_spam):
    for doc in raw_samples:
        feature_vector = get_feature_vector(features, raw_samples[doc])
        document = Document(doc, feature_vector, is_spam)  # HERE
        add_to_documents(is_training, document)


# Training Phase
def training():
    print('is reading spam training documents...')
    spam_training_raw_samples = get_training_documents(Directory.SpamTraining, True)
    print('is reading ham training documents...')
    ham_training_raw_samples = get_training_documents(Directory.HamTraining, False)

    print('is choosing features...')
    features = select_features()

    print('is setting feature vectors for spam training documents...')
    set_document(True, features, spam_training_raw_samples, True)
    print('is setting feature vectors for ham training documents...')
    set_document(True, features, ham_training_raw_samples, False)
    return features


def get_training_documents(directory, is_spam):
    documents_vectors = get_documents_vectors(True, directory, is_spam)
    for doc in documents_vectors:
        set_categorized_words_document_frequency(documents_vectors[doc], is_spam)
    return documents_vectors


# Preprocessing Training Data
def increase_dictionary_value(dictionary, key, amount):
    if key in dictionary:
        dictionary[key] += amount
    else:
        dictionary[key] = amount


def words_frequency_in_document(words):  # takes words set of a document and returns dictionary of their frequencies
    word_frequency = {}
    for word in words:
        increase_dictionary_value(word_frequency, word, 1)
    return word_frequency


def set_categorized_words_document_frequency(words_frequency, is_spam):
    for word in words_frequency.keys():
        if is_spam:
            increase_dictionary_value(spam_words_documents_frequency, word, 1)
        else:
            increase_dictionary_value(ham_words_documents_frequency, word, 1)


# Feature Selection Functions
def local_chi_square(word, is_spam):
    a = 0
    b = 0
    if is_spam:
        if word in spam_words_documents_frequency:
            a = spam_words_documents_frequency[word]
        if word in ham_words_documents_frequency:
            b = ham_words_documents_frequency[word]
        c = spam_training_samples_number - a
        d = ham_training_samples_number - b
    else:
        if word in ham_words_documents_frequency:
            a = ham_words_documents_frequency[word]
        if word in spam_words_documents_frequency:
            b = spam_words_documents_frequency[word]
        c = ham_training_samples_number - a
        d = spam_training_samples_number - b
    n = ham_training_samples_number + spam_training_samples_number  # number of all documents
    return (n * ((a * d - b * c) ** 2)) / ((a + c) * (b + d) * (a + b) * (c + d))


def chi_square(word):
    spam_chi_square = local_chi_square(word, True)
    ham_chi_square = local_chi_square(word, False)
    if spam_chi_square > ham_chi_square:
        return spam_chi_square
    return ham_chi_square


def select_features():  # returns 200 features for category
    words = {}
    for word in spam_words_documents_frequency.keys():
        words[word] = chi_square(word)
    for word in ham_words_documents_frequency.keys():
        words[word] = chi_square(word)
    sorted_words = dict(sorted(words.items(), key=lambda item: item[1], reverse=True))
    return list(sorted_words.keys())[0: vocabulary]


# Testing Phase
def testing(features):
    print('is reading spam testing documents...')
    spam_testing_raw_samples = get_documents_vectors(False, Directory.SpamTesting, True)
    print('is reading ham testing documents...')
    ham_testing_raw_samples = get_documents_vectors(False, Directory.HamTesting, False)

    print('is setting feature vectors for spam testing documents...')
    set_document(False, features, spam_testing_raw_samples, True)
    print('is setting feature vectors for ham testing documents...')
    set_document(False, features, ham_testing_raw_samples, False)

    print('Cosine Similarity')
    test(SimilarityMeasure.Cosine)
    print('tf-idf similarity')
    test(SimilarityMeasure.tf_idf)


def test(similarity_measure):
    k = choose_k(similarity_measure)
    wrong_spams = 0
    wrong_hams = 0
    print('is testing knn on spam documents...')
    for doc in spam_testing_documents:
        if knn(k, doc, similarity_measure) != 'spam':
            wrong_hams += 1
    print('is testing knn on ham documents...')
    for doc in ham_testing_documents:
        if knn(k, doc, similarity_measure) != 'ham':
            wrong_spams += 1
    evaluate(wrong_spams, wrong_hams)


# kNN Algorithm
def choose_k(similarity_measure):
    if similarity_measure == SimilarityMeasure.Cosine:
        return 51
    return 45


def knn(k, document, similarity_measure):
    neighbors = find_knn(k, document, similarity_measure)
    number_of_spam_neighbors = 0
    for neighbor in neighbors:
        if neighbor[0] == 's':  # it's spam
            number_of_spam_neighbors += 1
    if number_of_spam_neighbors > k / 2:
        return 'spam'
    return 'ham'


def find_knn(k, document, similarity_measure):
    neighbors = {}
    for doc in spam_training_documents:
        neighbors[doc.name] = similarity(document, doc, similarity_measure)
    for doc in ham_training_documents:
        neighbors[doc.name] = similarity(document, doc, similarity_measure)
    sorted_neighbors = dict(sorted(neighbors.items(), key=lambda item: item[1], reverse=True))
    return list(sorted_neighbors.keys())[0: k]


# Similarity Measures
def similarity(document1, document2, similarity_measure):
    if similarity_measure == SimilarityMeasure.Cosine:
        return document1.cosine_similarity(document2)
    return tf_idf_score(document1, document2)


def tf(document, word):
    return document.words_vector[word] + 1


def df(word):
    ans = 0
    if word in spam_words_documents_frequency:
        ans += spam_words_documents_frequency[word]
    if word in ham_words_documents_frequency:
        ans += ham_words_documents_frequency[word]
    return ans


def idf(word):
    n = spam_training_samples_number + ham_training_samples_number
    return math.log2(n / df(word))


def tf_idf_score(document1, document2):
    ans = 0
    for word in document1.words_vector:
        if document1.words_vector[word] > 0:
            ans += (tf(document2, word) * idf(word))
    return ans


# Evaluation Measures
def evaluate(wrong_spams, wrong_hams):
    right_spams = spam_testing_samples_number - wrong_hams
    right_hams = ham_testing_samples_number - wrong_spams
    print('Results:')
    show_confusion_matrix(right_spams, wrong_spams, wrong_hams, right_hams)
    print('Precision: ', precision(right_spams, wrong_spams))
    print('Recall: ', recall(right_spams, wrong_hams))
    print('F1-measure: ', f1(right_spams, wrong_spams, wrong_hams))


def show_confusion_matrix(a, b, c, d):
    print('Confusion Matrix:')
    print('           ', '      ', 'Actual ')
    print('           ', '      ', 'Spam ', 'ham')
    print('Predicated ', 'Spam: ', a, ' ',  b)
    print('           ', 'ham:  ', c, '  ', d)
    print()


def precision(a, b):
    return (a / (a + b)) * 100


def recall(a, c):
    return (a / (a + c)) * 100


def f1(a, b, c):
    p = precision(a, b)
    r = recall(a, c)
    return 2 * p * r / (p + r)


if __name__ == "__main__":
    print('is saving stop words...')
    stop_words = get_words_from_file(str(Directory.StopWords))  # reading stop words from file
    text_features = training()
    testing(text_features)
