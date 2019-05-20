import re
import string

import unicodedata
from nltk import word_tokenize
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))


def normalize_text(text):
    """
    Transform non ascii characters
    :param text:
    :return:
    """
    return unicodedata.normalize("NFKD", text)


def remove_punctuations(text):
    return text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))


def to_lowercase(text):
    """
    Transform all characters to lowercase
    :param text: input text
    :return: transformed text
    """
    return text.lower()


def remove_stopwords(tokenized_words):
    """
    Remove stopwords from tokenized words
    :param tokenized_words: tokenized words
    :return: original words without stop words
    """
    return [w for w in tokenized_words if not w in STOP_WORDS]


def remove_non_letters(text):
    """

    :param text:
    :return:
    """
    return re.sub("[^a-zA-Z0-9]", " ", text)


def text_to_words_list(text, stop_words=False):
    text = normalize_text(text)
    # text = remove_punctuations(text)
    # tokenized_text = word_tokenize(text)
    # text = ' '.join(tokenized_text)
    text = remove_non_letters(text)
    lower = to_lowercase(text)
    words = word_tokenize(lower)

    if stop_words:
        words = remove_stopwords(words)

    return words
