from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


class WordTokenizer(object):
    """
    Wrapper for a tokenizer
    """

    def __call__(self, doc):
        """
        Tokenize words
        :param doc: input text
        :return: tokenized words
        """
        return word_tokenize(doc)


class LemmaTokenizer(object):
    """
    Wrapper for a tokenizer with WordNetLemmatizer
    """

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        """
        Tokenize and lemmatize words
        :param doc: input text
        :return: tokenized and lemmatized words
        """
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


class StemTokenizer(object):
    """
    Wrapper for a tokenizer with PorterStemmer
    """

    def __init__(self):
        self.ps = PorterStemmer()

    def __call__(self, doc):
        """
        Tokenize and stem words
        :param doc: input text
        :return: tokenized and stemmed words
        """
        return [self.ps.stem(t) for t in word_tokenize(doc)]


class LemmaStemTokenizer(object):
    """
    Wrapper for a tokenizer with PorterStemmer and WordNetLemmatizer
    """

    def __init__(self):
        self.ps = PorterStemmer()
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        """
        Tokenize, lemmatize and stem words
        :param doc: input text
        :return: tokenized, lemmatized and stemmed words
        """
        return [self.wnl.lemmatize(self.ps.stem(t)) for t in word_tokenize(doc)]


class TokenizerFactory:
    """Factory class that instantiate Tokenizer"""

    @staticmethod
    def from_name(model_name):
        """
        Creates a tokenizer based on provided name
        :param model_name: tokenizer name
        :return: tokenizer instance
        """
        if model_name == "stem":
            return StemTokenizer()
        elif model_name == "lemma":
            return LemmaTokenizer()
        elif model_name == "lemmastem":
            return LemmaStemTokenizer()
        elif model_name == "default":
            return WordTokenizer()
        else:
            raise ValueError("Couldn't instantiate a tokenizer. Given `{}`, but supported values are "
                             "`stem`, `lemma`, `lemmastem`, `default`".format(model_name))
