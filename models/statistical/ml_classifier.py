import os

import shap
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split

from models.statistical.utility import text_to_words_list


class MLClassifier(object):
    def __init__(self, input_data, output_data, model, tokenizer,
                 stop_words=False, max_features=5000,
                 ngram=1, test_size=0.2, name=''):
        self.input_data = input_data
        self.output_data = output_data
        self.model = model
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.max_features = max_features
        self.ngram = ngram
        self.test_size = test_size
        self.vectorizer = CountVectorizer(analyzer="word",
                                          tokenizer=self.tokenizer,
                                          preprocessor=None,
                                          stop_words=None,
                                          max_features=self.max_features,
                                          ngram_range=(self.ngram - 1, self.ngram))
        self.name = name
        self.MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/{0}_'.format(self.name))
        self.VECTORIZER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            'models/vectorizer_{0}.pkl'.format(self.name))
        self.TF_IDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        'models/tfidf_{0}.pkl'.format(self.name))

    def __transform_features(self, clean_text_data, train=True):
        """
        Transform cleaned text into vectors
        :param clean_text_data:
        :param train:
        :return:
        """
        # transform feature data weights
        if train:
            self.vectorizer.fit(clean_text_data)
            joblib.dump(self.vectorizer, self.VECTORIZER_PATH)
            train_data_features = self.vectorizer.transform(clean_text_data)
            tf_transformer = TfidfTransformer(use_idf=False).fit(train_data_features)
            joblib.dump(tf_transformer, self.TF_IDF_PATH)
        else:
            self.vectorizer = joblib.load(self.VECTORIZER_PATH)
            train_data_features = self.vectorizer.transform(clean_text_data)
            tf_transformer = joblib.load(self.TF_IDF_PATH)
        train_data_features = tf_transformer.transform(train_data_features)
        train_data_features = train_data_features.toarray()
        return train_data_features

    def __clean_text(self, text):
        """
        Clean input text
        :param text: input text
        :return: cleaned text
        """
        clean_text = []
        for i in range(0, len(text)):
            clean_text.append(' '.join(text_to_words_list(text[i], self.stop_words)))

        return clean_text

    def create_features(self, data, train=True):
        """
        Vectorize input data
        :return:
        """
        clean_text = self.__clean_text(data)

        # print("Creating the bag of words...\n")
        return self.__transform_features(clean_text, train=train)

    def train(self, resampling=False):
        """
        Vectorize input data and train the model
        :return:
        """
        X = self.create_features(self.input_data)
        y = self.output_data
        if resampling:
            from collections import Counter
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=42)
            X, y = sm.fit_resample(X, y)
            print('Resampled dataset shape %s' % Counter(y))
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
        print('Test size is: %2.0f%%' % (self.test_size * 100), len(x_test), 'documents')
        print('Train size is: %2.0f%%' % ((1 - self.test_size) * 100), len(x_train), 'documents')
        self.model.train(x_train, y_train)
        self.model.save_model(self.MODELS_DIR)
        print('test metrics')
        self.model.get_metrics(x_test, y_test)
        print('total metrics')
        self.model.get_metrics(X, y)

        explainer = shap.TreeExplainer(self.model.model, X)
        shap_values = explainer.shap_values(X)
        X_test_array = X
        shap.summary_plot(shap_values, X_test_array, feature_names=self.vectorizer.get_feature_names(),
                          class_names=self.output_data.unique())

    def predict(self, X, plot=False):
        """
        Do a prediction on never seen data. The same pre-processing and vectorization will be applied
        :param X: Input data for prediction
        :return: predicted classes
        """
        self.model.load_model(self.MODELS_DIR)
        x = self.create_features(X, train=False)
        if plot:
            explainer = shap.TreeExplainer(self.model.model)
            shap_values = explainer.shap_values(x)
            shap.force_plot(explainer.expected_value[0], shap_values[0], features=x[0], link="identity",
                            feature_names=self.vectorizer.get_feature_names(), matplotlib=True)
        return self.model.predict(x)
