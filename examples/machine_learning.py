
from models.statistical.ml_classifier import MLClassifier
from models.statistical.models import ModelsFactory

from models.statistical.tokenizer import TokenizerFactory

# can be loaded from a config file
model_name = 'xgb'
tokenizer_name = 'default'
ngrams = 2
name = 'sst2'
stopwords = False
max_features = 5000
test_size = 0.2

model = ModelsFactory.from_name(model_name)  # init model
tokenizer = TokenizerFactory.from_name(tokenizer_name)  # init tokenizer


def predict_ml_classifier(text, plot=False):
    """
    Predict label from input text
    :param text: input text
    :param plot: show important words
    :return: predicted value
    """
    ml_classifier = MLClassifier(input_data=None,
                                         output_data=None,
                                         model=model,
                                         tokenizer=tokenizer,
                                         stop_words=stopwords,
                                         ngram=ngrams,
                                         max_features=max_features,
                                         test_size=test_size,
                                         name=name)
    return ml_classifier.predict([text], plot=plot)[0]


def train_ml_classifier(input_data, output_data):
    """
      Train a classifier
      :param input_data: input documents
      :param output_data: labels
      :return:
      """
    ml_classifier = MLClassifier(input_data=input_data,
                                         output_data=output_data,
                                         model=model,
                                         tokenizer=tokenizer,
                                         stop_words=stopwords,
                                         ngram=ngrams,
                                         max_features=max_features,
                                         test_size=test_size,
                                         name=name)

    ml_classifier.train(resampling=False)


if __name__ == '__main__':
    import pandas as pd
    from configs import SST2_DIR
    df = pd.read_csv(SST2_DIR + '/train.tsv', delimiter='\t')
    input_data = df['sentence']
    output_data = df['label']

    print('Training ML classifier')
    train_ml_classifier(input_data, output_data)