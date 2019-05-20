import os
import os.path as osp

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from xgboost.sklearn import XGBClassifier


class BaseModel(object):
    """
    Base class for all models
    """

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def get_metrics(self, x, y):
        y_predicted = self.predict(x)
        metrics = classification_report(y, y_predicted)
        print('Classification report:')
        print(metrics)
        return metrics

    def load_model(self, path):
        try:
            path += self.name + '.pkl'
            self.model = joblib.load(path)
        except Exception as e:
            print(e)
            print("Couldn't load scikit learn model on path {}!".format(path))

    def save_model(self, path):
        try:
            os.makedirs(osp.dirname(path), exist_ok=True)
            path += self.name + '.pkl'
            joblib.dump(self.model, path)
        except Exception as e:
            print(e)
            print("Couldn't save scikit learn model on path {}!".format(path))


class RandomForest(BaseModel):
    def __init__(self):
        super(RandomForest, self).__init__()
        self.model = RandomForestClassifier(n_estimators=100, n_jobs=4, max_depth=3, random_state=42)
        self.name = 'RF'


class NaiveBayes(BaseModel):
    def __init__(self):
        super(NaiveBayes, self).__init__()
        self.model = MultinomialNB()
        self.name = 'NB'


class LogisticRegression(BaseModel):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.model = LogisticRegression(random_state=0)
        self.name = 'LogisticRegression'


class SVM(BaseModel):
    def __init__(self):
        super(SVM, self).__init__()
        self.model = LinearSVC()
        self.name = 'SVM'


class XGBoost(BaseModel):
    def __init__(self):
        super(XGBoost, self).__init__()
        self.model = XGBClassifier(max_depth=3, n_jobs=4, random_state=42)
        self.name = 'XGB'


class ModelsFactory:
    """Factory class that instantiate a model"""

    @staticmethod
    def from_name(model_name):
        if model_name.lower() == "rf":
            return RandomForest()
        elif model_name.lower() == "nb":
            return NaiveBayes()
        elif model_name.lower() == "lr":
            return LogisticRegression()
        elif model_name.lower() == "svm":
            return SVM()
        elif model_name.lower() == "xgb":
            return XGBoost()
        else:
            raise ValueError("Couldn't instantiate a model. Given `{}`, but supported values are "
                             "`rf`, `nb`, `lr`, `svm`, `xgb`".format(model_name.lower()))
