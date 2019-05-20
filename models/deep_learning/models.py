import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Embedding, Dense, Bidirectional, Conv1D, Dropout, BatchNormalization, \
    MaxPooling1D, Input, Flatten
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from models.deep_learning.data_utils import Glove, DataUtils


class BaseModel(object):
    def __init__(self, x, y, maxlen=64, batch_size=128, epochs=100, model_name='model.hdf5'):
        self.x = x
        self.y = y
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.epochs = epochs
        self.classes = np.unique(self.y).size
        self.embedding_matrix = None
        self.model_name = model_name
        self.glove = Glove()
        self.tokenizer = Tokenizer()

    def create_features(self, x, train=True):
        if train:
            self.tokenizer.fit_on_texts(x)
        else:
            self.tokenizer = DataUtils.load_tokenizer('tokenizer.pkl')
        self.x = self.tokenizer.texts_to_sequences(x)
        self.x = pad_sequences(self.x, maxlen=self.maxlen, padding='post')
        if train:
            self.embedding_matrix = self.glove.build_embedding_matrix(self.tokenizer.word_index)
        else:
            self.embedding_matrix = DataUtils.load_embedding_matrix('embedding_matrix.npy')
        if train:
            DataUtils.save_tokenizer(self.tokenizer, 'tokenizer.pkl')
            DataUtils.save_embedding_matrix(self.embedding_matrix, 'embedding_matrix.npy')

    def __build_model__(self):
        raise NotImplementedError()

    def train(self):
        self.create_features(self.x)
        self.model = self.__build_model__()

        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        # Callbacks
        mc = ModelCheckpoint(self.model_name, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                      patience=3, min_lr=1e-6)

        # Split into train and test
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        self.model.fit(x_train, y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_data=(x_test, y_test),
                       # class_weight=class_weight,
                       callbacks=[mc, early_stopping, reduce_lr])

    def predict(self, x):
        self.create_features(x, train=False)
        self.load_model()
        return self.model.predict(self.x)

    def save_model(self):
        self.model.save(self.model_name)

    def load_model(self):
        self.model = load_model(self.model_name)


class CNN1DModel(BaseModel):
    def __build_model__(self):
        """
        Convolutional 1D network with pre-trained GloVe embedding
        :return:
        """
        DROPOUT_RATE = 0.2

        sequence_input = Input(shape=(self.x.shape[-1],), dtype='int32')

        embedding_layer = Embedding(input_dim=self.embedding_matrix.shape[0],
                                    output_dim=300,
                                    weights=[self.embedding_matrix],
                                    input_length=self.x.shape[-1],
                                    trainable=True)(sequence_input)

        x = Conv1D(128, 5, activation='elu', kernel_initializer='he_uniform')(embedding_layer)
        x = Dropout(DROPOUT_RATE)(x)
        x = MaxPooling1D(5)(x)
        x = BatchNormalization(axis=-1)(x)

        x = Conv1D(128, 5, activation='elu', kernel_initializer='he_uniform')(x)
        x = Dropout(DROPOUT_RATE)(x)
        x = MaxPooling1D(5)(x)
        x = BatchNormalization(axis=-1)(x)

        x = Conv1D(128, 1, activation='elu', kernel_initializer='he_uniform')(x)
        x = Dropout(DROPOUT_RATE)(x)
        x = MaxPooling1D(1)(x)  # global max pooling
        x = BatchNormalization(axis=-1)(x)

        x = Flatten()(x)
        x = Dense(512, activation='elu', kernel_initializer='he_uniform')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Dropout(DROPOUT_RATE)(x)

        dense = Dense(self.classes, activation='softmax')(x)  # no initialization in output layer
        self.model = Model(sequence_input, dense)
        return self.model


class RNNModel(BaseModel):
    def __build_model__(self):
        """
        Bidirectional LSTM model with pre-trained GloVe embeddings
        :return:
        """
        sequence_input = Input(shape=(self.x.shape[-1],), dtype='int32')

        embedding_layer = Embedding(input_dim=self.embedding_matrix.shape[0],
                                    output_dim=300,
                                    weights=[self.embedding_matrix],
                                    input_length=self.x.shape[-1])(sequence_input)

        bi_lstm1 = Bidirectional(LSTM(units=256, return_sequences=True))(embedding_layer)
        bi_lstm2 = Bidirectional(LSTM(units=256))(bi_lstm1)

        dense1 = Dense(512, activation='relu')(bi_lstm2)
        dense2 = Dense(self.classes, activation='softmax')(dense1)

        self.model = Model(sequence_input, dense2)

        return self.model
