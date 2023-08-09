from sklearn.ensemble import RandomForestClassifier
from raise_utils.learners.learner import Learner
from keras import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, BatchNormalization, Activation, SpatialDropout1D
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import numpy as np

class BILSTM(Learner):
    def __init__(self, epochs=10, max_words=1000, embedding=5,n_layers=1, input_lenght=1, learning_rate=0.001, dropout_rate=0.1, *args, **kwargs):
        """
        Initializes the BILSTM Classifier.
        :param epochs: Number of epochs to train for
        :param max_words: Maximum number of top words to consider
        :param embedding: Embedding dimensionality
        :param n_layers: Number of LSTM layers
        :param args: Args passed to Learner
        :param kwargs: Keyword args passed to Learner
        """
        super(BILSTM, self).__init__(*args, **kwargs)
        self.epochs = epochs
        self.max_words = max_words
        self.embed_dim = embedding
        self.n_layers = n_layers
        self.input_lenght = input_lenght
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.learner = self
        self.random_map = {
            "epochs": (10, 20),
            "learning_rate": [0.001, 0.01],
            "dropout_rate": [0.1, 0.05, 0.2]

        }
        self._instantiate_random_vals()
        
        
        
    def set_data(self, x_train, y_train, x_test, y_test) -> None:
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.y_train[self.y_train != 0] = 1
        self.y_test[self.y_test != 0] = 1
        
    def fit(self) -> None:
        self._check_data()
        model = Sequential() 
        model.add(Embedding(self.max_words, self.embed_dim,
                            input_length=self.input_lenght))
        model.add(SpatialDropout1D(self.dropout_rate))
        for _ in range(self.n_layers):
            model.add(Bidirectional(LSTM(150, dropout=self.dropout_rate)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        model.summary()

        self.learner = model
        if self.hooks is not None:
            if self.hooks.get('pre_train', None):
                for hook in self.hooks['pre_train']:
                    hook.call(self)

        model.fit(self.x_train, self.y_train,
                  batch_size=64, epochs=self.epochs, validation_split=0.2)

        if self.hooks is not None:
            if self.hooks.get('post_train', None):
                for hook in self.hooks['post_train']:
                    hook.call(model)
                    
    def predict(self, x_test):
        """
        Makes predictions
        :param x_test: Test data
        :return: np.ndarray
        """
        return (self.learner.predict(x_test) > 0.5).astype('int32')
        

class BILSTMGlove(Learner):
    def __init__(self, word_embd, epochs=10, max_words=1000, embedding=5,n_layers=1, input_lenght=1, learning_rate=0.001, dropout_rate=0.1, *args, **kwargs):
        """
        Initializes the BILSTM Classifier.
        :param epochs: Number of epochs to train for
        :param max_words: Maximum number of top words to consider
        :param embedding: Embedding dimensionality
        :param n_layers: Number of LSTM layers
        :param args: Args passed to Learner
        :param kwargs: Keyword args passed to Learner
        """
        super(BILSTM, self).__init__(*args, **kwargs)
        self.epochs = epochs
        self.max_words = max_words
        self.embed_dim = embedding
        self.n_layers = n_layers
        self.input_lenght = input_lenght
        self.word_embd = word_embd
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        
        self.learner = self

        
        self.random_map = {
            "epochs": (10, 20),
            "learning_rate": [0.001, 0.01, 0.1, 1.],
            "dropout_rate": [0.1, 0.05, 0.2]

        }
        self._instantiate_random_vals()
        
        
        
    def set_data(self, x_train, y_train, x_test, y_test) -> None:
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.y_train[self.y_train != 0] = 1
        self.y_test[self.y_test != 0] = 1
        
    def fit(self) -> None:
        self._check_data()
        model = Sequential() 
        model.add(Embedding(self.max_words, self.embed_dim,
                            input_length=self.input_lenght, weights=[self.word_embd]))
        model.add(SpatialDropout1D(self.dropout_rate))
        for _ in range(self.n_layers):
            model.add(Bidirectional(LSTM(150, dropout=self.dropout_rate)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        model.summary()

        self.learner = model
        
        if self.hooks is not None:
            if self.hooks.get('pre_train', None):
                for hook in self.hooks['pre_train']:
                    hook.call(self)

        model.fit(self.x_train, self.y_train,
                  batch_size=64, epochs=self.epochs, validation_split=0.2)

        if self.hooks is not None:
            if self.hooks.get('post_train', None):
                for hook in self.hooks['post_train']:
                    hook.call(model)
                    
    def predict(self, x_test):
        """
        Makes predictions
        :param x_test: Test data
        :return: np.ndarray
        """
        return (self.learner.predict(x_test) > 0.5).astype('int32')
        