# dnn_classifier.py
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow import keras
import numpy as np

class DNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=None, hidden_layers=(64, 32), epochs=30, batch_size=16, random_state=42):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        input_dim = X.shape[1] if self.input_dim is None else self.input_dim
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_dim))
        for units in self.hidden_layers:
            model.add(keras.layers.Dense(units, activation="relu"))
        model.add(keras.layers.Dense(len(np.unique(y)), activation="softmax"))
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.model = model
        return self

    def predict(self, X):
        proba = self.model.predict(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        return self.model.predict(X)