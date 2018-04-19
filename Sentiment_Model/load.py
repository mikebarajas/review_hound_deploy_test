import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    import h5py as h5py
    import json
    import io

from keras.models import Sequential
from keras.layers import MaxPooling1D, Conv1D, Flatten, Dropout, Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
# from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence

def init():
    PATH = './Sentiment_Model/optimal_dict3.json'
    with open(PATH) as json_data:
        d = json.load(json_data)
    word_dict = pd.Series(d)


    def encode_sentence(text):
        result = []
        arr = text_to_word_sequence(text, lower=True, split=" ")
        for word in arr:
            w = encode_word(word)
            if w is not None:
                result.append(w)
        return result

    def encode_word(word):
        if word not in word_dict:
            return None
        return word_dict[word]

    def build_model(words, vec_len, review_len):
        model = Sequential()
        model.add(Embedding(words, vec_len, input_length=review_len))
        model.add(Dropout(0.25))
        model.add(Conv1D(32, 3, padding="same"))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(16, 3, padding="same"))
        model.add(Flatten())
        model.add(Dropout(0.25))
        model.add(Dense(100, activation="sigmoid"))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        # model.summary()
        return model

    # Parameters
    version = 4
    words = len(word_dict)
    review_len = 1000
    vec_len = 300
    patience = 5
    batch_size = 40
    epochs = 3
    # Build model
    model = build_model(words, vec_len, review_len)

    # import h5py

    # Model
    from keras.preprocessing import sequence
    from keras.models import load_model
    model = load_model(("./Sentiment_Model/optimalfloyds3.h5"))


    def encode_batch(arr):
        result = []
        for sentence in arr:
            result.append(encode_sentence(sentence))
        return sequence.pad_sequences(result, maxlen=review_len)

    def predict_batch(arr):
        batch = encode_batch(arr)
        result = model.predict(batch, batch_size=len(batch), verbose=0)
        return print(result)

    
    return build_model(words, vec_len, review_len)

#print(predict_batch(["yes"]))
    # "good",
    # "this is the best thing ever",
    # "nice",
    # "bad",
    # "such a horrible judgement",
    # "no",
    # "shitty"
    # ]))