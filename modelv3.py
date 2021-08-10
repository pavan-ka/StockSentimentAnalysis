import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def getSentiment(headline):
    df = pd.read_csv("all-data.csv",encoding="latin-1")
    df.columns = ["Label","Headline"]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df["Headline"])

    sequences = tokenizer.texts_to_sequences(df["Headline"])

    max_length = np.max(list(map(lambda x: len(x), sequences)))

    sequences = pad_sequences(sequences, maxlen = max_length)

    X = sequences
    
    relabeling = {"negative":0,"neutral":1,"positive":2}
    df['Label'] = df["Label"].replace(relabeling)
    
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X,y)

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(X_train.shape[1],)))
    model.add(tf.keras.layers.Embedding(
        input_dim=10123,
        output_dim=128,
        input_length=X_train.shape[1]
    ))
    model.add(tf.keras.layers.Dense(128,activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64,activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(32,activation="relu"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(3))

    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

    history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=3,batch_size=32)

    @tf.autograph.experimental.do_not_convert
    def predict(t):
        token = tokenizer.texts_to_sequences(np.array([t]))

        padded = pad_sequences(token,maxlen=max_length)
        
        return model.predict(padded)

    vals = predict(headline)
    if max(vals[0]) == vals[0][0]:
        return 0
    elif max(vals[0]) == vals[0][1]:
        return 1
    else:
        return 2