#!/usr/bin/python3
# -*- coding: utf-8 -*-

import re
import os
import os.path
import datetime

import logging
import logging.handlers

import numpy as np
import pandas as pd


from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Input, Embedding, LSTM
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from nltk.corpus import stopwords
import tensorflow as tf


class SampleTraining:
    SEED = 7
    np.random.seed(SEED)
    MODEL_DATA_PATH = './dataset/imdb.csv'
    MODEL_FILE_PATH = 'model/tf-model-keras-trained-{}.h5'.format(datetime.datetime.now().strftime('%Y-%m-%d'))
    LOG_FILENAME = 'logs/sample-{}.log'.format(datetime.datetime.now().strftime('%Y-%m-%d'))
    MAX_FEATURES = 20000  # Only consider the top 20k words
    MAXLEN = 300  # Only consider the first 200 words of each movie review
    MODEL_TRAINING_EPOCHS = 5
    BATCH_SIZE = 512
    TEST_DIM = 0.10
    # dimensao de saida da camada Embedding
    EMBED_DIM = 128


    # Construtor
    def __init__(self, debug):
        self.configureLog(debug)

    # Configure the log system
    def configureLog(self, debug):
        # Creates and configure the log file
        self.logger = logging.getLogger('Sentiment Analysis with IMDb Review Dataset')
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        # Defines the format of the logger
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Configure the log rotation
        handler = logging.handlers.RotatingFileHandler(self.LOG_FILENAME, maxBytes=268435456, backupCount=50, encoding='utf8')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info('Starting LSTM Learner')

    def clean_str(self, string):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        cleanr = re.compile('<.*?>')
        string = re.sub(r'\d+', '', string)
        string = re.sub(cleanr, '', string)
        string = re.sub("'", '', string)
        string = re.sub(r'\W+', ' ', string)
        string = string.replace('_', '')

        return string.strip().lower()

    def prepare_data(self, data):
        data = data[['text', 'sentiment']]
        data['text'] = data['text'].apply(lambda x: x.lower())
        data['text'] = data['text'].apply(lambda x: self.clean_str(x))
        data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

        stop_words = set(stopwords.words('english'))
        text = []
        for row in data['text'].values:
            word_list = text_to_word_sequence(row)
            no_stop_words = [w for w in word_list if not w in stop_words]
            no_stop_words = " ".join(no_stop_words)
            text.append(no_stop_words)

        tokenizer = Tokenizer(num_words=self.MAX_FEATURES, split=' ')
        tokenizer.fit_on_texts(text)
        X = tokenizer.texts_to_sequences(text)
        X = pad_sequences(X, maxlen=self.MAXLEN)
        word_index = tokenizer.word_index
        Y = pd.get_dummies(data['sentiment']).values
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=self.TEST_DIM, random_state=42)

        return x_train, x_test, y_train, y_test, word_index, tokenizer

    def trainModel(self):
        data = pd.read_csv(self.MODEL_DATA_PATH)
        x_train, x_test, y_train, y_test, word_index, tokenizer = self.prepare_data(data)
        print(x_train.shape, y_train.shape)
        print(x_test.shape, y_test.shape)

        if os.path.exists('./{}'.format(self.MODEL_FILE_PATH)):
            try:
                self.logger.info('TensorFlow - Loading the Neural Network model')
                model.load_weights('./{}'.format(self.MODEL_FILE_PATH))
            except:
                self.logger.info('No such file or directory!')
        else:
            self.logger.info('TensorFlow - Creating the Neural Network model')
            input_shape = (self.MAXLEN,)
            model_input = Input(shape=input_shape, name="input")
            embedding = Embedding(self.MAX_FEATURES, self.EMBED_DIM, input_length=self.MAXLEN, name="embedding")(model_input)
            lstm = LSTM(self.EMBED_DIM, dropout=0.2, recurrent_dropout=0.2, name="lstm")(embedding)
            model_output = Dense(128, input_dim=64, kernel_initializer='uniform', activation='relu')(lstm)
            model_output = Dense(64, kernel_initializer='uniform', activation='relu')(model_output)
            model_output = Dense(2, kernel_initializer='uniform', activation='sigmoid')(model_output)
            self.model = Model(inputs=model_input, outputs=model_output)

            self.logger.info('TensorFlow - Compiling the Neural Network model')
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            #lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor = 0.9, patience=3, verbose = 1)
            #early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience = 8, verbose = 1, mode = 'auto')
            #checkpointer = ModelCheckpoint(self.MODEL_FILE_PATH, monitor='val_loss', verbose = 1, save_best_only=True)

            self.logger.debug('TensorFlow - Summary: {}'.format(self.model.summary()))

            self.logger.info('TensorFlow - Start training')
            start = datetime.datetime.now()
            with tf.device("/gpu:0"):
                hist = self.model.fit(
                    x_train,
                    y_train,
                    validation_split=0.10,
                    epochs=self.MODEL_TRAINING_EPOCHS,
                    batch_size=self.BATCH_SIZE) #,
                    #callbacks=[lr_reducer, early_stopper, checkpointer])
                
            end = datetime.datetime.now()
            diff = end - start
            self.logger.info('TensorFlow - TRAIN - Time: {}ms'.format(diff.total_seconds() * 1000))

            try:
                self.model.save_weights(self.MODEL_FILE_PATH)
            except:
                self.logger.debug('An error has occurred. Could not save the file!')
            
            start = datetime.datetime.now()
            score = self.model.evaluate(x_train, y_train, batch_size=self.BATCH_SIZE)
            end = datetime.datetime.now()

            diff = end - start
            self.logger.info('TensorFlow - TRAIN - Evaluation Time: {}ms'.format(diff.total_seconds() * 1000))

            for i in range(0, len(self.model.metrics_names)):
                self.logger.info('TensorFlow - TRAIN - {}: {}'.format(self.model.metrics_names[i], score[i]))

            score = self.model.predict(x_test, batch_size=self.BATCH_SIZE)

            self.logger.info('TensorFlow - TEST - F1_Score: {}'.format(f1_score(y_test, score, average="micro")))

            self.logger.info('TensorFlow - TEST - Evaluating dataset')

            start = datetime.datetime.now()
            score = self.model.evaluate(x_test, y_test, batch_size=self.BATCH_SIZE)
            end = datetime.datetime.now()

            diff = end - start
            self.logger.info('TensorFlow - TEST - Evaluation Time: {}ms'.format(diff.total_seconds() * 1000))

            for i in range(0, len(self.model.metrics_names)):
                self.logger.info('TensorFlow - TEST - {}: {}'.format(self.model.metrics_names[i], score[i]))

            score = self.model.predict(x_test, batch_size=self.BATCH_SIZE)

            self.logger.info('TensorFlow - TEST - F1_Score: {}'.format(f1_score(y_test, score, average="micro")))
