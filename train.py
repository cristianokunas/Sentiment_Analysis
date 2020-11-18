import os
import re
import time

import numpy as np
import pandas as pd

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Dense, Input, Embedding, LSTM
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import tensorflow as tf

seed = 7
np.random.seed(seed)
# O model sera exportado para este arquivo
filename = 'model/model_saved.h5'
epochs = 5 
# numero de amostras a serem utilizadas em cada atualizacao do gradiente - numero de instancias
batch_size = 32
# separa % para teste do modelo
test_dim = 0.20
# Quantidade maxima de palavras para manter no vocabulario
max_fatures = 20000 # 20000
# dimensao de saida da camada Embedding
embed_dim = 128

# Tamanho maximo das sentencas
max_sequence_length = 300

#Calcula tempo de execucao
def calcRuntime(totalTime):
  hours, rem = divmod(totalTime, 3600)
  minutes, seconds = divmod(rem, 60)
  formatTime = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
  return formatTime

# metodo para limpar as strings - tirar conteudo que nao agrega
def clean_str(string):
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


# Metodo para preparar os dados de treino e teste
# carrega csv, limpa as strings e remove as stop_wors
# Realiza a tokenizacao
def prepare_data(data):
    data = data[['text', 'sentiment']]
    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply(lambda x: clean_str(x))
    data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

    stop_words = set(stopwords.words('english'))
    text = []
    for row in data['text'].values:
        word_list = text_to_word_sequence(row)
        no_stop_words = [w for w in word_list if not w in stop_words]
        no_stop_words = " ".join(no_stop_words)
        text.append(no_stop_words)

    tokenizer = Tokenizer(num_words=max_fatures, split=' ')

    tokenizer.fit_on_texts(text)
    X = tokenizer.texts_to_sequences(text)

    X = pad_sequences(X, maxlen=max_sequence_length)

    word_index = tokenizer.word_index
    Y = pd.get_dummies(data['sentiment']).values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_dim, random_state=42)

    return X_train, X_test, Y_train, Y_test, word_index, tokenizer
# Especifica o device a ser utilizado
with tf.device("/gpu:0"):
    # Carrega o arquivo de dados .csv
    data = pd.read_csv('./dataset/imdb.csv')
    #descriptions
    data.describe()

    # chama metodo para preparar dados
    X_train, X_test, Y_train, Y_test, word_index, tokenizer = prepare_data(data)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)
    
    def model():
        input_shape = (max_sequence_length,)
        model_input = Input(shape=input_shape, dtype="int32")
        embedding = Embedding(max_fatures, embed_dim, input_length=max_sequence_length)(model_input)
        lstm = LSTM(embed_dim, dropout=0.2, recurrent_dropout=0.2)(embedding)
        model_output = Dense(128, input_dim=64, kernel_initializer='uniform', activation='relu')(lstm)
        model_output = Dense(64, kernel_initializer='uniform', activation='relu')(model_output)
        model_output = Dense(2, kernel_initializer='uniform', activation='sigmoid')(model_output)
        model = Model(inputs=model_input, outputs=model_output)
        return model

    # Criacao do modelo
    model = model()

    # compilacao do modelo
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    log_dir = "logs/fit/" + path1 + path2 +'/'
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # # Condicao de parada no treinamento da rede
    # lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor = 0.9, patience=3, verbose = 1)
    # early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience = 8, verbose = 1, mode = 'auto')
    # checkpointer = ModelCheckpoint(filename, monitor='val_loss', verbose = 1, save_best_only=True)

    print(model.summary())

    # Treinamento da rede neural
    # Verifica se existe um modelo treinado
    # True = carrega o modelo ja treinado
    # False = treina a rede e salva o modelo
    inicio = time.time()

    if os.path.exists('./{}'.format(filename)):
        try:
            model.load_weights('./{}'.format(filename))
            print('Successful model loading!')
        except:
            print('No such file or directory!')
    else:
        hist = model.fit(
            X_train,
            Y_train,
            validation_data=(X_test, Y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            shuffle=True,
            callbacks=[tensorboard_callback]) #lr_reducer, early_stopper, checkpointer,
        try:
            model.save_weights(filename)
        except:
            print('An error has occurred. Could not save the file!')

    fim = time.time()
    print(calcRuntime(fim - inicio))

    # Avaliando o modelo
    scores = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    print("Erro: %.2f%%" % (scores[0] * 100))

while True:
    print("\nType 0 to quit")
    sentence = input("input> ")
    if sentence == "0":
        break

    new_text = [sentence]
    new_text = tokenizer.texts_to_sequences(new_text)

    new_text = pad_sequences(new_text, maxlen=max_sequence_length, value=0)

    sentiment = model.predict(new_text, batch_size=1, verbose=2)[0]

    if (np.argmax(sentiment) == 0):
        pred_proba = "%.2f%%" % (sentiment[0] * 100)
        print("negativo => ", pred_proba)
    elif (np.argmax(sentiment) == 1):
        pred_proba = "%.2f%%" % (sentiment[1] * 100)
        print("positivo => ", pred_proba)
