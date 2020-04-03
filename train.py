import os
import re

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM  # , Bidirectional
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
#import matplotlib.pyplot as plt

seed = 7
np.random.seed(seed)
# O model sera exportado para este arquivo
filename = 'model/model_saved.h5'
# numero de iteracoes
epochs = 10  # email - 150
# numero de amostras a serem utilizadas em cada atualizacao do gradiente - numero de instancias
batch_size = 32  # email - 10
# separa % para teste do modelo
test_dim = 0.20
# Quantidade maxima de palavras para manter no vocabulario
max_fatures = 5000
# dimensao de saida da camada Embedding
embed_dim = 128

# Tamanho maximo das sentencas
max_sequence_length = 300

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


# Carrega o arquivo de dados .csv
data = pd.read_csv('./dataset/imdb.csv')

# chama metodo para preparar dados
X_train, X_test, Y_train, Y_test, word_index, tokenizer = prepare_data(data)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


# Cria o modelo
def model():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim, input_length=max_sequence_length))
    model.add(LSTM(embed_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm"))
    model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(2, init='uniform', activation='sigmoid'))
    return model


# Ccriacao do modelo
model = model()
# compilacao do modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# Treinamento da rede neural
# Verifica se existe um modelo treinado
# True = treina a rede e salva o modelo
# False =
if os.path.exists('./{}'.format(filename)):
    model.load_weights('./{}'.format(filename))
else:
    hist = model.fit(
        X_train,
        Y_train,
        validation_data=(X_test, Y_test),
        epochs=epochs,
        batch_size=batch_size)

    model.save_weights(filename)

# Avaliando o modelo
scores = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print("Acc: %.2f%%" % (scores[1] * 100))

while True:
    print("\nType 0 to quit")
    sentence = input("input> ")
    if sentence == "0":
        break

    new_text = [sentence]
    new_text = tokenizer.texts_to_sequences(new_text)

    new_text = pad_sequences(new_text, maxlen=max_sequence_length, dtype='int32', value=0)

    sentiment = model.predict(new_text, batch_size=1, verbose=2)[0]

    if (np.argmax(sentiment) == 0):
        pred_proba = "%.2f%%" % (sentiment[0] * 100)
        print("negativo => ", pred_proba)
    elif (np.argmax(sentiment) == 1):
        pred_proba = "%.2f%%" % (sentiment[1] * 100)
        print("positivo => ", pred_proba)
