import os
import re
import time

import numpy as np
import pandas as pd
import datetime

import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Dense, Input, Embedding, LSTM, Bidirectional
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# falta *
# lstm          = *2048 *1024 *512 *256 *128 *64 *32
# lstm-w50      = *2048 *1024 *512 *256 *128 *64 *32
# lstm-w100     = 2048 1024 512 256 128 *64 *32
# lstm-w200     = 2048 1024 512 256 128 64 32
# lstm-w300     = 2048 1024 512 256 128 64 32
# bilstm        = *2048 *1024 *512 *256 *128 *64 *32
# bilstm-w50    = 2048 1024 512 256 128 64 32
# bilstm-w100   = 2048 1024 512 256 128 64 32
# bilstm-w200   = 2048 1024 512 256 128 64 32
# bilstm-w300   = 2048 1024 512 256 128 64 32


seed = 7
np.random.seed(seed)
path1 = 'ep5gpu/'
path2 = 'bs32'
# O model sera exportado para este arquivo
filename = 'model/'+path1+'lstm/model_' + path2 + '.h5'

#log_dir = "logs/fit/" + path1 + "lstm/"+ path2 +'/'
#datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# filename = 'model/model_ep5bs32mf20000ed128ms300dr128-64dr64ds2-teste13.h5'#dr128-64dr64# numero de iteracoes
epochs = 5  # email - 150
# dimensionalidade do word embedding pre-treinado
word_embedding_dim = 100
# numero de amostras a serem utilizadas em cada atualizacao do gradiente - numero de instancias
batch_size = 1024
# separa % para teste do modelo
test_dim = 0.200
# Quantidade maxima de palavras para manter no vocabulario
max_features = 20000 # 20000
# dimensao de saida da camada Embedding
embed_dim = 128

# Tamanho maximo das sentencas
max_sequence_length = 300

pre_trained_wv = False

bilstm = True

#Calcula tempo de execucao
def calcRuntime(totalTime):
  hours, rem = divmod(totalTime, 3600)
  minutes, seconds = divmod(rem, 60)
  formatTime = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
  return formatTime

def load_pre_trained_wv(word_index, num_words, word_embedding_dim):
    embeddings_index = {}
    f = open(os.path.join('word_embedding', 'glove.6B.{0}d.txt'.format(word_embedding_dim)), encoding='utf-8')
    print("leu word")
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('%s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((num_words, word_embedding_dim))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


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

    # string = re.sub(r'RT+', '', string)
    # string = re.sub(r'@\S+', '', string)
    # string = re.sub(r'http\S+', '', string)

    cleanr = re.compile('<.*?>')

    string = re.sub(r'\d+', '', string)
    string = re.sub(cleanr, '', string)
    string = re.sub("'", '', string)
    string = re.sub(r'\W+', ' ', string)
    string = string.replace('_', '')

    # string = remove_emoji(string)

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

    tokenizer = Tokenizer(num_words=max_features, split=' ')

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
    # data.describe()

    # chama metodo para preparar dados
    X_train, X_test, Y_train, Y_test, word_index, tokenizer = prepare_data(data)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)
    
    def model():
        if pre_trained_wv is True:
            print("USE PRE TRAINED")
            num_words = min(max_features, len(word_index) + 1)
            weights_embedding_matrix = load_pre_trained_wv(word_index, num_words, word_embedding_dim)
            input_shape = (max_sequence_length,)
            model_input = Input(shape=input_shape, name="input", dtype='int32')
            embedding = Embedding(num_words, word_embedding_dim, input_length=max_sequence_length, name="embedding",
                                  weights=[weights_embedding_matrix], trainable=False)(model_input)

            if bilstm is True:
                lstm = Bidirectional(LSTM(word_embedding_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm"))(
                    embedding)
            else:
                lstm = LSTM(word_embedding_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm")(embedding)
        else:
            input_shape = (max_sequence_length,)
            model_input = Input(shape=input_shape, dtype="int32")
            embedding = Embedding(max_features, embed_dim, input_length=max_sequence_length)(model_input)
            if bilstm is True:
                lstm = Bidirectional(LSTM(embed_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm"))(embedding)
            else:
                lstm = LSTM(embed_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm")(embedding)

        model_output = Dense(128, input_dim=64, kernel_initializer='uniform', activation='relu')(lstm)
        model_output = Dense(64, kernel_initializer='uniform', activation='relu')(model_output)
        model_output = Dense(2, kernel_initializer='uniform', activation='sigmoid')(model_output)
        model = Model(inputs=model_input, outputs=model_output)
        return model

    # Criacao do modelo
    model = model()

    # compilacao do modelo
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #log_dir = "logs/fit/" + path1 + path2 +'/'
    #datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


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
            shuffle=True)#, callbacks=[tensorboard_callback]) #lr_reducer, early_stopper, checkpointer,
        try:
            model.save_weights(filename)
        except:
            print('An error has occurred. Could not save the file!')

    fim = time.time()
    print(calcRuntime(fim - inicio))


    # def plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de confusao', cmap=plt.cm.Blues):
    #
    #     plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #     plt.title(title)
    #     plt.colorbar()
    #     tick_marks = np.arange(len(classes))
    #     plt.xticks(tick_marks, classes, rotation=45)
    #     plt.yticks(tick_marks, classes)
    #
    #     fmt = '.2f' if normalize else 'd'
    #     thresh = cm.max() / 2.
    #     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #         plt.text(j, i, format(cm[i, j], fmt),
    #                  horizontalalignment="center",
    #                  color="white" if cm[i, j] > thresh else "black")
    #
    #     plt.tight_layout()
    #     plt.ylabel('Base de teste')
    #     plt.xlabel('Predicoes')
    #     plt.show()
    #
    # xx = model.predict(X_test)
    # rounded_predictions = np.argmax(xx, axis=1)
    # rounded_labels=np.argmax(Y_test, axis=1)
    # cm = confusion_matrix(rounded_labels, rounded_predictions)
    # print(cm)
    # # df_cm = pd.DataFrame(cm, range(2), range(2))
    # # sn.set(font_scale=1.4)
    # # svm = sn.heatmap(df_cm, annot=True, fmt='d')
    #
    #
    # # Compute confusion matrix
    # np.set_printoptions(precision=2)
    # # Plot non-normalized confusion matrix
    # plt.figure(figsize=(13, 5))
    # plot_confusion_matrix(cm, classes=['positivo=1', 'negativo=0'], normalize=False,  title='Matriz de confusao')

    # Avaliando o modelo
    scores = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    print("Erro: %.2f%%" % (scores[0] * 100))


    # while True:
    #     print("\nType 0 to quit")
    #     sentence = input("input> ")
    #     if sentence == "0":
    #         break
    #
    #     new_text = [sentence]
    #     new_text = tokenizer.texts_to_sequences(new_text)
    #
    #     new_text = pad_sequences(new_text, maxlen=max_sequence_length, value=0)
    #
    #     sentiment = model.predict(new_text, batch_size=1, verbose=2)[0]
    #
    #     if (np.argmax(sentiment) == 0):
    #         pred_proba = "%.2f%%" % (sentiment[0] * 100)
    #         print("negativo => ", pred_proba)
    #     elif (np.argmax(sentiment) == 1):
    #         pred_proba = "%.2f%%" % (sentiment[1] * 100)
    #         print("positivo => ", pred_proba)
