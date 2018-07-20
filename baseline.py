import sys, os, re, csv, codecs, numpy as np, pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

glove_file='glove file path e,g, ./dataset/glove.6B.50d.txt'
train_file='./dataset/train.csv'
test_file='./dataset/test.csv'

train=pd.read_csv(train_file)
test=pd.read_csv(test_file)

sent_train=train["comment_text"].fillna("nan")

classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y=train[classes].values

sent_test=test["comment_text"].fillna("nan")

##some parameters here:
max_words_count=20000
embedding_size=50
max_words_length=100

tokenizer=Tokenizer(num_words=max_words_count)
tokenizer.fit_on_texts(sent_train)
tokens_train = tokenizer.texts_to_sequences(sent_train)
tokens_test = tokenizer.texts_to_sequences(sent_test)

x_train=pad_sequences(tokens_train,maxlen=max_words_length)
x_test=pad_sequences(tokens_test,maxlen=max_words_length)

def index_to_embed(word,*embedding):
    return word,np.asarray(embedding,dtype='float32')

embed_dict=dict(index_to_embed(*o.strip().split())for o in open(glove_file))

all_embs = np.stack(embed_dict.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()

word_idx=tokenizer.word_index

embedding_matrix = np.random.normal(emb_mean, emb_std, (max_words_count, embedding_size))

for word,i in word_idx.items():
    if i < max_words_count:
        vec_temp=embed_dict.get(word)
        if vec_temp is not None:
            embedding_matrix[i]=vec_temp
            
            
            
inp=Input(shape=(max_words_length,))
x=Embedding(max_words_count,embedding_size,weights=[embedding_matrix])(inp)
x=Bidirectional(LSTM(embedding_size,return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
x=GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping, ModelCheckpoint


##file_path="lstm_preprocess_best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
callbacks_list = [checkpoint, early] #early
model.fit(x_train, y, batch_size=32, epochs=2, validation_split=0.1, callbacks=callbacks_list)

y_test = model.predict([x_test], batch_size=1024, verbose=1)
sample_submission = pd.read_csv('./dataset/sample_submission.csv')
sample_submission[classes] = y_test

## path name
sample_submission.to_csv('submission/submission_preprocess_lstm.csv', index=False)


from keras.layers import GRU
inp_1=Input(shape=(max_words_length,))
x_1=Embedding(max_words_count,embedding_size,weights=[embedding_matrix])(inp_1)
x_1=Bidirectional(GRU(embedding_size,return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x_1)
x_1=GlobalMaxPool1D()(x_1)
x_1 = Dense(50, activation="relu")(x_1)
x_1 = Dropout(0.1)(x_1)
x_1 = Dense(6, activation="sigmoid")(x_1)
model1 = Model(inputs=inp_1, outputs=x_1)
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

##file_path="gru_preprocess_best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
callbacks_list = [checkpoint, early] #early
model1.fit(x_train, y, batch_size=32, epochs=2, validation_split=0.1, callbacks=callbacks_list)

y_test_1 = model1.predict([x_test], batch_size=1024, verbose=1)
sample_submission = pd.read_csv('./dataset/sample_submission.csv')
sample_submission[classes] = y_test_1

#path name
sample_submission.to_csv('submission/submission_preprocess_GRU.csv', index=False)

## path name
file_lstm='submission/submission_preprocess_lstm.csv'
file_GRU='submission/submission_preprocess_GRU.csv'
p_lstm = pd.read_csv(file_lstm)
p_gru = pd.read_csv(file_GRU)

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
p_res_avg = p_lstm.copy()
p_res_avg[label_cols] = (p_gru[label_cols] + p_lstm[label_cols]) / 2

#path name
p_res_avg.to_csv('submission_preprocess_lstm+gru_avg.csv', index=False)
