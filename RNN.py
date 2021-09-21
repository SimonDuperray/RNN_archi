import collections, pandas as pd, numpy as np, tensorflow as tf, os, json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Dropout, LSTM
from keras.layers.embeddings import Embedding
from tensorflow.keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.callbacks import CSVLogger

class RNN_model:
   
   def __init__(self):
      self.in_data = None
      self.out_data = None
      self.preproc_in = None
      self.preproc_out = None
      self.in_tk = None
      self.out_tk = None
      self.model = None
      self.df_length = 0,
      self.percentage_true_neg = 0,

   def set_percent(self, perc):
      self.percentage_true_neg = perc

   def get_infos(self):
      in_words_counter = collections.Counter([word for sentence in self.in_data for word in sentence.split()])
      out_words_counter = collections.Counter([word for sentence in self.out_data for word in sentence.split()])

      self.preproc_in, self.preproc_out, self.in_tk, self.out_tk = self.preprocess(self.in_data, self.out_data)
      max_in_length = self.preproc_in.shape[1]
      max_out_length = self.preproc_out.shape[1]
      in_vocab_size = len(self.in_tk.word_index)
      out_vocab_size = len(self.out_tk.word_index)

      # in data
      print("Input data:")
      print(f"Total words: {len([word for sentence in self.in_data for word in sentence.split()])}")
      print(f"Unique words: {len(in_words_counter)}")
      # print(f"10 most commun words: {'" "'.join(list(zip(*in_words_counter.most_common(10))))}")
      print(f"Max input sentence length: {max_in_length}")
      print(f"Input vocabulary size: {in_vocab_size}")

      # out data
      print("Output data:")
      print(f"Total words: {len([word for sentence in self.out_data for word in sentence.split()])}")
      print(f"Unique words: {len(out_words_counter)}")
      # print(f"10 most commun words: {'" "'.join(list(zip(*out_words_counter.most_common(10))))}")
      print(f"Max output sentence length: {max_out_length}")
      print(f"Output vocabulary size: {out_vocab_size}")

   def tokenize(self, x):
      x_tk = Tokenizer()
      x_tk.fit_on_texts(x)
      return x_tk.texts_to_sequences(x), x_tk

   def pad(self, x, length=None):
      if length is None:
         length = max([len(sentence) for sentence in x])
      return pad_sequences(x, maxlen=length, padding='post', truncating='post')

   def preprocess(self, x, y):
      preprocess_x, x_tk = self.tokenize(x)
      preprocess_y, y_tk = self.tokenize(y)

      preprocess_x = self.pad(preprocess_x)
      preprocess_y = self.pad(preprocess_y)

      preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

      return preprocess_x, preprocess_y, x_tk, y_tk

   def logits_to_text(self, logits, tokenizer):
      index_to_words = {id: word for word, id in tokenizer.word_index.items()}
      index_to_words[0] = '<PAD>'
      return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

   def embed_model(self, input_shape, output_sequence_length, in_vocab_size, out_vocab_size):
      learning_rate = 0.001
      model = Sequential()
      model.add(Embedding(in_vocab_size, 100, input_length=input_shape[1], input_shape=input_shape[1:]))
      model.add(GRU(128, return_sequences=True))
      model.add(Dropout(0.5))
      model.add(GRU(128,  return_sequences=True))
      model.add(Dropout(0.5))
      model.add(TimeDistributed(Dense(256, activation='relu')))
      model.add(Dropout(0.5))
      model.add(TimeDistributed(Dense(out_vocab_size, activation='softmax'))) 

      model.compile(
         loss=sparse_categorical_crossentropy,
         optimizer=Adam(learning_rate),
         metrics=['accuracy']
      )
   
      self.model = model

      return model

   def model_summary(self):
      return self.model.summary()
   
   def run(self, nb_epochs, batch_size, validation_split):

      # read csv
      df = pd.read_csv('./datasets/dataset.csv')
      in_data = [item for item in df['in']]
      out_data = [item for item in df['out']]

      # store data in self object
      self.in_data = in_data
      self.out_data = out_data

      # preproc_in/out
      self.get_infos()

      # store df length
      if len(self.in_data)==len(self.out_data):
         self.df_length = len(self.in_data)
      else:
         self.df_length = max(len(self.in_data), len(self.out_data))

      # log message
      print(">>> Dataset loaded !")

      # rnn
      tmp_x = self.pad(self.preproc_in, self.preproc_out.shape[1])
      tmp_x = tmp_x.reshape((-1, self.preproc_out.shape[-2]))

      embed_rnn_model = self.embed_model(
         tmp_x.shape,
         self.preproc_out.shape[1],
         len(self.in_tk.word_index)+1,
         len(self.out_tk.word_index)+1
      )

      new_dict = {}

      history_const = embed_rnn_model.fit(
         tmp_x, 
         self.preproc_out, 
         batch_size=batch_size, 
         epochs=nb_epochs,
         validation_split=validation_split
      )

      history_const.history['epochs'] = nb_epochs
      history_const.history['df_length'] = self.df_length
      history_const.history['percentage_true_neg'] = self.percentage_true_neg
      history_const.history['batch_size'] = batch_size

      dir = "ep"+str(nb_epochs)+"-dflgt"+str(self.df_length)+"-perc"+str(int(self.percentage_true_neg))+"-btchsz"+str(batch_size)
      path = "histories/"+dir

      if not os.path.exists(path):
         os.mkdir(path)

      embed_rnn_model.save(path+"/m--"+dir+".h5")

      history_const.history['prediction'] = self.logits_to_text(embed_rnn_model.predict(tmp_x[:1])[0], self.out_tk)

      filename = path+"/f--"+dir+".json"

      # save history
      with open(filename, 'w') as outfile:
         json.dump(history_const.history, outfile, indent=3)