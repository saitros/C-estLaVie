# -*- coding: utf-8 -*-

from gensim.models import word2vec
import pickle
import re
import numpy as np
import tensorflow as tf
from konlpy.tag import Twitter

### 학습된 임베딩 메트릭스 로드

# WIKI 말뭉치 메트릭스
wiki_model = word2vec.Word2Vec.load('./wiki.model') #Path 확인

# 영화 도메인 메트릭스
movie_model = word2vec.Word2Vec.load('./2012movie.model') # Path 확인

### Dictionary Load

def load_dict():
  # movie - Path 확인
  with open('./movie.pickle', 'rb') as f:
    movie = pickle.load(f)

  # actor - Path 확인
  with open('./actor.pickle', 'rb') as f:
    actor = pickle.load(f)

  # director - Path 확인
  with open('./director.pickle', 'rb') as f:
    director = pickle.load(f)

  # genre - Path 확인
  with open('./genre_B.pickle', 'rb') as f:
    genre = pickle.load(f)

  # country - Path 확인
  with open('./country_B.pickle', 'rb') as f:
    country = pickle.load(f)

  dict_list = [movie, actor, director, genre, country]

  return dict_list

# ### 관련 변수 선언 => 향후 Class 안에 변수로 처리할 예정
#
twitter = Twitter()
max_sent_len = 16

# WIKI 말뭉치 관련
wiki_word2idx = {word: i for i, word in enumerate(wiki_model.wv.index2word)}
wiki_matrix = wiki_model.wv.vectors

# 영화 도메인 말뭉치 관련
movie_word2idx = {word: i for i, word in enumerate(movie_model.wv.index2word)}
movie_matrix = movie_model.wv.vectors

# POS 관련
pos2idx = {'Adjective': 5, 'Adverb': 10, 'Alpha': 11, 'Determiner': 6, 'Eomi': 0,
 'Exclamation': 7, 'Foreign': 8, 'Josa': 1, 'Modifier': 9, 'Noun': 13,
 'Number': 4, 'Punctuation': 12, 'Suffix': 3, 'Verb': 14, 'VerbPrefix': 2}

# 개체명 사전 관련
dict_list = load_dict()
year_regex = re.compile('\d+년|[12]\d{3}')
month_regex = re.compile('\d+월')

# 음절 관련 - Path 확인
max_word_len = 12
with open('./syl2idx.pickle', 'rb') as f:
  syl2idx = pickle.load(f)


### 음절 임베딩 관련 LSTM 모델
with tf.variable_scope('syl', reuse=tf.AUTO_REUSE):
  syl_hidden_size = 8
  sent_size = 1
  syl_size = len(syl2idx) # 8159

  syl_X = tf.placeholder(tf.float64, [sent_size, max_word_len, syl_size]) # (1, 12, 8159)

  # Bi-LSTM
  syl_fw_cell = tf.contrib.rnn.LSTMCell(syl_hidden_size)
  syl_bw_cell = tf.contrib.rnn.LSTMCell(syl_hidden_size)
  syl_outputs, syl_states = tf.nn.bidirectional_dynamic_rnn(syl_fw_cell, syl_bw_cell, syl_X, dtype=tf.float64)

### 전체 학습 모델 설계
batch_size = 1
input_n = 238
n_class = 8
hidden_size = 128
lr = 0.01

with tf.variable_scope('total', reuse=tf.AUTO_REUSE):
  X = tf.placeholder(tf.float64, [batch_size, max_sent_len, input_n])
  Y = tf.placeholder(tf.int64, [None, None])

  # LSTM output shape: (batch_size, max_sent_len, hidden_size)
  W = tf.Variable(tf.random_normal([batch_size, hidden_size, n_class], dtype=tf.float64))
  b = tf.Variable(tf.random_normal([n_class], dtype=tf.float64))

  # Bi-LSTM
  fw_cell = tf.contrib.rnn.LSTMCell(hidden_size)
  bw_cell = tf.contrib.rnn.LSTMCell(hidden_size)
  outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, X, dtype=tf.float64)

  # Softmax (향후 CRF로 변경 예정)
  logits = tf.matmul(outputs[1], W) + b
  cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y))
  optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
  prediction = tf.cast(tf.argmax(logits, 2), tf.int64)

# x_train, x_train_pos로 token화 함수
def input_data(x_data):
  x_train = []
  x_train_pos = []

  for sent in x_data:
    x_train.append([word for word, pos in twitter.pos(sent)])
    x_train_pos.append([pos for word, pos in twitter.pos(sent)])

  return (x_train, x_train_pos)
#
# ### 임베딩 함수 5개
#
# 1. WIKI 말뭉치 임베딩 함수
def wiki_embedding(x_train, wiki_word2idx, wiki_matrix, max_sent_len):
  X_train = []
  X_train_vect = []
  max_idx = len(wiki_word2idx)
  wiki_zeros = np.zeros([100])

  for sent in x_train:
    tmp = []
    for word in sent:
      try:
        tmp.append(wiki_word2idx[word])
      except:
        tmp.append(max_idx)
    X_train.append(tmp)

  X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_sent_len, padding='post', value=max_idx)

  for idx_list in X_train:
    tmp = []
    for idx in idx_list:
      try:
        tmp.append(wiki_matrix[idx])
      except:
        tmp.append(wiki_zeros)
    X_train_vect.append(tmp)

  return np.array(X_train_vect)

# 2. 영화 Domain 임베딩 함수
def movie_embedding(x_train, movie_word2idx, movie_matrix, max_sent_len):
  X_train = []
  X_train_vect = []
  max_idx = len(movie_word2idx)
  movie_zeros = np.zeros([100])

  for sent in x_train:
    tmp = []
    for word in sent:
      try:
        tmp.append(movie_word2idx[word])
      except:
        tmp.append(max_idx)
    X_train.append(tmp)

  X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_sent_len, padding='post', value=max_idx)

  for idx_list in X_train:
    tmp = []
    for idx in idx_list:
      try:
        tmp.append(movie_matrix[idx])
      except:
        tmp.append(movie_zeros)
    X_train_vect.append(tmp)

  return np.array(X_train_vect)

# 3. 품사 임베딩 (우선 one-hot 인코딩)
def pos_embedding(x_train_pos, pos2idx, max_sent_len):

  max_idx = len(pos2idx)
  X_train_pos = []
  X_train_pos_vect = []
  pos_matrix = np.eye(max_idx)
  pos_zeros = np.zeros([max_idx])

  for sent in x_train_pos:
    tmp = []
    for pos in sent:
      tmp.append(pos2idx[pos])
    X_train_pos.append(tmp)

  X_train_pos = tf.keras.preprocessing.sequence.pad_sequences(X_train_pos, maxlen=max_sent_len, padding='post', value=max_idx)

  for idx_list in X_train_pos:
    tmp = []
    for idx in idx_list:
      try:
        tmp.append(pos_matrix[idx])
      except:
        tmp.append(pos_zeros)
    X_train_pos_vect.append(tmp)

  return np.array(X_train_pos_vect)

# 4. 개체명 사전 임베딩 (one-hot 인코딩)
def entity_embedding(x_train, dict_list, max_sent_len):

  X_train_vect = []
  entity_zeros = [0] * 7

  for sentence in x_train:
    sentence_idx = []
    for word in sentence:
      word_idx = []
      # Check movie, actor, director, genre, country
      for one_dict in dict_list:
        if word in one_dict:
          word_idx.append(1)
        else:
          word_idx.append(0)
      # Year
      if len(year_regex.findall(word)) == 1:
        word_idx.append(1)
      else:
        word_idx.append(0)
      # Month
      if len(month_regex.findall(word)) == 1:
        word_idx.append(1)
      else:
        word_idx.append(0)

      sentence_idx.append(word_idx)
    while (len(sentence_idx) < max_sent_len):
      sentence_idx.append(entity_zeros)
    X_train_vect.append(sentence_idx)

  return np.array(X_train_vect)

# 5. 음절 임베딩
def syl_embedding(x_train, max_sent_len, max_word_len, syl2idx, syl_hidden_size):

  X_train = []
  X_train_vect = []
  max_idx = len(syl2idx)
  one_hot_matrix = np.eye(max_idx)
  one_hot_zeros = np.zeros([max_idx])
  word_zeros = np.zeros([syl_hidden_size*2])

  sess = tf.Session()
  saver = tf.train.Saver(tf.global_variables(scope='syl'))
  saver.restore(sess, './syl_lstm/syl_lstm.model') # Path 확인

  for sent in x_train:
    tmp = []
    for word in sent:
      tmp_2 = []
      for syl in word:
        try:
          tmp_2.append(syl2idx[syl])
        except:
          tmp_2.append(max_idx)
      tmp.append(tmp_2)
    X_train.append(tmp)

  for sent in X_train:
    sent = tf.keras.preprocessing.sequence.pad_sequences(sent, maxlen=max_word_len, padding='post', value=max_idx)
    sent_vect = []
    for word in sent:
      one_hot_word = []
      x_input = []

      for syl in word:
        try:
          one_hot_word.append(one_hot_matrix[syl])
        except:
          one_hot_word.append(one_hot_zeros)
      x_input.append(one_hot_word)
      syl_states_ = sess.run(syl_states, feed_dict={syl_X: x_input})
      word_vect = np.concatenate([syl_states_[1][0][0], syl_states_[0][0][0]])
      sent_vect.append(word_vect)

    while(len(sent_vect) < max_sent_len):
      sent_vect.append(word_zeros)

    X_train_vect.append(sent_vect)

  sess.close()

  return np.array(X_train_vect)

# ##
# 모듈 import 후, 아래 solution 함수 호출!
# input: String형 문장 , output: {'movie': [], 'actor': [], 'director': [], 'genre': [], 'country': [], 'year': [], 'month': [] }
# ##
def solution(chatbot_input):
  x_sentence = []
  x_sentence.append(chatbot_input)
  x_input, x_input_pos = input_data(x_sentence)
  
  wiki_embedding_ = wiki_embedding(x_input, wiki_word2idx, wiki_matrix, max_sent_len)
  movie_embedding_ = movie_embedding(x_input, movie_word2idx, movie_matrix, max_sent_len)
  pos_embedding_ = pos_embedding(x_input_pos, pos2idx, max_sent_len)
  entity_embedding_ = entity_embedding(x_input, dict_list, max_sent_len)
  syl_embedding_ = syl_embedding(x_input, max_sent_len, max_word_len, syl2idx, syl_hidden_size)
  input_embeddings = np.concatenate([wiki_embedding_, movie_embedding_, pos_embedding_, entity_embedding_, syl_embedding_], axis=-1)

  sess = tf.Session()
  saver = tf.train.Saver(tf.global_variables(scope='total'))
  saver.restore(sess, './total_lstm.model') # Path 확인
  
  output = sess.run(prediction, feed_dict={X: input_embeddings})
  output_dict = {'movie': [], 'actor': [], 'director': [], 'genre': [], 'country': [], 'year': [], 'month': [] }
  output_orders = ['movie', 'actor', 'director', 'genre', 'country', 'year', 'month']
  
  for idx, out in enumerate(output[0]):
    for i in range(7):
      if i == out:
        output_dict[output_orders[i]].append(x_input[0][idx])
  
  sess.close()
  
  return output_dict

print(solution("보헤미안 랩소디 개봉일 알려줘"))