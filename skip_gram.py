import tensorflow as tf
import jieba
from collections import Counter
import random
import utils
import numpy as np
import pandas as pd
import os

# 分词
input_words = []
f = open('5.5w_vector.txt','r',encoding= 'utf-8')
content = f.read().split()
aftersplit = []
for i in content:
    text_cut = jieba.cut(i)
    string = ' '.join(text_cut)
    aftersplit.append(string)
    str_list = string.split(' ')
    for j in str_list:
        input_words.append(j)

# 词数转换
word2int = {}
set_words = set(input_words)
for i,word in enumerate(set_words):
    word2int[word] = i
    
int2word = dict([(value,key) for key, value in word2int.items()])

# 把句子分成一个一个的类似英文的格式
sentences = []
for sentence in aftersplit:
    sentences.append(sentence.split())

# 构建center/target   
WINDOW_SIZE = 2
data = []
for sentence in sentences:
    for idx, word in enumerate(sentence):
        for neighbor in sentence[max(idx - WINDOW_SIZE, 0) : min(idx + WINDOW_SIZE, len(sentence)) + 1] : 
            if neighbor != word:
                data.append([word, neighbor])

                
for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j]=word2int[data[i][j]]

# batch generation
def get_batch_data(data,batch_size):
    center_batch = []
    target_batch = []
    
    random_index = np.random.randint(0,len(data),batch_size)
    for i in random_index:
        center_batch.append(data[i][0])
        target_batch.append(data[i][1])
    
    
    center_batch = np.array(center_batch)
    target_batch = np.reshape(np.array(target_batch),[batch_size,1])
    
    return center_batch, target_batch


class skipgram(object):
    def __init__(self,vocab_size, batch_size, embed_size, num_sampled, learning_rate,sess):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.lr = learning_rate
        self.build_model()
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.sess = sess
        self.chkpt_dir = os.getcwd()
        self.checkpoint_file = os.path.join(self.chkpt_dir,'nlpwork', 'SKIPGRAM.ckpt')
        self.saver = tf.train.Saver()
        
        
        
    
    def build_model(self):
        with tf.variable_scope('skipgram'):
            self.center_words = tf.placeholder(tf.int32, shape=[None], name="center_words")
            self.target_words = tf.placeholder(tf.int32, shape=[None, 1], name="taget_words")
            self.embedding_matrix = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0), 
                                                name="embedding_matrix")
            
            self.embed = tf.nn.embedding_lookup(self.embedding_matrix, self.center_words, name='embed')
            nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size],
                                                        stddev=1.0 / (self.embed_size ** 0.5)), 
                                                        name='nce_weight')
            nce_bias = tf.Variable(tf.zeros([self.vocab_size]), name='nce_bias')

            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, 
                                                        biases=nce_bias, 
                                                        labels=self.target_words, 
                                                        inputs=self.embed, 
                                                        num_sampled=self.num_sampled, 
                                                        num_classes=self.vocab_size), name='loss')
            
            self.norm = tf.sqrt(tf.reduce_sum(tf.square(self.embedding_matrix), 1, keep_dims = True))
            

    def train(self,inputs,outputs):
        return self.sess.run(self.optimizer, feed_dict = {self.center_words:inputs, self.target_words:outputs})
    
    def save_models(self):
        print("...Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)
        
    def load_models(self):
        print("...Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)
        

def test(model,session):
    model.load_models()
    test_index = np.random.randint(0,len(set_words),8)
    test_words = []
    
    norm = session.run(model.norm)
    matrix = session.run(model.embedding_matrix)
    normalized_mat = matrix/norm
    
    
    for i in test_index:
        test_words.append(list(set_words)[i])
        
    for i in test_words:
        int_ = word2int[i]
        int_ = np.array(int_)
        int_ = np.reshape(int_,[1])
        predicted = session.run(model.embed,feed_dict = {model.center_words:int_})
        similarity = np.matmul(predicted, np.transpose(normalized_mat))
        c = (-similarity).argsort()[0:8]
        for j in range(8):
            close_word = int2word[list(c[0])[j]]
            print(i,close_word)
    
    np.savetxt('embedding.txt',matrix)


def main():
    session = tf.Session()
    VOCAB_SIZE = 50000
    BATCH_SIZE = 128
    EMBED_SIZE = 128 # Dimention of word embedding vector
    SKIP_WINDOW = 1 # The context window
    NUM_SAMPLED = 64 # Number of negative examples to sample
    LEARNING_RATE = 0.001
    NUM_TRAINING_STEP = 1
    SKIP_STEP = 2000
    
    

    model = skipgram(len(set_words),BATCH_SIZE,EMBED_SIZE,NUM_SAMPLED,LEARNING_RATE,session)
    session.run(tf.global_variables_initializer())
    
    for i in range(NUM_TRAINING_STEP):
        centers, targets = get_batch_data(data,BATCH_SIZE)

        
        model.train(centers, targets)
        print(session.run(model.loss, feed_dict = {model.center_words:centers, model.target_words:targets}))
    
    model.save_models()
    
        
    
    test(model,session)
    
        

main()



    
