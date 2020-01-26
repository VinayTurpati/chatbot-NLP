# with open("vocab.pkl", 'rb') as f:
#     vocab_sort = pickle.load(f, encoding='latin1')
import re
import os
import sys
import json
import time
import string
import pickle
import models
import statistics
from keras.models import load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.losses import categorical_crossentropy
import numpy as np
np.random.seed(1234)
import warnings
warnings.filterwarnings("ignore") #ignore warnings
import config

with open("word_index_dic.pkl",'rb') as f:
    word_index = pickle.load(f, encoding = 'latin1')

with open("dec_labels.pkl", 'rb') as f:
    decoder_padded = pickle.load(f, encoding = 'latin1')

with open('inputs.pkl', 'rb') as f:
    question_padded = pickle.load(f, encoding = 'latin1')

with open('labels.pkl', 'rb') as f:
    answer_padded = pickle.load(f, encoding = 'latin1')

tic = time.time()
t = c = 0
batch_time = 0
epoch = config.epoch
step = config.step
batch_size = config.batch_size
dict_size = config.dictionary_length
loss_array = []
check_point_loss = 9999

model = models.seq2seq()

if os.path.isfile("model_weights.h5"):
    try:
        model.load_weights("model_weights.h5")
        print("Loaded previously trained weights!")
    except:
        pass

try:
    epoch = int(sys.argv[1])
except:
    pass

loss = []
print("Number of Batches for each epoch: {}".format(int(len(question_padded)/step)))

for k in range(epoch):
    print('-'*80)
    print('Epoch {}/{}'.format(k,epoch))
    print('-'*80)
    batch = 0
    tim = time.time()
    temp_loss = []

    for i in range(0, len(answer_padded),step):
        tic = time.time()
        batch += 1
        
        x_encoder = question_padded[i:i+step]     #create_input_sequence(q_padded[i:i+step],index_vec)
        x_decoder = decoder_padded[i:i+step]  #create_input_sequence(dec_padded[i:i+step],index_vec)
        y_train = np.eye(dict_size)[answer_padded[i:i+step].astype('int')]     #one_hot_encoding(a_padded[i:i+step])
        
        history = model.fit([x_encoder,x_decoder], y_train, batch_size=batch_size,epochs=1, verbose = 0)
        batch_time += time.time() - tic
        temp_loss.append(history.history['loss'][-1])
        
        if batch%1 == 0:
            print((" "*5+"Batch {}/{} ----- Loss: {:.5f} ------ Time: {:.3f}s").format(batch,int(len(answer_padded)/step),history.history['loss'][-1], batch_time))
            batch_time = 0


    epoch_loss = statistics.mean(temp_loss)

    if check_point_loss > epoch_loss:
        check_point_loss = epoch_loss
        print("Loss for epoch: ", epoch_loss)
        print("Model weights is saving to model_weights.h5")
        model.save_weights('model_weights.h5')

    loss.append(epoch_loss)
    loss_array.append(epoch_loss)

    print((" "*38+"Time for Epoch: {:.3f}s").format(time.time()-tim))

plt.plot(loss)
