
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import config
import pickle
import text_cleaning
import models
max_length = config.max_length
padding_type = config.padding_type
truncating_type = config.truncating_type

with open("cleaned_questions.pkl", "rb") as f:
    clean_questions = pickle.load(f)

def get_key(val): 
    for key, value in word_index.items():
        if val == value:
            return key

def get_answer(text):
    index_array = []
    for i in text_cleaning.clean_text(text).split(" "):
        index = word_index.get(i)
        if index == None:
            index_array.append(word_index.get('<OOV>'))
        else:
            index_array.append(index)
    index_array = pad_sequences(list([index_array]), padding = padding_type, 
                  truncating= truncating_type, maxlen=max_length)[0]
    dec_text = np.zeros(max_length)
    dec_text[0] = 1
    words = []
    for i in range(max_length):
        val_pred = model.predict([np.array([index_array]), np.array([dec_text])])[0][i]
        
        index_values = val_pred.argsort()[-10:][::-1]
        word = get_key(index_values[0])
        index_val = index_values[0]

        if word == "<OOV>":
            word = "("+get_key(index_values[1])+")"
            index_val = index_values[1]

        if i != max_length-1:
            dec_text[i+1] = index_val
        if word!= None:
            words.append(word)
    return words

with open("word_index_dic.pkl", 'rb') as f:
    word_index = pickle.load(f, encoding='latin1')

model = models.seq2seq()
model.load_weights("model_weights.h5")
print("weights are loaded..")

import random
limit = random.randint(0,len(clean_questions) -5)
for i in range(limit, limit+20):
	text = clean_questions[i]
	print("Question:  {}".format(text))
	print("Answer  :  {}".format(" ".join(get_answer(text))))
	print()