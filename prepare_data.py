
import re
import sys
import time
import json
import pickle
import numpy as np
np.random.seed(1234)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import config
from text_cleaning import clean_text

try:
    clear.clear_cache()
except:
    pass

min_line_length = config.min_line_length
max_length = config.max_length
padding_type = config.padding_type
truncating_type= config.truncating_type
dictionary_length = config.dictionary_length

def embedding(questions, answers):
    tokenizer = Tokenizer(num_words = dictionary_length, oov_token = '<OOV>')
    tokenizer.fit_on_texts(questions + answers)
    
    tokenizer.word_index = {e:i for e,i in tokenizer.word_index.items() if i <= dictionary_length-1}

    word_index = tokenizer.word_index
    word_index['<pad>'] = dictionary_length
    
    q_sequences = tokenizer.texts_to_sequences(questions)
    a_sequences = tokenizer.texts_to_sequences(answers)

    q_padded = pad_sequences(q_sequences, padding = padding_type, truncating= truncating_type, maxlen=max_length)
    a_padded = pad_sequences(a_sequences, padding = padding_type, truncating= truncating_type, maxlen=max_length)

    return q_padded, a_padded, word_index


with open(config.lines_path, 'rb') as f:
  lines = f.read().decode(encoding="ascii", errors="ignore").split("\n")

with open(config.conv_path, 'rb') as f:
  conv_lines = f.read().decode(encoding="ascii", errors="ignore").split("\n")


id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

# Create a list of all of the conversations' lines' ids.
convs = [ ]
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))
    
clean_questions = []
clean_answers = []
print("Cleaning the data.....")
for conv in convs[:int(len(convs))]:
    for i in range(len(conv)-1):
        question = id2line[conv[i]]
        answer = id2line[conv[i+1]]

        q_clean = clean_text(question)
        a_clean = clean_text(answer)
        q_len = len(q_clean.split())
        a_len = len(a_clean.split())
        if  q_len >= min_line_length and q_len <= max_length and a_len >= min_line_length and a_len <= max_length:
            clean_questions.append(q_clean)
            clean_answers.append(a_clean)

vocab = {}
for sentence in np.array(list(clean_answers)+list(clean_questions)):
    for word in sentence.split(" "):
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1
vocab_sort = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1])}

question_padded, answer_padded ,word_index = embedding(clean_questions, clean_answers)
decoder_padded = np.ones((len(answer_padded),len(answer_padded[0])))
decoder_padded[:,1:] = answer_padded[:,:-1]

with open("cleaned_questions.pkl", "wb") as f:
    pickle.dump(clean_questions, f)

with open("cleaned_answers.pkl", "wb") as f:
    pickle.dump(clean_answers, f)


with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab_sort, f)

with open("word_index_dic.pkl", "wb") as f:
    pickle.dump(word_index, f)

with open('dec_labels.pkl', 'wb') as f:
    pickle.dump(decoder_padded, f)

with open('inputs.pkl', 'wb') as f:
    pickle.dump(question_padded, f)

with open('labels.pkl', 'wb') as f:
    pickle.dump(answer_padded, f)
print("completed!")