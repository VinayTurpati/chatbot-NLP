import tensorflow as tf
import keras.backend as K
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.optimizers import Adam, SGD
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.layers.merge import concatenate
from keras.utils.vis_utils import plot_model
from keras.layers import Concatenate,Embedding
from tensorflow.keras.layers import Attention,Bidirectional
from keras.layers import Activation, Dense, Dropout, merge, LSTM, dot, TimeDistributed
from keras.layers import Input, Embedding, RepeatVector, Bidirectional
import config

def seq2seq():
	num_encoder_tokens = config.dictionary_length
	num_decoder_tokens = config.dictionary_length
	dict_size = config.dictionary_length
	dictionary_length = config.dictionary_length
	INPUT_LENGTH = config.max_length
	OUTPUT_LENGTH = config.max_length
	
	encoder_inputs = Input(shape=(INPUT_LENGTH,))
	decoder_inputs = Input(shape=(OUTPUT_LENGTH,))
	
	encoder = Embedding(dict_size, 128, input_length=INPUT_LENGTH, mask_zero=True)(encoder_inputs)
	encoder = LSTM(512, return_sequences=True)(encoder)
	encoder_last = encoder[:,-1,:]
	
	decoder = Embedding(dict_size, 128, input_length=OUTPUT_LENGTH, mask_zero=True)(decoder_inputs)
	decoder = LSTM(512, return_sequences=True)(decoder, initial_state=[encoder_last, encoder_last])
	
	
	attention = dot([decoder, encoder], axes=[2, 2])
	attention = Activation('softmax', name='attention')(attention)
	
	context = dot([attention, encoder], axes=[2,1])
	
	decoder_combined_context = concatenate([context, decoder])
	
	output = TimeDistributed(Dense(512, activation="tanh"))(decoder_combined_context)
	output = TimeDistributed(Dense(dictionary_length, activation="softmax"))(output)
	
	model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=[output])
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
	return model
#plot_model(model, show_shapes=True, show_layer_names=True)