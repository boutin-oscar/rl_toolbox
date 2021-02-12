import collections
import numpy as np

import tensorflow as tf

from tensorflow.keras import layers

class SimpleCritic ():
	def __init__ (self, env):
		self.obs_dim = env.obs_dim
		self.lstm_size = 256
		with tf.name_scope("init_critic"):
			input = layers.Input(shape=(None, env.obs_dim))
			init_state = [layers.Input(shape=(self.lstm_size, )), layers.Input(shape=(self.lstm_size, ))]
			
			obs_ph = input
			if hasattr(env, 'obs_mean'):
				obs_ph = (obs_ph-env.obs_mean)/env.obs_std
			else:
				print("WARNING (critic) : no obs range definded. Proceed with caution")
			
			mean = obs_ph
			mean = layers.Dense(1024, activation='relu')(mean)
			mean = layers.Dense(512, activation='relu')(mean)
			#mean = layers.Dense(256, activation='relu')(mean)
			lstm, *end_state = layers.LSTM(self.lstm_size, return_sequences=True, return_state=True)(mean, initial_state=init_state)
			#mean = layers.concatenate([mean, lstm])
			mean = tf.squeeze(layers.Dense(1, activation='linear')(mean), axis=[2])
			"""
			lstm, *end_state = layers.LSTM(self.lstm_size, return_sequences=True, return_state=True)(mean, initial_state=init_state)
			mean = mean = tf.squeeze(layers.Dense(1, activation='linear')(lstm), axis=[2])
			"""
		self.model = tf.keras.Model((input, ()), (mean, ()), name="critic")
		#self.model.summary()
		#self.model = tf.keras.Model((obs_ph, init_state), (mean, end_state), name="critic")
		
	def get_init_state(self, n_env):
		init_state_shape = (n_env, self.lstm_size)
		return (np.zeros(init_state_shape), np.zeros(init_state_shape))
	
	def get_weights (self):
		return self.model.get_weights()
		
	def set_weights (self, weights):
		self.model.set_weights(weights)
		
class RnnCritic ():
	def __init__ (self, env):
		self.obs_dim = env.obs_dim
		self.lstm_size = 256
		with tf.name_scope("init_critic"):
			input = layers.Input(shape=(None, env.obs_dim))
			init_state = (layers.Input(shape=(self.lstm_size, )), layers.Input(shape=(self.lstm_size, )))
			
			obs_ph = input
			if hasattr(env, 'obs_mean'):
				obs_ph = (obs_ph-env.obs_mean)/env.obs_std
			else:
				print("WARNING (critic) : no obs range definded. Proceed with caution")
				
			mean = layers.Dense(1024, activation='relu')(obs_ph)
			lstm, *end_state = layers.LSTM(self.lstm_size, return_sequences=True, return_state=True)(mean, initial_state=init_state)
			mean = tf.squeeze(layers.Dense(1, activation='linear')(lstm), axis=[2])
			
		#self.model = tf.keras.Model((input, ()), (mean, ()), name="critic")
		self.model = tf.keras.Model((input, init_state), (mean, end_state), name="critic")
		#self.model.summary()
		
	def get_init_state(self, n_env):
		init_state_shape = (n_env, self.lstm_size)
		return (np.zeros(init_state_shape, dtype=np.float32), np.zeros(init_state_shape, dtype=np.float32))
	
	def get_weights (self):
		return self.model.get_weights()
		
	def set_weights (self, weights):
		self.model.set_weights(weights)
		

class Critic (SimpleCritic):
	pass