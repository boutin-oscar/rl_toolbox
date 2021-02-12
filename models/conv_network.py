import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def enc_block (inp, features_nb):
	x = inp
	x = layers.TimeDistributed(layers.MaxPool2D())(x)
	x = layers.TimeDistributed(layers.Conv2D(features_nb, 3, activation='relu', padding='same'))(x)
	x = layers.TimeDistributed(layers.Conv2D(features_nb, 3, activation='relu', padding='same'))(x)
	return x

def dec_block (inp, shortcut, features_nb):
	x = layers.TimeDistributed(layers.UpSampling2D())(inp)
	x = layers.concatenate([x, shortcut], axis=-1)
	x = layers.TimeDistributed(layers.Conv2D(features_nb, 3, activation='relu', padding='same'))(x)
	x = layers.TimeDistributed(layers.Conv2D(features_nb, 3, activation='relu', padding='same'))(x)
	return x
	
from models.actor import BaseActor

class ConvActor (BaseActor):
	def __init__ (self, env, inp_dim=None):
		super().__init__(env)
		
		with tf.name_scope("input_process"):
			input = layers.Input(shape=(inp_dim, env.obs_dim))
			obs_ph = input
			
		with tf.name_scope("core_model"):
			obs_input = layers.Input(shape=(inp_dim, obs_ph.shape[-1]))
			mean = layers.Reshape(env.obs_shape)(obs_input)
			
			mask = mean# mean[:,:,:,:,0]
			
			first_features_nb = 8 # 64
			all_enc_features_nb = [32, 64] #, 128, 256] # [128, 256, 512, 1024]
			all_dec_features_nb = [32, 16] # [128, 64,  # [512, 256, 128, 64]
			out_features_nb = 5

			# --- first block ---
			x = mean
			l = layers.TimeDistributed(layers.Conv2D(first_features_nb, 3, activation='relu', padding='same'))(x)
			x = layers.TimeDistributed(layers.Conv2D(first_features_nb, 3, activation='relu', padding='same'))(x)

			# --- encoder blocks ---
			all_shortcuts = []
			for features_nb in all_enc_features_nb:
				all_shortcuts.append(x)
				x = enc_block(x, features_nb)

			# --- decoder blocks ---
			for shortcut, features_nb in zip(all_shortcuts[::-1], all_dec_features_nb):
				x = dec_block(x, shortcut, features_nb)

			# --- output ---
			last_layer = layers.TimeDistributed(layers.Conv2D(out_features_nb, 1, activation='softmax', padding='same'))
			x = last_layer(x)
			x = x / tf.reduce_sum(x, axis=-1, keepdims=True)
			action = layers.Reshape(env.act_shape)(x)
			
	
			self.core_model = tf.keras.Model((obs_input, ()), (action, ()), name="actor_core_model")
		
		
		self.model = tf.keras.Model((input, ()), (self.core_model(obs_ph)[0], ()), name="actor_model")
		self.test_model = tf.keras.Model(obs_input, mask, name="actor_model")
		#self.core_model.summary()
		
		last_layer.set_weights([x/10 for x in last_layer.get_weights()])
		
	def get_init_state(self, n_env):
		#init_state_shape = (n_env, self.lstm_size)
		return () #(np.zeros(init_state_shape), np.zeros(init_state_shape))



class ConvCritic ():
	def __init__ (self, env):
		self.obs_dim = env.obs_dim
		self.lstm_size = 128
		with tf.name_scope("init_critic"):
			input = layers.Input(shape=(None, env.obs_dim))
			init_state = [layers.Input(shape=(self.lstm_size, )), layers.Input(shape=(self.lstm_size, ))]
			
			first_features_nb = 16 # 64
			all_enc_features_nb = [32, 64] #, 128, 256] # [128, 256, 512, 1024]
			all_dec_features_nb = [32, 16] # [128, 64,  # [512, 256, 128, 64]
			out_features_nb = 5

			# --- first block ---
			#x = tf.reshape(input, tf.convert_to_tensor([tf.shape(input)[0], env.state.limits.width, env.state.limits.height, 4]))
			x = tf.reshape(input, [tf.shape(input)[0], tf.shape(input)[1], env.state.limits.width, env.state.limits.height, 4])
			x = layers.TimeDistributed(layers.Conv2D(first_features_nb, 3, activation='relu', padding='same'))(x)
			x = layers.TimeDistributed(layers.Conv2D(first_features_nb, 3, activation='relu', padding='same'))(x)
			
			# --- encoder blocks ---
			all_shortcuts = []
			for features_nb in all_enc_features_nb:
				all_shortcuts.append(x)
				x = enc_block(x, features_nb)
			
			"""
			# --- decoder blocks ---
			for shortcut, features_nb in zip(all_shortcuts[::-1], all_dec_features_nb):
				x = dec_block(x, shortcut, features_nb)
			"""
			# --- output ---
			"""
			x = tf.reshape(x, [tf.shape(input)[0], tf.shape(input)[1], env.state.limits.width * env.state.limits.height * 4]) # first_features_nb])
			x = layers.Dense(1024, activation='relu')(x)
			"""
			x = tf.reduce_mean(layers.Dense(1, activation='linear')(x), axis=[1,2,3])
			#x = tf.squeeze(layers.Dense(1, activation='linear')(x), axis=[2])
			"""
			mean = input
			mean = layers.Dense(1024, activation='relu')(mean)
			mean = layers.Dense(512, activation='relu')(mean)
			#mean = layers.Dense(256, activation='relu')(mean)
			lstm, *end_state = layers.LSTM(self.lstm_size, return_sequences=True, return_state=True)(mean, initial_state=init_state)
			#mean = layers.concatenate([mean, lstm])
			mean = tf.squeeze(layers.Dense(1, activation='linear')(mean), axis=[2])
			"""
		self.model = tf.keras.Model((input, ()), (x, ()), name="critic")
		#self.model.summary()
		#self.model = tf.keras.Model((obs_ph, init_state), (mean, end_state), name="critic")
		
	def get_init_state(self, n_env):
		init_state_shape = (n_env, self.lstm_size)
		return (np.zeros(init_state_shape), np.zeros(init_state_shape))
	
	def get_weights (self):
		return self.model.get_weights()
		
	def set_weights (self, weights):
		self.model.set_weights(weights)
		