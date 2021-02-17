import collections
import numpy as np

import tensorflow as tf

from tensorflow.keras import layers

class SimpleCritic ():
	def __init__ (self, env):
		self.obs_dim = env.obs_dim
		with tf.name_scope("init_critic"):
			input = layers.Input(shape=(None, env.obs_dim))
			
			obs_ph = input
			if hasattr(env, 'obs_mean'):
				obs_ph = (obs_ph-env.obs_mean)/env.obs_std
			else:
				print("WARNING (critic) : no obs range definded. Proceed with caution", flush=True)
			
			mean = obs_ph
			mean = layers.Dense(1024, activation='relu')(mean)
			mean = layers.Dense(512, activation='relu')(mean)
			mean = tf.squeeze(layers.Dense(1, activation='linear')(mean), axis=[2])
			
		self.model = tf.keras.Model(input, mean, name="critic")
	
	
	def get_weights (self):
		return self.model.get_weights()
		
	def set_weights (self, weights):
		self.model.set_weights(weights)
		

class Critic (SimpleCritic):
	pass