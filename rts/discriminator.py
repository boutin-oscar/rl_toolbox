import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class BaseDiscriminator ():
	def __init__ (self):
		self.save_path = "will not work"
		
	def get_weights (self):
		return self.model.get_weights()
		
	def set_weights (self, data):
		weights = data
		self.model.set_weights(weights)
		
	def save (self, path):
		self.model.save_weights(path.format("model"), overwrite=True)
		
	def load (self, path):
		self.model.load_weights(path.format("model"))


class Discriminator (BaseDiscriminator):
	def __init__ (self, obs_dim, obs_mean=None, obs_std=None, blindfold=None):
		
		self.obs_dim = obs_dim
		
		super().__init__()
		
		activation = "relu"
		
		obs_input = layers.Input(shape=(obs_dim, ))
		obs_ph = obs_input
		
		# scaling the input if mean and std are provided
		if obs_mean is not None:
			obs_ph = obs_ph - obs_mean
		if obs_std is not None:
			obs_ph = obs_ph / obs_std
		
		# hiding some inputs if a blindfold is provided
		if blindfold is not None:
			visible_obs = blindfold.select_visible(obs_ph)
			hidden_obs = blindfold.select_hidden(obs_ph)
		else:
			visible_obs = obs_ph
			hidden_obs = obs_ph
		
		
		mean = visible_obs
		mean = layers.Dense(1024, activation='relu')(mean)
		mean = layers.Dense(512, activation='relu')(mean)
		mean = layers.Dense(2, activation='linear')(mean)
		
		self.model = tf.keras.Model(obs_input, mean, name="discriminator")
		