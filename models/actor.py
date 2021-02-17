import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class BaseActor ():
	def __init__ (self):
		self.save_path = "will not work"
		self.logstd = tf.Variable(np.ones((self.act_dim,))*(-3), dtype=tf.float32, trainable=True)
		
	def get_weights (self):
		return (self.model.get_weights(), self.logstd.value())
		
	def set_weights (self, data):
		weights, logstd_value = data
		self.model.set_weights(weights)
		self.logstd.assign(logstd_value)
		
	def save (self, path):
		self.model.save_weights(path.format("actor"), overwrite=True)
		
	def load (self, path):
		self.model.load_weights(path.format("actor"))

#inp_dim should be kept to None in the training phase to allow for variable batch size, but tf needs us to set it to a fixed value when creating the lite model.
class SimpleActor (BaseActor):
	def __init__ (self, obs_dim, act_dim, obs_mean=None, obs_std=None, blindfold=None, inp_dim=None):
		
		self.obs_dim = obs_dim
		self.act_dim = act_dim
		
		super().__init__()
		
		activation = "relu"
		
		obs_input = layers.Input(shape=(inp_dim, obs_dim))
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
			print("WARNING (actor) : no blindfold available")
			visible_obs = obs_ph
			hidden_obs = obs_ph
		
		# creating the network
		self.action_layers = []
		
		self.action_layers.append(layers.Dense(256, activation=activation))
		self.action_layers.append(layers.Dense(128, activation=activation))
		self.action_layers.append(layers.Dense(64, activation=activation))
		
		full_repr = obs_ph
		for layer in self.action_layers:
			full_repr = layer(full_repr)
		
		skip = obs_ph
		
		last_layer = layers.Dense(act_dim, activation='sigmoid')
		self.action_layers.append(last_layer)
		action = last_layer(tf.concat((full_repr, skip), axis=-1))
		
		self.model = tf.keras.Model(obs_input, action, name="actor_model")
		
		last_layer.set_weights([x/10 for x in last_layer.get_weights()])
	
	
	
	
	
	
	
	
class LSTMActor (BaseActor): # not tested, should not be used...
	def __init__ (self, obs_dim, act_dim, obs_mean=None, obs_std=None, inp_dim=None):
		super().__init__(env)
		
		self.use_blindfold = use_blindfold
		self.use_lstm = use_lstm
		activation = "relu"
		
		self.lstm_size=128
		self.hidden_size = 64
		
		with tf.name_scope("core_model"):
			obs_input = layers.Input(shape=(inp_dim, env.obs_dim))
			obs_ph = (obs_input-env.obs_mean)/env.obs_std
			init_state = (layers.Input(shape=(self.lstm_size, )), layers.Input(shape=(self.lstm_size, )))
			
			if hasattr(env, 'blindfold'):
				visible_obs = env.blindfold.select_visible(obs_ph)
				hidden_obs = env.blindfold.select_hidden(obs_ph)
			else:
				print("WARNING (actor) : no blindfold available")
				visible_obs = obs_ph
				hidden_obs = obs_ph
				
			if self.use_blindfold and self.use_lstm:
				hidden_repr = visible_obs
				hidden_repr = layers.Dense(128, activation='relu')(hidden_repr)
				hidden_repr, *end_state = layers.LSTM(self.lstm_size, time_major=False, return_sequences=True, return_state=True)(hidden_repr, initial_state=init_state)
				hidden_repr = layers.Dense(self.hidden_size, activation="relu")(hidden_repr)
				
				full_repr_inp = tf.concat((hidden_repr, visible_obs), axis=-1)
				self.hiddden_repr = hidden_repr
			
			elif not self.use_blindfold:
				hidden_repr = hidden_obs
				hidden_repr = layers.Dense(128, activation='relu')(hidden_repr)
				hidden_repr = layers.Dense(self.hidden_size, activation="relu")(hidden_repr)
				end_state = init_state
				
				full_repr_inp = tf.concat((hidden_repr, visible_obs), axis=-1)
				self.hiddden_repr = hidden_repr
			
			else:
				end_state = init_state
				"""
				shape = tf.shape(visible_obs)
				print(shape)
				if inp_dim is None:
					hidden_repr = tf.zeros([shape[0], inp_dim, 16])
				else:
					hidden_repr = tf.zeros([shape[0], inp_dim, 16])
				full_repr_inp = tf.concat((hidden_repr, visible_obs), axis=-1)
				"""
				paddings = tf.constant([[0, 0], [0, 0], [self.hidden_size, 0]])
				full_repr_inp = tf.pad(visible_obs, paddings, "CONSTANT")
			
			self.action_layers = []
			self.action_layers.append(layers.Dense(256, activation=activation))
			self.action_layers.append(layers.Dense(128, activation=activation))
			self.action_layers.append(layers.Dense(64, activation=activation))
			
			full_repr = full_repr_inp
			for layer in self.action_layers:
				full_repr = layer(full_repr)
			
			skip = full_repr_inp
			
			last_layer = layers.Dense(self.act_dim, activation='tanh')
			self.action_layers.append(last_layer)
			action = (last_layer(tf.concat((full_repr, skip), axis=-1))+1)/2
			
			self.core_model = tf.keras.Model((obs_input, init_state), (action, end_state), name="actor_core_model")
			if (not self.use_blindfold) or self.use_lstm:
				print(self.use_blindfold, self.use_lstm)
				self.repr_model = tf.keras.Model((obs_input, init_state), self.hiddden_repr, name="repr_model")
			#self.action_model = tf.keras.Model(full_repr_inp, action, name="actor_action_model")
		
		
		self.model = self.core_model # tf.keras.Model((input, main_init_state), self.core_model(obs_ph), name="actor_model")
		if self.use_blindfold and False:
			self.core_model.summary()
		
		last_layer.set_weights([x/10 for x in last_layer.get_weights()])
		
	def get_init_state(self, n_env):
		init_state_shape = (n_env, self.lstm_size)
		return (np.zeros(init_state_shape), np.zeros(init_state_shape))
	