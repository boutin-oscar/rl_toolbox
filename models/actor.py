import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class ObsScaler:
	def __init__ (self, env):
		self.mean = np.ones(shape=(env.obs_dim,)) * 0.
		self.std = np.ones(shape=(env.obs_dim,)) * 1.
		self.is_init = True # False
		
		self.lamb = 0.99
		self.beta = 1
	
	def update (self, obs):
		"""
		#M = obs.max(axis=(0, 1))
		#m = obs.min(axis=(0, 1))
		#mean_hat = (M+m)/2 #np.mean(obs, axis=(0, 1))
		#std_hat = M-m #np.std(obs, axis=(0, 1))
		mean_hat = np.mean(obs, axis=(0, 1))
		std_hat = np.std(obs, axis=(0, 1))
		
		
		mean_lamb = self.lamb
		std_lamb = self.lamb
		if not self.is_init:
			#self.mean = mean_hat
			pass
			
		self.mean = mean_lamb * self.mean + (1-mean_lamb) * mean_hat
		self.std = std_lamb * self.std + (1-std_lamb) * std_hat
		self.is_init = True
		
		#self.std = np.maximum(self.std, std_hat)
		"""
		
		"""
		if self.is_init:
			cur_M = self.mean+self.std
			cur_m = self.mean-self.std
			
			M = obs.max(axis=(0, 1))
			m = obs.min(axis=(0, 1))
			
			new_M = np.maximum(M, cur_M)
			new_m = np.minimum(m, cur_m)
			
			self.mean = (new_M+new_m)/2
			self.std = (new_M-new_m)/2
		else:
			self.is_init = True
			
			M = obs.max(axis=(0, 1))
			m = obs.min(axis=(0, 1))
			
			self.mean = (M+m)/2
			self.std = (M-m)/2
		"""
		
	def scale_obs (self, obs):
		return (np.asarray(obs)).astype(np.float32)
		"""
		if self.is_init:
			return ((obs - self.mean)/(self.std + 1e-7)).astype(np.float32)
		else:
			return (np.asarray(obs)).astype(np.float32)
		"""
	
	def save (self, path):
		np.save(path.format("scaler_mean") + ".npy", self.mean)
		np.save(path.format("scaler_std") + ".npy", self.std)
	
	def load (self, path):
		self.mean = np.load (path.format("scaler_mean") + ".npy")
		self.std = np.load (path.format("scaler_std") + ".npy")
		self.is_init = True
		#print("scaler not loaded", flush=True)
		
	def get_weights (self):
		return (self.mean, self.std)
	
	def set_weights (self, data):
		self.is_init = True
		self.mean, self.std = data

class BaseActor ():
	def __init__ (self, env):
		self.save_path = "will not work"
		self.scaler = ObsScaler (env)
		self.act_dim = env.act_dim
		self.obs_dim = env.obs_dim
		
		self.logstd = tf.Variable(np.ones((self.act_dim,))*(-3), dtype=tf.float32, trainable=True)
		
	def get_weights (self):
		return (self.core_model.get_weights(), self.logstd.value(), self.scaler.get_weights())
		
	def set_weights (self, data):
		weights, logstd_value, scaler_data = data
		self.core_model.set_weights(weights)
		self.logstd.assign(logstd_value)
		self.scaler.set_weights(scaler_data)
		
	def save (self, path):
		self.core_model.save_weights(path.format("actor"), overwrite=True)
		self.scaler.save(path)
		
	def load (self, path):
		self.core_model.load_weights(path.format("actor"))
		self.scaler.load(path)

class SimpleActor (BaseActor):
	def __init__ (self, env, use_blindfold, use_lstm, inp_dim=None): # if use_blindeflod then we use a lstm to estimate the hidden state
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
			print(inp_dim, visible_obs.shape)
				
			if self.use_blindfold and self.use_lstm:
				print("lstm")
				hidden_repr = visible_obs
				hidden_repr = layers.Dense(128, activation='relu')(hidden_repr)
				hidden_repr, *end_state = layers.LSTM(self.lstm_size, time_major=False, return_sequences=True, return_state=True)(hidden_repr, initial_state=init_state)
				hidden_repr = layers.Dense(self.hidden_size, activation="relu")(hidden_repr)
				
				full_repr_inp = tf.concat((hidden_repr, visible_obs), axis=-1)
				self.hiddden_repr = hidden_repr
			
			elif not self.use_blindfold:
				print("no blind")
				hidden_repr = hidden_obs
				hidden_repr = layers.Dense(128, activation='relu')(hidden_repr)
				hidden_repr = layers.Dense(self.hidden_size, activation="relu")(hidden_repr)
				end_state = init_state
				
				full_repr_inp = tf.concat((hidden_repr, visible_obs), axis=-1)
				self.hiddden_repr = hidden_repr
			
			else:
				print("nothing")
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
	
class SimpleActor_old (BaseActor):
	def __init__ (self, env, use_blindfold, inp_dim=None):
		super().__init__(env)
		
		self.use_blindfold = use_blindfold
		activation = "relu"
		
		with tf.name_scope("input_process"):
			input = layers.Input(shape=(inp_dim, env.obs_dim))
			obs_ph = (input-env.obs_mean)/env.obs_std
			
			# using the optional blindfold
			if self.use_blindfold:
				if hasattr(env, 'blindfold'):
					obs_ph = env.blindfold.action_blindfold(obs_ph)
				else:
					print("WARNING (actor) : no blindfold available")
			else:
				print("WARNING (actor) : blindfold not used")
			
		with tf.name_scope("core_model"):
			obs_input = layers.Input(shape=(inp_dim, obs_ph.shape[-1]))
			"""
			mean = layers.Dense(first_size, activation=activation)(obs_input)
			mean = layers.Dense(secound_size, activation=activation)(mean)
			"""
			mean = layers.Dense(256, activation=activation)(obs_input)
			mean = layers.Dense(128, activation=activation)(mean)
			mean = layers.Dense(64, activation=activation)(mean)
			
			skip = obs_input
			
			last_layer = layers.Dense(self.act_dim, activation='tanh')
			action = (last_layer(tf.concat((mean, skip), axis=-1))+1)/2
			
			self.core_model = tf.keras.Model((obs_input, ()), (action, ()), name="actor_core_model")
		
		
		self.model = tf.keras.Model((input, ()), (self.core_model(obs_ph)[0], ()), name="actor_model")
		#self.core_model.summary()
		
		last_layer.set_weights([x/10 for x in last_layer.get_weights()])
		
	def get_init_state(self, n_env):
		#init_state_shape = (n_env, self.lstm_size)
		return () #(np.zeros(init_state_shape), np.zeros(init_state_shape))

class MixtureOfExpert (BaseActor):
	def __init__ (self, env, primitives, debug=False):
		super().__init__(env)
		
		self.primitive_nb = len(primitives)
		
		with tf.name_scope("input_process"):
			input = layers.Input(shape=(None, env.obs_dim))
			obs_ph = input
			
			# scaling the inputs
			if hasattr(env, 'obs_mean') and  hasattr(env, 'obs_std'):
				obs_ph = (obs_ph-env.obs_mean)/(env.obs_std+1e-7)
			else:
				print("WARNING (actor) : no obs range definded. Proceed with caution")
				
			# using the optional blindfold
			if hasattr(env, 'blindfold'):
				obs_ph = env.blindfold.action_blindfold(obs_ph)
			else:
				print("WARNING (actor) : no blindfold used")
			
		with tf.name_scope("core_model"):
			obs_input = layers.Input(shape=(None, obs_ph.shape[-1]))
			
			# influence
			influence = layers.Dense(512, activation='relu')(obs_input)
			influence = layers.Dense(256, activation='relu')(influence)
			influence = layers.Dense(self.primitive_nb, activation='softmax')(influence)
			influence = tf.expand_dims(influence, axis=2)
			
			# primitives
			self.primitives_cpy = []
			for i, prim in enumerate(primitives):
				#prim.name = str(i) + "coucou"
				self.primitives_cpy.append(tf.keras.Model(inputs=prim.core_model.input, outputs=prim.core_model.output, name='primitive_'+str(i)))
			
			for prim in self.primitives_cpy:
				for layer in prim.layers:
					layer.trainable = True
	
			
			primitives_actions = [prim(obs_input)[0] for prim in self.primitives_cpy]
			primitives_actions = tf.stack(primitives_actions, axis=3)
			
			# action
			action = tf.reduce_sum(primitives_actions*influence, axis=3)
			
			self.core_model = tf.keras.Model((obs_input, ()), (action, ()), name="actor_core_model")
			#self.model.summary()
		
		
		self.model = tf.keras.Model((input, ()), (self.core_model(obs_ph)[0], ()), name="actor_model")
		
		if debug:
			core_inf_model = tf.keras.Model((obs_input, ()), (influence, ()), name="core_inf_model")
			self.inf_model = tf.keras.Model((input, ()), (core_inf_model(obs_ph)[0], ()), name="inf_model")
		
	def get_init_state(self, n_env):
		#init_state_shape = (n_env, self.lstm_size)
		return () #(np.zeros(init_state_shape), np.zeros(init_state_shape))

class LSTMActor (BaseActor):
	def __init__ (self, env):
		super().__init__(env)
		self.lstm_size = 128
		
		with tf.name_scope("input_process"):
			input = layers.Input(shape=(None, env.obs_dim))
			main_init_state = (layers.Input(shape=(self.lstm_size, )), layers.Input(shape=(self.lstm_size, )))
			obs_ph = input
			
			# scaling the inputs
			if hasattr(env, 'obs_mean') and  hasattr(env, 'obs_std'):
				obs_ph = (obs_ph-env.obs_mean)/(env.obs_std+1e-7)
			else:
				print("WARNING (actor) : no obs range definded. Proceed with caution")
				
			# using the optional blindfold
			if hasattr(env, 'blindfold'):
				obs_ph = env.blindfold.select_visible(obs_ph)
			else:
				print("WARNING (actor) : no blindfold used")
			
		with tf.name_scope("core_model"):
			obs_input = layers.Input(shape=(None, obs_ph.shape[-1]))
			init_state = (layers.Input(shape=(self.lstm_size, )), layers.Input(shape=(self.lstm_size, )))
			
			skip = obs_input
			
			influence = layers.Dense(128, activation='relu')(obs_input)
			lstm, *end_state = layers.LSTM(self.lstm_size, return_sequences=True, return_state=True)(influence, initial_state=init_state)
			lstm = layers.Dense(128, activation="relu")(lstm)
			conc = tf.concat((lstm, skip), axis=-1)
			#conc = lstm
			
			last_layer = layers.Dense(self.act_dim, activation='tanh')
			action = (last_layer(conc)+1)/2
			
			self.core_model = tf.keras.Model((obs_input, init_state), (action, end_state), name="actor_core_model")
			#self.model.summary()
		
		
		self.model = tf.keras.Model((input, main_init_state), self.core_model((obs_ph, main_init_state)), name="actor_model")
		
		last_layer.set_weights([x/10 for x in last_layer.get_weights()])
		
		#last_layer.set_weights([x/100 for x in last_layer.get_weights()])
		
	def get_init_state(self, n_env):
		init_state_shape = (n_env, self.lstm_size)
		return (np.zeros(init_state_shape, dtype=np.float32), np.zeros(init_state_shape, dtype=np.float32))
