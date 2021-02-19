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


class Trainer:
	def __init__ (self, disc, tensorboard_log):
		self.disc = disc
		
		if not tensorboard_log == "":
			self.writer = tf.summary.create_file_writer(tensorboard_log)
		
		self.model_save_interval = 5
		self.debug_interval = 5
	
		self.create_learning_struct()
		
	def create_learning_struct (self):
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=2.5e-4, epsilon=1e-5)
		
		self.trainable_variables = []
		for model in [self.disc.model]:
			for layer in model.layers:
				if layer.trainable:
					self.trainable_variables = self.trainable_variables + layer.trainable_weights
		
	
	def compute_loss (self, n_step, do_log, trans, labels):
		with tf.name_scope("training"):
			with tf.name_scope("standard_cost"):			
				logits = self.disc.model(trans)
				cost = tf.nn.softmax_cross_entropy_with_logits(labels, logits, axis=-1)
				cost = tf.reduce_mean(cost)
				
			with tf.name_scope("loss"):
				loss = cost
			
			accuracy = tf.reduce_sum(tf.keras.metrics.categorical_accuracy(labels, logits))
		
		if self.writer is not None and do_log:
			with self.writer.as_default():
				with tf.name_scope("training"):
					tf.summary.scalar('standard_cost', cost, n_step)
					tf.summary.scalar('accuracy', accuracy, n_step)
		
		return loss, accuracy
	
	@tf.function 
	def train_step (self, n_step, do_log, trans, labels, learning_rate = 2.5e-4):
		with tf.GradientTape() as tape:
			loss, accuracy = self.compute_loss(n_step, do_log, trans, labels)
		max_grad_norm = 0.5
		gradients = tape.gradient(loss, self.trainable_variables)
		if max_grad_norm is not None:
			gradients, grad_norm = tf.clip_by_global_norm(gradients, max_grad_norm)
		grad_and_var = zip(gradients, self.trainable_variables)
		self.optimizer.learning_rate = learning_rate
		self.optimizer.apply_gradients(grad_and_var)
		
		return accuracy
	
	def train_network (self, n, full_trans, labels, train_step_nb):
		# --- training the networks ---
		for i in range(train_step_nb):
			#n_step = tf.constant(n, dtype=tf.int64)
			n_step = tf.constant(n*train_step_nb+i, dtype=tf.int64)
			#do_log = tf.convert_to_tensor((n%self.log_interval==0 and i == 0), dtype=tf.bool)
			do_log = tf.convert_to_tensor(True, dtype=tf.bool)
			
			accuracy = self.train_step(n_step = n_step, do_log=do_log, 
								trans = full_trans,
								labels = labels,
								learning_rate = 2.5e-4)

		# --- save the model ---
		if (n+1)%self.model_save_interval == 0:
			self.save()
		
		return int(accuracy.numpy())
	
	def save (self):
		path = self.disc.save_path# osp.join(self.actor.save_path, "{}")
		print("Model saved at : " + path.replace("\\", "\\\\"))
		self.disc.save(path)
	

