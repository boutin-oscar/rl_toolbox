import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from scipy import signal
import os.path as osp
import os
import datetime
import time
import sys
import shutil

#tensorboard --logdir=tensorboard --host localhost --port 8088

from models.critic import Critic

class Distillation:
	def __init__ (self, env, actor, teacher, tensorboard_log=""):
		
		if not tensorboard_log == "":
			self.writer = tf.summary.create_file_writer(tensorboard_log)
		
		self.model_save_interval = 5
		
		self.env = env
		self.actor = actor
		self.teacher = teacher
		
		self.USE_SYMETRY = hasattr(self.env, 'symetry')
		self.USE_REPR = hasattr(self.actor, 'repr_model')

		self.gamma = 0.95
		self.lam = 0.9
		
		self.debug_interval = 5
		self.log_interval = 1
		
		
		self.create_learning_struct ()
	
	
	
	# --- graph initialization ---
	
	def create_learning_struct (self):
		"""
		for layer in self.actor.action_layers:
			layer.trainable = False
		"""
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=2.5e-4, epsilon=1e-5)
		
		self.trainable_variables = []
		i = 0
		for model in [self.actor.model]:
			for layer in model.layers:
				if layer.trainable:
					print(i)
					i += 1
					self.trainable_variables = self.trainable_variables + layer.trainable_weights
		
	@tf.function
	def step (self, obs, state):
		action, final_state = self.actor.model((obs, state))
		return action, final_state
	
	@tf.function
	def teacher_step (self, obs, state):
		action, final_state = self.teacher.model((obs, state))
		return action, final_state
	
	
	
	def compute_loss (self, n_step, do_log, actor_init_state, teacher_init_state, obs, old_act, reward, mask, learning_rate = 2.5e-4):
		with tf.name_scope("training"):
			actor_clip_range = 0.1
			with tf.name_scope("teacher"):
				teacher_act = self.teacher.model((obs, teacher_init_state))[0]
				actor_act = self.actor.model((obs, actor_init_state))[0]
				delta_act = teacher_act - old_act
				delta_act = tf.maximum(tf.minimum(delta_act, actor_clip_range), -actor_clip_range)
				target_act = delta_act + old_act
				diff_act = target_act - actor_act
				teacher_loss = tf.reduce_mean(tf.multiply(tf.reduce_sum(tf.square(diff_act), axis=-1), mask))/tf.reduce_mean(mask)
			
			if self.USE_REPR:
				with tf.name_scope("repr"):
					teacher_repr = self.teacher.repr_model((obs, teacher_init_state))
					actor_repr = self.actor.repr_model((obs, actor_init_state))
					delta_repr = teacher_repr - actor_repr
					repr_loss = tf.reduce_mean(tf.multiply(tf.reduce_sum(tf.square(delta_repr), axis=-1), mask))/tf.reduce_mean(mask)
				
				
			if self.USE_SYMETRY:
				with tf.name_scope("symetry"):
					symetry_loss = self.env.symetry.loss(self.actor, obs, actor_init_state, mask)
					
				
			with tf.name_scope("loss"):
				loss = teacher_loss
				if self.USE_REPR:
					loss = loss + repr_loss
				if self.USE_SYMETRY:
					loss = loss #+ symetry_loss
		
		if self.writer is not None and do_log:
			with self.writer.as_default():
				
				with tf.name_scope("training"):
					tf.summary.scalar('teacher_loss', teacher_loss, n_step)
					if self.USE_REPR:
						tf.summary.scalar('repr_loss', repr_loss, n_step)
					tf.summary.scalar('mean_ep_len', tf.reduce_mean(tf.reduce_sum(mask, axis=1)), n_step)
				with tf.name_scope("optimized"):
					if self.USE_SYMETRY:
						tf.summary.scalar('symetry_loss', symetry_loss, n_step)
					tf.summary.scalar('mean_rew', tf.reduce_mean(tf.multiply(reward, mask))/tf.reduce_mean(mask), n_step)
					#tf.summary.scalar('entropy_loss', entropy_loss, n_step)
		
		
		return loss
	
	@tf.function 
	def train_step (self, n_step, do_log, actor_init_state, teacher_init_state, obs, old_act, reward, mask, learning_rate = 2.5e-4):
		with tf.GradientTape() as tape:
			loss = self.compute_loss(n_step = n_step, do_log = do_log,
									actor_init_state = actor_init_state,
									teacher_init_state = teacher_init_state,
									obs = obs,
									old_act = old_act,
									reward = reward,
									mask = mask,
									learning_rate = learning_rate)
		max_grad_norm = 0.5
		gradients = tape.gradient(loss, self.trainable_variables)
		if max_grad_norm is not None:
			gradients, grad_norm = tf.clip_by_global_norm(gradients, max_grad_norm)
		grad_and_var = zip(gradients, self.trainable_variables)
		self.optimizer.learning_rate = learning_rate
		self.optimizer.apply_gradients(grad_and_var)
	
	def get_rollout (self, rollout_len, current_s=None):
		# --- simulating the environements ---
		if current_s is None:
			current_s = self.env.reset()
			current_s = np.expand_dims(np.stack(current_s), axis=1)
		current_actor_init_state = self.actor.get_init_state(self.env.num_envs)
		current_teacher_init_state = self.teacher.get_init_state(self.env.num_envs)
		
		is_env_done = [False for i in range(self.env.num_envs)]
		all_s = [[] for i in range(self.env.num_envs)]
		all_a = [[] for i in range(self.env.num_envs)]
		all_neglog = [[] for i in range(self.env.num_envs)]
		all_r = [[] for i in range(self.env.num_envs)]
		all_masks = [[] for i in range(self.env.num_envs)]
		
		n_env_done = 0
		t = 0
		
		while t < rollout_len:#config.training["rollout_len"]:
			t += 1
			current_s = np.asarray(current_s, dtype=np.float32)
			scaled_s = self.actor.scaler.scale_obs(current_s)
			current_a, current_actor_init_state = self.step (scaled_s, current_actor_init_state)
			current_teacher_a, current_teacher_init_state = self.teacher_step(scaled_s, current_teacher_init_state)
			if np.random.random() < 0.8:
				current_a = current_a.numpy()
			else:
				current_a = current_teacher_a.numpy()
			current_a += np.random.normal(size=current_a.shape) * np.exp(-3)
			#print(current_a.shape)
			current_new_s, current_r, current_done = self.env.step(current_a)
			
			n_env_done = 0
			
			for i, (s, a, r, done) in enumerate(zip(current_s, current_a, current_r, current_done)):
				all_s[i].append(s[0])
				all_a[i].append(a[0])
				if not is_env_done[i]:
					all_r[i].append(r)
					all_masks[i].append(1)
					is_env_done[i] = done
				else:
					all_r[i].append(r)
					all_masks[i].append(0)
					n_env_done += 1
			
			current_s = current_new_s
			current_s = np.expand_dims(np.stack(current_s), axis=1)
			
			
		
		# --- reshaping the logs ---
		all_s = np.asarray(all_s, dtype=np.float32)
		all_a = np.asarray(all_a, dtype=np.float32)
		all_r = np.asarray(all_r, dtype=np.float32)
		all_masks = np.asarray(all_masks)
		all_masks[:,-1] = np.zeros(all_masks[:,-1].shape)
		all_masks = all_masks.astype(np.float32)
		
		return (all_s, all_a, all_r, all_masks)
		
	def train_network (self, n, all_s, all_a, all_r, all_masks, train_step_nb):
		num_envs = all_s.shape[0]
		#all_last_values, all_gae, all_new_value = calc_gae(all_s, all_r)
		
		self.actor.scaler.update(all_s)#*np.expand_dims(all_masks, axis=2))
		
		if self.USE_SYMETRY:
			self.actor.scaler.update(self.env.symetry.state_symetry(all_s))
		
		scaled_s = self.actor.scaler.scale_obs(all_s)
		
		# --- training the networks ---
		for i in range(train_step_nb):
			n_step = tf.constant(n, dtype=tf.int64)
			do_log = tf.convert_to_tensor((n%self.log_interval==0 and i == 0), dtype=tf.bool)
			
			self.train_step(n_step = n_step, do_log=do_log, 
								actor_init_state = self.actor.get_init_state(num_envs),
								teacher_init_state = self.teacher.get_init_state(num_envs),
								obs = scaled_s,
								old_act = all_a,
								reward = all_r,
								mask = all_masks,
								learning_rate = 2.5e-4)

		# --- save the model ---
		if (n+1)%self.model_save_interval == 0:
			self.save()
	
	def save (self):
		path = self.actor.save_path# osp.join(self.actor.save_path, "{}")
		print("Model saved at : " + path.replace("\\", "\\\\"))
		self.actor.save(path)
		
	def get_weights (self):
		return self.actor.get_weights()
			
	def set_weights (self, weights):
		self.actor.set_weights(weights)

import warehouse

class DistillationNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
	
		env = input_dict['Environment'][0]
		actor = input_dict['Actor'][0]
		teacher = input_dict['Teacher'][0]
		
		# --- set the weights of the actor action_layers to those of the teacher's ---
		for actor_layer, teacher_layer in zip(actor.action_layers, teacher.action_layers):
			actor_layer.set_weights(teacher_layer.get_weights())
			actor_layer.trainable  = False
		
		USE_ADR = hasattr(env, 'adr')
		
		if self.mpi_role == 'main':
			tensorboard_path = os.path.join(save_path['tensorboard'], self.data['tensorboard_name_prop'])
			os.makedirs(tensorboard_path)
			
			trainer = Distillation(env, actor, teacher, tensorboard_path)
			trainer.model_save_interval = int(self.data['model_save_interval_prop'])
			train_step_nb = int(self.data['train_step_nb_prop'])
			
			start_time = time.time()
			desired_rollout_nb = int(self.data['rollout_nb_prop'])
			
			for n in range(int(self.data['epoch_nb_prop'])):
				# send the network weights
				# and get the latest rollouts
				req = ["s", "a", "r", "mask", "dumped", "adr"]
				msg = {"node":proc_num, "weights" : trainer.get_weights(), "rollout_nb":desired_rollout_nb, "request" : req}
				data = warehouse.send(msg)
				all_s = data["s"]
				all_a = data["a"]
				all_r = data["r"]
				all_masks = data["mask"]
				dumped_rollout_nb = data["dumped"]
				if USE_ADR:
					env.adr.update(data["adr"])
					env.adr.log()
				
				# update the network weights
				trainer.train_network(n, all_s, all_a, all_r, all_masks, train_step_nb)
				
				#debug
				n_rollouts = all_s.shape[0]
				rollout_len = all_s.shape[1]
				print("Epoch {} :".format(n), flush=True)
				print("Loaded {} rollouts for training while dumping {}.".format(n_rollouts, dumped_rollout_nb), flush=True)
				dt = time.time() - start_time
				start_time = time.time()
				if dt > 0:
					print("fps : {}".format(n_rollouts*rollout_len/dt), flush=True)
				print("mean_rew : {}".format(np.sum(all_r * all_masks)/np.sum(all_masks)), flush=True)
				
				if USE_ADR:
					env.adr.save()
					
		elif self.mpi_role == 'worker':
			trainer = Distillation(env, actor, teacher)
			rollout_len = int(self.data['rollout_len_prop'])
			#data = warehouse.send({"request":["node"]}) ; self.data['name'] == data['node']"
			msg = {"request" : ["weights", "node"]}
			data = warehouse.send(msg)
			
			while proc_num > data["node"]:
				time.sleep(0.3)
				data = warehouse.send(msg)
			
			while proc_num == data["node"]:
				test_adr = USE_ADR and np.random.random() < float(self.data['adr_prob_prop'])
				
				env.test_adr = test_adr
				
				trainer.set_weights (data["weights"])
				
				if test_adr:
					# simulate rollout
					all_s, all_a, all_r, all_mask = trainer.get_rollout(env.adr_rollout_len)
					
					msg = {"node":proc_num, 
							"adr" : env.adr.get_msg(),
							"request" : ["weights", "adr", "node"]}
				else:
					# simulate rollout
					all_s, all_a, all_r, all_mask = trainer.get_rollout(rollout_len)
					
					# send rollout back to warehouse
					# and get network weights and update actor
					msg = {"node":proc_num, 
							"s" : all_s,
							"a" : all_a,
							"r" : all_r,
							"mask" : all_mask,
							"request" : ["weights", "adr", "node"]}
					
				data = warehouse.send(msg)
				
				if USE_ADR:
					env.adr.update(data["adr"])
		
		
		for actor_layer, teacher_layer in zip(actor.action_layers, teacher.action_layers):
			actor_layer.trainable  = True
		
		output_dict['Trained actor'] = trainer.actor
	
