import os
import warehouse

from models.actor import SimpleActor

class SimpleActorNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		# create the actor
		env = input_dict['Env'][0]
		use_blindfold = self.data['blindfold_prop'] == "1"
		use_lstm = self.data['lstm_prop'] == "1"
		actor = SimpleActor (env, use_blindfold, use_lstm)
		output_dict['Actor'] = actor
		
		# save the actor
		if self.mpi_role == 'main':
			actor.save_path = os.path.join(save_path['models'], self.data['save_name_prop'])
			os.makedirs(actor.save_path)
			actor.save_path += "/{}"
			actor.save(actor.save_path)
			
			data_out = {str(proc_num)+":actor_weight":warehouse.Entry(action="set", value=actor.get_weights())}
			data = warehouse.send(data_out)
		
		data_out = {str(proc_num)+":actor_weight":warehouse.Entry(action="get", value=None)}
		data = warehouse.send(data_out)
		actor.set_weights(data[str(proc_num)+":actor_weight"].value)

from models.actor import MixtureOfExpert

class MixActorNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		# create the actor
		env = input_dict['Env'][0]
		primitives = input_dict['Primitive']
		actor = MixtureOfExpert (env, primitives)
		output_dict['Actor'] = actor

		# save the actor
		if self.mpi_role == 'main':
			actor.save_path = os.path.join(save_path['models'], self.data['save_name_prop'])
			os.makedirs(actor.save_path)
			actor.save_path += "/{}"
			actor.save(actor.save_path)
		
from models.actor import LSTMActor

class LSTMActorNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		# create the actor
		env = input_dict['Env'][0]
		actor = LSTMActor (env)
		output_dict['Actor'] = actor
		
		# save the actor
		if self.mpi_role == 'main':
			actor.save_path = os.path.join(save_path['models'], self.data['save_name_prop'])
			os.makedirs(actor.save_path)
			actor.save_path += "/{}"
			actor.save(actor.save_path)
			
from models.conv_network import ConvActor

class ConvActorNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		# create the actor
		env = input_dict['Env'][0]
		actor = ConvActor (env)
		output_dict['Actor'] = actor
		
		# save the actor
		if self.mpi_role == 'main':
			actor.save_path = os.path.join(save_path['models'], self.data['save_name_prop'])
			os.makedirs(actor.save_path)
			actor.save_path += "/{}"
			actor.save(actor.save_path)
			
			
			
class LoadActorNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		# create the actor
		actor = input_dict['Actor'][0]

		if self.mpi_role == 'main' or True:
			path = self.data['model_path_prop']
			actor.load(path + "/{}")
			
			#save the actor
			actor.save(actor.save_path)
		print("loaded the right actor", flush=True)
		output_dict['Actor'] = actor

class SaveActorNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		# create the actor
		actor = input_dict['Actor'][0]
		output_dict['Actor'] = actor

		if self.mpi_role == 'main':
			path = os.path.join(save_path['models'], self.data['model_path_prop'])
			os.makedirs(path)
			path += "/{}"
			
			#save the actor
			actor.save(path)

from models.conv_network import ConvCritic

class ConvCriticNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		# create the critic
		env = input_dict['Env'][0]
		critic = ConvCritic (env)
		output_dict['Critic'] = critic

from models.critic import Critic
class LoadCriticNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		# create the actor
		env = input_dict['Env'][0]
		path = self.data['model_path_prop'] + "/{}"
		critic = Critic (env)
		if self.mpi_role == 'main':
			critic.model.load_weights(path.format("critic"))
		output_dict['Critic'] = critic

# free the whole net
class FreePrimitiveNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		actor = input_dict['Actor'][0]
		for layer in actor.model.layers:
			layer.trainable = (self.data['free_prop'] == "1")
		output_dict['Actor'] = actor

		if self.mpi_role == 'main':
			
			#save the actor
			actor.save(actor.save_path)


class SimpleEnvNode:
	def __init__ (self, mpi_role):

		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		from environments import simple_env
		env = simple_env.SimpleEnv()
		output_dict['Env'] = env

class CartPoleNode:
	def __init__ (self, mpi_role):

		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		from environments import cartpole_old
		env = cartpole_old.CartPoleEnv()
		#env.mode = int(self.data['mode_prop'])
		output_dict['Env'] = env

class DogEnvNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		from environments import dog_env

		env = dog_env.DogEnv()
		#env.training_mode = int(self.data['mode_prop'])
		output_dict['Env'] = env

class BotEnvNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		from environments import bot_env

		env = bot_env.BotEnv()
		output_dict['Env'] = env

from ppo import PPO
from ppo_cat import PPO_cat
import warehouse
import time
import numpy as np
from models.critic import Critic

class TrainPPONode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, proc_num, input_dict, output_dict):
		
		pnid = str(proc_num)+":"
		
		env = input_dict['Environment'][0]
		actor = input_dict['Actor'][0]
		log_std = float(self.data['log_std_prop'])
		
		USE_ADR = hasattr(env, 'adr')
		
		if self.mpi_role == 'main':
			tensorboard_path = os.path.join(save_path['tensorboard'], self.data['tensorboard_name_prop'])
			os.makedirs(tensorboard_path)
			
			if "Critic" in input_dict:
				critic = input_dict['Critic'][0]
			else:
				critic = Critic(env)
			
			if self.data['ppo_cat_prop'] == "0":
				trainer = PPO(env, actor, critic, tensorboard_path, init_log_std=log_std)
			if self.data['ppo_cat_prop'] == "1":
				trainer = PPO_cat(env, actor, critic, tensorboard_path, init_log_std=log_std)
			trainer.model_save_interval = int(self.data['model_save_interval_prop'])
			train_step_nb = int(self.data['train_step_nb_prop'])
			
			start_time = time.time()
			desired_rollout_nb = int(self.data['rollout_nb_prop'])
			
			for n in range(int(self.data['epoch_nb_prop'])):
				# send the network weights
				# and get the latest rollouts
				req = ["s", "a", "r", "neglog", "mask", "dumped", "adr"]
				msg = {	pnid+"weights" : warehouse.Entry(action="set", value=trainer.get_weights()),
						pnid+"adr" : warehouse.Entry(action="get", value=None),
						pnid+"s" : warehouse.Entry(action="get_l", value=desired_rollout_nb),
						pnid+"a" : warehouse.Entry(action="get_l", value=desired_rollout_nb),
						pnid+"r" : warehouse.Entry(action="get_l", value=desired_rollout_nb),
						pnid+"neglog" : warehouse.Entry(action="get_l", value=desired_rollout_nb),
						pnid+"mask" : warehouse.Entry(action="get_l", value=desired_rollout_nb),
						}
				data = warehouse.send(msg)
				all_s = np.concatenate(data[pnid+"s"].value, axis=0)
				all_a = np.concatenate(data[pnid+"a"].value, axis=0)
				all_r = np.concatenate(data[pnid+"r"].value, axis=0)
				all_neglog = np.concatenate(data[pnid+"neglog"].value, axis=0)
				all_masks = np.concatenate(data[pnid+"mask"].value, axis=0)
				#dumped_rollout_nb = data["dumped"].value
				
				if USE_ADR:
					env.adr.update(data[pnid+"adr"].value)
					env.adr.log()
				
				# update the network weights
				all_last_values, all_gae, all_new_value = trainer.calc_gae(all_s, all_r, all_masks)
				trainer.train_networks(n, all_s, all_a, all_r, all_neglog, all_masks, train_step_nb, all_last_values, all_gae, all_new_value)
				
				#debug
				n_rollouts = all_s.shape[0]
				rollout_len = all_s.shape[1]
				print("Epoch {} :".format(n), flush=True)
				dumped_rollout_nb = "?"
				print("Loaded {} rollouts for training while dumping {}.".format(n_rollouts, dumped_rollout_nb), flush=True)
				dt = time.time() - start_time
				start_time = time.time()
				if dt > 0:
					print("fps : {}".format(n_rollouts*rollout_len/dt), flush=True)
				print("mean_rew : {}".format(np.sum(all_r * all_masks)/np.sum(all_masks)), flush=True)
				
				if USE_ADR:
					env.adr.save()
					
		elif self.mpi_role == 'worker':
			if self.data['ppo_cat_prop'] == "0":
				trainer = PPO(env, actor, Critic(env), init_log_std=log_std)
			if self.data['ppo_cat_prop'] == "1":
				trainer = PPO_cat(env, actor, Critic(env), init_log_std=log_std)
			rollout_len = int(self.data['rollout_len_prop'])
			#data = warehouse.send({"request":["node"]}) ; self.data['name'] == data['node']"
			msg = {	pnid+"weights" : warehouse.Entry(action="get", value=None),
					pnid+"adr" : warehouse.Entry(action="set", value={}),
					"pnid" : warehouse.Entry(action="get", value=None)}
			data = warehouse.send(msg)
			
			while proc_num == data["pnid"].value and not warehouse.is_work_done:
				test_adr = USE_ADR and np.random.random() < float(self.data['adr_prob_prop'])
				
				env.test_adr = test_adr
				
				#print(data["weights"][0], flush=True)
				trainer.set_weights (data[pnid+"weights"].value)
				
				if test_adr:
					# simulate rollout
					all_s, all_a, all_r, all_neglog, all_mask = trainer.get_rollout(env.adr_rollout_len)
					
					msg = {	pnid+"adr" : warehouse.Entry(action="update", value=env.adr.get_msg()),
							pnid+"weights" : warehouse.Entry(action="get", value=None),
							"pnid" : warehouse.Entry(action="get", value=None)}
				else:
					# simulate rollout
					all_s, all_a, all_r, all_neglog, all_mask = trainer.get_rollout(rollout_len)
					
					# send rollout back to warehouse
					# and get network weights and update actor
					msg = {	pnid+"s" : warehouse.Entry(action="add", value=all_s),
							pnid+"a" : warehouse.Entry(action="add", value=all_a),
							pnid+"r" : warehouse.Entry(action="add", value=all_r),
							pnid+"neglog" : warehouse.Entry(action="add", value=all_neglog),
							pnid+"mask" : warehouse.Entry(action="add", value=all_mask),
							pnid+"weights" : warehouse.Entry(action="get", value=None),
							pnid+"adr" : warehouse.Entry(action="get", value=None), 
							"pnid" : warehouse.Entry(action="get", value=None)}
					
				data = warehouse.send(msg)
				
				if USE_ADR:
					env.adr.update(data[pnid+"adr"].value)
		
		
		output_dict['Trained actor'] = trainer.actor
		output_dict['Critic'] = trainer.critic

from distillation import DistillationNode

type_dict = {
		'SimpleActorNode':SimpleActorNode,
		'MixActorNode':MixActorNode,
		'LSTMActorNode':LSTMActorNode,
		'ConvActorNode':ConvActorNode,
		'LoadActorNode':LoadActorNode,
		'SaveActorNode':SaveActorNode,
		'FreePrimitiveNode':FreePrimitiveNode,
		'ConvCriticNode':ConvCriticNode,
		'LoadCriticNode':LoadCriticNode,
		'SimpleEnvNode':SimpleEnvNode,
		'CartPoleNode':CartPoleNode,
		'DogEnvNode':DogEnvNode,
		'BotEnvNode':BotEnvNode,
		'TrainPPONode':TrainPPONode,
		'DistillationNode': DistillationNode
		}

def get_process (type):
	if type not in type_dict:
		raise NameError('Node type {} not known'.format(type))
	
	return type_dict[type]
