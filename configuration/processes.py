import functools
import numpy as np

import warehouse

my_rank = 0
mpi_role = 'not set'
proc_num = 0
pnid = "0:"


def process_node (func):
	@functools.wraps(func)
	def node_wrapper(*args, **kwargs):
		global proc_num
		global pnid
		proc_num += 1
		pnid = str(proc_num)+":"
		warehouse.send({"proc_num": warehouse.Entry(action="set_max", value=proc_num)})
		return func(*args, **kwargs)
	return node_wrapper

@process_node
def setup_exp ():
	import sys
	import os
	import shutil
	
	exp_name = "default"
	if len(sys.argv) > 1:
		exp_name = sys.argv[1]
	save_dir_path = os.path.join("results", exp_name)

	tensorboard_log = os.path.join(save_dir_path, "tensorboard")
	model_path = os.path.join(save_dir_path, "models")
	env_path = os.path.join(save_dir_path, "env")
		
	paths = {'tensorboard':tensorboard_log, 'models':model_path, 'env':env_path}

	if mpi_role == 'main':
		if os.path.exists(save_dir_path) and os.path.isdir(save_dir_path): # del dir if exists
			shutil.rmtree(save_dir_path)
		
		os.makedirs(tensorboard_log)
		os.makedirs(model_path)
		os.makedirs(env_path)
	
	return save_dir_path
	
@process_node
def dog_env ():
	from environments import dog_env
	env = dog_env.DogEnv()
	return env

@process_node
def simple_actor (obs_dim, act_dim, obs_mean=None, obs_std=None, blindfold=None, inp_dim=None, save_path=None):
	import os
	from models.actor import SimpleActor
	
	if save_path is None:
		raise NameError("A save_path should always be given to an actor")
		
	actor = SimpleActor (obs_dim, act_dim, obs_mean=obs_mean, obs_std=obs_std, blindfold=blindfold, inp_dim=inp_dim)
	actor.save_path = save_path
	
	if mpi_role == 'main':
		os.makedirs(actor.save_path)
		actor.save(actor.save_path)
		
		data_out = {pnid+":actor_weight":warehouse.Entry(action="set", value=actor.get_weights())}
		data = warehouse.send(data_out)
	
	data_out = {pnid+":actor_weight":warehouse.Entry(action="get", value=None)}
	data = warehouse.send(data_out)
	actor.set_weights(data[pnid+":actor_weight"].value)
	
	return actor

@process_node
def load_actor (actor, path):
	if mpi_role == 'main':
		actor.load(path)
		
		data_out = {pnid+":actor_weight":warehouse.Entry(action="set", value=actor.get_weights())}
		data = warehouse.send(data_out)
	
	data_out = {pnid+":actor_weight":warehouse.Entry(action="get", value=None)}
	data = warehouse.send(data_out)
	actor.set_weights(data[pnid+":actor_weight"].value)





