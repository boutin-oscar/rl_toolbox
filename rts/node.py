import numpy as np
import os

import configuration.processes as nodes
import warehouse

import time

@nodes.process_node
def generate_trans_batch (env, actor, rollout_nb, rollout_len, log_std, save_path):
	
	mpi_role = nodes.mpi_role
	proc_num = nodes.proc_num
	pnid = nodes.pnid
	
	if mpi_role == 'main':
		os.makedirs(save_path)
		
		msg = {pnid+"trans" : warehouse.Entry(action="get_l", value=rollout_nb)}
		data = warehouse.send(msg)
		
		all_trans = np.stack(data[pnid+"trans"].value)
		
		np.save(os.path.join(save_path, "all_trans.npy"), all_trans)
		
	elif mpi_role == 'worker':
		
		msg = {"proc_num" : warehouse.Entry(action="get", value=None)}
		data = warehouse.send(msg)
		
		while proc_num >= data["proc_num"].value and not warehouse.is_work_done:
		
			all_trans = []
			
			
			obs = env.reset ()
			obs = np.asarray(obs).reshape((1,1,-1))
			
			for i in range(rollout_len):
				
				trans = [obs.flatten()]
				
				act = actor.model(obs).numpy()
				act = act + np.random.normal(size=act.flatten().shape[0]).reshape(act.shape) * np.exp(log_std)
				obs, rew, done = env.step(act)
				obs = np.asarray(obs).reshape((1,1,-1))
				
				trans.append((obs.flatten()-trans[0])*10)
				trans.append(act.flatten())
				all_trans.append(np.concatenate(trans))
				
			all_trans = np.asarray(all_trans).reshape((rollout_len,-1))
			
			msg = {	pnid+"trans" : warehouse.Entry(action="add", value=all_trans),
					"proc_num" : warehouse.Entry(action="get", value=None)}
			data = warehouse.send(msg)

def format_trans (all_trans_raw, lab):
	all_trans = all_trans_raw.reshape((-1, all_trans_raw.shape[-1]))
	all_labs = np.ones((all_trans.shape[0], 1)) @ lab
	return all_trans, all_labs

@nodes.process_node
def train_discrim (disc, real_trans_path, all_env, actor, epoch_nb, train_step_per_epoch, rollout_per_epoch, rollout_len, log_std, model_save_interval, tensorboard_path):
	
	
	mpi_role = nodes.mpi_role
	proc_num = nodes.proc_num
	pnid = nodes.pnid
	
	if mpi_role == 'main':
		os.makedirs(tensorboard_path)
		
		from rts.discriminator import Trainer
		trainer = Trainer(disc, tensorboard_path)
		trainer.model_save_interval = model_save_interval
		
		real_lab = np.asarray([1, 0]).reshape((1,2))
		synth_lab = np.asarray([0, 1]).reshape((1,2))
		
		all_real_trans = np.load(os.path.join(real_trans_path, "all_trans.npy"))
		all_real_trans, all_real_labs = format_trans(all_real_trans, real_lab)
		
		start_time = time.time()
		for n in range(epoch_nb):
		
			# get the latest rollouts
			msg = {	pnid+"trans" : warehouse.Entry(action="get_l", value=rollout_per_epoch), 
					"dumped" : warehouse.Entry(action="get", value=None) }
			data = warehouse.send(msg)
			dumped_rollout_nb = data["dumped"].value

			all_synth_trans_raw = data[pnid+"trans"].value
			all_synth_trans, all_synth_labs = format_trans(np.concatenate(all_synth_trans_raw, axis=0), synth_lab)
			
			# put the training data together
			all_trans = np.concatenate([all_real_trans, all_synth_trans], axis=0)
			all_labs = np.concatenate([all_real_labs, all_synth_labs], axis=0)
			
			# random offset for regularisation
			all_trans += np.random.normal(size=all_trans.shape) * 0.003
			
			# update the network weights
			accuracy = trainer.train_network(n, all_trans, all_labs, train_step_per_epoch)
			
			#debug
			n_rollouts = len(all_synth_trans_raw)
			print("Epoch {} :".format(n), flush=True)
			print("Loaded {} synthetic rollouts for training while dumping {} for a total of {} transitions.".format(n_rollouts, dumped_rollout_nb, all_synth_trans.shape[0]), flush=True)
			dt = time.time() - start_time
			start_time = time.time()
			if dt > 0:
				print("fps : {}".format(all_synth_trans.shape[0]/dt), flush=True)
			print("accuracy : {}/{}".format(accuracy, all_trans.shape[0]), flush=True)
		
		
		msg = {pnid+":disc_weight":warehouse.Entry(action="set", value=disc.get_weights())}
		data = warehouse.send(msg)
	
	elif mpi_role == "worker":
		
		msg = {"proc_num" : warehouse.Entry(action="get", value=None)}
		data = warehouse.send(msg)
		
		while proc_num >= data["proc_num"].value and not warehouse.is_work_done:
		
			all_trans = []
			
			env = all_env[np.random.randint(len(all_env))]
			obs = env.reset ()
			obs = np.asarray(obs).reshape((1,1,-1))
			done = [False]
			
			i = 0
			while i < rollout_len and not done[0]:
				i += 1
				trans = [obs.flatten()]
				
				act = actor.model(obs).numpy()
				act = act + np.random.normal(size=act.flatten().shape[0]).reshape(act.shape) * np.exp(log_std)
				obs, rew, done = env.step(act)
				obs = np.asarray(obs).reshape((1,1,-1))
				
				trans.append((obs.flatten()-trans[0])*10)
				trans.append(act.flatten())
				all_trans.append(np.concatenate(trans))
				
			all_trans = np.asarray(all_trans).reshape((i,-1))
			
			msg = {	pnid+"trans" : warehouse.Entry(action="add", value=all_trans),
					"proc_num" : warehouse.Entry(action="get", value=None)}
			data = warehouse.send(msg)
		
	
	msg = {pnid+":disc_weight":warehouse.Entry(action="get", value=None)}
	data = warehouse.send(msg)
	disc.set_weights(data[pnid+":disc_weight"].value)
	
	
	
	
	
	
	