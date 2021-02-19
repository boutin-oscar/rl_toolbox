import numpy as np

import configuration.processes as nodes
import warehouse

@nodes.process_node
def train_ppo (actor, env, epoch_nb, rollout_per_epoch, rollout_len, train_step_per_epoch, init_log_std, model_save_interval, adr_test_prob, tensorboard_path):
	
	mpi_role = nodes.mpi_role
	proc_num = nodes.proc_num
	pnid = nodes.pnid
	
	import os
	import time
	
	from ppo import PPO
	from models.critic import Critic
	
	USE_ADR = hasattr(env, 'adr') and adr_test_prob > 1e-7
	
	if mpi_role == 'main':
		os.makedirs(tensorboard_path)
		
		critic = Critic(env)
		
		trainer = PPO(env, actor, critic, tensorboard_path, init_log_std=init_log_std)
		trainer.model_save_interval = model_save_interval
		
		start_time = time.time()
		
		for n in range(epoch_nb):
			# send the network weights
			# and get the latest rollouts
			msg = {	pnid+"weights" : warehouse.Entry(action="set", value=trainer.get_weights()),
					pnid+"adr" : warehouse.Entry(action="get", value=None),
					pnid+"s" : warehouse.Entry(action="get_l", value=rollout_per_epoch),
					pnid+"a" : warehouse.Entry(action="get_l", value=rollout_per_epoch),
					pnid+"r" : warehouse.Entry(action="get_l", value=rollout_per_epoch),
					pnid+"neglog" : warehouse.Entry(action="get_l", value=rollout_per_epoch),
					pnid+"mask" : warehouse.Entry(action="get_l", value=rollout_per_epoch),
					"dumped" : warehouse.Entry(action="get", value=None)
					}
			data = warehouse.send(msg)
			all_s = np.concatenate(data[pnid+"s"].value, axis=0)
			all_a = np.concatenate(data[pnid+"a"].value, axis=0)
			all_r = np.concatenate(data[pnid+"r"].value, axis=0)
			all_neglog = np.concatenate(data[pnid+"neglog"].value, axis=0)
			all_masks = np.concatenate(data[pnid+"mask"].value, axis=0)
			dumped_rollout_nb = data["dumped"].value
			
			if USE_ADR:
				env.adr.update(data[pnid+"adr"].value)
				env.adr.log()
			
			# update the network weights
			all_last_values, all_gae, all_new_value = trainer.calc_gae(all_s, all_r, all_masks)
			trainer.train_networks(n, all_s, all_a, all_r, all_neglog, all_masks, train_step_per_epoch, all_last_values, all_gae, all_new_value)
			
			#debug
			n_rollouts = all_s.shape[0]
			cur_rollout_len = all_s.shape[1]
			print("Epoch {} :".format(n), flush=True)
			#dumped_rollout_nb = "?"
			print("Loaded {} rollouts for training while dumping {}.".format(n_rollouts, dumped_rollout_nb), flush=True)
			dt = time.time() - start_time
			start_time = time.time()
			if dt > 0:
				print("fps : {}".format(n_rollouts*cur_rollout_len/dt), flush=True)
			print("mean_rew : {}".format(np.sum(all_r * all_masks)/np.sum(all_masks)), flush=True)
			
			if USE_ADR:
				env.adr.save()
				
	elif mpi_role == 'worker':
		trainer = PPO(env, actor, Critic(env), init_log_std=init_log_std)
		
		msg = {	pnid+"weights" : warehouse.Entry(action="get", value=None),
				pnid+"adr" : warehouse.Entry(action="set", value={}),
				"proc_num" : warehouse.Entry(action="get", value=None)}
		data = warehouse.send(msg)
		
		while proc_num >= data["proc_num"].value and not warehouse.is_work_done:
			test_adr = USE_ADR and np.random.random() < adr_test_prob
			
			env.test_adr = test_adr
			
			trainer.set_weights (data[pnid+"weights"].value)
			
			if test_adr:
				# simulate rollout
				all_s, all_a, all_r, all_neglog, all_mask = trainer.get_rollout(env.adr_rollout_len)
				
				msg = {	pnid+"adr" : warehouse.Entry(action="update", value=env.adr.get_msg()),
						pnid+"weights" : warehouse.Entry(action="get", value=None),
						"proc_num" : warehouse.Entry(action="get", value=None)}
			else:
				# simulate rollout
				all_s, all_a, all_r, all_neglog, all_mask = trainer.get_rollout(rollout_len)
				
				# send rollout back to warehouse
				# and get network weights to update actor
				msg = {	pnid+"s" : warehouse.Entry(action="add", value=all_s),
						pnid+"a" : warehouse.Entry(action="add", value=all_a),
						pnid+"r" : warehouse.Entry(action="add", value=all_r),
						pnid+"neglog" : warehouse.Entry(action="add", value=all_neglog),
						pnid+"mask" : warehouse.Entry(action="add", value=all_mask),
						pnid+"weights" : warehouse.Entry(action="get", value=None),
						pnid+"adr" : warehouse.Entry(action="get", value=None), 
						"proc_num" : warehouse.Entry(action="get", value=None)}
				
			data = warehouse.send(msg)
			
			if USE_ADR:
				env.adr.update(data[pnid+"adr"].value)