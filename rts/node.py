import numpy as np
import os

import configuration.processes as nodes
import warehouse

import time

@nodes.process_node
def generate_trans_batch (env, actor, rollout_nb, rollout_len, save_path):
	
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
				obs, rew, done = env.step(act)
				obs = np.asarray(obs).reshape((1,1,-1))
				
				trans.append(obs.flatten())
				trans.append(act.flatten())
				all_trans.append(np.concatenate(trans))
				
			all_trans = np.asarray(all_trans).reshape((rollout_len,-1))
			
			msg = {	pnid+"trans" : warehouse.Entry(action="add", value=all_trans),
					"proc_num" : warehouse.Entry(action="get", value=None)}
			data = warehouse.send(msg)
