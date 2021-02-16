import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from mpi4py import MPI

import configuration.processes as nodes
import warehouse

"""
mpiexec -n 4 python start_training.py exp_0

tensorboard --logdir=results/exp_0/tensorboard --host localhost --port 6006
"""

def main_programm ():
	
	exp_path = nodes.setup_exp()
	
	env = nodes.dog_env ()
	actor = nodes.simple_actor(env, save_path = exp_path+"\\models\\expert\\{}", use_blindfold = True)
	
	ppo_config = dict(	epoch_nb = 2, # 20000,
						rollout_per_epoch = 10,
						rollout_len = 100, #400,
						train_step_per_epoch = 6,
						init_log_std = -1,
						model_save_interval = 10,
						adr_test_prob = 0.3)
	
	#trained_actor = nodes.train_ppo(actor, env, **dict)




if __name__ == "__main__":

	comm = MPI.COMM_WORLD
	my_rank = comm.Get_rank()
	my_name = MPI.Get_processor_name()
	mpi_role = 'main' if my_rank == 0 else ('wh' if my_rank == 1 else 'worker')
	nodes.mpi_role = mpi_role
	
	
	warehouse.start_warehouse(comm, my_rank, 1)
	
	if not mpi_role == 'wh':
		main_programm ()
		warehouse.send({}, work_done=True)
	
	
	