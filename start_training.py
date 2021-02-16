import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from mpi4py import MPI

import configuration.processes as nodes
from configuration.config import main_programm
import warehouse

"""
mpiexec -n 4 python start_training.py exp_0
mpiexec -n 32 python start_training.py exp_0

tensorboard --logdir=results/exp_0/tensorboard --host localhost --port 6006

ps -aux

"""


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
	
	print("Thread {} has ended cleanly.".format(my_rank)) 
	
	
	