import sys
import os
import shutil
import xml.etree.ElementTree as ET
from mpi4py import MPI

import configuration.nodes
import warehouse

"""
exp_0 : no critic knowlege
exp_1 : critic knows the act offset
"""

"""
mpiexec -n 10 python start_sequence.py exp_0

tensorboard --logdir=results/exp_0/tensorboard --host localhost --port 6006
"""

class ProcessNode:
	def __init__ (self, mpi_role, data, mpi_rank):
		self.input_dict = {}
		self.output_dict = {}
		self.input_links = []
		self.output_links = []
		self.data = data
		
		self.proc = configuration.nodes.get_process(data['type'])(mpi_role)
		self.proc.data = data
		self.proc.mpi_rank = mpi_rank
		
	def run (self, save_paths, proc_num):
		self.proc.run(save_paths, proc_num, self.input_dict, self.output_dict)

DEBUG = True

if __name__ == "__main__":
	comm = MPI.COMM_WORLD
	my_rank = comm.Get_rank()
	my_name = MPI.Get_processor_name()
	mpi_role = 'main' if my_rank == 0 else ('wh' if my_rank == 1 else 'worker')
	
	
	warehouse.start_warehouse(comm, my_rank, 1)
	
	
	if not mpi_role == 'wh':
		
		# reading the xml
		config_file_name = "configuration/config.xml"
		tree = ET.parse(config_file_name)
		root = tree.getroot()

		proc_dict = {}

		for child in root.iter('Process'):
			attrib = child.attrib
			proc_dict[attrib['name']] = ProcessNode(mpi_role, attrib, my_rank)

		for child in root.iter('Link'):
			attrib = child.attrib
			proc_dict[attrib['from_node']].output_links.append(attrib)
			proc_dict[attrib['to_node']].input_links.append(attrib)

		if DEBUG and mpi_role == 'main':
			type_outputs = {}
			for node in proc_dict.values():
				if not node.data['type'] in type_outputs:
					type_outputs[node.data['type']] = set()
				for link in node.output_links:
					type_outputs[node.data['type']].add(link['from_socket'])
			print("Expected node outputs : ", flush=True)
			print(type_outputs, flush=True)

		# calculating the execution order
		requirement_nb = {}
		dest_nb = {}
		target_nodes = []
		for key in proc_dict.keys():
			requirement_nb[key] = len(proc_dict[key].input_links)
			dest_nb[key] = len(proc_dict[key].output_links)
			if len(proc_dict[key].output_links) == 0:
				target_nodes.append(key)

		proc_stack = []
		while len(target_nodes) > 0:
			if requirement_nb[target_nodes[-1]] == 0:
				to_remove = target_nodes.pop()
				proc_stack.append(to_remove)
				for link in proc_dict[to_remove].output_links:
					requirement_nb[link['to_node']] -= 1
			else:
				for link in proc_dict[target_nodes[-1]].input_links:
					if dest_nb[link['from_node']] == 1:
						target_nodes.append(link['from_node'])
						dest_nb[link['from_node']] -= 1
					else:
						dest_nb[link['from_node']] -= 1

		# refresh dicts
		exp_name = "default"
		if len(sys.argv) > 1:
			exp_name = sys.argv[1]
		save_dir_path = os.path.join("results", exp_name)#datetime.datetime.now().strftime("snoopx_%Y_%m_%d_%Hh%Mm%Ss"))

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


		# executing the stacked processes
		for proc_num, proc in enumerate(proc_stack):
			proc_dict[proc].run(paths, proc_num)
			
			for link in proc_dict[proc].output_links:
				if link['to_socket'] in proc_dict[link['to_node']].input_dict:
					proc_dict[link['to_node']].input_dict[link['to_socket']].append(proc_dict[proc].output_dict[link['from_socket']])
				else:
					proc_dict[link['to_node']].input_dict[link['to_socket']] = [proc_dict[proc].output_dict[link['from_socket']]]
		
		
		# closing everything
		warehouse.send({"node":len(proc_stack)}, work_done=True)
	
	