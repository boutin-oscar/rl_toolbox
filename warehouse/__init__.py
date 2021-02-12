# Saving and parsing data between mpi threads

from mpi4py import MPI
import numpy as np

rollout_comp = ["s", "a", "prim_a", "r", "neglog", "mask", "last_values", "gae", "new_value"]


# ------- STORAGE SYSTEM -------

# network weights storage system

_has_weights = True
_weights = ""

def store_weights (weights):
	global _has_weights
	global _weights
	_weights = weights
	_has_weights = True

def weights_sendable ():
	return _has_weights

def send_weights ():
	return _weights

# critic weights storage system

_has_critic = True
_critic = ""

def store_critic (critic):
	global _has_critic
	global _critic
	_critic = critic
	_has_critic = True

def critic_sendable ():
	return _has_critic

def send_critic ():
	return _critic

# primitive weights storage system

_has_primitive = True
_primitive = []

def store_primitive (primitive):
	global _has_primitive
	global _primitive
	_primitive = primitive
	_has_primitive = True

def primitive_sendable ():
	return _has_primitive

def send_primitive ():
	return _primitive

# rollout storage system

rollouts = {key:[] for key in rollout_comp}
rollouts_nb = {key:0 for key in rollout_comp}
sent_rollout_nb = 0
dumped_rollouts = 0

def store_rollout (x, key):
	rollouts[key].append(x)
	rollouts_nb[key] += x.shape[0]

def rollout_sendable (key):
	return rollouts_nb[key] >= sent_rollout_nb

def send_rollout (key):
	global dumped_rollouts
	to_return = np.concatenate(rollouts[key])[-sent_rollout_nb:]
	dumped_rollouts = rollouts_nb[key] - sent_rollout_nb
	
	rollouts[key] = []
	rollouts_nb[key] = 0
	
	return to_return

def set_rollout_nb (x):
	global sent_rollout_nb
	sent_rollout_nb = x


# adr storage system

adr_values = {}

def store_adr (new_values):
	adr_values.update(new_values)

def adr_sendable ():
	return True

def send_adr ():
	return adr_values

# current node storage system

cur_node = 0

def store_node (new_values):
	global cur_node
	if new_values > cur_node:
		for key in rollout_comp:
			rollouts[key] = []
			rollouts_nb[key] = 0
	cur_node = max(cur_node, new_values)

def node_sendable ():
	return not cur_node == -1

def send_node ():
	return cur_node
	


# tags
DEFAULT = 0
WORK_DONE = 1

# global var
is_work_done = False

store_dict = {"weights" : store_weights, "critic" : store_critic, "primitive" : store_primitive, "rollout_nb" : set_rollout_nb, "adr" : store_adr, "node" : store_node}
for key in rollout_comp:
	store_dict[key] = lambda x, key=key : store_rollout(x, key)

sendable_dict = {"weights" : weights_sendable, "critic" : critic_sendable, "primitive" : primitive_sendable, "dumped" : (lambda : True), "adr" : adr_sendable, "node" : node_sendable}
for key in rollout_comp:
	sendable_dict[key] = lambda key=key : rollout_sendable(key)

send_dict = {"weights" : send_weights, "critic" : send_critic, "primitive" : send_primitive, "dumped" : (lambda : dumped_rollouts), "adr" : send_adr, "node" : send_node}
for key in rollout_comp:
	send_dict[key] = lambda key=key : send_rollout(key)

def start_warehouse (comm, my_rank, wh_rank):
	global _comm
	global _my_rank
	global _wh_rank
	_comm = comm
	_my_rank = my_rank
	_wh_rank = wh_rank
	
	if _my_rank == _wh_rank:
		print("starting warehouse on rank {}".format(_my_rank), flush=True)
		work_loop()
		print("Warehouse on rank {} closed".format(_my_rank), flush=True)

def work_loop ():
	global is_work_done
	request_stack = []
	status = MPI.Status()
	
	notified_procs = 1
	num_procs = _comm.Get_size()
	while notified_procs < num_procs:
		# wait for new message
		data =_comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
		if status.Get_tag() == WORK_DONE:
			notified_procs += 1
			print("notified_procs", notified_procs, flush=True)
		
		do_store = not "node" in data or data["node"] >= cur_node
		
		# process and store the message's data
		# and add the message's request to the stack
		for key, value in data.items():
			if key == "request":
				request_stack.append((status.Get_source(), value))
				
			elif do_store:
				store_dict[key](value)
					
					
		# try to process the requests that can be
		not_processed = []
		while request_stack:
			rank, request = request_stack.pop()
			feasable = np.all([sendable_dict[req]() for req in request])
			if feasable:
				data = {req:send_dict[req]() for req in request}
				tag = DEFAULT
				_comm.send(data, dest=rank, tag=tag)
			else:
				not_processed.append((rank, request))
		
		for x in not_processed:
			request_stack.append(x)
			"""
			rank, req = x
			print("wh: msg {} not processed".format(x), flush=True)
			print("wh: feasable = {}".format(str([sendable_dict[req]() for req in request])))
			print("wh: rollout_nb =", rollouts_nb)
			#print("wh: rollouts =", rollouts)
			"""
		
	
	
def send (data, work_done=False):
	global is_work_done
	
	# send data to the main warehouse
	tag = WORK_DONE if work_done else DEFAULT
	_comm.send(data, dest=_wh_rank, tag=tag)
	
	# wait for its response if there was a request in the msg
	if "request" in data:
		status = MPI.Status()
		out_data =_comm.recv(source=_wh_rank, tag=MPI.ANY_TAG, status=status)
		is_work_done = is_work_done or status.Get_tag() == WORK_DONE
		return out_data
