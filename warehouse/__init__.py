
from mpi4py import MPI
import numpy as np
from collections import namedtuple

# the singe element that holds all the data
store_dict = {}


# a msg is always a dict of Entries. 
Entry = namedtuple('Entry', ['action', 'value'])
"""
action can be :
set		-> sets the value
add		-> adds the value (excepts a list in the right slot)
update	-> updates the dict (expects a dict or nothing in the right slot)
get		-> asks the value of the field (wait fot the field to be present)
get_l	-> get last value elems of the list (wait for the list to fill up)

"""

# a tuple to store the functions per action
class Action:
	def process (key, value):
		raise NameError('not implemented')
	def feasable (key, value):
		raise NameError('not implemented')
	def data (key, value):
		raise NameError('not implemented')
action_dict = {}


class SetAction (Action):
	def process (key, value):
		store_dict[key] = value
	def feasable (key, value):
		return True
	def data (key, value):
		return
action_dict['set'] = SetAction
	
class AddAction (Action):
	def process (key, value):
		if not key in store_dict:
			store_dict[key] = []
		store_dict[key].append(value)
	def feasable (key, value):
		return True
	def data (key, value):
		return
action_dict['add'] = AddAction
		
class UpdateAction (Action):
	def process (key, value):
		if not key in store_dict:
			store_dict[key] = {}
		store_dict[key].update(value)
	def feasable (key, value):
		return True
	def data (key, value):
		return
action_dict['update'] = UpdateAction
		
class GetAction (Action):
	def process (key, value):
		pass
	def feasable (key, value):
		return key in store_dict
	def data (key, value):
		return store_dict[key]
action_dict['get'] = GetAction

class GetLAction (Action):
	def process (key, value):
		pass
	def feasable (key, value):
		return key in store_dict and len(store_dict[key]) >= value
	def data (key, value):
		to_return = store_dict[key][-value:]
		store_dict[key] = []
		return to_return
action_dict['get_l'] = GetLAction
		

# tags to check for programm end
DEFAULT = 0
WORK_DONE = 1

is_work_done = False

# entry points :

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
		
		# process and store the message's data
		# and add the message's request to the stack
		process_incomming (data)
		
		request_stack.append((status.Get_source(), data))
		
					
		# try to process the requests that can be
		not_processed = []
		while request_stack:
			rank, request = request_stack.pop()
			if calc_feasable(request):
				data = calc_data (request)
				tag = WORK_DONE if is_work_done else DEFAULT
				_comm.send(data, dest=rank, tag=tag)
			else:
				not_processed.append((rank, request))
		
		for x in not_processed:
			request_stack.append(x)


def send (data, work_done=False):
	global is_work_done
	
	# send data to the main warehouse
	tag = WORK_DONE if work_done else DEFAULT
	_comm.send(data, dest=_wh_rank, tag=tag)
	
	# wait for its response if there was a request in the msg
	status = MPI.Status()
	out_data =_comm.recv(source=_wh_rank, tag=MPI.ANY_TAG, status=status)
	is_work_done = is_work_done or status.Get_tag() == WORK_DONE
	return out_data


# actual data-processing helpers

def process_incomming (request):
	for key, (action, value) in request.items():
		action_dict[action].process(key, value)

def calc_feasable (request):
	return all([action_dict[action].feasable(key, value) for key, (action, value) in request.items()])

def calc_data (request):
	return {key:Entry(action=action, value=action_dict[action].data(key, value)) for key, (action, value) in request.items()}