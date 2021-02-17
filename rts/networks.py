
import configuration.processes as nodes
import warehouse

def create_actor (sim, save_path):
	return nodes.simple_actor(	obs_dim = sim.obs_dim,
								act_dim = sim.act_dim, 
								save_path = save_path)

def create_dq_actor (sim, save_path):
	return nodes.simple_actor(	obs_dim = sim.obs_dim + sim.act_dim,
								act_dim = sim.obs_dim, 
								save_path = save_path)

@nodes.process_node
def create_disc (sim, save_path):
	import os
	from rts.discriminator import Discriminator
	
	if save_path is None:
		raise NameError("A save_path should always be given to a discriminator")
		
	disc = Discriminator (sim.obs_dim*2 + sim.act_dim)
	disc.save_path = save_path
	
	if nodes.mpi_role == 'main':
		os.makedirs(disc.save_path)
		disc.save(disc.save_path)
		
		data_out = {nodes.pnid+":actor_weight":warehouse.Entry(action="set", value=disc.get_weights())}
		data = warehouse.send(data_out)
	
	data_out = {nodes.pnid+":actor_weight":warehouse.Entry(action="get", value=None)}
	data = warehouse.send(data_out)
	disc.set_weights(data[nodes.pnid+":actor_weight"].value)
	
	return disc