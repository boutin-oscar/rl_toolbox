import os



def test_ppo ():
	import configuration.processes as nodes
	from ppo.node import train_ppo

	exp_path = nodes.setup_exp()
	
	env = nodes.dog_env ()
	actor = nodes.simple_actor(	obs_dim = env.obs_dim,
								act_dim = env.act_dim, 
								obs_mean = env.obs_mean, 
								obs_std = env.obs_std, 
								blindfold = env.blindfold,
								save_path = os.path.join(exp_path, "models", "expert", "{}"))
					
	ppo_config_test = dict(	epoch_nb = 2, # 20000,
							rollout_per_epoch = 10,
							rollout_len = 100, #400,
							train_step_per_epoch = 6,
							init_log_std = -1,
							model_save_interval = 10,
							adr_test_prob = 0.3,
							tensorboard_path = os.path.join(exp_path, "tensorboard", "expert"))
			
	ppo_config = dict(	epoch_nb = 20000,
						rollout_per_epoch = 10,
						rollout_len = 400,
						train_step_per_epoch = 6,
						init_log_std = -1,
						model_save_interval = 10,
						adr_test_prob = 0.3,
						tensorboard_path = os.path.join(exp_path, "tensorboard", "expert"))
					
	train_ppo(actor, env, **ppo_config_test) # for testing purposes
	#train_ppo(actor, env, **ppo_config)


def test_rts ():
	import configuration.processes as nodes
	from ppo.node import train_ppo
	from rts.env import Simulator, RealEnv, SuperEnv, TunableEnv
	from rts.networks import create_actor, create_dq_actor, create_disc
	from rts.node import generate_trans_batch, train_discrim

	exp_path = nodes.setup_exp()
	
	ep_scaler = 1
	
	ppo_config_main = dict(	epoch_nb = 1000//ep_scaler,
							rollout_per_epoch = 10,
							rollout_len = 200,
							train_step_per_epoch = 6,
							init_log_std = -1,
							model_save_interval = 10,
							adr_test_prob = 0.3)
	
	ppo_config_dq = dict(	epoch_nb = 1000//ep_scaler,
							rollout_per_epoch = 10,
							rollout_len = 50,
							train_step_per_epoch = 6,
							init_log_std = -2,
							model_save_interval = 10,
							adr_test_prob = 0.3)
						
	
	
	sim = Simulator()
	
	actor = create_actor(sim, save_path = os.path.join(exp_path, "models", "expert", "{}"))
	dq_actor = create_dq_actor(sim, save_path = os.path.join(exp_path, "models", "dq", "{}"))
	disc = create_disc(sim, save_path = os.path.join(exp_path, "models", "disc", "{}"))
	
	nodes.load_actor (actor, os.path.join("results", "safe", "models", "expert", "{}"))
	nodes.load_actor (dq_actor, os.path.join("results", "safe", "models", "dq", "{}"))
	# nodes.load_actor (dq_actor, os.path.join("results", "first_dq", "models", "dq", "{}"))
	nodes.load_actor (disc, os.path.join("results", "safe", "models", "disc", "{}"))
	
	real_env = RealEnv(sim)
	super_env = SuperEnv(sim, dq_actor)
	tunable_env = TunableEnv(sim, actor, disc)
	
	all_dq_actors = [dq_actor]
	all_super_env = [super_env]
	"""
	# training a first actor
	ppo_config_main["tensorboard_path"] = os.path.join(exp_path, "tensorboard", "expert_super")
	train_ppo(actor, super_env, **ppo_config_main)
	"""
	# interacting with the real world
	real_trans_path = os.path.join(exp_path, "data", "real_batch_0")
	rollout_nb = 100//ep_scaler
	generate_trans_batch(	env = real_env, 
							actor = actor, 
							rollout_nb = rollout_nb, 
							rollout_len = 50,
							log_std = -2,
							save_path = real_trans_path)
		
	for i in range(10):
		# training the discriminator
		train_discrim (	disc = disc,
						real_trans_path = real_trans_path,
						all_env = all_super_env,
						# all_env = [real_env],
						actor = actor,
						epoch_nb = 1, #30//ep_scaler,
						train_step_per_epoch = 100,
						rollout_per_epoch = rollout_nb,
						rollout_len = 50,
						log_std = -2,
						model_save_interval = 10,
						tensorboard_path = os.path.join(exp_path, "tensorboard", "disc_"+str(i)))
		
		# creating a new dq_actor
		dq_actor = create_dq_actor(sim, save_path = os.path.join(exp_path, "models", "dq_"+str(i), "{}"))
		super_env = SuperEnv(sim, dq_actor)
		all_dq_actors.append(dq_actor)
		all_super_env.append(super_env)
		
		
		ppo_config_dq["tensorboard_path"] = os.path.join(exp_path, "tensorboard", "dq_tunable_"+str(i))
		train_ppo(dq_actor, tunable_env, **ppo_config_dq)
		
		# ----------- debug --------------
		generate_trans_batch(	env = super_env, 
								actor = actor, 
								rollout_nb = 100//ep_scaler, 
								rollout_len = 200,
								log_std = -2,
								save_path = os.path.join(exp_path, "data", "synth_batch_"+str(i)))

	if nodes.mpi_role == "main":
		print("End of training", flush=True)
	

# entry point of the distributed programm, selection of the algorithm
main_programm = test_rts