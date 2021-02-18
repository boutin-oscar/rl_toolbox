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
	from rts.node import generate_trans_batch

	exp_path = nodes.setup_exp()
		
	ppo_config_main = dict(	epoch_nb = 1000,
							rollout_per_epoch = 10,
							rollout_len = 200,
							train_step_per_epoch = 6,
							init_log_std = -1,
							model_save_interval = 10,
							adr_test_prob = 0.3)
	
	ppo_config_dq = dict(	epoch_nb = 1000,
							rollout_per_epoch = 10,
							rollout_len = 200,
							train_step_per_epoch = 6,
							init_log_std = -1,
							model_save_interval = 10,
							adr_test_prob = 0.3)
						
	
	
	sim = Simulator()
	
	actor = create_actor(sim, save_path = os.path.join(exp_path, "models", "expert", "{}"))
	dq_actor = create_dq_actor(sim, save_path = os.path.join(exp_path, "models", "dq", "{}"))
	disc = create_disc(sim, save_path = os.path.join(exp_path, "models", "disc", "{}"))
	
	nodes.load_actor (actor, os.path.join("results", "safe", "models", "expert", "{}"))
	nodes.load_actor (dq_actor, os.path.join("results", "safe", "models", "dq", "{}"))
	nodes.load_actor (disc, os.path.join("results", "safe", "models", "disc", "{}"))
	
	real_env = RealEnv(sim)
	super_env = SuperEnv(sim, dq_actor)
	tunable_env = TunableEnv(sim, actor, disc)
	
	"""
	ppo_config_main["tensorboard_path"] = os.path.join(exp_path, "tensorboard", "expert_super")
	train_ppo(actor, super_env, **ppo_config_main)
	"""
	if nodes.mpi_role == "main":
		print("Starting generation", flush=True)
	
	generate_trans_batch(	env = real_env, 
							actor = actor, 
							rollout_nb = 100, 
							rollout_len = 200,
							save_path = os.path.join(exp_path, "data", "batch_0"))
	
	
	"""
	ppo_config_test["tensorboard_path"] = os.path.join(exp_path, "tensorboard", "expert_real")
	train_ppo(actor, real_env, **ppo_config_test)
	ppo_config_test["tensorboard_path"] = os.path.join(exp_path, "tensorboard", "expert_super")
	train_ppo(actor, super_env, **ppo_config_test)
	ppo_config_test["tensorboard_path"] = os.path.join(exp_path, "tensorboard", "dq_tunable")
	train_ppo(dq_actor, tunable_env, **ppo_config_test)
	"""

# entry point of the distributed programm, selection of the algorithm
main_programm = test_rts