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
	
	#train_ppo(actor, env, **ppo_config_test)
	train_ppo(actor, env, **ppo_config)


def test_rts ():
	import configuration.processes as nodes
	from ppo.node import train_ppo
	from rts.env import Simulator, RealEnv, SuperEnv, TunableEnv
	from rts.networks import create_actor, create_dq_actor, create_disc

	exp_path = nodes.setup_exp()
	
	
	sim = Simulator()
	
	actor = create_actor(sim, save_path = os.path.join(exp_path, "models", "expert", "{}"))
	dq_actor = create_dq_actor(sim, save_path = os.path.join(exp_path, "models", "dq", "{}"))
	disc = create_disc(sim, save_path = os.path.join(exp_path, "models", "disc", "{}"))
	



# entry point of the distributed programm, selection of the algorithm
main_programm = test_rts
