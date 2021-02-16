import os

import configuration.processes as nodes

def test_ppo ():
	
	exp_path = nodes.setup_exp()
	
	env = nodes.dog_env ()
	actor = nodes.simple_actor(env, save_path = os.path.join(exp_path, "models", "expert", "{}"), use_blindfold = True)
	
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
	
	#nodes.train_ppo(actor, env, **ppo_config_test)
	nodes.train_ppo(actor, env, **ppo_config)




# entry point of the distributed programm, selection of the algorithm
main_programm = test_ppo
