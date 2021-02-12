#nmap -A -iL hosts_64.txt -p T:22

import tensorflow as tf
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import time

import pybullet as p

from environments.dog_env import DogEnv
from models.actor import SimpleActor, MixtureOfExpert, LSTMActor
import models.lite_model as lite
#lite.load('distribution/exp_2/model.tflite')

import game_controller

if __name__ == '__main__':
	tf.config.threading.set_inter_op_parallelism_threads(1)
	print("inter_op_parallelism_threads : {}".format(tf.config.threading.get_inter_op_parallelism_threads()))
	
	debug = True
	render = False
	load_trained = True
	#actor_type = "simple"
	actor_type = "simple"
	
	env = DogEnv(debug=debug, render=render)

	#path = "results\\dog_working\\models\\expert\\{}"
	#path = "results\\exp_0_0\\models\\expert\\{}"
	#path = "results\\exp_2_best_guess\\models\\expert\\{}"
	path = "results\\exp_0\\models\\expert\\{}"
	#path = "results\\exp_0\\models\\teacher\\{}"
	
	
	if actor_type=="mix":
		primitives = [SimpleActor(env) for i in range(2)]
		actor = MixtureOfExpert(env, primitives, debug=True)
	elif actor_type == "simple":
		actor = SimpleActor(env, use_blindfold=True, use_lstm=False)
	elif actor_type == "lstm":
		actor = LSTMActor(env)
	
	if load_trained:
		actor.load(path)
		
	env.test_adr = True
	env.training_change_cmd = False
	env.carthesian_act = True
	#env.state.target_speed =  np.asarray([1, 0])*1
	#env.state.target_rot_speed = 0
	#env.training_mode = 1
	loc_config = {'init_floating':False} # {'gain_p': 0.00698, 'gain_d': 0.30}
	obs = env.reset (config = loc_config)
	obs = actor.scaler.scale_obs(obs)
	init_state = actor.get_init_state(env.num_envs)
	
	all_rew = []
	all_rew2 = []
	all_done = []
	all_stuff = [[] for i in range(1000)]
	all_obs = []
	all_influence = []
	all_act = []
	all_inf = []
	all_dev = []
	all_speed = []
	all_sim = []
	all_delta_pos = []
	
	all_anim = []
	
	"""
	env.reset(1)
	obs1, _, _ = env.step(np.asarray([0.5]*12))
	obs1 = np.expand_dims(np.asarray(obs1, dtype=np.float32), axis=1)
	act1, init_state = actor.model((obs1, init_state))
	print(act1)
	env.reset(-1)
	obs2, _, _ = env.step(np.asarray([0.5]*12))
	obs2 = np.expand_dims(np.asarray(obs2, dtype=np.float32), axis=1)
	act2, init_state = actor.model((obs2, init_state))
	print(act2)
	
	
	
	print(1/0)
	"""
	last_time = time.time()
	env.state.target_speed =  np.asarray([1, 0])*1
	env.state.target_rot_speed = 0
	fac = 1
	for i in range(10000):
		events = p.getKeyboardEvents()
		speed = 1
		rot = 0
		if 113 in events:
			rot += 1
		if 100 in events:
			rot -= 1
		if 115 in events:
			speed -=1
		if 122 in events:
			speed +=1
		p_y = 0.5
		y = env.state.base_pos[1]
		y_targ = 0
		theta_targ = min(max(p_y*(y_targ-y), -np.pi/3), np.pi/3)
		
		theta = np.arctan2(env.state.planar_speed[1], env.state.planar_speed[0])
		
		delta_theta = theta_targ-theta
		while delta_theta < np.pi:
			delta_theta += np.pi*2
		while delta_theta > np.pi:
			delta_theta -= np.pi*2
		
		p_theta = 1
		#rot = min(max(p_theta*delta_theta, -0.5), 0.5)
		"""
		if speed == 0:
			rot = 0
		"""
		"""
		theta = rot
		speed = np.cos(theta)
		rot = np.sin(theta)
		"""
		#speed = 0
		#rot = 0
		
		task = [-1, -1,-1, -1, 0, 0, 1, 1, 1, 1, 0, 0]
		#rot = task[(i//30)%len(task)]
		
		#env.set_cmd(2, rot)
		theta = np.pi*2*i/400 * 0
		targ_speed_x = 0.4 #* i/400
		targ_speed_y = 0.3
		env.state.target_speed =  np.asarray([np.cos(theta)*speed*targ_speed_x, np.sin(theta)*speed*targ_speed_y])
		env.state.target_rot_speed = 0#.25# rot/2#rot
		
		if True:
			targ_speed_x = 0.4 #* i/400
			targ_speed_y = 0.2
			max_rot = 1
			control = game_controller.get_action()
			env.state.target_speed =  np.asarray([-control[1]*targ_speed_x, control[0]*targ_speed_y])
			env.state.target_rot_speed = -control[2]*max_rot
			print(env.state.target_speed)
		if False:
			env.state.target_speed =  np.asarray([0, 0.])
			env.state.target_rot_speed = 0
		
		#print(env.state.base_clearance)
		#print(env.state.target_rot_speed)
		#print(env.adr.success)
		
		#obs = env.symetry.state_symetry(obs)
		obs = np.expand_dims(np.asarray(obs, dtype=np.float32), axis=1)
		start = time.time()
		#act, init_state = actor.model((obs, init_state))
		act, init_state = actor.model((obs, init_state))
		
		if actor_type=="mix":
			all_inf.append(actor.inf_model((obs, init_state))[0].numpy())
		dur = time.time()-start
		
		act = act.numpy()
		#act = env.symetry.action_symetry(act)
		all_act.append(act)
		act = act # + np.random.normal(size=12).reshape(act.shape) * np.exp(-1)
		#act = np.asarray([0.5]*12)
		#act = env.state.target_pose
		#act = np.asarray([0.5, 0, 0.2]*4)
		obs, rew, done = env.step(act)
		
		to_add = []
		for x in env.rewards[0].all_rew_inst:
			to_add.append(x.step() * x.a)
		all_rew2.append(to_add)
		
		obs = actor.scaler.scale_obs(np.asarray(obs))
		all_obs.append(obs)
		all_rew.append(rew[0])
		all_dev.append(env.dev)
		all_speed.append(env.state.loc_planar_speed)
		#print(rew)
		while (time.time()-last_time < 1/30) and True:
			pass
		last_time = time.time()
		
		#sim = np.mean(np.square((obs - env.symetry.state_symetry(obs))))
		#all_sim.append(sim)
		
		cur_anim = []
		for i in range(3):
			cur_anim.append(env.state.base_pos[i])
		for i in range(4):
			cur_anim.append(env.state.base_rot[i])
		for i in range(12):
			cur_anim.append(env.state.joint_rot[i])
		all_anim.append(cur_anim)
		
		phase = env.state.foot_phases[0]
		r = 0.4
		k = int(phase/(2*np.pi))
		res = phase - k*np.pi*2
		if res > 2*np.pi*r:
			z = 0
		else:
			z = (1-np.cos(res/r))
		h = 0.05
		all_delta_pos.append(h*z)
		
		#print(env.state.loc_up_vect)
	
	
	plt.plot(env.sim.band)
	plt.show()
	
	"""
	all_gen_obs = np.asarray(env.obs_generator.to_plot)
	
	for i in range(2): 
		plt.plot(all_gen_obs[:,i+12+12+12+8+3+3])
	plt.show()
	"""
	"""
	# --- actions ---
	all_act = np.asarray(all_act).reshape((-1, 12))
	for i in range(4):
		plt.plot(all_act[:,0+3*i], all_act[:,2+3*i], 'o')
	#plt.plot(all_act[:,1])
	plt.show()
	"""
	"""
	to_plot = np.asarray(env.sim.to_plot)
	for i in range(1, 12, 3):#all_gen_obs.shape[1]):
		plt.plot(all_gen_obs[:,i], label=str(i)+"obs")
		
	for i in range(1, 12, 3):
		plt.plot(to_plot[i], label=str(i)+"tru")
	
	plt.legend()
	plt.show()
	"""
	"""
	# --- foot distance to ground ---
	for i in range(4):
		l = [np.sum(np.square(v)[:2]) if d < 0.02 else 0 for d, v in zip(env.sim.to_plot[i+24], env.sim.to_plot[i+24+8+2])]
		#plt.plot(l, label=str(i))
		plt.plot(env.sim.to_plot[i+24], label=str(i)+"dist")
		#plt.plot(np.sqrt(np.sum(np.square(env.sim.to_plot[i+24+8+2])[:,:2], axis=1)), label=str(i)+"speed")
	plt.plot(all_delta_pos)
	plt.legend()
	plt.show()
	"""
	"""
	# --- speed ---
	plt.plot(np.asarray(all_speed)[:,0])
	plt.show()
	"""
	"""
	print(np.asarray(all_anim).shape)
	all_anim = np.asarray(all_anim)[50:]
	np.save("anim.npy", all_anim)
	"""
	if len(all_rew) > 0:
		print("rew :", np.mean(all_rew))
		#print("speed :", np.mean(env.sim.to_plot[8+24]
		print("speed (mean) :", np.mean(all_speed, axis=0))
		print("speed (max) :", np.max(all_speed, axis=0))
		#print("rew :", np.mean(all_rew2))
		#print("speed :", np.mean(env2.sim.to_plot[8+24]))
	print(env.kin.standard_rot(env.state.joint_rot))
	
	
	all_obs = np.asarray(env.blindfold.select_visible(all_obs))[:,0,:]
	plt.plot(all_obs[:,19:26])
	plt.show()
	
	"""
	plt.plot(all_rew)
	plt.show()
	
	for i in range(12):
		plt.plot(env.sim.to_plot[i+12])
	#plt.plot(env.sim.to_plot[8+24])
	plt.show()
	"""
	"""
	# --- rewards ---
	all_rew2 = np.asarray(all_rew2)
	for i in range(all_rew2.shape[1]):
		plt.plot(all_rew2[:,i], label=str(i))
	plt.plot(env.rewards[0].all_rew_inst[4].to_plot[0])
	plt.plot(env.rewards[0].all_rew_inst[4].to_plot[1])
	plt.legend()
	plt.show()
	"""
	"""
	all_speed = np.asarray(all_speed)
	plt.plot(all_speed[:,0])
	plt.plot(all_speed[:,1])
	plt.show()
	"""
	"""
	# --- pos/torque ---
	for i in range(12):
		#plt.plot(env.sim.to_plot[i*3], label=str(i))
		plt.plot(env.sim.to_plot[i+40], env.sim.to_plot[i+40+12], label=str(i))
		#plt.plot(env.sim.to_plot[i+40], label=str(i))
	plt.show()
	"""
	"""
	ori = env.sim.to_plot[2+40][-1]
	plt.plot([i/240 for i in range(len(env.sim.to_plot[2+40]))], np.asarray(env.sim.to_plot[2+40])-ori, label=str(2))
	mov = np.load("mov.npy")
	plt.plot(mov[:,1]+0.05, mov[:,0]-mov[:,0][-1])
	"""
	"""
	for i in range(12):
		plt.plot(np.asarray(env.sim.to_plot[i+40])/np.asarray(env.sim.to_plot[i+40+12]), label=str(i))
	
	#plt.plot(env.sim.to_plot[4+24])
	plt.legend()
	plt.show()
	"""
	print(env.state.base_clearance-0.32174531481227875)
	if len(env.sim.raw_frames) > 0:
		with open("results/video/raw.out", "wb") as f:
			f.write(np.stack(env.sim.raw_frames).tostring())
	
	
	
	
	