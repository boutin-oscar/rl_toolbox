
import numpy as np
import time
from pathlib import Path


from .dogState import DogState
from .kinematics import Kinematics
from .simulator import Simulator
from .reward import NotDyingsRew, SpeedRew, TurningRew, PoseRew, JointAccRew, BaseAccRew, FootClearRew, BaseRotRew, BaseClearRew
from .reward_standard import FullRewStandard
from .obs_gen import FullObsGenerator


from .adr import Adr

class DogEnv():
	def __init__(self, debug=False, render=False):
		
		self.debug = debug
		self.render = render
		
		self.adr = Adr()
		self.test_adr = False
		self.adr_rollout_len = 400
		self.frame_at_speed = 0
		
		self.kin = Kinematics()
		self.state = DogState(self.adr)
		self.obs_generator = FullObsGenerator(self.state, debug)
		#self.symetry = self.obs_generator.symetry
		self.blindfold = self.obs_generator.blindfold
		self.sim = Simulator(self.state, self.adr, self.debug, self.render)
		"""
		# not dying - moving at the right speed - turning at the right speed - keeping joint acceleration low - keeping body acceleration low
		self.rewards = [NotDyingsRew(self.state), 
						SpeedRew(self.state), 
						TurningRew(self.state), 
						PoseRew(self.state), 
						JointAccRew(self.state), 
						BaseAccRew(self.state),
						BaseRotRew(self.state),
						FootClearRew(self.state),
						BaseClearRew(self.state)]
		"""
		self.rewards = [FullRewStandard(self.state)]
		
		# multiple obs and act in one obs
		self.obs_pool = []
		self.pool_len = self.obs_generator.obs_transition_len # TODO : unify this with obs_gen in a way that makes sense
		
		self.obs_dim = self.blindfold.obs_dim
		self.act_dim = 12
		self.num_envs = 1
		self.obs_mean = self.obs_generator.mean
		self.obs_std = self.obs_generator.std
		
		
		if self.debug:
			self.to_plot = [[] for i in range(100)]
		
		
		# --- setting up the adr ---
		self.adr.add_param("theta", 0, 0.01, 1)
		
		"""
		# --- training settings ---
		self.train_continuous = True
		self.train_speed = []
		self.train_rot_speed = []
		self.training_change_cmd = True
		self.only_forward = False
		self.has_rand_act_delta = False
		self.carthesian_act = True
		"""
		
		self.training_mode = 0
		
	
	def step(self, action):
		
		act = action.flatten()
		act_delta = self.state.act_offset # -np.ones(act.shape) * 0.1
		#act_delta = np.random.normal(size=act.shape) * self.adr.value("act_rand_std") + self.act_offset
		act_rand = act + act_delta
		self.state.foot_phases += 2*np.pi*self.state.foot_f0/30
		self.state.foot_phases = list(np.fmod(self.state.foot_phases, 2*np.pi))
		#legs_angle = self.kin.calc_joint_target (act_rand, self.state.foot_phases, self.state.new_loc_up_vect)
		legs_angle = self.kin.calc_joint_target (act_rand, self.state.foot_phases)
		#print(legs_angle)
		#legs_angle = act_rand
		self.sim.step(act_rand, legs_angle)
		
		all_rew = [reward.step() for reward in self.rewards]
		rew = np.sum(all_rew)
		done = np.any([reward.done() for reward in self.rewards])
		
		#self.obs_pool = self.obs_pool[2:] + self.state.calc_obs() + [act]
		self.obs_pool = self.obs_pool[1:] + self.obs_generator.generate()
		
		
		if self.debug:
			for reward, s in zip(all_rew, self.to_plot):
				s.append(reward)
		
		if self.test_adr:
			pos_speed_deviation = np.sum(np.square(self.state.target_speed - self.state.mean_planar_speed))
			rot_speed_deviation = np.square(self.state.target_rot_speed - self.state.mean_z_rot_speed)
			self.dev = pos_speed_deviation + rot_speed_deviation
			if pos_speed_deviation + rot_speed_deviation < np.square(0.2):
				self.frame_at_speed += 1
			adr_success = not done and self.frame_at_speed > 400*0.7
			self.adr.step(adr_success, not adr_success)
			#self.adr.step(adr_success, False)
		else:
			self.adr.step(False, False)
		
		if not self.debug:
			self.reset_cmd()
		
		
		return self.calc_obs(), [rew], [done]
		
	def reset(self, config={}):
		self.state.config = config
		
		self.adr.reset()
		self.cum_rew = 0
		self.frame_at_speed = 0
		
		
		self.all_cmd = []
			
		
		#target_pose =  np.asarray([0.7, 0.7, 0.1, 0.7, 0.3, 0.1] + [0.3, 0.7, 0.2, 0.3, 0.3, 0.2])
		#target_pose =  np.asarray([0.7, 0.7, 0.1, 0.7, 0.3, 0.1] + [0.3, 0.7, 0.1, 0.3, 0.3, 0.1])
		target_pose =  np.asarray([0.5, 0.5, 0.2] * 4)
		
		
		des_v = 0
		des_clear = 0
		
		act = target_pose
		legs_angle = np.asarray(self.kin.calc_joint_target (act, [0, np.pi, np.pi, 0], [0, 0, 1]))
		
		self.sim.reset(des_v, des_clear, legs_angle)
		self.state.joint_fac = self.kin.standard_rot([1]*12)
		
		self.state.target_pose = target_pose
		self.state.mean_action = target_pose*1
		
		# --- setting the target speed and rot ---
		if not self.debug:
			self.reset_cmd()
		
		self.obs_pool = self.obs_generator.generate()# + [act]
		
		for i in range(self.pool_len-1):
			#self.sim.step(act, legs_angle)
			self.obs_pool += self.obs_generator.generate()# + [act]
		
		return self.calc_obs()
	
	def choose_cmd (self, step=0):
		# self.training_mode (0:onlyforward, 1:smartforward)
		if self.training_mode == 0:
			"""
			max_v_targ = 0.2
			theta = np.random.random()*np.pi*2
			rot = np.random.uniform(-0.5, 0.5)
			#theta = np.random.randint(2) * np.pi
			return (np.cos(theta)*max_v_targ, np.sin(theta)*max_v_targ, rot)
			"""
			found = False
			if np.random.random() < 0.5:
				while not found:
					vx = np.random.uniform(-1, 1)
					vy = np.random.uniform(-1, 1)
					found = (vx*vx + vy*vy < 1)
			else:
				theta = np.random.random()*2*np.pi
				vx = np.cos(theta)
				vy = np.sin(theta)
			rot = np.random.uniform(-1, 1)
			vx *= 0.4
			vy *= 0.3
			rot *= 1
			return (vx, vy, rot)
			
		elif self.training_mode == 1:
			r = np.random.random()
			theta = self.adr.value("theta")
			is_test = self.adr.is_test_param("theta")
			theta = 1
			if r < 1/3:
				return (max_v_targ, 0, theta * (-1 if is_test else -np.random.random()))
			elif r < 2/3:
				return (max_v_targ, 0, 0)
			elif r < 3/3:
				return (max_v_targ, 0, theta * (1 if is_test else np.random.random()))
			elif r < 4/4:
				return (0, 0, 0)
		else:
			print("training_mode {} not supported".format(self.training_mode))

	def reset_cmd (self):
		i = self.state.frame//100
		if i >= len(self.all_cmd):
			self.all_cmd.append(self.choose_cmd(i))
			
		targ_speed_x, targ_speed_y, targ_rot = self.all_cmd[i]
		self.state.target_speed = np.asarray([targ_speed_x, targ_speed_y])
		self.state.target_rot_speed = targ_rot
	
	def calc_obs (self):
		return [np.concatenate(self.obs_pool)]
	
	def close(self):
		self.adr.close()
		self.sim.close()
	
	def set_epoch (self, ep):
		pass
	
	