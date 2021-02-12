import numpy as np
import time
import matplotlib.pyplot as plt
import pybullet as p

class RewardFunc ():
	def set_epoch (self, e):
		pass

class FullRewStandard (RewardFunc):
	def __init__ (self, state):
		self.all_rew_class = [NotDyingsRew, PoseRew, Rlv, Rav, Rb, Rfc, Rbc, Rtau]
		self.all_rew_inst = [x(state) for x in self.all_rew_class]
		
	def step (self):
		return np.sum([x.step() * x.a for x in self.all_rew_inst])
	
	def done (self):
		return np.any([x.done() for x in self.all_rew_inst])

class NotDyingsRew(RewardFunc):
	def __init__(self, state):
		self.state = state
		self.min_base_clearance = 0.08
		self.all_foot_id = [2, 5, 8, 11]
		#self.all_test_id = sum([[x-1, x, x+1] for x in self.all_foot_id], [])
		self.all_test_id = self.all_foot_id
		
		
		self.a = 0.5
		
	def step (self):
		return -1 if self.done() else 0
	
	def done (self):
		base_done = self.state.base_clearance < self.min_base_clearance
		#foot_done = np.any([x < 0.01 for x in self.state.foot_clearance[::2]])
		
		contact_done = False
		
		for x in self.state.contacts:
			if x[3] not in self.all_test_id:
				contact_done = True
		
		return base_done or contact_done
	
class PoseRew(RewardFunc):
	def __init__(self, state):
		self.state = state
		
		self.mask = np.asarray([1, 1, 1] * 4)
		self.a = 0.02 # 0.05
		
	def step (self):
		a = np.exp(-np.sum(np.square(self.state.mean_planar_speed))/0.5)
		return -np.sum(np.square(self.state.target_pose - self.state.mean_action) * self.mask) * a
	
	def done (self):
		return False
	
class Rlv (RewardFunc):
	def __init__ (self, state):
		self.state = state
		self.max_vel = 0.2
		self.a = 0.05
		
	def step (self):
		"""
		norm = np.sqrt(np.sum(np.square(self.state.target_speed)))
		if norm < 0.01:
			return 0
		
		dir = self.state.target_speed / norm
		
		loc_v = np.asarray(self.state.loc_planar_speed)
		vpr = np.sum(loc_v * dir)
		if vpr > self.max_vel:
			return 1
		else:
			#return np.exp(-2.0 * np.square(vpr - self.max_vel))
			return np.exp(-(1/0.08) * np.square(vpr - self.max_vel))
		"""
		return np.exp(-(1/0.02) * np.sum(np.square(self.state.loc_planar_speed - self.state.target_speed)))
		#return np.exp(-(1/0.04) * np.sum(np.square(self.state.loc_planar_speed - self.state.target_speed)))
		#return np.exp(-2 * np.sum(np.square(self.state.loc_planar_speed - self.state.target_speed)))
	def done (self):
		return False


class Rav (RewardFunc):
	def __init__ (self, state):
		self.state = state
		self.max_vel = 0.6
		self.a = 0.05
		
	def step (self):
		"""
		norm = self.state.target_rot_speed
		if norm < 0.01:
			return 0
		
		dir = self.state.target_rot_speed / norm
		wpr = self.state.mean_z_rot_speed * dir
		if wpr > self.max_vel:
			return 1
		else:
			return np.exp(-1.5 * np.square(wpr - self.max_vel))
		"""
		return np.exp(-1.5 * np.square(self.state.base_rot_speed[2] - self.state.target_rot_speed))
	
	def done (self):
		return False


class Rb (RewardFunc):
	def __init__ (self, state):
		self.state = state
		self.max_vel = 0.6
		self.a = 0.04
		self.c = 0
		self.to_plot = [[] for i in range(100)]
		
	def step (self):
		norm_vel = np.sqrt(np.sum(np.square(self.state.target_speed)))
		norm_rot = self.state.target_rot_speed
		if norm_vel < 0.01:
			v0_2 = np.square(norm_vel)
		else:
			dir_vel = self.state.target_speed / norm_vel
			loc_v = np.asarray(self.state.loc_planar_speed)
			v0_2 = np.sum(np.square(loc_v - loc_v * dir_vel))
		speed_rew = np.exp(-1.5 * v0_2)
		rot_rew = np.exp(-1.5 * np.sum(np.square(self.state.base_rot_speed[:2])))
		"""
		if self.c%2 == 0:
			self.to_plot[0].append(self.state.base_rot_speed[0])
			self.to_plot[1].append(self.state.base_rot_speed[1])
		self.c += 1
		"""
		return speed_rew + rot_rew
	
	def done (self):
		return False

class Rfc (RewardFunc): # < --  a diminumer
	def __init__ (self, state):
		self.state = state
		self.a = 0.4 # 0.1
	
	def step (self):
		n_swing = 0
		rew = 0
		for i in range(4):
			rew += np.minimum(self.state.foot_clearance[i], 0.02) # 0.04
		return rew
		"""
		for i in range(4):
			r = 0.2
			phase = self.state.foot_phases[i]
			k = int(phase/(2*np.pi))
			res = phase - k*np.pi*2
			z = 1-np.cos(res/r)
			h = 0.1
			if res < r*2*np.pi and z >= 0.:
				n_swing += 1
				if self.state.foot_clearance[i] >= z*h*0 + 0.01 - 1e-4:
					rew += 1 #1
				rew=z
		if n_swing == 0:
			return 0
		return rew/n_swing
		"""
	
	def done (self):
		return False
"""
class Rfc (RewardFunc):
	def __init__ (self, state):
		self.state = state
		self.a = 0.01
	
	def step (self):
		n_swing = 0
		rew = 0
		for i in range(4):
			if np.sin(self.state.foot_phases[i]) > 0.:
				n_swing += 1
				rew += self.state.foot_clearance[i]/0.01
		if n_swing == 0:
			return 0
		return rew/n_swing
	
	def done (self):
		return False
"""
class Rbc (RewardFunc):
	def __init__ (self, state):
		self.state = state
		self.a = 0.02
		
		self.all_foot_id = [2, 5, 8, 11]
		self.all_test_id = sum([[x] for x in self.all_foot_id], [])
	
	def has_contact (self):
		contact = False
		for x in self.state.contacts:
			if x[3] not in self.all_test_id:
				contact = True
		return contact
	
	def step (self):
		return -1 if self.has_contact() else 0
	
	def done (self):
		return False

class Rtau (RewardFunc):
	def __init__ (self, state):
		self.state = state
		self.a = 5e-5
	
	def step (self):
		return - np.sum(np.abs(self.state.joint_torque))
	
	def done (self):
		return False