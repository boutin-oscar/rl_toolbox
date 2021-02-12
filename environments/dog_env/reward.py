import numpy as np
import time
import matplotlib.pyplot as plt
import pybullet as p

class RewardFunc ():
	def set_epoch (self, e):
		pass

class NotDyingsRew(RewardFunc):
	def __init__(self, state):
		self.state = state
		self.min_base_clearance = 0.08
		self.all_foot_id = [9, 20, 31, 42]
		self.all_test_id = sum([[x-1, x, x+1] for x in self.all_foot_id], [])
		
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
	
	
class SpeedRew(RewardFunc):
	def __init__(self, state):
		self.state = state
		self.a = 0.5
		self.sigma_2 = 1
		self.c1 = 1*np.exp(-1)
		self.b = 0.5
	def step (self):
		return -np.sqrt(np.sum(np.square(self.state.target_speed - self.state.mean_planar_speed)))*self.c1 + self.b
	
	def done (self):
		return False
	
class TurningRew(RewardFunc):
	def __init__(self, state):
		self.state = state
		
		self.a = 0.5
		self.sigma_2 = 1
		
	def step (self):
		return np.exp(-np.sum(np.square(self.state.target_rot_speed - self.state.mean_z_rot_speed))/self.sigma_2) * self.a
	
	def done (self):
		return False
	
class PoseRew(RewardFunc):
	def __init__(self, state):
		self.state = state
		
		self.mask = np.asarray([1, 1, 1] * 4)
		self.a = 0.5 # 0.5
		
	def step (self):
		a = self.a * np.exp(-np.sum(np.square(self.state.mean_planar_speed))/0.5)
		return -np.sum(np.square(self.state.target_pose - self.state.mean_action) * self.mask) * a
	
	def done (self):
		return False
	
class JointAccRew(RewardFunc):
	def __init__(self, state):
		self.state = state
		self.target_rot_speed = 0
		
		self.a = 0#1e-6
		
	def step (self):
		return -np.sum(np.square(self.state.acc_joint_rot)) * self.a
	
	def done (self):
		return False
	
class BaseAccRew(RewardFunc):
	def __init__(self, state):
		self.state = state
		self.target_rot_speed = 0
		
		self.a_0 = 1/200
	
	def step (self):
		fac = self.a_0 * np.exp(-np.sum(np.square(self.state.mean_planar_speed))/0.5)
		return -np.sum(np.square(self.state.base_pos_acc)) * fac
	
	def done (self):
		return False
	
class BaseRotRew(RewardFunc):
	def __init__(self, state):
		self.state = state
		self.a = 0.05 # 0.05
		
	def step (self):
		
		fac = self.a
		return -np.sum(np.square(self.state.base_rot_speed[:2])) * fac
	
	def done (self):
		return False
class FootClearRew(RewardFunc):
	def __init__(self, state):
		self.state = state
		self.target_rot_speed = 0
		self.a = 0.1 # 0.05
		
		self.des_clear = 0.05
		
	def step (self):
		to_return = 0
		#for i in [1, 3, 5, 7]
		for i in [0, 1, 2, 3]:
			clear = self.state.foot_clearance[i]
			speed = self.state.foot_vel[i]
			if clear < self.des_clear:
				to_return -= np.sum(np.square(speed[:2]))*(1-clear/self.des_clear)
		return to_return*self.a
	
	def done (self):
		return False


class BaseClearRew(RewardFunc):
	def __init__(self, state):
		self.state = state
		self.target_rot_speed = 0
		self.a = 5 # 20
		
	def step (self):
		return min(self.state.base_pos[2]-0.33, 1)
	
	def done (self):
		return False
	
	