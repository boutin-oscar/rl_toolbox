import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

class DogState:
	def __init__ (self, adr, config={}):
		self.adr = adr
		self.adr.add_param("max_random_rot", 2, 10/1000, 10) # degrees
		
		self.config = config
		self.reset()
		
		
	
	def reset (self):
		self.base_pos = [0, 0, 0.25]
		self.base_rot = [0, 0, 0, 1]
		self.joint_rot = [0]*12
		self.joint_target = [0]*12
		self.fake_target = [0]*12
		
		self.base_pos_speed = [0, 0, 0]
		self.base_rot_speed = [0, 0, 0]
		self.joint_rot_speed = [0]*12
		
		self.base_clearance = 0
		self.foot_clearance = [0]*4
		self.foot_vel = [0]*4
		self.foot_phases = [0, np.pi, np.pi, 0]
		self.foot_f0 = np.asarray([1, 1, 1, 1]) * (self.config['foot_f0'] if 'foot_f0' in self.config else 1.5) # 1.5
		self.joint_deadband = np.asarray([0.02*0.3, 0.02, 0.04]*4) * 1# 0.3
		
		
		self.base_rot_mat = np.identity(3)
		self.planar_speed = [0, 0]
		self.loc_planar_speed = [0, 0]
		self.loc_up_vect = [0, 0, 1]
		self.new_loc_up_vect = [0, 0, 1]
		self.loc_pos_speed = [0, 0, 0]
		self.loc_rot_speed = [0, 0, 0]
		
		self.rot_speed_noise = np.asarray([0.004, 0.004, 0.01]) * 1
		
		self.mean_planar_speed = np.asarray([0, 0])
		self.mean_z_rot_speed = 0
		self.mean_joint_rot = [0]*12
		self.mean_action = [0.5]*12
		self.target_pose = np.asarray([0.5, 0.5, 0.3] * 4)
		self.all_kp = [100]*12
		
		self.acc_joint_rot = [0]*12
		self.last_joint_rot_speed = [0]*12
	
		self.target_speed = np.asarray([0, 0])*1
		self.target_rot_speed = 0
		
		self.frame = 0
		
		self.joint_torque = [0]*12
		#self.contact_force = [0]*4
		self.base_pos_acc = [0, 0, 0]
		self.base_rot_acc = [0, 0, 0]
		self.act_offset = np.zeros((12,))
		s = 2*np.pi/180
		self.joint_offset = np.zeros((12,)) + np.random.uniform(-s, s, size=(12,))
		if 'joint_offset' in self.config:
			self.joint_offset = self.config['joint_offset']
			
		s = 0.05
		self.loc_up_vect_offset = [0, 0, 0] # np.random.uniform(-s, s, size=3)
		s = self.adr.value("max_random_rot")
		self.random_rot = R.from_euler('zyx', [np.random.uniform(-s, s), np.random.uniform(-s, s), np.random.uniform(-s, s)], degrees=True)
		
		self.offset_force = [0, 0, 0]
		
		self.joint_fac = [1]*12
		
		self.delay = 6
		
		self.friction = 1
		
		self.contacts = ()
		
	def calc_obs (self):
		pass
		#return self.obs_generator.generate()