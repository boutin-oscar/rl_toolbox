import numpy as np
import pybullet as p
import matplotlib.image as mpimg
import time
from pathlib import Path
import os, sys

"""
jointType : 
JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
"""

class Simulator():

	def __init__(self, state, adr, debug=False, render=False):
		# --- Step related ---
		self.state = state
		self.adr = adr
		
		self.timeStep = 1/240
		self.frameSkip = 8
		
		self.lowpass_joint_f = 15 # Hz
		self.lowpass_joint_alpha = min(1, self.timeStep*self.lowpass_joint_f)
		
		self.lowpass_rew_f = 5 # 5 # Hz
		self.lowpass_rew_alpha = min(1, self.timeStep*self.lowpass_rew_f*self.frameSkip)
		
		# --- Render-related ---
		self.debug = debug
		self.render = render
		self.first_render = True
		self.render_path = None
		self.raw_frames = []
		self.frame = 0
		self.frame_per_render = 4
	
		# --- Connecting to the right server ---
		if self.debug:
			self.pcId = p.connect(p.GUI)
			p.resetDebugVisualizerCamera (1, 0, 0, [0, 0, 0.3])
			#p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
			p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
			
			self.to_plot = [[] for i in range(100)]
		else:
			self.pcId = p.connect(p.DIRECT)
			
		# --- Loading the meshes ---
		urdf_path = str(Path(__file__).parent) + "/urdf"
		self.groundId = p.loadURDF(urdf_path + "/plane_001/plane.urdf", physicsClientId=self.pcId)
		#self.robotId = p.loadURDF(urdf_path + "/robot_001/robot.urdf", [0,0,1], physicsClientId=self.pcId)
		#self.robotId = p.loadURDF(urdf_path + "/robot_002_1/robot.urdf", [0,0,1], physicsClientId=self.pcId)
		self.robotId = p.loadURDF(urdf_path + "/robot_002_1/robot.urdf", [0,0,0], flags=p.URDF_MERGE_FIXED_LINKS, physicsClientId=self.pcId)
		
		self.urdf_joint_indexes = [0,1,2,6,7,8,9,10,11,3,4,5]
		#self.urdf_joint_indexes = [2,4,8,13,15,19,24,26,30,35,37,41]
		
		p.setPhysicsEngineParameter(fixedTimeStep=self.timeStep, physicsClientId=self.pcId)
		
		# --- setting the right masses ---
		p.changeDynamics(self.robotId, -1, mass=6.7, physicsClientId=self.pcId)
		for i in range(p.getNumJoints(self.robotId, physicsClientId=self.pcId)):
			mass = p.getDynamicsInfo(self.robotId, i, physicsClientId=self.pcId)[0]
			p.changeDynamics(self.robotId, i, mass=mass*2/1.789, physicsClientId=self.pcId)
		if False:
			for i in range(-1, 12):
				print("mass of link {} : {}".format(i, p.getDynamicsInfo(self.robotId, i, physicsClientId=self.pcId)[0]))
		
		# --- setting up the adr ---
		self.default_kp = 60  # 108
		self.min_kp = self.default_kp * 0.8
		self.max_kp = self.default_kp * 1.2
		self.min_knee_dkp = self.default_kp * 0.6
		self.adr.add_param("min_kp", self.default_kp, -1, self.min_kp)
		self.adr.add_param("max_kp", self.default_kp, 1, self.max_kp)
		self.adr.add_param("knee_min_kp", self.default_kp, -1, self.min_knee_dkp)
		
		self.adr.add_param("max_offset_force", 0, 0.1, 100)
		self.adr.add_param("max_perturb_force", 0, 1, 1000)
		
		self.adr.add_param("min_friction", 0.7, -1/1000, 0.1)
		
		self.band = []
		
		
	def step (self, action, joint_target):
		
		delay = 0 # self.state.delay
		for t in range(delay):
			self.add_perturb_force ()
			self.update_joint_drive()
			p.stepSimulation (physicsClientId=self.pcId)
			if self.render:
				self.render_frame()
			
			if self.debug:
				base_pos, base_rot = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.pcId)
				p.resetDebugVisualizerCamera (1, 30, -15, base_pos, physicsClientId=self.pcId)
				
			
		self.update_joint_lowpass (joint_target)
			
		for t in range(self.frameSkip-delay):
			self.add_perturb_force ()
			self.update_joint_drive()
			p.stepSimulation (physicsClientId=self.pcId)
		
			if self.render:
				self.render_frame()
			
			if self.debug:
				base_pos, base_rot = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.pcId)
				#p.resetDebugVisualizerCamera (1, 90, 0, base_pos, physicsClientId=self.pcId)
				p.resetDebugVisualizerCamera (1, 30, -15, base_pos, physicsClientId=self.pcId)
				
				
		self.update_state(action)
		
		if self.debug:
			for i in range(12):
				self.to_plot[i].append(self.state.joint_rot[i]*self.state.joint_fac[i])
				#self.to_plot[i+12].append(self.state.joint_rot_speed[i])mean_action
				#self.to_plot[i+12].append(action[i])
				self.to_plot[i+12].append(self.state.joint_rot_speed[i])
				#self.to_plot[i+12].append(self.state.joint_torque[i])
			for i in range(4):
				self.to_plot[i+24].append(self.state.foot_clearance[i])
			self.to_plot[24+4].append(self.state.base_clearance)
			for i in range(4):
				self.to_plot[i+24+8+2].append(self.state.foot_vel[i])
			self.to_plot[8+24].append(self.state.mean_planar_speed[0])
			self.to_plot[9+24].append(self.state.mean_planar_speed[1])
	
	def add_perturb_force (self):
		
		perturb_force_norm = self.adr.value("max_perturb_force")*np.random.random()
		found = False
		while not found:
			perturb_force = np.random.normal(size=3)
			norm_2 = np.sum(np.square(perturb_force))
			if norm_2 >= 1e-5:
				found = True
				perturb_force = perturb_force*perturb_force_norm/np.sqrt(norm_2)
		
		if self.state.frame%30 == 0:
			perturb_force = [perturb_force[i] + self.state.offset_force[i] for i in range(3)]
		else:
			perturb_force = [perturb_force[i] for i in range(3)]
		
		p.applyExternalForce(self.robotId, -1, perturb_force, [0, 0, 0], p.LINK_FRAME, physicsClientId=self.pcId)
	
	def update_joint_drive (self):
		max_vel = 2000
		timestep = 1/240
		kd = 0.1 # 0.1 # 0.13
		gain_d = timestep * kd / (1/240)
		if 'gain_d' in self.state.config:
			gain_d = self.state.config['gain_d']
		for i in range(12):
			true_target = self.state.joint_target[i] + self.state.joint_offset[i]
			urdf_joint_id = self.urdf_joint_indexes[i]
			
			kp = self.state.all_kp[i]
			if i%3 == 2:
				kp *= 0.8
			gain_p = timestep * timestep * kp / (0.27) # 0.003789
			if 'gain_p' in self.state.config:
				gain_p = self.state.config['gain_p']
			
			deadband = self.state.joint_deadband[i]
			data = p.getJointState(self.robotId, urdf_joint_id, physicsClientId=self.pcId)
			cur_pos = data[0]
			delta_target = cur_pos - true_target
			if delta_target > deadband:
				delta_target = deadband
			elif delta_target < -deadband:
				delta_target = -deadband
			fake_target = true_target+delta_target
			#f = 10 # 8 # abs(delta_target/deadband)*8
			f = 10 * abs(delta_target/deadband)
			self.state.fake_target[i] = fake_target
			p.setJointMotorControl2(self.robotId, urdf_joint_id, p.POSITION_CONTROL, targetPosition=fake_target, force=f, maxVelocity=max_vel, positionGain=gain_p, velocityGain=gain_d, physicsClientId=self.pcId)
			
			if self.debug:
				self.to_plot[i+40].append(data[0])
				self.to_plot[i+40+12].append(data[3])
	
	def update_joint_lowpass (self, joint_target):
		
		lamb = 1 # 0.403
		for i in range(12):
			delta = (joint_target[i] - self.state.joint_target[i])
			self.state.joint_target[i] = self.state.joint_target[i] + delta*lamb
		#self.state.joint_target += 0.3 * np.asarray([1, 0, 0, -1, 0, 0, -1, 0, 0, 1, 0, 0])
		
	def update_state (self, action):
		# experimental
		new_pos_speed, new_rot_speed = p.getBaseVelocity(self.robotId, physicsClientId=self.pcId)
		new_pos_speed = np.asarray(new_pos_speed)
		new_rot_speed = np.asarray(new_rot_speed)
		self.state.base_pos_acc = (new_pos_speed-self.state.base_pos_speed)/(self.timeStep*self.frameSkip)
		self.state.base_rot_acc = (new_rot_speed-self.state.base_rot_speed)/(self.timeStep*self.frameSkip)
		
		# --- base pos and rot ---
		self.state.base_pos, self.state.base_rot = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.pcId)
		self.state.base_pos = list(self.state.base_pos)
		self.state.base_rot = list(self.state.base_rot)
		
		# --- base speed ---
		self.state.base_pos_speed, self.state.base_rot_speed = p.getBaseVelocity(self.robotId, physicsClientId=self.pcId)
		self.state.base_pos_speed = list(self.state.base_pos_speed)
		self.state.base_rot_speed = list(self.state.base_rot_speed)
		
		# --- joint pos and speed ---
		for i in range(12):
			urdf_joint_id = self.urdf_joint_indexes[i]
			data = p.getJointState(self.robotId, urdf_joint_id, physicsClientId=self.pcId)
			self.state.joint_rot[i] = data[0] - self.state.joint_offset[i]
			self.state.joint_rot_speed[i] = data[1]
			self.state.joint_torque[i] = data[3]
		
		# --- body clearances ---
		all_contact_point = p.getClosestPoints(self.robotId, self.groundId, 100, linkIndexA=-1, physicsClientId=self.pcId)
		if len(all_contact_point) == 0:
			self.state.base_clearance = 0
		else:
			_, _, _, _, _, point_pos, _, _, dist, _, _, _, _, _ = all_contact_point[0]
			self.state.base_clearance = dist
		
		# --- foot clearances ---
		#all_foot_id = [9, 20, 31, 42]#[1, 2, 4, 5, 7, 8, 10, 11]
		all_foot_id = [2, 5, 8, 11]
		for i, link_index in enumerate(all_foot_id):
			all_contact_point = p.getClosestPoints(self.robotId, self.groundId, 100, linkIndexA=link_index, physicsClientId=self.pcId)
			if len(all_contact_point) == 0:
				self.state.foot_clearance[i] = 0
				self.state.foot_vel[i] = [0,0,0]
			else:
				_, _, _, _, _, point_pos, _, _, dist, _, _, _, _, _ = all_contact_point[0]
				self.state.foot_clearance[i] = dist
				
				linkWorldPosition, linkWorldOrientation, _, _, _, _, worldLinkLinearVelocity, worldLinkAngularVelocity = p.getLinkState(self.robotId, link_index, computeLinkVelocity=True, computeForwardKinematics=True, physicsClientId=self.pcId)
				rel_pos = np.asarray(point_pos) - np.asarray(linkWorldPosition)
				world_vel = np.asarray(worldLinkLinearVelocity) + self.vect_prod(worldLinkAngularVelocity, rel_pos)
				self.state.foot_vel[i] = world_vel
		
		
		# --- local speed ---
		self.state.base_rot_mat = np.asarray(p.getMatrixFromQuaternion(self.state.base_rot)).reshape((3,3))
		self.state.planar_speed = np.asarray(self.state.base_pos_speed[:2])
		
		lamb = 0.2
		self.state.new_loc_up_vect = (np.asarray((0, 0, 1)).reshape((1,3)) @ self.state.base_rot_mat).flatten()#.tolist()
		self.state.loc_up_vect = np.asarray(self.state.loc_up_vect) * (1-lamb) + self.state.new_loc_up_vect * lamb
		self.state.loc_pos_speed = (np.asarray(self.state.base_pos_speed).reshape((1, 3)) @ self.state.base_rot_mat).flatten().tolist()
		new_loc_rot_speed = (np.asarray(self.state.base_rot_speed).reshape((1, 3)) @ self.state.base_rot_mat).flatten()
		self.state.loc_rot_speed = np.asarray(self.state.loc_rot_speed) * (1-lamb) + new_loc_rot_speed * lamb
		
		s = 0.05
		#self.state.loc_up_vect = (self.state.loc_up_vect + np.random.uniform(-s, s, size=3) + np.asarray([0.1, 0.1, 0])).tolist()
		#self.state.loc_up_vect +=np.random.uniform(-s, s, size=3)
		self.state.loc_up_vect += self.state.loc_up_vect_offset
		self.state.loc_up_vect = self.state.random_rot.apply(self.state.loc_up_vect)
		self.state.loc_up_vect /= np.sqrt(np.sum(np.square(self.state.loc_up_vect)))
		
		
		# --- local vectors ---
		planar_front = np.zeros((2,))
		planar_left = np.zeros((2,))
		
		planar_front[0] = self.state.base_rot_mat[0, 0]
		planar_front[1] = self.state.base_rot_mat[1, 0]
		planar_front /= np.sqrt(np.sum(np.square(planar_front)))
		planar_left[0] = planar_front[1]
		planar_left[1] = -planar_front[0]
		
		# --- local planar speed ---
		self.state.loc_planar_speed = [np.sum(self.state.planar_speed * planar_front), np.sum(self.state.planar_speed * planar_left)]
		
		self.state.mean_planar_speed = self.state.mean_planar_speed*(1-self.lowpass_rew_alpha) + np.asarray(self.state.loc_planar_speed)*self.lowpass_rew_alpha
		self.state.mean_z_rot_speed = self.state.mean_z_rot_speed*(1-self.lowpass_rew_alpha) + self.state.base_rot_speed[2]*self.lowpass_rew_alpha
		
		for i in range(12):
			self.state.mean_joint_rot[i] = (1-self.lowpass_rew_alpha)*self.state.mean_joint_rot[i] + self.lowpass_rew_alpha*self.state.joint_rot[i]
			self.state.mean_action[i] = (1-self.lowpass_rew_alpha)*self.state.mean_action[i] + self.lowpass_rew_alpha*action[i]
		
		for i in range(12):
			self.state.acc_joint_rot[i] = (self.state.joint_rot_speed[i] - self.state.last_joint_rot_speed[i])/(self.timeStep*self.frameSkip)
			self.state.last_joint_rot_speed[i] = self.state.joint_rot_speed[i]
		
		self.state.frame += 1
		
		
		self.state.contacts = p.getContactPoints(self.robotId, self.groundId)
		
	def reset (self, des_v, des_clear, legs_angle):
	
		self.init_floating = self.state.config['init_floating'] if 'init_floating' in self.state.config else False
		if not self.init_floating:
			p.setGravity(0, 0, -9.81, physicsClientId=self.pcId)
			
		if 'sim_per_step' in self.state.config:
			self.frameSkip = self.state.config['sim_per_step']
		
		h0 = 5
		self.state.reset()
		if self.init_floating:
			self.state.base_rot = [0.2, 0, 0, 0.9]
		p.resetBasePositionAndOrientation(self.robotId, [0, 0, h0], self.state.base_rot, physicsClientId=self.pcId)
		p.resetBaseVelocity(self.robotId, self.state.base_pos_speed, self.state.base_rot_speed, physicsClientId=self.pcId)
		for i in range(12):
			legs_angle[i] += self.state.joint_offset[i]
			self.state.joint_rot[i] = legs_angle[i]
			self.state.joint_target[i] = legs_angle[i]
			self.state.mean_joint_rot[i] = legs_angle[i]
			urdf_joint_id = self.urdf_joint_indexes[i]
			p.resetJointState(self.robotId, urdf_joint_id, legs_angle[i], 0, physicsClientId=self.pcId)
		
		act_clear = self.get_clearance ()
		h = des_clear-act_clear+h0 + (0.2 if self.init_floating else 0)
		self.state.base_pos = [0, 0, h]
		self.state.base_pos_speed = [des_v, 0, 0]
		p.resetBasePositionAndOrientation(self.robotId, self.state.base_pos, self.state.base_rot, physicsClientId=self.pcId)
		p.resetBaseVelocity(self.robotId, self.state.base_pos_speed, self.state.base_rot_speed, physicsClientId=self.pcId)
	
		# --- friction ---
		min_friction = self.adr.value("min_friction")
		self.state.friction = np.random.uniform(min_friction, 0.9) # np.random.uniform(0.7, 2) # 0.6 # np.random.uniform(0.4, 1)
		if self.adr.is_test_param("min_friction"):
			self.state.friction = min_friction
		friction = self.state.friction
		restitution = 0.95 # 0.95 # np.random.random()*0.95
		#all_foot_id = [9, 20, 31, 42]
		all_foot_id = [2, 5, 8, 11]
		#for foot_id in all_foot_id:
		for foot_id in range(12):
			for i in [0]: #[-1, 0, 1]:
				urdf_joint_id = foot_id + i
				p.changeDynamics(self.robotId, urdf_joint_id, lateralFriction=friction, restitution=restitution, physicsClientId=self.pcId)
		p.setDebugObjectColor(self.robotId, 5, (0, 0, 1), physicsClientId=self.pcId)
		#print(p.getDynamicsInfo(self.groundId, -1, physicsClientId=self.pcId)[5])
		p.changeDynamics(self.groundId, -1, lateralFriction=friction, restitution=restitution, physicsClientId=self.pcId)
		
		# --- joint kp ---
		for i in range(12):
			cur_max_kp = self.adr.value("max_kp")
			min_name = "knee_min_kp" if i%3 == 2 else "min_kp"
			cur_min_kp = self.adr.value(min_name)
			
			kp = np.random.random()*(cur_max_kp-cur_min_kp) + cur_min_kp
			if self.adr.is_test_param("max_kp"):
				kp = cur_max_kp
			if self.adr.is_test_param(min_name):
				kp = cur_min_kp
			
			self.state.all_kp[i] = kp
		
		# --- external force ---
		perturb_force_norm = self.adr.value("max_offset_force")
		if not self.adr.is_test_param("max_offset_force"):
			perturb_force_norm *= np.random.random()
		found = False
		while not found:
			offset_force = np.random.normal(size=3)
			norm_2 = np.sum(np.square(offset_force))
			if norm_2 >= 1e-5:
				found = True
				self.state.offset_force = offset_force*perturb_force_norm/np.sqrt(norm_2)
				self.state.offset_force = [self.state.offset_force[i] for i in range(3)]
		
		
		self.state.delay = 3 + np.random.randint(5)
		
	def get_clearance (self):
		to_return = 100
		all_foot_id = list(range(p.getNumJoints(self.robotId, physicsClientId=self.pcId)))
		for i, link_index in enumerate(all_foot_id):
			all_contact_point = p.getClosestPoints(self.robotId, self.groundId, 100, linkIndexA=link_index, physicsClientId=self.pcId)
			if len(all_contact_point) > 0:
				_, _, _, _, _, point_pos, _, _, dist, _, _, _, _, _ = all_contact_point[0]
				to_return = min(to_return, dist)
		return to_return
	
	def vect_prod (self, a, b):
		to_return = np.empty((3,), dtype=np.float32)
		to_return[0] = a[1]*b[2] - a[2]*b[1]
		to_return[1] = a[2]*b[0] - a[0]*b[2]
		to_return[2] = a[0]*b[1] - a[1]*b[0]
		return to_return
	
	def render_frame (self):
		self.frame += 1
		if self.frame%self.frame_per_render != 0:
			return
			
		cameraTargetPosition = list(p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.pcId)[0])
		
		if self.first_render:
			self.first_render = False
			self.first_height = cameraTargetPosition[2]
			self.cam_alpha = 0
			self.cam_r = 2
		
		self.cam_alpha += np.pi*2/(30*10)
		
		base_pos = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.pcId)[0]
		cameraEyePosition = [base_pos[0]+self.cam_r*np.sin(self.cam_alpha), base_pos[1]+self.cam_r*np.cos(self.cam_alpha), 0.7]
		
		
		cameraTargetPosition[2] = self.first_height
		cameraUpVector = [0,0,1]
		width = 960
		height = 640
		
		viewMatrix = p.computeViewMatrix(cameraEyePosition, cameraTargetPosition, cameraUpVecto, physicsClientId=self.pcId)
		projectionMatrix = p.computeProjectionMatrixFOV(fov=40.0, aspect=width/height, nearVal=0.1, farVal=10, physicsClientId=self.pcId)
		
		width, height, rgbImg, depthImg, segImg = p.getCameraImage(width=width,	 height=height, viewMatrix=viewMatrix, projectionMatrix=projectionMatrix, physicsClientId=self.pcId)
		self.raw_frames.append(rgbImg.reshape((width,height,-1))[:,:,:3])
	
	def close (self):
		p.disconnect(physicsClientId=self.pcId)