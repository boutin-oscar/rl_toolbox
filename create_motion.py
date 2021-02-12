
import pybullet as p

import time
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np

from environments.dog_env.kinematics import Kinematics
from scipy import optimize as opt

debug_delta = [0.1, 0, 0]
delta_foot = [[0.01, 0, -0.2], [0.01, 0, -0.2], [-0.01, 0, -0.2], [-0.01, 0, -0.2]]
urdf_joint_indexes = [0,1,2,6,7,8,9,10,11,3,4,5]
kin = Kinematics()

def start_bullet ():
	global robotId
	
	pcId = p.connect(p.GUI)
	p.resetDebugVisualizerCamera (1, 0, 0, [0, 0, 0.3])
	p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
	urdf_path = "environments/dog_env/urdf"
	#groundId = p.loadURDF(urdf_path + "/plane_001/plane.urdf")
	robotId = p.loadURDF(urdf_path + "/robot_002_1/robot.urdf", [0,0,0], flags=p.URDF_MERGE_FIXED_LINKS)
	
	base_pos, base_rot = p.getBasePositionAndOrientation(robotId)
	p.resetDebugVisualizerCamera (1, 30, -15, base_pos)

def get_end_leg_pos (debug=False):
	to_return = []
	foot_link_id = [2, 5, 8, 11]
	for i, foot_id in enumerate(foot_link_id):
		urdf_joint_index = urdf_joint_indexes[foot_id]
		com_pos, com_ori, loc_frame_pos, loc_frame_ori, link_pos, link_ori = p.getLinkState(robotId, urdf_joint_index, computeForwardKinematics=1)
		#foot_pos = link_pos + ori * delta_foot
		r = R.from_quat(link_ori)
		foot_pos = np.asarray(link_pos) + r.apply(delta_foot[i])
		to_return.append(foot_pos)
		if debug:
			p.addUserDebugLine(foot_pos, [x+y for x, y in zip(foot_pos, debug_delta)], lineWidth=3, lineColorRGB=[1, 0, 1])
	
	p.setDebugObjectColor(robotId, urdf_joint_indexes[11], [1, 0, 0])
	
	return np.asarray(to_return).flatten()

def set_body (pos, euler_angle):
	r = R.from_euler('zyx', euler_angle, degrees=False)
	p.resetBasePositionAndOrientation(robotId, pos, r.as_quat())

def set_leg_angle (angles):
	for i, angle in enumerate(angles):
		urdf_joint_index = urdf_joint_indexes[i]
		p.resetJointState(robotId, urdf_joint_index, angle)

def set_leg_action (actions):
	set_leg_angle(kin.calc_joint_target(actions, [0]*4))

def f(act):
	set_leg_action(act)
	obs_pos = np.asarray(get_end_leg_pos())
	cost = 0.5*np.sum(np.square(obs_pos-des_pos))
	return obs_pos-des_pos



if __name__ == "__main__":
	start_bullet ()
	
	zero_act = np.asarray([0.5, 0.5, 0.2]*4)
	cur_act = zero_act
	
	set_body([0, 0, 0], [0, 0, 0])
	set_leg_action(zero_act)
	des_pos = np.asarray(get_end_leg_pos(debug=True))
	
	all_pos = []
	all_rot = []
	all_act = []
	
	T = 4
	dt = 1/30
	N = int(T/dt)
	for i in range(N):
		phi = np.pi*i/N
		#cur_body_pos = [0, 0, -np.square(np.sin(phi)) * 0.05]
		#cur_body_rot = [0, 0, 0]
		cur_body_pos = [0, 0, -np.square(np.sin(phi)) * 0.05]
		cur_body_rot = [0, 0, 3*np.square(np.sin(phi))*np.cos(phi) * 0.1]
		#cur_body_rot = [0, 0, 0]
		set_body(cur_body_pos, cur_body_rot)
		
		
		res = opt.least_squares(f, cur_act, diff_step=1e-5)
		cur_act = np.maximum(np.minimum(res.x, 1), 0)
		
		if np.any(res.x> 1) or np.any(res.x < 0):
			print("Movement not viable !!!")
			
		all_pos.append(cur_body_pos)
		all_rot.append(cur_body_rot)
		all_act.append(res.x)
	
	all_act = np.asarray(all_act)
	
	np.save("environments/dog_env/movement/rot_x.npy", all_act)
	
	i = 0
	while (True):
		i += 1
		p.getLinkState(robotId, 0, computeForwardKinematics=1)
		if len(all_pos) > 0:
			set_body(all_pos[i%len(all_pos)], all_rot[i%len(all_rot)])
			set_leg_action(all_act[i%len(all_act)])
		time.sleep(0.03)
	