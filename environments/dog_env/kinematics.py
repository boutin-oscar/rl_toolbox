from .leg_ik import Leg
import numpy as np

class Kinematics():
	def __init__(self):
		self.all_legs = [Leg(inv_x=True, inv_y=False, inv_1=True, inv_2=False, inv_3=False),
					Leg(inv_x=True, inv_y=True, inv_1=False, inv_2=True, inv_3=False),
					Leg(inv_x=False, inv_y=False, inv_1=False, inv_2=True, inv_3=False),
					Leg(inv_x=False, inv_y=True, inv_1=True, inv_2=False, inv_3=False)]
			
	def calc_joint_target (self, raw_action, phases, loc_up_vect=[0,0,1]):
		action = np.maximum(np.minimum(raw_action, 1), 0)
		return self.motor_pos(action, phases, loc_up_vect)

	def motor_pos (self, raw_action, phases, loc_up_vect):
		legs_actions = self.split_legs(raw_action)
		to_return = []
		for leg, leg_action, phase in zip(self.all_legs, legs_actions, phases):
			to_return += leg.motor_pos (leg_action, phase)
		return np.asarray(to_return)

	def split_legs (self, action):
		to_return = []
		for i in range(4):
			to_return.append([action[i*3+j] for j in range(3)])
		return to_return

	def standard_rot (self, rot):
		legs_rot = self.split_legs(rot)
		to_return = []
		for leg, leg_rot in zip(self.all_legs, legs_rot):
			to_return += leg.standard_rot (leg_rot)
		return np.asarray(to_return)