import numpy as np
from .symetry import create_symetry
from .blindfold import create_blindfold


class FullObsGenerator:
	def __init__ (self, state, debug=False):
		self.obs_generators_ref = [JointTargetPos,
								ObsJointDelta,
								TrueJointDelta,
								JointPos,
								JointOffset,
								KPS,
								FootPhase,
								LocalUp,
								#ObsRotVel,
								RotVel,
								
								Cmd_PosVel,
								Cmd_RotVel,
								
								Height,
								PosVel,
								MeanPosVel,
								MeanRotVel,
								Friction,
								
								OffsetForce,
								]
		self.obs_generators = [obs_ref(state) for obs_ref in self.obs_generators_ref]
		
		self.debug = debug
		if self.debug:
			self.to_plot = []
		
		self.obs_transition_len = 3
		
		self.symetry = create_symetry ([obs.symetry() for obs in self.obs_generators], self.obs_transition_len)
		self.blindfold = create_blindfold ([obs.blindfold() for obs in self.obs_generators], self.obs_transition_len)
		
		self.mean = []
		self.std = []
		for i in range(self.obs_transition_len):
			for gen in self.obs_generators:
				self.mean.append(gen.mean)
				self.std.append(gen.std)
		self.mean = np.concatenate(self.mean, axis=0)
		self.std = np.concatenate(self.std, axis=0)
		
	def generate (self):
		to_return = []
		for obs_generator in self.obs_generators:
			obs_generator.generate(to_return)
		if self.debug:
			self.to_plot.append(np.asarray(to_return))
		return [np.asarray(to_return)]

class FlexGenerator:
	def __init__ (self, state, child_generator_classes):
		self.state = state
		self.obs_generators = [obs_ref(state) for obs_ref in child_generator_classes]
	
	def generate (self):
		to_return = []
		for obs_generator in self.obs_generators:
			obs_generator.generate(to_return)
		return [np.asarray(to_return)]

class ObsGenerator:
	def __init__ (self, state):
		self.state = state
		self.raw = False
		if not self.raw:
			self.init_dev ()
	"""
	def generate (self, to_return):
		interm_ret = []
		self._generate (interm_ret)
		if self.raw:
			to_return += interm_ret[:]
		else:
			to_return += list((np.asarray(interm_ret)-self.mean)/self.std)
	"""
	def generate (self, to_return):
		self._generate(to_return)
		

class JointTargetPos (ObsGenerator):
	def init_dev (self):
		self.mean = np.asarray([0, -0.75, -1] * 4)
		self.std = np.asarray([1, 1, 1] * 4)
	def _generate (self, to_return):
		for i in range(12):
			to_return.append(self.state.joint_target[i]*self.state.joint_fac[i])
	def symetry (self):
		return np.asarray([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
							[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
							[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
	def blindfold (self):
		return [True] * 12

class ObsJointDelta (ObsGenerator):
	def init_dev (self):
		self.mean = np.asarray([0, 0, 0] * 4)
		self.std = np.asarray([1, 1, 1] * 4)
	def _generate (self, to_return):
		for i in range(12):
			#to_return.append((self.state.joint_rot[i] - self.state.joint_offset[i] - self.state.joint_target[i])*self.state.joint_fac[i])
			to_return.append((self.state.joint_rot[i] - self.state.joint_offset[i] - self.state.fake_target[i])*self.state.joint_fac[i])
	def symetry (self):
		return np.asarray([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
							[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
							[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
	def blindfold (self):
		return [True] * 12

class TrueJointDelta (ObsGenerator):
	def init_dev (self):
		self.mean = np.asarray([0, 0, 0] * 4)
		self.std = np.asarray([1, 1, 1] * 4)
	def _generate (self, to_return):
		for i in range(12):
			to_return.append((self.state.joint_rot[i] - self.state.joint_offset[i] - self.state.joint_target[i])*self.state.joint_fac[i])
			#to_return.append((self.state.joint_rot[i] - self.state.joint_offset[i] - self.state.fake_target[i])*self.state.joint_fac[i])
	def symetry (self):
		return np.asarray([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
							[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
							[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
	def blindfold (self):
		return [False] * 12

class JointPos (ObsGenerator):
	def init_dev (self):
		self.mean = np.asarray([0, -0.75, -1] * 4)
		self.std = np.asarray([1, 1, 1] * 4)
	def _generate (self, to_return):
		for i in range(12):
			to_return.append(self.state.joint_rot[i]*self.state.joint_fac[i])
	def symetry (self):
		return np.asarray([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
							[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
							[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
	def blindfold (self):
		return [False] * 12

class JointOffset (ObsGenerator):
	def init_dev (self):
		self.mean = np.asarray([0, 0, 0] * 4)
		self.std = np.asarray([1, 1, 1] * 4)
	def _generate (self, to_return):
		for i in range(12):
			to_return.append(self.state.joint_offset[i]*self.state.joint_fac[i])
	def symetry (self):
		return np.asarray([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
							[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
							[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
	def blindfold (self):
		return [False] * 12

class KPS (ObsGenerator):
	def init_dev (self):
		self.mean = np.asarray([60] * 12)
		self.std = np.asarray([100] * 12)
	def _generate (self, to_return):
		for i in range(12):
			to_return.append(self.state.all_kp[i])
	def symetry (self):
		return np.asarray([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
							[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
							[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
	def blindfold (self):
		return [False] * 12
		
class FootPhase (ObsGenerator):
	def init_dev (self):
		self.mean = np.asarray([0] * 8)
		self.std = np.asarray([1] * 8)
	def _generate (self, to_return):
		to_return += list(np.sin(self.state.foot_phases))
		to_return += list(np.cos(self.state.foot_phases))
	def symetry (self):
		return np.asarray([[0, 1, 0, 0, 0, 0, 0, 0],
							[1, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 1, 0, 0, 0, 0],
							[0, 0, 1, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 1, 0, 0],
							[0, 0, 0, 0, 1, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 1],
							[0, 0, 0, 0, 0, 0, 1, 0]])
	def blindfold (self):
		return [True] * 8

class LocalUp (ObsGenerator):
	def init_dev (self):
		self.mean = np.asarray([0] * 3)
		self.std = np.asarray([1] * 3)
	def _generate (self, to_return):
		#to_return += [0, 0, 1]
		#to_return += [self.state.loc_up_vect[i] + self.state.loc_up_vect_offset[i] for i in range(3)]
		to_return += [self.state.loc_up_vect[i] for i in range(3)]
	def symetry (self):
		return np.diag([1, -1, 1])
	def blindfold (self):
		return [True] * 3
		
class ObsRotVel (ObsGenerator):
	def init_dev (self):
		self.mean = np.asarray([0] * 3)
		self.std = np.asarray([1] * 3)
	def _generate (self, to_return):
		to_return += [self.state.loc_rot_speed[i] + np.random.normal()*self.state.rot_speed_noise[i] for i in range(3)]
		#to_return += [0, 0, 0]
	def symetry (self):
		return np.diag([-1, 1, -1])
	def blindfold (self):
		return [True] * 3
		
class RotVel (ObsGenerator):
	def init_dev (self):
		self.mean = np.asarray([0] * 3)
		self.std = np.asarray([1] * 3)
	def _generate (self, to_return):
		to_return += [self.state.loc_rot_speed[0], self.state.loc_rot_speed[1], self.state.loc_rot_speed[2]]
		#to_return += [0, 0, 0]
	def symetry (self):
		return np.diag([-1, 1, -1])
	def blindfold (self):
		return [False] * 3
		
class Cmd_PosVel (ObsGenerator):
	def init_dev (self):
		self.mean = np.asarray([0] * 2)
		self.std = np.asarray([1] * 2)
	def _generate (self, to_return):
		to_return += [self.state.target_speed[0], self.state.target_speed[1]]
	def symetry (self):
		return np.diag([1, -1])
	def blindfold (self):
		return [True] * 2
		
class Cmd_RotVel (ObsGenerator):
	def init_dev (self):
		self.mean = np.asarray([0] * 1)
		self.std = np.asarray([1] * 1)
	def _generate (self, to_return):
		to_return += [self.state.target_rot_speed]
	def symetry (self):
		return np.diag([-1])
	def blindfold (self):
		return [True] * 1
		
class Height (ObsGenerator):
	def init_dev (self):
		self.mean = np.asarray([0.3] * 1)
		self.std = np.asarray([1] * 1)
	def _generate (self, to_return):
		to_return += [self.state.base_pos[2]]
	def symetry (self):
		return np.diag([1])
	def blindfold (self):
		return [False] * 1
		
class PosVel (ObsGenerator):
	def init_dev (self):
		self.mean = np.asarray([0.] * 3)
		self.std = np.asarray([1] * 3)
	def _generate (self, to_return):
		to_return += [self.state.loc_pos_speed[0], self.state.loc_pos_speed[1], self.state.loc_pos_speed[2]]
	def symetry (self):
		return np.diag([1, -1, 1])
	def blindfold (self):
		return [False] * 3
		
class MeanPosVel (ObsGenerator):
	def init_dev (self):
		self.mean = np.asarray([0.] * 2)
		self.std = np.asarray([1] * 2)
	def _generate (self, to_return):
		to_return += [self.state.mean_planar_speed[0], self.state.mean_planar_speed[1]]
	def symetry (self):
		return np.diag([1, -1])
	def blindfold (self):
		return [False] * 2
		
class MeanRotVel (ObsGenerator):
	def init_dev (self):
		self.mean = np.asarray([0.] * 1)
		self.std = np.asarray([1] * 1)
	def _generate (self, to_return):
		to_return += [self.state.mean_z_rot_speed]
	def symetry (self):
		return np.diag([-1])
	def blindfold (self):
		return [False] * 1

class Friction (ObsGenerator):
	def init_dev (self):
		self.mean = np.asarray([np.sqrt(.5)] * 1)
		self.std = np.asarray([1] * 1)
	def _generate (self, to_return):
		to_return += [self.state.friction]
	def symetry (self):
		return np.diag([1])
	def blindfold (self):
		return [False] * 1

class OffsetForce (ObsGenerator):
	def init_dev (self):
		self.mean = np.asarray([0] * 3)
		self.std = np.asarray([10] * 3)
	def _generate (self, to_return):
		to_return += [self.state.offset_force[i] for i in range(3)]
	def symetry (self):
		return np.diag([1, -1, 1])
	def blindfold (self):
		return [False] * 3
		