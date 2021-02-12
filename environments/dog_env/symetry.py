import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt


def create_symetry (all_trans_A, obs_transition_len):
	
	obs_len = np.sum([A.shape[0] for A in all_trans_A]) * obs_transition_len
	obs_A = np.zeros((obs_len, obs_len))
	a = 0
	for i in range(obs_transition_len):
		for A in all_trans_A:
			b = a + A.shape[0]
			obs_A[a:b,a:b] = A
			a = b
	
	act_A = np.asarray([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
						[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						[0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
						[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
	act_B = np.asarray([0, 1, 0] * 4)
	
	return Symetry (act_A, act_B, obs_A)
	

class Symetry:
	def __init__ (self, act_A, act_B, obs_A):
		self.act_A = act_A
		self.act_B = act_B
		self.obs_A = obs_A
	
	def action_symetry (self, input_action):
		return input_action @ self.act_A + self.act_B

	def state_symetry (self, input_obs):
		return input_obs @ self.obs_A

	def loss (self, actor, input_obs, init_state, mask):
		obs_sym = self.state_symetry(input_obs)
		
		diff = actor.model((input_obs, init_state))[0] - self.action_symetry(actor.model((obs_sym, init_state))[0])
		return tf.reduce_mean(tf.multiply(tf.reduce_sum(tf.square(diff), axis=-1), mask))/tf.reduce_mean(mask)