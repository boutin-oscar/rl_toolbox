import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

def create_blind_matrix (all_observed):
	out_len = len([1 for x in all_observed if x])
	
	act_A = []
	a = 0
	for observed in all_observed:
		to_add = np.zeros((out_len,))
		if observed:
			to_add[a] = 1
			a += 1
		act_A.append(to_add)
	act_A = np.asarray(act_A)
	return act_A
	

def create_blindfold (all_observed, obs_transition_len):
	all_observed = sum(all_observed, []) * obs_transition_len
	all_non_observed = [not x for x in all_observed]
	inp_len = len(all_observed)
	
	visible_A = create_blind_matrix(all_observed)
	hidden_A = create_blind_matrix(all_non_observed)
	

	return Blindfold (inp_len, visible_A, hidden_A)
	
	
class Blindfold:
	def __init__ (self, obs_dim, visible_A, hidden_A):
		self.obs_dim = obs_dim
		self.visible_A = visible_A
		self.hidden_A = hidden_A
		
	def select_visible (self, input_obs):
		return input_obs @ self.visible_A
		
	def select_hidden (self, input_obs):
		return input_obs @ self.hidden_A
	
	
