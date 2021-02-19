import numpy as np
import tensorflow as tf

class SuperEnv:
	def __init__ (self, sim, dq_actor, debug=False):
		
		self.dq_actor = dq_actor
		self.sim = sim
		
		self.obs_dim = self.sim.obs_dim
		self.act_dim = self.sim.act_dim
		
		self.num_envs = 1
		
		self.debug = debug
		
		self.max_f = 0.01
	
	@tf.function
	def run_internal_model (self, inp):
		return self.dq_actor.model(inp)
	
	def calc_dq (self, act):
		dq_act = self.run_internal_model(self.dq_obs(self.q, act))[0].numpy()
		dq = (2*dq_act-1) * self.max_f
		dq = dq.reshape(self.q.shape)
		self.dq = dq
		#self.dq[0,0] = 0
		
	def calc_dq_real (self, act): # calculating the dq that simulates solid friction
		lin_s, u = self.sim.linear_step (self.q, act)
		dq = np.asarray([0, -lin_s[1,0]]).reshape(lin_s.shape)
		dq = np.maximum(np.minimum(dq, self.max_f), -self.max_f)
		self.dq = dq
		
	
	def step (self, act):
	
		self.calc_dq(act)
		to_print = str(self.dq)
		# self.calc_dq_real(act)
		# print("{}, {}".format(to_print, self.dq), flush=True)
		nq, u = self.sim.step(self.q, self.dq, act)
		
		rew = self.sim.calc_rew(self.q, u, nq)
		self.q = nq
		return self.calc_obs(), [rew], [self.sim.calc_done(self.q)]
	
	def dq_obs (self, q, act):
		return np.concatenate([q.flatten(), act.flatten()]).reshape((1, 1, -1))
	
	def calc_obs (self):
		return [self.q.flatten()]
	
	def reset (self):
		self.q = self.sim.reset()
		return self.calc_obs ()