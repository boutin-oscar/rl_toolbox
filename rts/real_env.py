import numpy as np

class RealEnv: # we usually do not have access to this.
	def __init__ (self, sim, debug=False):
		self.sim = sim
		
		self.obs_dim = self.sim.obs_dim
		self.act_dim = self.sim.act_dim
		
		self.num_envs = 1
		
		self.debug = debug
		
		self.max_f = 0.01
	
	def calc_dq (self, act): # calculating the dq that simulates solid friction
		lin_s, u = self.sim.linear_step (self.q, act)
		dq = np.asarray([0, -lin_s[1,0]]).reshape(lin_s.shape)
		dq = np.maximum(np.minimum(dq, self.max_f), -self.max_f)
		self.dq = dq
		
	def step (self, act):
		self.calc_dq(act)
		nq, u = self.sim.step(self.q, self.dq, act)
		
		rew = self.sim.calc_rew(self.q, u, nq)
		self.q = nq
		return self.calc_obs(), [rew], [self.sim.calc_done(self.q)]
	
	def calc_obs (self):
		return [self.q.flatten()]
	
	def reset (self):
		self.q = self.sim.reset()
		return self.calc_obs ()