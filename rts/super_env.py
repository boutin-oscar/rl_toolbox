import numpy as np
import tensorflow as tf

class SuperEnv:
	def __init__ (self, dq_actor, debug=False):
		
		self.dq_actor = dq_actor
		
		self.range_q = sim.range_q
		self.kill_range_q = sim.kill_range_q

		
		self.obs_dim = sim.obs_dim
		self.act_dim = sim.act_dim
		
		self.obs_mean = sim.obs_mean
		self.obs_std = sim.obs_std
		
		self.num_envs = 1
		
		self.debug = debug
		
		self.max_f = 0.01
	
	
	@tf.function
	def run_internal_model (self, inp):
		return self.dq_actor.model(inp)
	
	def calc_dq (self, act):
		u = np.maximum(np.minimum(2*act-1, 1), -1)
		#dq_act = self.q * 0 + 0.5
		dq_act = self.run_internal_model(self.dq_obs(self.q, act))[0].numpy()
		dq = (2*dq_act-1) * self.max_f
		dq = dq.reshape(self.q.shape)
		self.dq = dq
		return dq
		
	
	def step (self, act):
	
		dq = self.calc_dq(act)
		self.q, u = sim.step(self.q, dq, act)
		
		rew = -np.mean(np.square(self.q/self.kill_range_q)) - np.mean(np.square(u))
		rew = rew if not self.calc_done() else -2
		
		return self.calc_obs(), [rew], [self.calc_done()]
	
	def dq_obs (self, q, act):
		return (np.concatenate([q.flatten(), act.flatten()]).reshape((1, 1, -1)), self.dq_actor.get_init_state(self.num_envs))
	
	def calc_obs (self):
		return [self.q.flatten()]
	
	def calc_rew (self):
		return -np.mean(np.square(self.q/self.kill_range_q)) if not self.calc_done() else -1
	
	def calc_done (self):
		return np.any(self.q > self.kill_range_q) or np.any(self.q < -self.kill_range_q)
	
	def reset (self):
		self.q = np.random.uniform(-self.range_q, self.range_q)
		self.q = np.asarray([0, 1]).reshape(self.q.shape)
		if self.debug:
			self.q = np.asarray([0, 1]).reshape(self.q.shape)
		
		return self.calc_obs ()