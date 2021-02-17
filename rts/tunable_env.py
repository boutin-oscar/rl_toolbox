import numpy as np
import tensorflow as tf

class TunableEnv:
	def __init__ (self, sim, actor, disc, debug=False):
		
		self.sim = sim
		self.actor = actor
		self.disc = disc
		
		self.obs_dim = self.sim.obs_dim + self.sim.act_dim
		self.act_dim = self.sim.obs_dim
		
		self.num_envs = 1
		
		self.debug = debug
		
		self.max_f = 0.01
	
	@tf.function
	def run_internal_disc (self, full_trans):
		return tf.nn.softmax(self.disc.model(full_trans))[:,0]
	
	@tf.function
	def run_internal_actor (self, inp):
		return self.actor.model(inp)
	
	
	def step (self, dq_act):
		
		full_trans = [self.q.flatten()]
		
		dq_act = np.maximum(np.minimum(dq_act, 1), -1)
		dq = (dq_act*2-1)*self.max_f
		self.dq = dq.reshape(self.q.shape)
		
		"""# --- real dq --- 
		lin_s, u = sim.linear_step (self.q, self.a)
		dq = np.asarray([0, -lin_s[1,0]]).reshape(lin_s.shape)
		dq = np.maximum(np.minimum(dq, self.max_f), -self.max_f)
		"""# ---------------
		
		self.q, u = self.sim.step(self.q, self.dq, self.act)
		
		full_trans.append(self.q.flatten())
		full_trans.append(self.act.flatten())
		full_trans = np.concatenate(full_trans).reshape((1, -1))
		
		rew = self.run_internal_disc (full_trans)
		rew = rew.numpy().flatten()[0]
		
		self.act = self.run_internal_actor(np.expand_dims(self.q.flatten(), axis=(0,1))).numpy().flatten()
		
		return self.calc_obs(), [rew], [self.sim.calc_done(self.q)]
	
	def calc_obs (self):
		return [np.concatenate([self.q.flatten(), self.act.flatten()])]
	
	def reset (self):
		self.q = self.sim.reset()
		self.act = self.run_internal_actor(np.expand_dims(self.q.flatten(), axis=(0,1))).numpy().flatten()
		
		return self.calc_obs ()