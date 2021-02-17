import numpy as np
import tensorflow as tf

class TunableEnv:
	def __init__ (self, debug=False):
		
		self.range_q = sim.range_q
		self.kill_range_q = sim.kill_range_q

		self.obs_dim = sim.obs_dim + sim.act_dim
		self.act_dim = sim.obs_dim
		
		self.obs_mean = np.zeros(self.obs_dim)
		self.obs_std = np.ones(self.obs_dim)
		
		self.num_envs = 1
		
		self.debug = debug
		
		self.max_f = 0.01
		
		
		real_env = RealEnv()
		self.actor = SimpleActor(real_env, use_blindfold=False, use_lstm=False)
		self.actor.load("results\\safe\\models\\expert\\{}")
		
		self.discriminator = Discriminator(sim.obs_dim*2 + sim.act_dim)
		self.discriminator.load("results\\discrim\\models\\{}")
		#print(tf.nn.softmax(discriminator.model(full_trans))[:,0])
		
	
	@tf.function
	def run_internal_disc (self, full_trans):
		return tf.nn.softmax(self.discriminator.model(full_trans))[:,0]
	
	@tf.function
	def run_internal_actor (self, inp):
		return self.actor.model(inp)[0]
	
	
	
	def step (self, dq_act):
		
		full_trans = [self.q.flatten()]
		
		dq = (dq_act*2-1)*self.max_f
		dq = np.maximum(np.minimum(dq, self.max_f), -self.max_f).reshape(self.q.shape)
		
		"""# --- real dq --- 
		lin_s, u = sim.linear_step (self.q, self.a)
		dq = np.asarray([0, -lin_s[1,0]]).reshape(lin_s.shape)
		dq = np.maximum(np.minimum(dq, self.max_f), -self.max_f)
		"""# ---------------
		
		self.q, u = sim.step(self.q, dq, self.a)
		
		full_trans.append(self.q.flatten())
		full_trans.append(self.a.flatten())
		full_trans = np.concatenate(full_trans).reshape((1, -1))
		
		rew = self.run_internal_disc (full_trans)
		rew = rew.numpy().flatten()[0]
		self.dq = dq
		
		return self.calc_obs(), [rew], [self.calc_done()]
	
	def calc_obs (self):
		init_state = self.actor.get_init_state(self.num_envs)
		#self.a = self.actor.model((self.q.reshape((1, 1, 2)), init_state))[0].numpy()
		self.a = self.run_internal_actor((self.q.reshape((1, 1, 2)), init_state)).numpy()
		return [np.concatenate([self.q.flatten(), self.a.flatten()])]
	
	def calc_rew (self):
		raise NameError("Not implemented yet...")
		return -np.mean(np.square(self.q/self.kill_range_q)) if not self.calc_done() else -1
	
	def calc_done (self):
		return np.any(self.q > self.kill_range_q) or np.any(self.q < -self.kill_range_q)
	
	def reset (self):
		self.q = np.random.uniform(-self.range_q, self.range_q)
		self.q = np.asarray([0, 1]).reshape(self.q.shape)
		if self.debug:
			self.q = np.asarray([0, 1]).reshape(self.q.shape)
		
		return self.calc_obs ()