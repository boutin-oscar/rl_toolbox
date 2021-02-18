import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt

from models.actor import SimpleActor

from rts.simulator import Simulator
from rts.super_env import SuperEnv
from rts.real_env import RealEnv

sim = Simulator()
"""
actor = SimpleActor (sim.obs_dim, sim.act_dim)
dq_actor = SimpleActor (sim.obs_dim + sim.act_dim, sim.obs_dim)

actor.load("results/safe/models/expert/{}")
dq_actor.load("results/safe/models/dq/{}")

super_env = SuperEnv(sim, dq_actor)
real_env = RealEnv(sim)
env = real_env

all_obs = []

obs = env.reset ()
N = 200
for i in range(N):
	obs = np.asarray(obs).reshape((1,1,-1))
	
	all_obs.append(obs)
	
	
	
	act = actor.model(obs).numpy()
	obs, rew, done = env.step(act)



all_obs = np.asarray(all_obs).reshape((N, -1))

plt.plot(all_obs)
plt.show()
"""

all_trans = np.load("results/exp_0/data/batch_0/all_trans.npy")

def calc_dq (sim, trans):
	q = trans[:sim.obs_dim].reshape(sim.range_q.shape)
	nq = trans[sim.obs_dim:sim.obs_dim*2].reshape(sim.range_q.shape)
	act = trans[sim.obs_dim*2:]
	
	lin_nq, u = sim.linear_step(q, act)
	dq = nq - lin_nq
	return q, dq

def show_dq (sim, inp_trans):
	all_trans = inp_trans.reshape((-1, inp_trans.shape[-1]))
	
	all_q = []
	all_dq = []
	for trans in all_trans:
		q, dq = calc_dq(sim, trans)
		all_q.append(q.flatten())
		all_dq.append(dq.flatten())
		
	all_q = np.stack(all_q)
	all_dq = np.stack(all_dq)
	
	plt.quiver(all_q[:,0], all_q[:,1], all_dq[:,0], all_dq[:,1])
	plt.show()
	
show_dq(sim, all_trans)
	






