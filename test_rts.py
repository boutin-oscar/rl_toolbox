import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt

from models.actor import SimpleActor

from rts.simulator import Simulator
from rts.super_env import SuperEnv
from rts.real_env import RealEnv

def calc_dq (sim, trans):
	q = trans[:sim.obs_dim].reshape(sim.range_q.shape)
	nq = q + trans[sim.obs_dim:sim.obs_dim*2].reshape(sim.range_q.shape)/10
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
	print(np.std(all_dq, axis=0))
	
	plt.quiver(all_q[:,0], all_q[:,1], all_dq[:,0], all_dq[:,1])
	plt.show()
	

"""

sim = Simulator()
actor = SimpleActor (sim.obs_dim, sim.act_dim)
dq_actor = SimpleActor (sim.obs_dim + sim.act_dim, sim.obs_dim)

actor.load("results/safe/models/expert/{}")
dq_actor.load("results/exp_0/models/dq/{}")

super_env = SuperEnv(sim, dq_actor)
real_env = RealEnv(sim)
env = super_env

all_obs = []
all_trans = []

obs = env.reset ()
N = 200
for i in range(N):
	obs = np.asarray(obs).reshape((1,1,-1))
	trans = [obs.flatten()]
	all_obs.append(obs)
	
	
	
	act = actor.model(obs).numpy()
	obs, rew, done = env.step(act)
	trans.append(np.asarray(obs).flatten())
	trans.append(act.flatten())
	all_trans.append(np.concatenate(trans, axis=0))

all_trans = np.asarray(all_trans)
print(all_trans.shape)

all_obs = np.asarray(all_obs).reshape((N, -1))

show_dq(sim, all_trans)

"""

all_trans = np.load("results/exp_2/data/synth_batch_9/all_trans.npy")
# all_trans = np.load("results/exp_2/data/real_batch_0/all_trans.npy")
sim = Simulator()
show_dq(sim, all_trans[:])
	






