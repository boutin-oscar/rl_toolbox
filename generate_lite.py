import tensorflow as tf
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import time
import shutil

import pybullet as p

from environments.dog_env import DogEnv
from models.actor import SimpleActor, MixtureOfExpert, LSTMActor

import sys

import models.lite_model as lite

def from_model_to_lite (model, lite_path):
	save_model_path = "save_model_tmp"
	tf.saved_model.save(model, save_model_path)
	
	converter = tf.lite.TFLiteConverter.from_saved_model(save_model_path) # path to the SavedModel directory
	tflite_model = converter.convert()

	with open(lite_path.format('model.tflite'), 'wb') as f:
		f.write(tflite_model)

def model_function (model, inp):
	out = model(inp)
	to_return = {}
	to_return['main_output'] = out[0]
	to_return['hidden_0'] = out[1][0]
	to_return['hidden_1'] = out[1][1]
	return to_return
	
def from_keras_to_lite (model, lite_path):
	run_model = tf.function(lambda x: model_function(model, x))
	# This is important, let's fix the input size.
	BATCH_SIZE = 1
	STEPS = 1
	concrete_func = run_model.get_concrete_function(
		(tf.TensorSpec([BATCH_SIZE, STEPS, 78+3*12], model.inputs[0].dtype, name="main_input"),
		tf.TensorSpec([BATCH_SIZE, 128], model.inputs[0].dtype, name="hidden_0"), 
		tf.TensorSpec([BATCH_SIZE, 128], model.inputs[0].dtype, name="hidden_1")))

	# model directory.
	save_model_path = "save_model_tmp"
	model.save(save_model_path, save_format="tf", signatures=concrete_func)
	
	m_model = tf.keras.models.load_model(save_model_path)
	
	
	converter = tf.lite.TFLiteConverter.from_saved_model(save_model_path)
	tflite_model = converter.convert()

	with open(lite_path.format('model.tflite'), 'wb') as f:
		f.write(tflite_model)
	
class FakeEnv ():
	def __init__ (self, obs_dim, act_dim, obs_mean, obs_std):
		self.obs_dim = obs_dim
		self.act_dim = act_dim
		self.obs_mean = obs_mean
		self.obs_std = obs_std

if __name__ == '__main__':
	if len(sys.argv) > 1:
		target_name = sys.argv[1]
		
		debug = False
		render = False
		actor_type = "simple"
		
		
		env = DogEnv(debug=debug, render=render)
		obs = env.reset()
		obs = np.asarray(obs, dtype=np.float32)

		#path = "results\\baseline\\models\\expert\\{}"
		src_path = "results\\" + target_name + "\\models\\expert\\{}"
		lite_path = "distribution\\" + target_name + "\\"
		
		if os.path.exists(lite_path) and os.path.isdir(lite_path): # del dir if exists
			shutil.rmtree(lite_path)
		os.makedirs(lite_path)
		
		
		mean_blind = env.obs_mean @ env.blindfold.visible_A
		std_blind = env.obs_std @ env.blindfold.visible_A
		print(mean_blind.shape)
		
		fake_env = FakeEnv (mean_blind.shape[0], env.act_dim, mean_blind, std_blind)
		
		if actor_type == "simple":
			old_actor = SimpleActor(env, use_blindfold=True, use_lstm=False)
			new_actor = SimpleActor(fake_env, use_blindfold=True, use_lstm=False, inp_dim=1)
		else:
			raise NameError ('Actor type not yet supported')
		
		new_actor.model.summary()
		old_actor.model.summary()
		
		new_actor.load(src_path)
		old_actor.load(src_path)
		
		z = np.zeros((1,128), dtype=np.float32)
		
		print(obs.shape)
		act = old_actor.model((np.asarray(obs.reshape((1,1,-1)), dtype=np.float32), (z, z)))
		print(act[0].shape)
		act = new_actor.model((np.asarray(env.blindfold.select_visible(obs).reshape((1,1,-1)), dtype=np.float32), (z, z)))
		print(act[0].shape)
		#print(new_actor.model(np.asarray(env.blindfold.action_blindfold(obs), dtype=np.float32)))
		
		
		#from_model_to_lite(new_actor.model, lite_path+"{}") # <---- used to work
		
		from_keras_to_lite(new_actor.model, lite_path+"{}")
		
		lite.load(lite_path+"model.tflite")
		"""
		print("inputs")
		print(lite.input_details)
		print("outputs")
		print(lite.output_details)
		"""
		
		obs = np.expand_dims(np.asarray(obs, dtype=np.float32), axis=1)*0
		old_result = old_actor.model((obs, old_actor.get_init_state(1)))
		old_step = (('main_output', old_result[0].numpy()),
					('hidden_0', old_result[1][0].numpy()),
					('hidden_1', old_result[1][1].numpy()))
		
		lite_obs = [('serving_default_main_input:0', np.asarray(env.blindfold.select_visible(obs)*0, dtype=np.float32).reshape((1,1,-1))),
					('serving_default_hidden_0:0', np.zeros((1,128), dtype=np.float32)), 
					('serving_default_hidden_1:0', np.zeros((1,128), dtype=np.float32))]
		
		lite_step = lite.step(lite_obs)
		print("inputs : ")
		for detail in lite.input_details:
			print(detail['name'])
		
		print("outputs : ")
		
		for old_name, old_value in old_step:
			for lite_name, lite_value in lite_step:
				if lite_value.shape == old_value.shape:
					if np.mean(np.square(lite_value-old_value)) < 1e-10:
						print(old_name, "->", lite_name)
		
		
		print()
		print("Successfull conversion to : " + lite_path)
		
		
		
		
		
		
	else:
		print("I need to have the target name (exp : python generate_lite.py exp_0)")
	
	