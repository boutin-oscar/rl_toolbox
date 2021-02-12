import numpy as np
from pathlib import Path
import sys



class Adr:
	def __init__ (self):
		self.all_values = {}
		self.all_deltas = {}
		self.all_min = {}
		self.all_max = {}
		self.logger = {}
		self.tested_param = ""
		
		self.sum_rew = 0
		self.n_rew = 0
		self.premature_end = False
		self.success = False
		self.failure = False
	
	def add_param(self, name, start_value, delta, bound):
		if name in self.all_values.keys():
			print("ERROR : multiple params with same name :", name)
			print(1/0)
		self.all_values[name] = start_value
		self.all_deltas[name] = delta
		self.all_min[name] = min(start_value, bound)
		self.all_max[name] = max(start_value, bound)
		self.logger[name] = []
	
	def log (self):
		for name in self.all_values.keys():
			self.logger[name].append(self.all_values[name])
	
	def value(self, param):
		return self.all_values[param]
		
	def is_test_param (self, name):
		return name == self.tested_param
	
	def step (self, success, failure):
		self.success = success
		self.failure = failure
	
	def reset (self):
		if self.tested_param != "":
			self.all_values[self.tested_param] = self.updated_tested_value()
		
		self.success = False
		self.failure = False
		
		if np.random.random() < 1:
			self.set_test_param ()
		else:
			self.tested_param = ""
		
		self.log()
	
	def updated_tested_value (self):
		value = self.all_values[self.tested_param]
		
		if self.success:
			value += self.all_deltas[self.tested_param]
			#print("success !!", flush=True)
		elif self.failure:
			value -= self.all_deltas[self.tested_param]
		
		if value > self.all_max[self.tested_param]:
			value = self.all_max[self.tested_param]
		if value < self.all_min[self.tested_param]:
			value = self.all_min[self.tested_param]
		
		return value
	
	
	def set_test_param (self):
		keys = list(self.all_values.keys())
		param_nb = len(keys)
		if param_nb > 0:
			id = int(np.random.random()*param_nb)
			self.tested_param = keys[id]
	
	def save (self):
		exp_name = "default"
		if len(sys.argv) > 1:
			exp_name = sys.argv[1]
		save_path = "adr/rank_" + exp_name #+str(self.rank)
		names = []
		values = []
		for name in self.logger.keys():
			names.append(name)
			values.append(self.logger[name])
		
		with open(save_path+".txt", "w") as f:
			for name in names:
				f.write(name+"$")
		values = np.asarray(values)
		np.save(save_path, values)
	
	def close (self):
		pass
	
	
	def get_msg (self):
		if self.tested_param != "":
			return {self.tested_param : self.updated_tested_value()}
		else:
			return {}
	
	def update (self, new_values):
		self.all_values.update(new_values)
