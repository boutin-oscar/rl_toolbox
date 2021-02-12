import tensorflow.lite as tflite
import numpy as np


def load (path):
	global interpreter
	global input_details
	global output_details
	
	# Load the TFLite model and allocate tensors.
	interpreter = tflite.Interpreter(model_path=path.format('model.tflite'))
	interpreter.allocate_tensors()
	# Get input and output tensors.
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

def step (obs):
	for name, arr in obs:
		for detail in input_details:
			if name == detail['name']:
				interpreter.set_tensor(detail['index'], arr)
	interpreter.invoke()
	
	to_return = []
	for detail in output_details:
		to_return.append((detail['name'], interpreter.get_tensor(detail['index'])))
	
	return to_return

def step_old (obs):
	interpreter.set_tensor(input_details[0]['index'], obs.astype(np.float32))
	interpreter.invoke()
	act = interpreter.get_tensor(output_details[0]['index'])
	
	return act

if __name__ == "__main__":
	load("distribution/exp_2/model.tflite")
	print(step(np.load("o.npy")))