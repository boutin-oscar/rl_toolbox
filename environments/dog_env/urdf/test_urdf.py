import numpy as np
import pybullet as p
import time

class Simulator:
	def __init__ (self):
		self.pcId = p.connect(p.GUI)
		p.resetDebugVisualizerCamera (1, 0, 0, [0, 0, 0.3])
		p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
		#p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

		urdf_path = ""
		self.groundId = p.loadURDF(urdf_path + "/plane_001/plane.urdf", [0,0,-1], physicsClientId=self.pcId)
		#self.robotId = p.loadURDF(urdf_path + "/robot_001/robot.urdf", [0,0,1], physicsClientId=self.pcId)
		self.robotId = p.loadURDF(urdf_path + "/robot_002_1/robot.urdf", [0,0,0], flags=p.URDF_MERGE_FIXED_LINKS, physicsClientId=self.pcId)
		#self.robotId = p.loadURDF(urdf_path + "/robot_002_1/robot.urdf", [0,0,0], physicsClientId=self.pcId)
		
		#self.urdf_joint_indexes = [0,1,2,3,4,5,6,7,8,9,10,11]
		self.urdf_joint_indexes = [2,4,8,13,15,19,24,26,30,35,37,41]
		
		self.timeStep = 1/240
		p.setPhysicsEngineParameter(fixedTimeStep=self.timeStep, physicsClientId=self.pcId)
		
		
		p.resetBasePositionAndOrientation(self.robotId, [0, 0, 0], [0, 0, 0, 1], physicsClientId=self.pcId)
		base_pos, base_rot = p.getBasePositionAndOrientation(self.robotId, physicsClientId=self.pcId)
		p.resetDebugVisualizerCamera (1, 30, -15, base_pos, physicsClientId=self.pcId)
		
		print(p.getNumJoints(self.robotId, physicsClientId=self.pcId))
		zero_pos = -np.asarray(p.getLinkState(self.robotId, 0, physicsClientId=self.pcId)[0])
		print("body_mass :", p.getDynamicsInfo(self.robotId, -1, physicsClientId=self.pcId)[0])
		leg_mass = 0
		# p.getNumJoints(self.robotId, physicsClientId=self.pcId)
		for i in range(3):
			leg_mass += p.getDynamicsInfo(self.robotId, i, physicsClientId=self.pcId)[0]
		print("leg_mass :", leg_mass)
		
if __name__ == "__main__":
	sim = Simulator()

	
	while (1):
		p.getLinkState(sim.robotId, 0, physicsClientId=sim.pcId)
		time.sleep(0.3)
	