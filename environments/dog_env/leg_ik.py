import numpy as np

class Leg:
	def __init__ (self, inv_x=False, inv_y=False, inv_1=False, inv_2=False, inv_3=False):
		self.l1 = 0.196
		#self.l2 = 0.180
		self.l2 = 0.200
		self.e = 0.02 + 0.098/2 # 0.069
		self.max_norm_2 = (self.l1+self.l2)**2 + self.e**2 - 0.01
		self.min_norm_2 = self.e**2 + 0.01
		
		self.fac_1 = -1 if inv_1 else 1
		self.fac_2 = -1 if inv_2 else 1
		self.fac_3 = -1 if inv_3 else 1
		
		self.fac_x = -1 if inv_x else 1
		self.fac_y = -1 if inv_y else 1
		self.inv_y = inv_y
		
		self.create_range ()
	
	def motor_pos (self, action, phase):
		foot_coord = self.calc_coord(action) + self.phase_to_pos(phase) # + self.up_to_pos(loc_up_vect)
		
		norm_2 = np.sum(np.square(foot_coord))
		if norm_2 >= self.max_norm_2:
			foot_coord = foot_coord*np.sqrt(self.max_norm_2/norm_2)
		if norm_2 <= self.min_norm_2:
			foot_coord = foot_coord*np.sqrt(self.min_norm_2/norm_2)
		
		motor_angle = self.calc_angle(foot_coord[:,0])
		motor_angle = self.cap_angle (motor_angle)
		return motor_angle
	
	def up_to_pos (self, loc_up_vect):
		delta_up = (np.asarray([0, 0, 1])-loc_up_vect).reshape((3,1))
		delta_up = np.maximum(np.minimum(delta_up, 0.5), -0.5)
		if self.inv_y:
			delta_up[1] = -delta_up[1]
		return  delta_up*0.3
	
	def phase_to_pos (self, phase):
		#k = 2*(phase - np.pi)/np.pi
		r = 0.4
		k = int(phase/(2*np.pi))
		res = phase - k*np.pi*2
		if res > 2*np.pi*r:
			z = 0
		else:
			z = (1-np.cos(res/r))
		h = 0.05 # 0.05
		#print(h*z)
		return np.asarray([0, 0, h*z]).reshape((3,1))
	
	# --- From raw action to cathesian coordinate ---
	def calc_coord (self, action):
		if self.inv_y:
			action[1] = 1-action[1]
		a = np.diag(action)
		u = (np.identity(3)-a) @ self.u_m + a @ self.u_M
		b = (np.identity(3)-a) @ self.b_m + a @ self.b_M
		x = np.linalg.solve (u, b)
		return x
	
	# --- From cathesian coordinate to rotation angle ---
	def calc_angle (self, coord): # coord : [x, y, z]
		coord[0] *= self.fac_x
		
		lpx = np.sqrt(coord[1]**2 + coord[2]**2 - self.e**2)
		theta_1 = np.arctan2(lpx, self.e)
		theta_2 = np.arctan2(coord[1], -coord[2])
		a1 = np.pi/2 - theta_1 - theta_2
		
		d2 = lpx**2 + coord[0]**2
		d = np.sqrt(d2)
		a3 = np.pi - np.arccos((self.l1*self.l1 + self.l2*self.l2 - d2)/(2*self.l1*self.l2))
		
		a_aux = np.arccos((self.l1*self.l1 - self.l2*self.l2 + d2)/(2*self.l1*d))
		a2 = np.arcsin(coord[0]/d) - a_aux
		
		# inv calculations
		a1 = a1*self.fac_1
		a2 = a2*self.fac_2
		a3 = a3*self.fac_3
		
		return [a1, a2, -a3]
		
	# --- Creating the work volume ---
	def create_range (self):
		
		dx = 0.25 # 0.069
		dz = 0.20
		
		xm = -dx - 0.08 * self.fac_x
		xM = dx - 0.08 * self.fac_x
		ym = -0.069 # 0
		yl = 3*0.069 #2*0.069
		yh = yl
		zm = -(self.l1+self.l2)
		zM = zm + dz
		#zM = zm+2*0.069
		"""
		dx = 0.15 # 0.069
		dz = 0.20
		
		xm = -dx - 0.1 * self.fac_x
		xM = dx - 0.1 * self.fac_x
		ym = 0
		yl = 2*0.069
		yh = yl
		zm = -(self.l1+self.l2)
		zM = zm + dz
		#zM = zm+2*0.069
		"""
		
		
		umx, bmx = self.face_from_point([xm, yl, zm], [xm, ym, zm], [xm, yh, zM])
		uMx, bMx = self.face_from_point([xM, yl, zm], [xM, ym, zm], [xM, yh, zM])
		
		umy, bmy = self.face_from_point([xm, ym, zm], [xM, ym, zm], [xM, ym, zM])
		uMy, bMy = self.face_from_point([xm, yl, zm], [xM, yl, zm], [xM, yh, zM])
		
		umz, bmz = self.face_from_point([xm, yl, zm], [xM, yl, zm], [xM, ym, zm])
		uMz, bMz = self.face_from_point([xm, yh, zM], [xM, yh, zM], [xM, ym, zM])
		
		self.u_m = np.stack([umx, umy, umz])
		self.u_M = np.stack([uMx, uMy, uMz])
		
		self.b_m = np.asarray([bmx, bmy, bmz]).reshape((3,1))
		self.b_M = np.asarray([bMx, bMy, bMz]).reshape((3,1))
		
	def face_from_point (self, p1, p2, p3):
		p1 = np.asarray(p1).reshape((3,))
		p2 = np.asarray(p2).reshape((3,))
		p3 = np.asarray(p3).reshape((3,))
		v1 = p1-p2
		v2 = p3-p2
		u = np.cross(v1,v2)
		u /= np.sqrt(np.sum(np.square(u)))
		b = np.sum(p2*u)
		return u, b

	def standard_rot (self, rot):
		return [rot[0]*self.fac_1, rot[1]*self.fac_2, rot[2]*self.fac_3]
		
	def cap_angle (self, angles):
		return angles