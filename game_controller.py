import pygame
import time

pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() > 0:
	joystick = pygame.joystick.Joystick(0)
	joystick.init()
	print("got a  joystick")
else:
	print("no joystick")
	joystick = None

def get_action ():
	if joystick is not None:
		pygame.event.get()
		to_return = [joystick.get_axis(i) for i in range(6)]
		#print(to_return)
		return to_return
	return [0] * 6

if __name__ == "__main__":

	done = False


	started = False

	while 1:
		print(get_action())
		time.sleep(0.3)

"""
import pygame
import time

if __name__ == "__main__":
	print("coucou")
	
	pygame.init()
	pygame.joystick.init()

	while 1:
		get_action()
		time.sleep(0.3)
"""