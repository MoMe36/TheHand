import pygame
from pygame.locals import *
import numpy as np 
from OpenGL.GL import *
from OpenGL.GLU import *



def get_articulation(p0,p1, width = 1.):

	# print(p0)
	# print(p1)


	points = []
	decals = np.array([[0.,0., 1.],
					   [0.,1., 1.],
					   [0,1., 0.]
					   ])

	for i, p in enumerate([p0,p1]): 
		points.append(p)
		for j in range(decals.shape[0]): 
			moved_point = p + decals[j]*width
			points.append(moved_point.tolist())

	edges = ((0,1),
			 (0,3),
			 (0,4),
			 (2,1),
			 (2,3),
			 (2,6),
			 (6,7),
			 (6,5),
			 (5,1),
			 (5,4),
			 (7,3),
			 (7,4))
	

	return points, edges


class Finger:

	def __init__(self, pos, nb_joints, length): 
		
		self.pos = pos 
		self.nb_joints = nb_joints
		self.length = length

		self.angles = np.random.uniform(0.,np.pi*0.5, (self.nb_joints))*0.5
	
	def get_joints_positions(self): 
	
		base_pos = self.pos[:2]

		joints = np.hstack([np.cos(self.angles).reshape(-1,1), np.sin(self.angles).reshape(-1,1)])
		joints *= self.length
		joints = np.vstack([base_pos, joints])
		joints = np.cumsum(joints, 0)
		
		z_pos = np.ones((joints.shape[0], 1))*self.pos[-1]


		joints = np.hstack([joints, z_pos])

		cubes = []
		edges = []
		for i in range(joints.shape[0] -1):
			articulation, art_edges = get_articulation(joints[i], joints[i+1]) 

			cubes.append(articulation)
			edges.append(art_edges)

		return cubes, edges

	def move(self, action): 

		self.angles += action*0.01
		self.angles = np.clip(self.angles, 0., np.pi/2.)
	

class Hand: 

	def __init__(self, nb_fingers, nb_joints, joints_length): 
		
		self.nb_fingers = nb_fingers
		self.nb_joints =nb_joints
		self.joints_length = joints_length

		self.fingers = [Finger(np.array([0.0,0.5,2.*_]), self.nb_joints, self.joints_length) for _ in range(self.nb_fingers)]

	def compute_draw_infos(self): 

		fingers = [f.get_joints_positions() for f in self.fingers]

		return fingers

	def move(self, action): 

		for f,a in zip(self.fingers, action): 
			f.move(a)
		
colors = ((1.,0.,0.), 
		  (0.,1.,0.), 
		  (0.,0.,1.))

def draw(hand): 

	glBegin(GL_LINES)

	data = hand.compute_draw_infos()

	for finger_data in data: 
		cube, cube_edges = finger_data
		color = 0
		for c, ce in zip(cube, cube_edges):
			glColor3fv(colors[color]) 

			
			for edge in ce: 
				for v in edge: 
					glVertex3fv(c[v])
		
			color += 1


	glEnd()




def main():
	pygame.init()
	display = (800,600)
	pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

	hand = Hand(3,3,3.)
	

	gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
	glTranslatef(0.0,-2, -20)
	glRotatef(45,1.,0,0.)

	incs = [0.1, -0.2, 0.1]

	while True:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()

		glRotatef(0.2, 0, 1, 0)
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
		draw(hand)
		hand.move(incs)
		pygame.display.flip()
		pygame.time.wait(10)


main()
