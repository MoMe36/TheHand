import pygame
from pygame.locals import *
import numpy as np 
from OpenGL.GL import *
from OpenGL.GLU import *



def get_articulation(p0,p1, angle, width = 1.):

	# print(p0)
	# print(p1)


	points = []

	# ====== Shows the points order ========
	# decals = np.array([[0.,0., 1.],
	# 				   [0.,1., 1.],
	# 				   [0,1., 0.]
	# 				   ])


	# ====== Real distance based on angles =====
	decals = np.array([[0.,0., 1.],
					   [-np.sin(angle),np.cos(angle), 1.],
					   [-np.sin(angle), np.cos(angle), 0.]
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
	
	surfaces = ((0,1,2,3),
				(3,2,6,7),
				(2,1,5,6),
				(4,5,6,7),
				(0,4,7,3),
				(0,1,5,4) )

	return points, edges, surfaces


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
		surfaces = []
		for i in range(joints.shape[0] -1):
			articulation, art_edges, surf = get_articulation(joints[i], joints[i+1], self.angles[i]) 

			cubes.append(articulation)
			edges.append(art_edges)
			surfaces.append(surf)

		return cubes, edges, surfaces

	def move(self, action): 

		self.angles += action*0.01
		self.angles = np.clip(self.angles, 0., np.pi/2.)
	
class Target: 

	def __init__(self, pos):

		self.pos = pos 
	
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

def draw(hand, quadratic): 

	data = hand.compute_draw_infos()
	glColor3fv((1,1,1)) 
	glBegin(GL_QUADS)
	for finger_data in data:
		cube, cube_edges, all_surfaces = finger_data
		for surfaces, c in zip(all_surfaces, cube):
			for surface in surfaces: 
				for vertex in surface: 
					# input(vertex)
					glVertex3fv(c[vertex])
	glEnd()

	glColor3fv((0,0,0)) 
	glBegin(GL_LINES)
	for finger_data in data: 
		cube, cube_edges, surfaces = finger_data
		color = 0
		for c, ce in zip(cube, cube_edges):
			# glColor3fv(colors[color]) 
			for edge in ce: 
				for v in edge: 
					glVertex3fv(c[v])
		
			color += 1

	glEnd()


	glColor3fv((1.,0.5,0.1))
	glPushMatrix()
	# glLoadIdentity()
	glTranslatef(1,1,1)
	gluSphere(quadratic,0.5,32,32)		
	glPopMatrix()

def draw_text(screen, font): 

	label = font.render('Saluuuuut', 1, (255,255,255))
	screen.blit(label, (120,120))


def main():
	pygame.init()
	display = (800,600)
	screen = pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

	font = pygame.font.SysFont('monospace', 15)


	hand = Hand(3,3,3.)
	
	quadratic = gluNewQuadric()
	gluQuadricNormals(quadratic, GLU_SMOOTH)		# Create Smooth Normals (NEW)
	gluQuadricTexture(quadratic, GL_TRUE)
	gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
	glTranslatef(-7.0,-5, -30)
	glRotatef(45,1.,0,0.)

	incs = [0.7, -0.8, 0.5]

	while True:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()

		# glRotatef(0.2, 0, 1, 0)
		# glTranslatef(0.1,0,0)
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
		draw(hand, quadratic)
		hand.move(incs)

		# draw_text(screen, font)

		pygame.display.flip()
		pygame.time.wait(10)


main()

