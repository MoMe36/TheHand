import pygame
from pygame.locals import *
import numpy as np 
from OpenGL.GL import *
from OpenGL.GLU import *



def get_articulation(p0,p1, angle, scale, width = 1.):

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
			moved_point = p + decals[j]*width*scale
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

	def __init__(self, pos, nb_joints, length, width): 
		
		self.pos = pos 
		self.nb_joints = nb_joints
		self.length = length
		self.width = width 
		self.angles = np.random.uniform(0.,np.pi*0.5, (self.nb_joints))*0.5
	
	def get_joints_positions(self, scale): 
	
		base_pos = self.pos[:2]

		joints = np.hstack([np.cos(self.angles).reshape(-1,1), np.sin(self.angles).reshape(-1,1)])
		joints *= self.length
		joints = np.vstack([base_pos, joints])
		joints = np.cumsum(joints, 0)
		
		z_pos = np.ones((joints.shape[0], 1))*self.pos[-1]


		joints = np.hstack([joints, z_pos])*scale

		cubes = []
		edges = []
		surfaces = []
		for i in range(joints.shape[0] -1):
			articulation, art_edges, surf = get_articulation(joints[i], joints[i+1], self.angles[i], scale, self.width) 

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

	def __init__(self, nb_fingers, nb_joints, joints_length, width, spacing_ratio): 
		
		self.nb_fingers = nb_fingers
		self.nb_joints =nb_joints
		self.joints_length = joints_length
		self.spacing_ratio = spacing_ratio

		self.fingers = [Finger(np.array([0.0,0.0,f*spacing_ratio*width]), self.nb_joints, self.joints_length, width) for f in range(self.nb_fingers)]

	def compute_draw_infos(self, scale): 


		fingers = [f.get_joints_positions(scale) for f in self.fingers]

		return fingers

	def move(self, action): 

		for f,a in zip(self.fingers, action): 
			f.move(a)
		
class World: 

	def __init__(self, nb_fingers = 3, nb_joints = 3, joints_length = 0.2, scale = 15., width = 0.05, spacing_ratio = 1.5): 
		
		
		self.scale = scale 
		self.nb_joints = nb_joints
		self.nb_fingers = nb_fingers
		self.joints_length = joints_length
		
		self.hand = Hand(nb_fingers, nb_joints, joints_length, width, spacing_ratio)

		self.targets = np.array([[np.random.uniform(0.3, 0.7), np.random.uniform(-0.5,0.3), f*spacing_ratio*width] for f in range(nb_fingers)])


		self.render_ready = False 

	def init_render(self): 

		pygame.init()
		display = (800,600)
		screen = pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

		glEnable(GL_DEPTH_TEST)
		glEnable(GL_LIGHTING)

		light_direction = [0.8, 0.8, 1.0, 1.0]
		light_intensity = [0.9, 0.9, 0.9, 1.0]
		ambient_intensity = [0.8, 0.8, 0.8, 1.0]

		glLightModelfv(GL_LIGHT_MODEL_AMBIENT,  ambient_intensity)
		glEnable(GL_LIGHT0)

		glLightfv(GL_LIGHT0, GL_POSITION, light_direction)
		glLightfv(GL_LIGHT0, GL_DIFFUSE, light_intensity)

		glEnable(GL_COLOR_MATERIAL)
		glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)


		self.quadratic = gluNewQuadric()
		gluQuadricNormals(self.quadratic, GLU_SMOOTH)		# Create Smooth Normals (NEW)
		gluQuadricTexture(self.quadratic, GL_TRUE)
		gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
		glTranslatef(-7.0,-5, -30)
		glRotatef(45,1.,0,0.)

		self.render_ready = True

	def render(self): 

		if not self.render_ready: 
			self.init_render()

		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

		self.draw()
		pygame.display.flip()
		pygame.time.wait(10)

	def draw_targets(self): 

		for target_pos in self.targets:
			glPushMatrix()

			pos = target_pos*self.scale

			glColor3fv((0.8,0.6,0.3))
			glTranslatef(*pos)
			gluQuadricDrawStyle(self.quadratic, GLU_FILL);
			gluSphere(self.quadratic,0.65,16,16)		
			glColor3fv((1,1,1))
			gluQuadricDrawStyle(self.quadratic, GLU_LINE);
			gluSphere(self.quadratic,0.7,8,6)

			glPopMatrix()


	def draw(self): 

		data = self.hand.compute_draw_infos(self.scale)
		glColor3fv((0.3,0.75,0.5)) 
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

		self.draw_targets()
		# glColor3fv((0.8,0.6,0.3))
		# glPushMatrix()
		# # glLoadIdentity()
		# glTranslatef(1,1,1)
		# gluQuadricDrawStyle(self.quadratic, GLU_FILL);
		# gluSphere(self.quadratic,0.65,16,16)		
		# glColor3fv((1,1,1))
		# gluQuadricDrawStyle(self.quadratic, GLU_LINE);
		# gluSphere(self.quadratic,0.7,8,6)
		# glPopMatrix()





# colors = ((1.,0.,0.), 
# 		  (0.,1.,0.), 
# 		  (0.,0.,1.))

def draw(world, quadratic): 

	hand = world.hand
	scale = world.scale 
	data = hand.compute_draw_infos(scale)
	glColor3fv((0.3,0.75,0.5)) 
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


	glColor3fv((0.8,0.6,0.3))
	glPushMatrix()
	# glLoadIdentity()
	glTranslatef(1,1,1)
	gluQuadricDrawStyle(quadratic, GLU_FILL);
	gluSphere(quadratic,0.65,16,16)		
	glColor3fv((1,1,1))
	gluQuadricDrawStyle(quadratic, GLU_LINE);
	gluSphere(quadratic,0.7,8,6)
	glPopMatrix()



	# input()



def draw_text(screen, font): 

	text = "Random value: {}".format(np.random.uniform(0.,1.))
	position = (0,15,2)
	textSurface = font.render(text, True, (255,255,255,255), (0,0,0,255))     
	textData = pygame.image.tostring(textSurface, "RGBA", True)     
	glRasterPos3d(*position)     
	glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)


	# Method called several times because \n is not supported 

	text = "Random value: {}".format(np.random.uniform(0.,1.))
	position = (0,14,2)
	textSurface = font.render(text, True, (255,255,255,255), (0,0,0,255))     
	textData = pygame.image.tostring(textSurface, "RGBA", True)     
	glRasterPos3d(*position)     
	glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)


def main():
	# pygame.init()
	# display = (800,600)
	# screen = pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

	# font = pygame.font.SysFont('monospace', 15)


	world = World()
	#hand = Hand(3,3,3.)

	# glEnable(GL_DEPTH_TEST)
	# glEnable(GL_LIGHTING)

	# light_direction = [0.8, 0.8, 1.0, 1.0]
	# light_intensity = [0.9, 0.9, 0.9, 1.0]
	# ambient_intensity = [0.8, 0.8, 0.8, 1.0]

	# glLightModelfv(GL_LIGHT_MODEL_AMBIENT,  ambient_intensity)
	# glEnable(GL_LIGHT0)

	# glLightfv(GL_LIGHT0, GL_POSITION, light_direction)
	# glLightfv(GL_LIGHT0, GL_DIFFUSE, light_intensity)

	# glEnable(GL_COLOR_MATERIAL)
	# glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)


	# quadratic = gluNewQuadric()
	# gluQuadricNormals(quadratic, GLU_SMOOTH)		# Create Smooth Normals (NEW)
	# gluQuadricTexture(quadratic, GL_TRUE)
	# gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
	# glTranslatef(-7.0,-5, -30)
	# glRotatef(45,1.,0,0.)



	incs = [0.7, -0.8, 0.5]
	counter = 0 
	while True:
		# for event in pygame.event.get():
		# 	if event.type == pygame.QUIT:
		# 		pygame.quit()
		# 		quit()

		# glRotatef(0.2, 0, 1, 0)
		# glTranslatef(0.1,0,0)
		# glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
		# glShadeModel(GL_FLAT)

		# draw(world, quadratic)
		world.render()
		world.hand.move(incs)
		counter += 1 
		if counter > 80: 
			incs = np.random.uniform(-1.,1., (3,3))
			counter = 0

		# draw_text(screen, font)

		# pygame.display.flip()
		# pygame.time.wait(10)



main()

