import pygame
from pygame.locals import *
import numpy as np 
from OpenGL.GL import *
from OpenGL.GLU import *
import gym 
from gym import spaces, error, utils
from gym.utils import seeding 

np.set_printoptions(formatter={'float':'{:0.2f}'.format})
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
	
	def get_joints_draw_positions(self, scale): 
	
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

	def get_joints_pos(self): 

		base_pos = self.pos[:2]

		joints = np.hstack([np.cos(self.angles).reshape(-1,1), np.sin(self.angles).reshape(-1,1)])
		joints *= self.length
		joints = np.vstack([base_pos, joints])
		joints = np.cumsum(joints, 0)
		
		z_pos = np.ones((joints.shape[0], 1))*self.pos[-1]


		joints = np.hstack([joints, z_pos])

		return joints

	def move(self, action): 

		self.angles += action*0.01
		self.angles = np.clip(self.angles, -np.pi*0.5, np.pi*0.5)
#		self.angles[0] = np.clip(self.angles[0], -np.pi/3., np.pi/6)
#		for i in range(1, self.angles.shape[0]): 
#			self.angles[i] = np.clip(self.angles[i], -np.pi/3., self.angles[i-1])
	
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

		fingers = [f.get_joints_draw_positions(scale) for f in self.fingers]

		return fingers

	def move(self, action): 

		for f,a in zip(self.fingers, action): 
			f.move(a)
		
class World(gym.Env): 

	metadata = {'render.modes':['human']}

	def __init__(self, nb_fingers = 1, nb_joints = 3, joints_length = 0.2, scale = 15., width = 0.05, spacing_ratio = 1.5, max_steps = 500): 
		
		
		super().__init__()

		self.scale = scale 
		self.nb_joints = nb_joints
		self.nb_fingers = nb_fingers
		self.joints_length = joints_length
		self.fingers_width = width
		self.fingers_spacing = spacing_ratio
	

		self.initialize_spaces()

		self.max_steps = max_steps

		self.render_ready = False 

	def initialize_spaces(self): 

		obs = np.ones((self.nb_joints + 2))
		low_obs = -np.pi*0.5*obs 
		high_obs = np.pi*0.5*obs

		ac = np.ones((self.nb_joints))
		low_ac = -ac
		high_ac = ac

		self.observation_space = spaces.Box(low_obs, high_obs, dtype = np.float)
		self.action_space = spaces.Box(low_ac, high_ac, dtype = np.float)

	def create_hand(self): 
		self.hand = Hand(self.nb_fingers, self.nb_joints, self.joints_length, self.fingers_width, self.fingers_spacing)

	def create_targets(self): 

		max_length = self.joints_length*self.nb_joints
		distances = np.random.uniform(0.4,0.9, (self.nb_fingers))*max_length 
		angles = np.random.uniform(-np.pi*0.5, np.pi/6, (self.nb_fingers))

		self.targets = np.array([[np.cos(a)*d, np.sin(a)*d, i*self.fingers_spacing*self.fingers_width] for i, (a,d) in enumerate(zip(angles, distances))])
		#self.targets = np.array([[np.random.uniform(0.3, 0.7), np.random.uniform(-0.5,0.1), f*self.fingers_spacing*self.fingers_width] for f in range(self.nb_fingers)])

	def reset(self): 

		self.steps = 0 
		self.create_hand()
		self.create_targets()

		return self.observe()[0]
	
	def observe(self): 

		state = []
		for f in self.hand.fingers: 

			current_angles = f.angles.tolist()
			for a in current_angles:
				state.append(a)
			
		for t in self.targets: 
			for tp in t: 
				state.append(tp)

		reward = 0 
		for f,t in zip(self.hand.fingers, self.targets):
			effector_position = f.get_joints_pos()[-1] 
			distance = np.sqrt(np.sum(np.power(effector_position - t, 2)))
			finger_reward = 1 - np.min([distance, 1.])
			reward += finger_reward

		reward /= float(len(self.hand.fingers))
		done = False 
		infos = {}
		if self.steps > self.max_steps:
			done = True
	
		self.info_to_text = 'Reward: {:.3f}'.format(reward)
		return state, reward, done, infos


	def step(self, action):  

		self.steps += 1

		self.hand.move(action)

		return self.observe()


	def init_render(self): 

		pygame.init()
		display = (800,600)
		self.screen = pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
		self.font = pygame.font.SysFont('monospace', 15)

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
		glTranslatef(-7.0,0., -20)
		glRotatef(0.,1.,0,0.)

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
		self.move_camera()
		self.draw_text(	)

	def draw_text(self): 
	

		base_position = np.array([1, 5, 2])
		for i,(f,t) in enumerate(zip(self.hand.fingers, self.targets)): 
			text = self.get_fingers_target_infos(f,t,i)
			position = base_position + np.array([0,i*0.5, 0])
			self.render_text(text,position)
			

		text = "Steps: {}/{} {}".format(self.steps, self.max_steps, self.info_to_text)
		position = base_position + np.array([0,-1,2])
		self.render_text(text,position)

	def render_text(self, text, position): 

		textSurface = self.font.render(text, True, (255,255,255,255), (0,0,0,255))     
		textData = pygame.image.tostring(textSurface, "RGBA", True)     
		glRasterPos3d(*position)     
		glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)


	def get_fingers_target_infos(self, finger, target, num): 

		text = "Random value: {}".format(np.random.uniform(0.,1.))
		text = "ID: {} Effector pos: {} Target pos {}".format(num, finger.get_joints_pos()[-1][:-1], target[:-1])
		return text

	def move_camera(self): 

		glPushMatrix()
		translation = np.zeros((3))

		event = pygame.event.get()

		keys = pygame.key.get_pressed()

		if keys[273] == 1: 
			translation[2] += 1
		if keys[274] == 1:
			translation[2] -= 1
		if keys[275] == 1: 
			translation[0] -= 1
		if keys[276] == 1: 
			translation[0] += 1
		if keys[117] == 1: 
			translation[1] -= 1
		if keys[106] == 1: 
			translation[1] += 1	

		translation *= 0.2

		glTranslatef(*translation)

		rotation = np.zeros_like(translation)
		if keys[113] == 1: 
			rotation[1] += 1
		if keys[100] == 1: 
			rotation[1] -= 1

		glRotatef(1,*rotation)
		# glRotatef(1, 0.,1.,0.)

		glPopMatrix()


def main():
	
	world = World()
	world.reset()
	
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
		ns, r, done, _ = world.step(incs)
		
		if done: 
			world.reset()
		counter += 1 
		if counter > 80: 
			incs = np.random.uniform(-1.,1., (3,3))
			#incs[:,1:] = 0.
			incs = np.ones_like(incs)*(-1.)	
			counter = 0

		# draw_text(screen, font)

		# pygame.display.flip()
		# pygame.time.wait(10)



main()

