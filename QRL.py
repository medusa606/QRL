# from __future__ import division	
import sys
import numpy as np
import numpy.ma as ma #for mask arrays
import cv2
import time
import matplotlib.pyplot as plt
import random
import os.path
import datetime

class Environment(object):
	
	def __init__(self, gridH, gridW, end_positions, end_rewards, blocked_positions, start_position, default_reward, road_positions, road_rewards, scale=25):
		
		self.action_space = 4
		self.state_space = gridH * gridW	
		self.gridH = gridH
		self.gridW = gridW
		self.scale = scale 

		self.end_positions = end_positions
		self.end_rewards = end_rewards
		self.blocked_positions = blocked_positions

		self.road_positions = road_positions
		self.road_rewards = road_rewards

		#perceptions
		self.on_road = 0
		self.diff_x = 0
		self.diff_y = 0
		self.euclid = 0
		self.inv_euclid = 0
		self.inv_euclid2 = 0
		self.inv_euclid3 = 0
		self.last_av_pos = -1
		
		self.start_position = start_position
		if self.start_position == None:
			self.position = self.init_start_state()
		else:
			self.position = self.start_position
						
		self.state2idx = {}
		self.idx2state = {}
		self.idx2reward = {}
		for i in range(self.gridH):
			for j in range(self.gridW):
				idx = i*self.gridW + j
				self.state2idx[(i, j)] = idx
				self.idx2state[idx]=(i, j)
				self.idx2reward[idx] = default_reward
				
		# set the AV reward
		for position, reward in zip(self.end_positions, self.end_rewards):
			self.idx2reward[self.state2idx[position]] = reward

		#update road rewards
		for position, reward in zip(self.road_positions, self.road_rewards):
			self.idx2reward[self.state2idx[position]] = reward

		self.frame = np.zeros((self.gridH * self.scale, self.gridW * self.scale, 3), np.uint8)	
		
		# for position in self.blocked_positions:			
		# 	y, x = position			
		# 	cv2.rectangle(self.frame, (x*self.scale, y*self.scale), ((x+1)*self.scale, (y+1)*self.scale), (100, 100, 100), -1)
		
		for position, reward in zip(self.road_positions, self.road_rewards):
			text = str(int(reward))
			if reward > 0.0: text = '+' + text			
			if reward > 0.0: color = (0, 255, 0)
			else: color = (0, 0, 255)
			font = cv2.FONT_HERSHEY_SIMPLEX
			y, x = position		
			(w, h), _ = cv2.getTextSize(text, font, 1, 2)
			# cv2.rectangle(self.frame, (x*self.scale, y*self.scale), ((x+1)*self.scale, (y+1)*self.scale), (100, 100, 100), -1) #from blocked positions
			# cv2.putText(self.frame, text, (int((x+0.5)*self.scale-w/2), int((y+0.5)*self.scale+h/2)), font, 1, color, 2, cv2.LINE_AA)	

		for position, reward in zip(self.end_positions, self.end_rewards):
			text = str(int(reward))
			if reward > 0.0: text = '+' + text			
			if reward > 0.0: color = (0, 255, 0)
			else: color = (0, 0, 255)
			font = cv2.FONT_HERSHEY_SIMPLEX
			y, x = position		
			(w, h), _ = cv2.getTextSize(text, font, 1, 2)
			cv2.putText(self.frame, text, (int((x+0.5)*self.scale-w/2), int((y+0.5)*self.scale+h/2)), font, 1, color, 2, cv2.LINE_AA)	
			#cv2.putText(self.frame, text, (int((x+0.5)*self.scale)-w/2, int((y+0.5)*self.scale+h/2)), font, 1, color, 2, cv2.LINE_AA)
			
        # colour the pavements
		pavement_rows = [0,1,10,11]
		for y in pavement_rows:
			for x in range(self.gridW+1):
				cv2.rectangle(self.frame, (x*self.scale, y*self.scale), ((x+1)*self.scale, (y+1)*self.scale), (100, 100, 100), -1)



	def init_start_state(self):
		
		while True:
			
			preposition = (np.random.choice(self.gridH), np.random.choice(self.gridW))
			
			if preposition not in self.end_positions and preposition not in self.blocked_positions:
				
				return preposition

	def get_state(self):
		
		return self.state2idx[self.position]

	def update_state(self):
		
		#clear the board of previous blocked positions
		self.frame = np.zeros((self.gridH * self.scale, self.gridW * self.scale, 3), np.uint8)	

		# #update blocked positions
		# for position in self.blocked_positions:			
		# 	y, x = position			
		# 	cv2.rectangle(self.frame, (x*self.scale, y*self.scale), ((x+1)*self.scale, (y+1)*self.scale), (100, 100, 100), -1)

		# update the road rewards
		for position, reward in zip(self.road_positions, self.road_rewards):
			text = str(int(reward))
			if reward > 0.0: text = '+' + text			
			if reward > 0.0: color = (0, 255, 0)
			else: color = (0, 0, 255)
			font = cv2.FONT_HERSHEY_SIMPLEX
			y, x = position		
			(w, h), _ = cv2.getTextSize(text, font, 1, 2)
			#cv2.rectangle(self.frame, (x*self.scale, y*self.scale), ((x+1)*self.scale, (y+1)*self.scale), (100, 100, 100), -1) #from blocked positions
			#cv2.putText(self.frame, text, (int((x+0.5)*self.scale-w/2), int((y+0.5)*self.scale+h/2)), font, 0.5, color, 1, cv2.LINE_AA)	

		#GC update the position of the AV and rewards
		for position, reward in zip(self.end_positions, self.end_rewards):
			self.idx2reward[self.state2idx[position]] = reward
			#update grid
			text = str(int(reward))
			if reward > 0.0: text = '+' + text			
			if reward > 0.0: color = (0, 255, 0)
			else: color = (0, 0, 255)
			font = cv2.FONT_HERSHEY_SIMPLEX
			y, x = position		
			(w, h), _ = cv2.getTextSize(text, font, 1, 2)
			cv2.rectangle(self.frame, (x*self.scale, y*self.scale), ((x+1)*self.scale, (y+1)*self.scale), (100, 100, 100), -1) #from blocked positions
			# cv2.putText(self.frame, text, (int((x+0.5)*self.scale-w/2), int((y+0.5)*self.scale+h/2)), font, 0.5, color, 1, cv2.LINE_AA)

		# colour the pavements
		pavement_rows = [0,1,10,11]
		for y in pavement_rows:
			for x in range(self.gridW+1):
				cv2.rectangle(self.frame, (x*self.scale, y*self.scale), ((x+1)*self.scale, (y+1)*self.scale), (100, 100, 100), -1)
			

	def percepts(self, AV_state):
		
		#shorthand - pedestrian position
		xp = self.position[1]
		yp = self.position[0]
		
		#av position
		xa = AV_state[1]
		ya = AV_state[0]

		# is agent on road
		if (xp > 1) | (xp < 4):
			self.on_road = 1
		if (xp <= 1) | (xp >= 4):
			self.on_road = 0
		
		# distance to AV
		self.diff_x = (xp - xa)
		self.diff_y = (yp - ya)
		self.euclid = np.sqrt(np.square(self.diff_x) + np.square(self.diff_y))

		# inverse distance to AV
		if self.euclid == 0:
			self.inv_euclid = 1
			self.inv_euclid2 = 1
			self.inv_euclid3 = 1
		else:
			self.inv_euclid = 1/self.euclid
			self.inv_euclid2 = 1/np.square(self.euclid)
			self.inv_euclid3 = 1/np.power(self.euclid,3)

		# set memory of last known position of the av
		# todo

		# find time to kerb
		# todo

		# find av time to samekerb level
		# todo

		# print("~~~~ env ~~~~")
		# print("xp yp ",xp, yp)
		# print("xa ya ",xa, ya)
		# print("on road ",self.on_road)
		# print("dif y ", self.diff_y)
		# print("diff x ", self.diff_x)
		# print("euclid ", self.euclid)
		# print("inv_euclid ", self.inv_euclid)
		# print("inv_euclid2 ", self.inv_euclid2)
		# print("inv_euclid3 ", self.inv_euclid3)
		# print("~~~~~~~~~~~~~~")

		#return (self.on_road, self.diff_x, self.diff_y, self.euclid, self.inv_euclid, self.inv_euclid2, self.inv_euclid3)
		return (self.on_road, self.diff_x, self.diff_y, self.euclid,0,0,0)

	def one_step_ahead_features(self, future_actions, AV_state):
		
		#shorthand - pedestrian position
		curr_xp = self.position[1]
		curr_yp = self.position[0]
		loop =0

		#print("old position ", curr_xp, curr_yp)
		#print("future_actions ", list(future_actions))

		for action in future_actions:

			#print("action from future_actions", action)

			# Update position based on future_action
			if action == 0:
				xp = curr_xp
				yp = curr_yp + 1 #proposed = (self.position[0] +1, self.position[1])			
			elif action == 1:
				xp = curr_xp
				yp = curr_yp - 1 #proposed = (self.position[0] -1, self.position[1])			
			elif action == 2:
				xp = curr_xp + 1 #proposed = (self.position[0], self.position[1] +1)			
				yp = curr_yp
			elif action == 3:
				xp = curr_xp - 1 #proposed = (self.position[0], self.position[1] -1)
				yp = curr_yp
			#print("new position ", xp, yp)

			#av position
			xa = AV_state[1]
			ya = AV_state[0]

			# is agent on road
			if (xp > 1) | (xp < 4):
				on_road = 1
			if (xp <= 1) | (xp >= 4):
				on_road = 0
			
			# distance to AV
			diff_x = (xp - xa)
			diff_y = (yp - ya)
			euclid = np.sqrt(np.square(diff_x) + np.square(diff_y))

			# inverse distance to AV
			if euclid==0:
				inv_euclid = 1
				inv_euclid2 = 1
				inv_euclid3 = 1
			else:
				inv_euclid = 1/euclid
				inv_euclid2 = 1/np.square(euclid)
				inv_euclid3 = 1/np.power(euclid,3)

			curr_features = np.array([on_road, diff_x, diff_y,euclid, inv_euclid, inv_euclid2, inv_euclid3])
			#print("curr_features ", curr_features)
			#print("loop ", loop)
			if loop == 0:
				features = curr_features
				loop =1
			else:
				features = np.vstack((features,curr_features))
				loop = loop + 1
			#print("features shape ", features.shape)
			#print("features ", features)
		return features

	def get_possible_actions(self):
		
		'''
		pos = self.position
		possible_actions = []
		
		if pos[0]+1<self.nH: and (pos[0]+1,pos[1]) not in self.blocked_positions:
			possible_actions.append(0)
				
		if pos[0]-1>=0 and (pos[0]-1,pos[1]) not in self.blocked_positions:
			possible_actions.append(1)
		
		if pos[1]+1<self.nW and (pos[0],pos[1]+1) not in self.blocked_positions:
			possible_actions.append(2)
				
		if pos[1]-1>=0 and (pos[0],pos[1]-1) not in self.blocked_positions:
			possible_actions.append(3)
		
		return possible_actions		
		'''
		
		return range(self.action_space) 
	
	def step(self, action):

		# Actions are:
		# 0 = down (+Y)
		# 1 = up   (-Y)
		# 2 = right (+X)
		# 3 = left  (-X)
		
		# Check if action is blocked, then update position
		if action >= self.action_space:
			return
		if action == 0:
			proposed = (self.position[0] +1, self.position[1])			
		elif action == 1:
			proposed = (self.position[0] -1, self.position[1])			
		elif action == 2:
			proposed = (self.position[0], self.position[1] +1)			
		elif action == 3:
			proposed = (self.position[0], self.position[1] -1)			
		y_within = proposed[0] >= 0 and proposed[0] < self.gridH
		x_within = proposed[1] >= 0 and proposed[1] < self.gridW
		free = proposed not in self.blocked_positions		
		if x_within and y_within and free:			
			self.position = proposed
			
		next_state = self.state2idx[self.position] 
		reward = self.idx2reward[next_state]
		# print("###STEP### self.position",self.position)
		# print("###STEP### next_state",next_state)
		
		if self.position in self.end_positions:
			done = True
		else:
			done = False
			
		return next_state, reward, done
		
	def reset_state(self):
		
		#print("\n ~ GAME RESTARTING ~\n")
		if self.start_position == None:
			self.position = self.init_start_state()
		else:
			self.position = self.start_position
	
	def render(self, qvalues_matrix, running_score, simTime, nA, agentState):
		
		frame = self.frame.copy()

		# for each state cell
		
		for idx, qvalues in enumerate(qvalues_matrix):
			
			position = self.idx2state[idx]
		
			if position in self.end_positions or position in self.blocked_positions:
				continue
			
			qvalues = np.tanh(qvalues*0.1) # for vizualization only
        	
        	# for each action in state cell
	        		
			for action, qvalue in enumerate(qvalues):

				# draw (state, action) qvalue traingle
				
				if action == 0:
					dx2, dy2, dx3, dy3 = 0.0, 1.0, 1.0, 1.0				
				if action == 1:
					dx2, dy2, dx3, dy3 = 0.0, 0.0, 1.0, 0.0				
				if action == 2:
					dx2, dy2, dx3, dy3 = 1.0, 0.0, 1.0, 1.0				
				if action == 3:
					dx2, dy2, dx3, dy3 = 0.0, 0.0, 0.0, 1.0	
					
				x1 = int(self.scale*(position[1] + 0.5))			
				y1 = int(self.scale*(position[0] + 0.5))				
				
				x2 = int(self.scale*(position[1] + dx2))
				y2 = int(self.scale*(position[0] + dy2))
				
				x3 = int(self.scale*(position[1] + dx3))
				y3 = int(self.scale*(position[0] + dy3))		
				
				pts = np.array([[x1, y1], [x2, y2], [x3, y3]], np.int32)
				pts = pts.reshape((-1, 1, 2))
				
				if qvalue > 0: color = (0, int(qvalue*255),0)
				elif qvalue < 0: color = (0,0, -int(qvalue*255))
				else: color = (0, 0, 0)

				cv2.fillPoly(frame, [pts], color)
			
			# # draw crossed lines
			
			# x1 = int(self.scale*(position[1]))			
			# y1 = int(self.scale*(position[0]))
			
			# x2 = int(self.scale*(position[1] + 1.0))			
			# y2 = int(self.scale*(position[0] + 1.0))

			# x3 = int(self.scale*(position[1]+ 1.0 ))			
			# y3 = int(self.scale*(position[0]))
			
			# x4 = int(self.scale*(position[1]))			
			# y4 = int(self.scale*(position[0] + 1.0))
			
			# cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
			# cv2.line(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
			
			# # draw arrow indicating best action
						
			# best_action = 0
			# best_qvalue = qvalues[0]
			# for action, qvalue in enumerate(qvalues):
			# 	if qvalue > best_qvalue:
			# 		best_qvalue = qvalue
			# 		best_action = action
			
			# if best_action == 0:
			# 	dx1, dy1, dx2, dy2 = 0.0, -0.25, 0.0, 0.25	
							
			# elif best_action == 1:
			# 	dx1, dy1, dx2, dy2 = 0.0, 0.25, 0.0, -0.25
								
			# elif best_action == 2:
			# 	dx1, dy1, dx2, dy2 = -0.25, 0.0, 0.25, 0.0
								
			# elif best_action == 3:
			# 	dx1, dy1, dx2, dy2 = 0.25, 0.0, -0.25, 0.0				
									
			# x1 = int(self.scale*(position[1] + 0.5 + dx1))			
			# y1 = int(self.scale*(position[0] + 0.5 + dy1))	
					
			# x2 = int(self.scale*(position[1] + 0.5 + dx2))			
			# y2 = int(self.scale*(position[0] + 0.5 + dy2))	
							
			# cv2.arrowedLine(frame, (x1, y1), (x2, y2), (255, 100, 0), 8, line_type=8, tipLength=0.5)		
			
		# draw horizontal lines		
		for i in range(self.gridH+1):
			cv2.line(frame, (0, i*self.scale), (self.gridW * self.scale, i*self.scale), (255, 255, 255), 1)
		
		# draw vertical lines		
		for i in range(self.gridW+1):
			cv2.line(frame, (i*self.scale, 0), (i*self.scale, self.gridH * self.scale), (255, 255, 255), 1)
		
		#openCV rectangle function
		# cv2.rectangle(img, pt1, pt2, color, thickness, lineType, shift)
		# Parameters
		#     img   Image.
		#     pt1   Vertex of the rectangle.
		#     pt2    Vertex of the rectangle opposite to pt1 .
		#     color Rectangle color or brightness (grayscale image).
		#     thickness  Thickness of lines that make up the rectangle. Negative values,
		#     like CV_FILLED , mean that the function has to draw a filled rectangle.
		#     lineType  Type of the line. See the line description.
		#     shift   Number of fractional bits in the point coordinates.
		# Must be integers
		# Must have order (left, top) and (right, bottom)

		# # draw agent
		# for i in range(0,2):		
		# 	y, x = self.position
		# 	x = x + (i * 2)		 
		# 	y = y + (i * 2)		 
		# 	y1 = int((y + 0.3)*self.scale)
		# 	x1 = int((x + 0.3)*self.scale)
		# 	y2 = int((y + 0.7)*self.scale)
		# 	x2 = int((x + 0.7)*self.scale)
		# 	cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), -1)
			
		# 	cv2.imshow('frame', frame)
		# 	cv2.moveWindow('frame', 0, 0)
		# 	key = cv2.waitKey(1)
		# 	if key == 27: sys.exit()
		# 	#print('### RENDER1 ### xy',x,y,x1,x2,y1,y2)		
		# 	# cv2.rectangle(frame, (x1+50, y1+50), (x2+50, y2+50), (0, 255, 255), -1)


		#======================================
		# draw agent
		#======================================
		#print("simTime, nA",simTime, nA)
		for agentID in range(0,nA):
			y,x = agentState[simTime, agentID,:]
			y1 = int((y + 0.3)*self.scale)
			x1 = int((x + 0.3)*self.scale)
			y2 = int((y + 0.7)*self.scale)
			x2 = int((x + 0.7)*self.scale)
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), -1)
			#print('### RENDER2 ### xy',x,y,x1,x2,y1,y2)	
			#print("### RENDER ### simTime, agentID, x,y,x1,x2,y1,y2",simTime, agentID,x,y,x1,x2,y1,y2)
			#time.sleep(1)
			
		#======================================
		cv2.imshow('frame', frame)
		cv2.moveWindow('frame', 0, 0)
		key = cv2.waitKey(10)
		if key == 27: sys.exit()



		#print score
		text = 'score = ' + str(int(running_score))
		# if running_score > 0.0: text = '+' + text			
		if running_score > 0.0: color = (0, 255, 0)
		else: color = (0, 0, 255)
		font = cv2.FONT_HERSHEY_SIMPLEX
		# y, x = position		
		y = self.gridH - 1
		x = self.gridW - 1
		(w, h), _ = cv2.getTextSize(text, font, 1, 2)

		# cv2.putText(img, running_score, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) 
		cv2.putText(frame, text, (int((x+0.5)*self.scale-w/2), int((y+0.5)*self.scale+h/2)), font, 0.5, color, 1, cv2.LINE_AA)


		cv2.imshow('frame', frame)
		cv2.moveWindow('frame', 0, 0)
		key = cv2.waitKey(1)
		if key == 27: sys.exit()    

class FeatAgent:

	# This class uses a featured based representation of the world rather than explicit states
	# as such perception is required to inform the agent on the environment
	
	def __init__(self, alpha, epsilon, discount, action_space, state_space):
 
		self.feat_space = 7 #set this to the number of features
		self.action_space = action_space
		self.alpha = alpha
		self.epsilon = epsilon
		self.discount = discount
		# we remove the explicit state space and replace with feature based representation
		#self.qvalues = np.zeros((state_space, action_space), np.float32)
		#self.feat_weights = np.zeros((self.feat_space), np.float32)
		self.feat_weights = np.random.uniform(size=(self.feat_space),low=-1,high=1) #set random feature weights
		self.qvalues = np.zeros((self.feat_space, action_space), np.float32)

		# print("feat_weights ", self.feat_weights)
		# print("qvalues ", self.qvalues)
		# print("feat_space ", self.feat_space)
		# print("action_space ", self.action_space)

	def feat_q_update(self, state, AV_state, action, reward, next_state, next_state_possible_actions, done, features, q_val_dash):

		# calculate Q-values based on the feature representation
		# Q(s,a) = w1.f1(s,a) + w2.f2(s,a) + ... wi.fi(s,a)

		# features are:
		# f1 = on_road
		# f2 = x distance between av and ped
		# f3 = y distance between av and ped
		# f4 = euclidean distance between av and ped
		# f5 = inverse euclidean distance between av and ped
		# f6 = inverse euclidean distance^2 between av and ped
		# f7 = inverse euclidean distance^3 between av and ped

		qval = np.sum(np.multiply(self.feat_weights, features))

		# now update the feature weights given the reward
		difference = (reward + self.alpha * q_val_dash) - qval
		# print("features x weights ", np.multiply(self.feat_weights, features))
		# print("#############################")
		# print("reward ", reward)
		# print("self.discount ", self.discount)
		#print("next_state ", next_state)
		#print("next_state_possible_actions ", list(next_state_possible_actions))
		# print("qval ", qval)
		# print("q_val_dash ", q_val_dash)
		# print("self.feat_weights", self.feat_weights)
		# print("difference", difference)
		#print("self.feat_weights.shape", self.feat_weights.shape)

		for i in range(self.feat_weights.shape[0]):
			wi = self.feat_weights[i]
			self.feat_weights[i] = wi + self.alpha * difference * features[i]
			# print("~~~~~~~~~~~~~~~~~~~~~~~~")
			# print("i", i)
			# print("i w_old new ", i, wi, self.feat_weights[i])
			# print("self.feat_weights[i]", c)
			# print("self.alpha", self.alpha)
			# print("difference", difference)

			# print("wi shape", wi.shape)
			# print("self.feat_weights[i] shape", self.feat_weights[i].shape)
			# print("self.alpha shape", self.alpha)
			# print("difference shape", difference.shape)

			# print("features[i]", features[i])

			


	def update(self, state, action, reward, next_state, next_state_possible_actions, done):

		# Q(s,a) = (1.0 - alpha) * Q(s,a) + alpha * (reward + discount * V(s'))

		if done==True:
			qval_dash = reward
		else:
			qval_dash = reward + self.discount * self.get_value(next_state, next_state_possible_actions)
			
		qval_old = self.qvalues[state][action]      
		qval = (1.0 - self.alpha)* qval_old + self.alpha * qval_dash
		self.qvalues[state][action] = qval

	# def get_best_action(self, state, possible_actions, features):

	# 	print("------ QVAL ------")
	# 	# calculate q-val for all actions
	# 	all_q_val = np.sum(np.multiply(self.feat_weights, features),axis=1)

	# 	# find the best q-val and return the index
	# 	q_val_dash = np.max(all_q_val)
	# 	idx_best_q = np.argmax(all_q_val)

	# 	#retun the action for the best q-val
	# 	best_action = possible_actions[idx_best_q]

	# 	print("all_q_val ", all_q_val)
	# 	print("idx_best_q ", idx_best_q)
	# 	print("best_action ", best_action)

	# 	return best_action, q_val_dash, all_q_val


	def calc_new_feature_func(action, features):

		# if I take this action what will my new feature functions be?

		return new_features



	def get_action(self, state, possible_actions, features):
         
		# with probability epsilon take random action, otherwise - the best policy action
		epsilon = self.epsilon
		# find the best action an associated q-value

		# calculate q-val for all actions
		all_q_val = np.sum(np.multiply(self.feat_weights, features),axis=1)

		# find the best q-val and return the index
		#q_val_dash = np.max(all_q_val)
		idx_best_q = np.argmax(all_q_val)

		#retun the action for the best q-val
		best_action = possible_actions[idx_best_q]

		#chosen_action = self.get_best_action(state, possible_actions, features)
		# print("------------------")
		# print("------ QVAL ------")
		if epsilon > np.random.uniform(0.0, 1.0):
			chosen_action = random.choice(possible_actions)
			action_index = np.where(np.isclose(possible_actions,chosen_action))
			# print("possible_actions ", possible_actions)
			# print("chosen_action ", chosen_action)
			# print("action_index ", action_index)
			# print("action_index shape ", np.shape(action_index))
			# print("action_index[0] ", action_index[0])
			# print("action_index ", action_index)
			# print("random action taken")
			q_val_dash = all_q_val[action_index[0]]
			
		else:
			chosen_action = best_action
			q_val_dash = np.max(all_q_val)

		
		# print("all_q_val ", all_q_val)
		# print("idx_best_q ", idx_best_q)
		# print("best_action ", best_action)
		# print("chosen_action ", chosen_action)
		# print("q_val_dash ", q_val_dash)
		return chosen_action, q_val_dash
        
	def get_value(self, state, possible_actions):
		
		pass

def randomStart(startLocations, simTime, nA, agentState, rsLog, pLog, nExp):
	#print("### randomStart ###")
	# initialise each agent with random position based on "deadzone"
	log_string = ""
	start_array = startLocations[nExp,:]
	# print("nExp start_array",nExp,start_array)

	for agentID in range(0,nA):

		# read start location from master table
		x = start_array[(0+2*agentID)]
		y = start_array[(1+2*agentID)]
		# add locations to agent state array
		agentState[simTime,agentID,0] = x
		agentState[simTime,agentID,1] = y
		log_string = log_string + ", %4i, %4i" % (x,y)
		#print(log_string)		
		#print("initial state for agent", agentID," is ",agentState[simTime,agentID,:])
	rsindex = "%4i" % (nExp)
	index = "%4i, %4i, %4i" % (0, nExp, simTime)
	rsLog.write(rsindex + log_string + "\n")
	pLog.write(index + log_string + "\n")
	
def moveGen(simTime, agentID, rLog):

	ran = random.randint(1,5)
	#print("ran",ran)
	x, y = 0, 0
	if ran==1:
		log = "moving UP"
		y = y - 1
	if ran==2:
		log = "moving DOWN"
		y = y + 1
	if ran==3:
		log = "moving LEFT"
		x = x - 1
	if ran==4:
		log = "moving RIGHT"
		x = x + 1
	if ran==5:
		log = "moving NONE"
	#store the random numbers to check consistency	
	rLog.write("%7i, %4i \n" % (simTime, ran)) 
	return x,y #WARNING X and Y are used the wrong way around

def randomMove(simTime, nA, agentState, pLog, rLog, nExp, AV_y):
	#print("### randomMove ###")
	log_string = ""
	for agentID in range(0,nA):

		illegal_move=True
		while(illegal_move):
			
			#get a delta move randomly
			dx, dy = moveGen(simTime, agentID, rLog)

			# Add delta to previous state
			new_x = int(agentState[simTime-1,agentID,0] + dx)
			new_y = int(agentState[simTime-1,agentID,1] + dy)
			#print("new x y ", new_x, new_y)

			# check if agent has moved off the board
			if (new_x<0):
				#print("ILLEGAL move 1")
				continue
			elif(new_x>gridH-1):
				#print("ILLEGAL move 2")
				continue
			elif(new_y<0):
				#print("ILLEGAL move 3")
				continue
			elif(new_y>gridW-1):
				#print("ILLEGAL move 4")
				#print("y value ",new_y," is greater than grid limit ", gridH)
				continue
			else:
				illegal_move=False

		# Add delta to previous state
		agentState[simTime,agentID,0] = new_x
		agentState[simTime,agentID,1] = new_y
		# Log position data
		# pLog.write("%d, %d, %d, %d \n" % (simTime, agentID, x, y))
		#print("state for agent", agentID," is ",agentState[simTime,agentID,:])
		log_string = log_string + ", %4i, %4i" % (new_x,new_y)
		#print(log_string)
		
		#print("initial state for agent", agentID," is ",agentState[simTime,agentID,:])
	index = "%4i, %4i, %4i" % (nExp, simTime, AV_y)
	pLog.write(index + log_string + "\n")

def randomBehaviour(simTime, nA, agentState, XR_WD_status, pLog, rLog, nExp, AV_y, diag=True):
	# Agent walks along pavement and randomly choose to cross the road
	for agentID in range(0,nA):
		walk_direction = 0
		crossing_road = 0
		log_string = ""
		# ran = random.randint(1,5) #roll 5-sided dice
		ran = random.randint(1,11) #roll 11-sided dice
		rLog.write("%7i, %4i \n" % (simTime, ran))  #log random number
		old_ax = agentState[simTime-1,agentID,0]
		old_ay = agentState[simTime-1,agentID,1]


		#if first step then set agents down pavement
		if int(simTime)==1:
			XR_WD_status[agentID,0] = 0
			XR_WD_status[agentID,1] = 0
			# if old_ay>int(round(gridH/2)): #walking direction
			if ran>1 or ran<7:
				walk_direction = -1
				#if diag:print("Agent is East side")
			elif ran>6:
				walk_direction = 1
				#if diag:print("Agent is West side")
			#set 1/5 chance of crossing road
			if ran==1:
				if old_ax>9: crossing_road = -1 #if on lower pavement, move up
				if old_ax<2: crossing_road = 1 #if on upper pavement, move down
			else:
				crossing_road = 0
			XR_WD_status[agentID,1] = walk_direction
			XR_WD_status[agentID,0] = crossing_road

			#print("old_xy=%2i,%2i xy=%2i,%2i n=%3i t=%2i WD=%2i XR=%2i" % (old_ax, old_ay, 0, 0, nExp, simTime,walk_direction,crossing_road))


		#find walk direction if sim started
		if simTime>1:
			old2_ax = agentState[simTime-2,agentID,0]
			old2_ay = agentState[simTime-2,agentID,1]
			old_ax = agentState[simTime-1,agentID,0]
			old_ay = agentState[simTime-1,agentID,1]
			#if diag:print("Agent old XY=%d %d new XY=%d,%d" % (old_ax, old_ay, ax, ay))
			if old2_ay>old_ay: #walking direction
				walk_direction = -1 #walking 'left'
				if diag:print("Left walking detected")
			if old2_ay<old_ay:
				walk_direction = 1  #walking 'right'
				if diag:print("Right walking detected")
			if old2_ay==old_ay:
				#find if agent is crossing road
				if old2_ax>old_ax:
					crossing_road=-1 #moving 'up'
					if diag:print("Agent is mid-crossing going up")
				elif old2_ax<old_ax:
					crossing_road=1  #moving 'down'				
					if diag:print("Agent is mid-crossing going down")
				else:
					print("##RB## WARNING: Unrecognised agent behaviour")
			#set 1/5 chance of crossing road
			if ran==1:
				if old_ax>9:
					crossing_road = -1 #if on lower pavement, move up
					if diag:print("Agent has decided to cross UP")
				if old_ax<2:
					crossing_road = 1 #if on upper pavement, move down
					if diag:print("Agent has decided to cross DOWN")
			XR_WD_status[agentID,1] = walk_direction
			XR_WD_status[agentID,0] = crossing_road


		# Move the agent based on the walk and crossing direction
		dx = 0
		dy = 0
		if crossing_road == -1:
			dx = -1
			dy = 0
		if crossing_road ==  1:
			dx =  1
			dy = 0
		if crossing_road ==  0:
			if walk_direction == -1:
				dy =  -1
				dx = 0
			elif walk_direction == 1:
				dy = 1
				dx = 0
			else:
				print("##RB## WARNING: No valid move found")
		if diag:print("Agent dx=%2i dy=%2i" % (dx, dy))

		# Determine new position
		new_x = int(old_ax + dx)
		new_y = int(old_ay + dy)


		# Reverse direction if agent hits edge
		if (new_y == 0) or (new_y > gridW-1):
			dy = dy * -1
			if diag:print("Agent at y-limit reversing")
			new_y = int(old_ay + dy)
		if (new_x == 0) or (new_x > gridH-1):
			dx = dx * -1
			new_x = int(old_ax + dx)
			if diag:print("Agent at x-limit reversing")

		# print("nExp=%3i simTime=%2i WD=%2i XR=%2i" % (nExp, simTime,walk_direction,crossing_road))
		if diag: print("old_xy=%2i,%2i xy=%2i,%2i n=%3i t=%2i WD=%2i XR=%2i" % (old_ax, old_ay, new_x, new_y, nExp, simTime,walk_direction,crossing_road))
		# print("walk_direction=%d" % walk_direction)
		# print("crossing_road=%d" % crossing_road)

		# Add delta to previous state
		agentState[simTime,agentID,0] = new_x
		agentState[simTime,agentID,1] = new_y
		log_string = log_string + ", %4i, %4i" % (new_x,new_y)

	# write position log
	index = "%4i, %4i, %4i" % (nExp, simTime, AV_y)
	pLog.write(index + log_string + "\n")

def Proximity(simTime, nA, agentState, pLog, rLog, nExp, AV_y, trigger_radius=15, diag=False):
	# Agent walks along pavement and randomly choose to cross the road
	from scipy.spatial import distance #for cityblock distance
	
	for agentID in range(0,nA):
		walk_direction = 0
		crossing_road = 0
		log_string = ""
		
		ran = random.randint(1,10) #roll 10-sided dice
		rLog.write("%7i, %4i \n" % (simTime, ran))  #log random number
		old_ax = agentState[simTime-1,agentID,0]
		old_ay = agentState[simTime-1,agentID,1]


		#if first step then set agents down pavement
		if int(simTime)==1:
			# if old_ay>int(round(gridH/2)): #walking direction
			if ran<6:
				walk_direction = -1
				#if diag:print("Agent is East side")
			elif ran>5:
				walk_direction = 1
				#if diag:print("Agent is West side")
			else:
				walk_direction = 1
			#check if at edge/corner
			if old_ay==0:
				walk_direction = 1
			if old_ay==gridW-1:
				walk_direction = -1

		#find walk direction if sim started
		if simTime>1:
			crossing_road, walk_direction = detectAction(crossing_road, walk_direction,simTime,agentID)
			# if walk_direction==0 and crossing_road==0:
			# 	print("post detect-action check")
			# 	print("old_xy=%2i,%2i n=%3i t=%2i WD=%2i XR=%2i" % (old_ax, old_ay, nExp, simTime,walk_direction,crossing_road))
			
			
			#If AV is within radius then cross the road
			# AV coordiantes are: [2,3,4,5], AV_y
			pt = np.zeros(shape=(4,1))
			for AV_x in range (2, 6):
				AV_coord = np.array([AV_x,AV_y])
				AG_coord = np.array([old_ax, old_ay])
				#print(AV_coord, type(AV_coord))
				#print(AG_coord, type(AG_coord))
				temp = distance.cityblock(AV_coord, AG_coord)
				pt[AV_x-2] = temp
				#print("AV=%s AG=%s PT=%d" % (AV_coord, AG_coord, temp))	
			prox_MIN = np.min(pt)

			if prox_MIN<trigger_radius:
				if(diag): print("proximity triggered for agent %d at %d m" % (agentID, 1.5*prox_MIN))

				if old_ax>9:
					crossing_road = -1 #if on lower pavement, move up
					if diag:print("Agent has decided to cross UP")
				if old_ax<2:
					crossing_road = 1 #if on upper pavement, move down
					if diag:print("Agent has decided to cross DOWN")


		# calculate new xy positions
		if walk_direction==0 and crossing_road==0:
			print("##Proximity## WARNING: walk_direction overwritten")
			print("old_xy=%2i,%2i n=%3i t=%2i WD=%2i XR=%2i" % (old_ax, old_ay, nExp, simTime,walk_direction,crossing_road))
			#raw_input("Press Enter to continue...")

		new_x, new_y = moveXR(old_ax, old_ay, crossing_road, walk_direction, diag)
		new_x, new_y = checkEdge(gridW, gridH, old_ax, old_ay, new_x, new_y, diag)

				
		if diag: print("old_xy=%2i,%2i xy=%2i,%2i n=%3i t=%2i WD=%2i XR=%2i" % (old_ax, old_ay, new_x, new_y, nExp, simTime,walk_direction,crossing_road))

		# Add delta to previous state
		agentState[simTime,agentID,0] = new_x
		agentState[simTime,agentID,1] = new_y
		log_string = log_string + ", %4i, %4i" % (new_x,new_y)

	# write position log
	index = "%4i, %4i, %4i" % (nExp, simTime, AV_y)
	pLog.write(index + log_string + "\n")	

def Election(simTime, nA, agentState, XR_WD_status, pLog, rLog, nExp, AV_y, CP=True, ECA=True, trigger_radius=15, diag=True):
	# Agent walks along pavement and randomly choose to cross the road
	from scipy.spatial import distance #for cityblock distance

	electionArray = np.zeros(shape=(nA,4)) #store election results
	# XR_WD_status = np.zeros(shape=(nA,2))
	agentElected = False
	use_closest_agent = ECA
	CP_array = np.zeros(shape=(nA,))


	for agentID in range(0,nA):
		walk_direction = 0
		crossing_road = 0
		log_string = ""
		dx = 0
		dy = 0
		
		ran = random.randint(1,10) #roll 10-sided dice
		rLog.write("%7i, %4i \n" % (simTime, ran))  #log random number
		old_ax = agentState[simTime-1,agentID,0]
		old_ay = agentState[simTime-1,agentID,1]


		#if first step then set agents down pavement
		if int(simTime)==1:
			#reset the election parameters
			allAgentsXR = 0
			agentElected = False
			XR_WD_status[agentID,0] = 0
			XR_WD_status[agentID,1] = 0

			# randomly set walking direction
			if ran<6:
				walk_direction = -1
			elif ran>5:
				walk_direction = 1
			else:
				walk_direction = 1
			#check if at edge/corner
			if old_ay==0:
				walk_direction = 1
			if old_ay==gridW-1:
				walk_direction = -1
			# execute move orders
			new_x, new_y = moveXR(old_ax, old_ay, crossing_road, walk_direction)
			new_x, new_y = checkEdge(gridW, gridH, old_ax, old_ay, new_x, new_y)
			# Add delta to previous state
			agentState[simTime,agentID,0] = new_x
			agentState[simTime,agentID,1] = new_y
			XR_WD_status[agentID,1] = walk_direction
			log_string = log_string + ", %4i, %4i" % (new_x,new_y)	


		#find walk direction if sim started
		if simTime>1:
			crossing_road, walk_direction = detectAction(crossing_road, walk_direction,simTime,agentID)
			# XR_WD_status[agentID,0] = crossing_road #don't want to update this as is election specific
			XR_WD_status[agentID,1] = walk_direction

			#see which agents on pavement closest to AV
			all_agents_upper_pavement = agentState[simTime-1,:,0] < 2
			
			
			#Cityblock distance to AV
			pt = np.zeros(shape=(4,1))
			for AV_x in range (2, 6):
				AV_coord = np.array([AV_x,AV_y])
				AG_coord = np.array([old_ax, old_ay])
				#print(AV_coord, type(AV_coord))
				#print(AG_coord, type(AG_coord))
				temp = distance.cityblock(AV_coord, AG_coord)
				pt[AV_x-2] = temp
				#print("AV=%s AG=%s PT=%d" % (AV_coord, AG_coord, temp))	
			#prox_MIN = np.min(pt)
			#print("proximity_test output is %d %d %d %d" % (pt[0],pt[1],pt[2],pt[3]))
			electionArray[agentID,:] = pt[0],pt[1],pt[2],pt[3]
	
	if (simTime>1):

		# Show the minimum proximity per agent
		PT_over_agents = np.min(electionArray, axis=1)
		#if(diag):print("min PT per agent ", PT_over_agents )

		for agentID in range (0,nA):
			
			old_ax = agentState[simTime-1,agentID,0]
			old_ay = agentState[simTime-1,agentID,1]
			curr_XR = XR_WD_status[agentID,0]
			allAgentsXR = np.any(XR_WD_status[:,0])
			#choose the min PT
			min_PT_per_Agent = np.min(electionArray[agentID,:])
			best_candidate = -1
			
			
			# find if any proximity test is within the trigger radius
			trigger = (np.min(electionArray[agentID,:])<=trigger_radius)
			if(diag):print(" ID %d PT=%d trigger=%d XR %d and anyXR= %d ectd=%d" % (agentID, min_PT_per_Agent, trigger, curr_XR, allAgentsXR, agentElected))


			#========================================================
			#
			#   ~~~~~~~~~~~    Hold the election!    ~~~~~~~~~~~~~
			#
			#========================================================
			
			# if there is a single agent, you have a Rotten Borough
			if nA==1 and trigger and not(allAgentsXR):
				best_candidate = 0
				#print("best_candidate ID=%d" % best_candidate)
			# otherwise hold an election to find the best candidate 
			elif nA>1 and trigger and not(allAgentsXR):			
				shortestPT_per_agent = np.min(electionArray,axis=1)
				shortList = shortestPT_per_agent<trigger_radius
				p_shortList = np.logical_and(shortList, all_agents_upper_pavement)
				
				#sum the no of True in this array
				no_eligible_candidates = np.sum(shortList)
				# print("no_eligible_candidates = %d" % no_eligible_candidates)


				# sum the pavement shortlist
				sum_p_shortList = np.sum(p_shortList)
				if sum_p_shortList==0:
					# if all zero then set all True to allow subsequent logic 
					sum_p_shortList=1
				# print("sum_p_shortList",sum_p_shortList)


				# find if any agent meets both criteria
				shortAnd = np.logical_and(shortList, p_shortList)
				p_shortAnd = np.sum(shortAnd)
				
				if no_eligible_candidates==1:
					best_candidate = np.where(shortList)[0]
				
				if no_eligible_candidates>1 and p_shortAnd==0:					
					#choose to elect the closest or farthest agent from the AV
					if use_closest_agent:
						best_candidate = np.where(shortestPT_per_agent == np.amin(shortestPT_per_agent))
					else:
						# chose the agent within the radius but furthest from the AV
						best_candidate = np.where(shortestPT_per_agent == np.amax(shortestPT_per_agent))
			
					#print("best_candidate ID=" , best_candidate)
					#print(type(best_candidate))
					bc_arr = np.asarray(best_candidate)
					#print(bc_arr)
					#print(np.shape(bc_arr))
					flat_bc = bc_arr.flatten()
					# print("best_candidate ID=" , flat_bc)
					best_candidate = flat_bc[0]



				# If there is a candidate on the closer pavement use this one
				if no_eligible_candidates>1 and p_shortAnd>0:					
					
					# print("\nChoice for agent on upper pavement")

					# return the indexes of where shortAnd==True
					shortAnd_idx = np.argwhere(shortAnd==True)
					# print("shortAnd_idx",shortAnd_idx)

					#chose any candidate from the list
					best_candidate = shortAnd_idx[0]
					bc_arr = np.asarray(best_candidate)
					#print(bc_arr)
					#print(np.shape(bc_arr))
					flat_bc = bc_arr.flatten()
					# print("best_candidate ID=" , flat_bc)
					best_candidate = flat_bc[0]


					#Is there more the 1 agent with best PT?
					no_best_candidate = np.shape(best_candidate)
					# print("no_best_candidate =" , no_best_candidate)

					# if(diag):raw_input("Press Enter to continue...")
			else:
				best_candidate = -1


			a = best_candidate==agentID
			b = not(agentElected)
			c = not(allAgentsXR)
			#print("a=%s b=%s c=%s " % (a, b, c))
			


			# if trigger and not(agentElected) and not(allAgentsXR):
			if best_candidate==agentID and not(agentElected) and not(allAgentsXR):
				agentElected = True
				if(diag):print("agent elected to XR xy=%2i,%2i n=%3i t=%2i" % (old_ax, old_ay,nExp,simTime))


				#set XR from the agent state 
				crossing_road = XR_WD_status[agentID,0]
				
				#agentIDs_with_lowest_values = np.argmin(min_PT_per_Agent) # return agnetID(s)
				#print("min_PT_per_Agent=%s" % min_PT_per_Agent)
				# set move orders for each agent
				if old_ax>9:
					crossing_road = -1 #if on lower pavement, move up
					#if diag:print("Agent %d has decided to cross UP" % agentID)
				if old_ax<2:
					crossing_road = 1 #if on upper pavement, move down
					#if diag:print("Agent %d has decided to cross DOWN" % agentID)

				# set the agent state so no other agent crosses
				XR_WD_status[agentID,0] = crossing_road
				# execute move orders for each agent
				new_x, new_y = moveXR(old_ax, old_ay, crossing_road, walk_direction)
				new_x, new_y = checkEdge(gridW, gridH, old_ax, old_ay, new_x, new_y)
				# Add delta to previous state
				agentState[simTime,agentID,0] = new_x
				agentState[simTime,agentID,1] = new_y
				log_string = log_string + ", %4i, %4i" % (new_x,new_y)
			elif curr_XR!=0:
				# If already crossing continue
				crossing_road = curr_XR
				new_x, new_y = moveXR(old_ax, old_ay, crossing_road, walk_direction)
				new_x, new_y = checkEdge(gridW, gridH, old_ax, old_ay, new_x, new_y)
				# Add delta to previous state
				agentState[simTime,agentID,0] = new_x
				agentState[simTime,agentID,1] = new_y
				log_string = log_string + ", %4i, %4i" % (new_x,new_y)

			else:
				# if no agent in range then continue on current direction
				# set move order
				crossing_road = XR_WD_status[agentID,0] 
				walk_direction = XR_WD_status[agentID,1] 

				# execute move orders
				new_x, new_y = moveXR(old_ax, old_ay, crossing_road, walk_direction)
				new_x, new_y = checkEdge(gridW, gridH, old_ax, old_ay, new_x, new_y)
				# Add delta to previous state
				agentState[simTime,agentID,0] = new_x
				agentState[simTime,agentID,1] = new_y
				log_string = log_string + ", %4i, %4i" % (new_x,new_y)
			
			if diag: print("old_xy=%2i,%2i xy=%2i,%2i n=%3i t=%2i WD=%2i XR=%2i" % (old_ax, old_ay, new_x, new_y, nExp, simTime,walk_direction,crossing_road))
			#print("\n")


	# if(diag):raw_input("Press Enter to continue...")

	# write position log
	index = "%4i, %4i, %4i" % (nExp, simTime, AV_y)
	pLog.write(index + log_string + "\n")	

def detectAction(crossing_road, walk_direction,simTime,agentID, diag=False):	

	old2_ax = agentState[simTime-2,agentID,0]
	old2_ay = agentState[simTime-2,agentID,1]
	old_ax = agentState[simTime-1,agentID,0]
	old_ay = agentState[simTime-1,agentID,1]
	
	#if diag:print("Agent old XY=%d %d new XY=%d,%d" % (old_ax, old_ay, ax, ay))
	if old2_ay>old_ay: #walking direction
		walk_direction = -1 #walking 'left'
		if diag:print("Left walking detected")
		if old_ay==0:walk_direction=1 #bounce off edges
	if old2_ay<old_ay:
		walk_direction = 1  #walking 'right'
		if diag:print("Right walking detected")
		if old_ay==gridW-1:walk_direction=-1 #bounce off edges
	if old2_ay==old_ay:
		#find if agent is crossing road
		if old2_ax>old_ax:
			crossing_road=-1 #moving 'up'
			if diag:print("Agent is mid-crossing going up")
		elif old2_ax<old_ax:
			crossing_road=1  #moving 'down'				
			if diag:print("Agent is mid-crossing going down")
		elif old_ax==0:
			crossing_road=1  #moving 'down'				
		elif old_ax==gridH-1:
			crossing_road=-1  #moving 'down'				
		else:
			print("##detectAction## WARNING: Unrecognised agent behaviour")
	#check if at edge/corner
	if old_ay==0:
		walk_direction = 1
	if old_ay==gridW-1:
		walk_direction = -1
	return crossing_road, walk_direction

def checkEdge(gridW, gridH, old_ax, old_ay, new_x, new_y, diag=False):	
	dx, dy = 0,0
	if (new_y == 0) or (new_y >= gridW):
		dy = dy * -1
		if diag:print("Agent at y-limit reversing")
		new_y = int(old_ay + dy)
	if (new_x == 0) or (new_x >= gridH):
		dx = dx * -1
		new_x = int(old_ax + dx)
		if diag:print("Agent at x-limit reversing")
	return new_x, new_y

def moveXR(old_x, old_y, XR, WD, diag=False):
	# Move the agent based on the walk and crossing direction
	dx, dy = 0, 0
	crossing_road = XR
	walk_direction = WD
	if crossing_road == -1:
		dx = -1
		dy = 0
	if crossing_road ==  1:
		dx =  1
		dy = 0
	if crossing_road ==  0:
		if walk_direction == -1:
			dy =  -1
			dx = 0
		elif walk_direction == 1:
			dy = 1
			dx = 0
		else:
			print("##RB## WARNING: No valid move found")
	if diag:print("Agent dx=%2i dy=%2i" % (dx, dy))

	# Determine new position
	new_x = int(old_x + dx)
	new_y = int(old_y + dy)
	return new_x, new_y

def checkReward(nA, simTime, agentState, agentScores, nExp, roadPenaltyMaxtrix):
	#Check penalties and living costs
	rewardArr = np.zeros(shape=nA)
	for agentID in range (0,nA):
		reward = 0
		# Check agent location against penalty matrix
		Ag_x = int(agentState[simTime,agentID,0])
		Ag_y = int(agentState[simTime,agentID,1])
		# print("###REWARD### Ag_x, Ag_y ", Ag_x, Ag_y)
		# next_state = env.state2idx[(Ag_x,Ag_y)]
		# reward = env.idx2reward[next_state]
		# print("###REWARD### Agent ID reward ", agentID, reward)
		# Update agent score profile
		reward = roadPenaltyMaxtrix[Ag_y, Ag_x]
		curr_score = agentScores[nExp,agentID]
		agentScores[nExp,agentID] = curr_score + reward	
		rewardArr[agentID]=reward
		#print("###REWARD### ID curr_score, reward ",agentID, curr_score, reward)
		#if reward==vt: break #no double accounting, only first agent counts!
	#print("Agent scores are: ", reward)
	return rewardArr

def checkValidTest(nA, simTime, agentState):
	for agentID in range (0,nA):
		Ag_x = int(agentState[simTime,agentID,0])
		Ag_y = int(agentState[simTime,agentID,1])	
		# Check if valid test generated
		if (Ag_x,Ag_y) in env.end_positions:
			done = True
		else:
			done = False
		return done

def moveAV(gridW,gridH,AV_y):
	AVpositionMaxtrix = np.zeros(shape=(gridH,gridW))
	AVpositionMaxtrix[[2,3,4,5],AV_y]=1
	return AVpositionMaxtrix

def MASrender(simTime, nA, agentState, score, nExp):
		
	frame = env.frame.copy()
		
	# draw horizontal lines		
	for i in range(env.gridH+1):
		cv2.line(frame, (0, i*env.scale), (env.gridW * env.scale, i*env.scale), (255, 255, 255), 1)
	
	# draw vertical lines		
	for i in range(env.gridW+1):
		cv2.line(frame, (i*env.scale, 0), (i*env.scale, env.gridH * env.scale), (255, 255, 255), 1)
	
	#======================================
	# draw agent
	#======================================
	#print("simTime, nA",simTime, nA)
	for agentID in range(0,nA):
		y,x = agentState[simTime, agentID,:]
		y1 = int((y + 0.3)*env.scale)
		x1 = int((x + 0.3)*env.scale)
		y2 = int((y + 0.7)*env.scale)
		x2 = int((x + 0.7)*env.scale)
		cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), -1)
		#print('### RENDER2 ### xy',x,y,x1,x2,y1,y2)	
		#print("### RENDER ### simTime, agentID, x,y,x1,x2,y1,y2",simTime, agentID,x,y,x1,x2,y1,y2)
		#time.sleep(1)
		
	#======================================
	#print score
	text = 'ACC = ' + str(int(score)) + ' / ' + str(int(nExp+1))
	color = (0,0,255)
	font = cv2.FONT_HERSHEY_SIMPLEX
	y = gridH - 2
	x =  7
	(w, h), _ = cv2.getTextSize(text, font, 1, 2)
	x1, y1  = (int((x+0)*env.scale-w/2), int((y+1)*env.scale+h/2))
	x2, y2  = (int((x+12)*env.scale-w/2), int((y-0.9)*env.scale+h/2))
	cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
	cv2.putText(frame, text, (int((x+0.5)*env.scale-w/2), int((y+0.5)*env.scale+h/2)), font, 1, color, 2, cv2.LINE_AA)

	# colour the pavements
	pavement_rows = [0,1,10,11]
	for y in pavement_rows:
		for x in range(env.gridW+1):
			cv2.rectangle(env.frame, (x*env.scale, y*env.scale), ((x+1)*env.scale, (y+1)*env.scale), (100, 100, 100), -1)	
	
	cv2.imshow('frame', frame)
	cv2.moveWindow('frame', 0, 0)
	key = cv2.waitKey(10)
	if key == 27: sys.exit()

	cv2.imshow('frame', frame)
	cv2.moveWindow('frame', 0, 0)
	key = cv2.waitKey(1)
	if key == 27: sys.exit()

def initLocation(nA, nTests):
	startLocations = np.zeros(shape=(nTests,nA*2))
	# print("startLocations",startLocations)
	for experimentID in range(0,nTests):
		tempArray = []
		for agentID in range(0,nA):
			x, y = 0, 0
			rand = random.randint(0,3)
			if rand==0:
				x = 0
				y = random.randint(18,65)
			if rand==1:
				x = 1
				y = random.randint(12,65)
			if rand==2:
				x = 10
				y = random.randint(36,65)
			if rand==3:
				x = 11
				y = random.randint(54,65)
			tempArray.extend([x,y])			
			# print("tempArray",tempArray)
		startLocations[experimentID,:] = np.asarray(tempArray)
		#print("tempArray",tempArray)
	return startLocations

def getActionSpace(simTime,nA,agentState,XR_WD_status, AV_y, diag):

	# store possible actions for each agent: UP(-x), DOWN(+x), L(-y), R(+y), NONE
	actionSpace = np.zeros(shape=(nA,5))
	futureStates = np.zeros(shape=(nA,5,2))
	if(diag):print("###getActionSpace agentState", agentState[simTime-1,:,:])

	for agentID in range(0,nA):
		count = 0
		xp = int(agentState[simTime-1,agentID,0])
		yp = int(agentState[simTime-1,agentID,1])
		XR = XR_WD_status[agentID,0]
		WD = XR_WD_status[agentID,1]

		if simTime>=1: #if this is the first move then find valid moves
			# print("getActionSpace first move t=%d" % simTime)
			# print("###getActionSpace middle futureStates", futureStates.reshape(-1, 2))
			xp = int(agentState[0,agentID,0])
			yp = int(agentState[0,agentID,1])
			actionSpace[agentID, :] = 1 #set all to valid and check if valid
			for sID in range(0, 5):
				if sID==0:
					new_x, new_y = xp-1, yp
				if sID==1:
					new_x, new_y = xp+1, yp
				if sID==2:
					new_x, new_y = xp, yp-1
				if sID==3:
					new_x, new_y = xp, yp+1
				if sID==4:
					new_x, new_y = xp, yp

				if (new_y < 0) or (new_y > gridW-1):
					actionSpace[agentID, sID] = 0
					new_x, new_y = -1, -1
				if (new_x < 0) or (new_x > gridH-1):
					actionSpace[agentID, sID] = 0
					new_x, new_y = -1, -1
				futureStates[agentID,sID,:] = new_x, new_y
				# if(diag):print('###getActionSpace### actionSpace',actionSpace)
				# if(diag):print('###getActionSpace### futureStates',futureStates)
	
	return actionSpace, futureStates

def updatefeatures(simTime, agentState, XR_WD_status, AV_y, nA, nF, gridW, gridH, future=False, diag=False):	
	import numpy as np

	if future:
		nA = nA * 5
	else:
		nA = np.ma.size(agentState,1)
	features = np.zeros(shape=(nA, nF))
	#print("updatefeatures for nA=%d future=%d" % (nA,future))	

	for agentID in range(0,nA):
		#shorthand - pedestrian position
		if not(future):
			xp = int(agentState[simTime-1,agentID,0])
			yp = int(agentState[simTime-1,agentID,1])
		else:
			if(diag):print("future extrapolation...")
			xp = int(agentState.reshape(-1, 2)[agentID,0])
			yp = int(agentState.reshape(-1, 2)[agentID,1])
			if xp==-1 or yp==-1:
				if(diag):print("action invalid, skipping...")
				features[agentID,:] =  [0]*nF
				continue # next ID

		#av position
		xa = np.array([2,3,4,5])
		ya = AV_y

		# f1 = is agent on road
		if (xp > 1) | (xp < 10):
			on_road = 1
		if (xp <= 1) | (xp >= 10):
			on_road = 0
		
		# delta XY to AV
		diff_x = xa - xp
		dy = ya - yp
		dy = dy/float(gridW) #normalise to grid world to prevent large numbers
		abs_diff_x = abs(diff_x)
		min_abs_diff_x = np.min(abs(diff_x))
		dx = diff_x[np.argwhere(min_abs_diff_x==abs_diff_x)]
		dx = dx[0]
		dx = dx[0]
		dx = dx / float(gridH)  #normalise to grid world

		# find euclidean distance
		euclid = np.sqrt(np.square(dy) + np.square(dx))
		
		# see what actions are active
		if not(future):
			XR = XR_WD_status[agentID,0]  # x-dir
			WD = XR_WD_status[agentID,1]  # y-dir

		# get inverse dx and dy
		if dx == 0:
			inv_dx=1
		else:
			inv_dx=(1/dx)/10
		if dy == 0:
			inv_dy=1
		else:
			inv_dy=(1/dy)/10

		# inverse distance to AV
		if euclid == 0:
			inv_euclid = 1
			inv_euclid2 = 1
			inv_euclid3 = 1
		else:
			inv_euclid  = (1/euclid)/10
			inv_euclid2 = (1/np.square(euclid))/1000
			inv_euclid3 = (1/np.power(euclid,3))/1000000

		# AV towards me indicator
		if dy < 0:
			toward = 1
		else:
			toward = 0

		# AV on opposite road
		if dx > 3:
			AV_opp = 1
		else:
			AV_opp = 0

		# AV within radius 15
		if euclid < (15 / float(gridH)):
			trigger = 1
		else:
			trigger = 0


		# features[agentID,:] =  dx, dy, euclid
		# features[agentID,:] =  dx, dy, euclid, inv_euclid      
		# features[agentID,:] =  dx, dy, euclid, inv_euclid, inv_euclid2 
		# features[agentID,:] =  dx, dy, euclid, inv_euclid, inv_euclid2, AV_opp 
		# features[agentID,:] =  dx, dy, euclid, inv_euclid, inv_euclid2, inv_euclid3
		# features[agentID,:] =  dx, dy, euclid, inv_euclid, inv_euclid2, inv_euclid3, toward
		# features[agentID,:] =  on_road, dx, dy, euclid, inv_euclid, inv_euclid2, inv_euclid3
		# features[agentID,:] =  dx, dy, euclid, inv_dx, inv_dy, inv_euclid
		# features[agentID,:] =  dx, dy, euclid, inv_euclid, toward
		# features[agentID,:] =  dx, dy, euclid, inv_euclid, toward, on_road
		
		features[agentID,:] =  dx, dy, euclid,inv_euclid,toward
		# features[agentID,:] =  dx, dy, euclid,inv_euclid,toward, trigger
		# features[agentID,:] =  dx, dy, euclid,inv_euclid,toward,AV_opp
		# features[agentID,:] =  dx, dy, euclid,inv_euclid,on_road
		# features[agentID,:] =  dx, dy, euclid,inv_euclid,toward,on_road
		# features[agentID,:] =  dx, dy, euclid, inv_euclid, inv_euclid2, inv_euclid3,toward,on_road
		
		if(diag):print("###updatefeatures### features ",features[agentID,:])
		if diag:
			print("~~~~ percepts ~~~~")
			print("xp yp ",xp, yp)
			print("xa ya ",xa, ya)
			print("on road=\t%d"%on_road)
			print("delta y=\t%f "% dy)
			print("delta x=\t%f "% dx)
			print("XR=\t\t%d"% XR)
			print("WD=\t\t%d" % WD)
			print("euclid=\t\t%.2f" % euclid)
			print("inv_euclid=\t%.2f" % inv_euclid)
			print("inv_euclid2=\t%.2f " % inv_euclid2)
			print("inv_euclid3=\t%.2f " % inv_euclid3)
			print("~~~~~~~~~~~~~~")
	
	# tags = ('dx', 'dy', 'euc') 		
	# tags = ('dx', 'dy', 'euc','i_euc/10') 
	# tags = ('dx', 'dy', 'euc', 'i_euc/10','i_euc2/10^3') 		
	# tags = ('dx', 'dy', 'euc', 'i_euc/10','i_euc2/10^3', 'AV_opp') 		
	# tags = ('dx', 'dy', 'euc', 'i_euc/10', 'i_euc2/10^3', 'i_euc3/10^6') 		
	# tags = ('dx', 'dy', 'euc', 'i_euc/10', 'i_euc2/10^3', 'i_euc3/10^6','toward') 		
			
	# tags = ('dx', 'dy', 'euc', 'inv_dx/10', 'inv_dy/10','i_euc/10') 
	tags = ('dx', 'dy', 'euc', 'i_euc/10','toward') 
	# tags = ('dx', 'dy', 'euc', 'i_euc/10','toward','trigger') 
	# tags = ('dx', 'dy', 'euc', 'i_euc/10','toward','AV_opp') 
	# tags = ('dx', 'dy', 'euc', 'i_euc/10','toward','on_road') 
	# tags = ('dx', 'dy', 'euc','i_euc/10','toward') 
	# tags = ('dx', 'dy', 'euc','i_euc/10','on_road') 
	# tags = ('dx', 'dy', 'euc','i_euc/10','toward','on_road') 
	# tags = ('dx', 'dy', 'euc', 'i_euc/10', 'i_euc2/10^3', 'i_euc3/10^6','toward','on_road') 		
		


	return features, tags

def featuresOfFutureActions(simTime,nA, XR_WD_status, AV_y, nF, agentState, futureStates, future=True):	

	# if(diag): print('###featuresOfFutureActions### futureStates',futureStates)
	f,t = updatefeatures(simTime, futureStates, XR_WD_status, AV_y, nA, nF, gridW, gridH, future=True)
	# print("###featuresOfFutureActions", f)
	# raw_input("Press Enter to continue...")
	return f

def qValForFutureFeats(nA,nF, futureFeatures,futureActions,AV_y,feat_weights):
	# qval = np.zeros(shape=(nA*5))
	qval = np.zeros(shape=(nA,5))
	q_argmax = np.zeros(shape=nA) 		#index of best action
	# mask = np.logical_not(np.array(futureActions,dtype=bool)) #boolean mask to indicate which values NOT to use
	# print('###qValForFutureFeats### mask',mask)
	# raw_input()
	counter=0
	# calculate Q-values based on the feature representation
	# Q(s,a) = w1.f1(s,a) + w2.f2(s,a) + ... wi.fi(s,a)
	for agentID in range(0,nA):
		for fidx in range(0, 5):			
			w = feat_weights[agentID,:]
			f = futureFeatures[fidx,:]
			valid_actions = futureActions[agentID,:] #logic of valid actions
			mask = np.logical_not(np.array(futureActions[agentID,:],dtype=bool)) #boolean mask to indicate which values NOT to use
			valid_actions = ma.masked_array(futureActions[agentID,:],mask)
			product = np.multiply(w,f)
			temp = np.sum(product)
			# print("ID=%d fidx=%d" %(agentID, fidx))
			# print("weights", w)
			# print("feats", f)
			# print("product", product)
			# print("temp", temp)
			# print("temp", temp)
			qval[agentID,fidx] = temp
			counter = counter +1

		# argMAX for index of best q-value
		# print("chosen action index is",np.argmax(qval[agentID,:]))
		# use the valid actions to prune all possible actions
		all_qvals = qval[agentID,:]
		pruned_actions = np.multiply(all_qvals,valid_actions)
		q_argmax[agentID] = np.argmax(pruned_actions)
		# print("all_qvals",all_qvals)
		# print("valid_actions",valid_actions)
		# print("pruned_actions",pruned_actions)
		
		# index of highest q-value is best action
		# print("qval[agentID,:]", qval[agentID,:])
		# print("q_argmax", q_argmax[agentID])
	# print("q_argmax", q_argmax)

	# raw_input("Press Enter to continue...")
	return qval, q_argmax

def qUpdate():
	# update the feature weights given the reward
	difference = (reward + self.alpha * q_val_dash) - qval

	for i in range(self.feat_weights.shape[0]):
		wi = self.feat_weights[i]
		self.feat_weights[i] = wi + self.alpha * difference * features[i]

def updateWeights(features,q_vals_future,q_argmax,feat_weights,reward,alpha,current_qval):
	# ================================================
	# update feature weights
	#
	# difference = (reward + discount * q_val_dash) - qval
	# new_weight(i) = old_weight(i) + alpha * difference * feature(i)
	# ================================================
	# print("q_argmax.shape",q_argmax.shape)
	# print("q_argmax[agentID]",q_argmax[agentID])
	numFeat = np.shape(features)[1]
	# print("numFeat",numFeat)
	for agentID in range(0, nA):
		q_val_dash = q_vals_future[agentID, (int)(q_argmax[agentID])]			
		for wi in range(0, numFeat):
			old_weight = (float)(feat_weights[agentID,wi])
			difference = (reward[agentID] + discount * q_val_dash) - current_qval[agentID] 
			new_weight = (float)(old_weight + alpha * difference * features[agentID,wi])
			feat_weights[agentID,wi] = new_weight
			# print("********************************")
			# print("***      Weight Update       ***")
			# print("********************************")						
			# print("agent = ", agentID)
			# print("weight index = ", wi)
			# print("reward[agentID]", reward[agentID])
			# print("discount", discount)
			# print("current_qval", current_qval[agentID])
			# print("q_val_dash", q_val_dash)
			# print("difference", difference)
			# print("***************")
			# print("old_weight", old_weight)
			# print("alpha", alpha)
			# print("difference", difference)
			# print("feature", features[agentID,wi])
			# print("new weight", new_weight)

def resetEnv(simTime,nExp,nTests,test_gen_time,done,running_score,exclusions,AV_y,display_grid,nA,agentState,gridW,startLocations,rsLog,pLog):
	# reset environment 
	test_gen_time[nExp] = simTime
	# nExp = nExp + 1
	# if (nExp>nTests-1):
	random.seed(nExp)
	np.random.seed(nExp)
	env.reset_state()
	done = False 
	running_score = 0
	exclusions = np.empty(shape=(nA,2)) #ID, xy
	AV_y=0
	if display_grid:
		MASrender(simTime, nA, agentState, validTests, nExp)
	maxT = (int)(round(gridW / vAV)+1)
	agentState = np.empty(shape=(maxT,nA,2)) #state is [time,ID,position(x,y)]
	# reste the timer and actors on the board
	simTime = 0
	randomStart(startLocations,simTime,nA,agentState,rsLog,pLog,nExp) 
	return simTime,nExp,test_gen_time,done,running_score,exclusions,AV_y,display_grid,nA,agentState,gridW,startLocations,rsLog,pLog

def plot_Features(fig,features,counter,y):
	x = np.arange(len(features[0,:]))
	if counter==1: 
		y=features
		counter=counter+1
	else: 
		temp = np.transpose(features[0,:])
		y = np.vstack((y,temp))
		counter=counter+1
	if counter>2:				
		plt.plot(y, label=tags)
		plt.plot(y, label=['on_road', 'dx', 'dy', 'euc', 'i_euc', 'i_euc2', 'i_euc3', 'score'])
		plt.title('Agent Features') 
		fig.canvas.draw()
		img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,	sep='')
		img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
		img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
		cv2.imshow("Features",	img)
		# key = cv2.waitKey(10)
		# if key == 27: sys.exit()
	return counter, y

def plot_Weights(Wfig,feat_weights,Wcounter):
	x = np.arange(len(feat_weights[0,:]))
	if Wcounter==1: 
		y=feat_weights
		Wcounter=Wcounter+1
	else: 
		temp = np.transpose(feat_weights[0,:])
		y = np.vstack((y,temp))
		Wcounter=Wcounter+1
	if Wcounter>2:				
		plt.plot(y, label=tags)
		plt.title('Weights') 
		Wfig.canvas.draw()
		Wimg = np.fromstring(Wfig.canvas.tostring_rgb(), dtype=np.uint8,	sep='')
		Wimg = Wimg.reshape(Wfig.canvas.get_width_height()[::-1] + (3,))
		Wimg = cv2.cvtColor(Wimg,cv2.COLOR_RGB2BGR)
		cv2.imshow("Weights",	Wimg)

def plot_Qvalues(Qfig,current_qval,Qcounter):				
	x = np.shape(current_qval)
	if Qcounter==1: 
		y=current_qval
		Qcounter=Qcounter+1
	else: 
		temp = current_qval#np.transpose(current_qval[0,:])
		y = np.vstack((y,temp))
		Qcounter=Qcounter+1
	if Qcounter>2:				
		plt.plot(y, label=tags)
		plt.title('Q-values') 
		Qfig.canvas.draw()
		Qimg = np.fromstring(Qfig.canvas.tostring_rgb(), dtype=np.uint8,	sep='')
		Qimg = Qimg.reshape(Qfig.canvas.get_width_height()[::-1] + (3,))
		Qimg = cv2.cvtColor(Qimg,cv2.COLOR_RGB2BGR)
		cv2.imshow("Q-Values",	Qimg)

# ======================================================================
# --- User Experiment Params -----------------------------------------

nTests = 100					# Number of experiements to run
gridH, gridW = 12, 66			# Each grid unit is 1.5m square
pavement_rows = [0,1,10,11] 	# grid row of each pavement
vAV = 6 						# 6u/s ~9.1m/s ~20mph
vPed = 1 						# 1u/s ~1.4m/s ~3mph
nA = 1							# Number of agents
delay = 0.01 					# delay between each frame, slows sim down
vt = 1000						# points for a valid test
AV_y = 0						# AV start position along road
default_reward= -1 				# Living cost
road_pen = -5					# Penalty for being in road
nF = 5							# number of features per agent


display_grid  = True 			# Show the grid
display_chart = True 			# Show plotted data
diag 		  = False			# What level of CL diagnostics to show
loopAgentList = False 			# use nAlist to loop through nA

plotFeatures  = False 			# plot features
plotWeights   = False			# plot feature weights
plotQvalues   = False 			# plot q-values chart
plotAccuracy  = True 			# plot the accuracy averaged over nExp
SaveCharts    = True 			# save the plots produced
display_chart_modulo = 1

# Choose the type of agent behaviour
# 	RandAction	= take random actions
# 	RandBehaviour = walk pavements, randomly cross road with 1/11 chance
# 	Proximity = cross when agent within specified radius
#	Election = elects a single agent within range to cross road

agentChoices = ['RandAction', 'RandBehaviour','Proximity','Election','Q_Agent']
agentBehaviour = agentChoices[4] 	# TODO replace with CL arg
TR = 15 							# Proximity/Election Trigger Radius
ECA = True							# If election is held, choose closest to AV, else furthest
CP = True 							# Elect agents on pavement closest to the AV

# Q-Learning Feature Set
FeatSet = 1						# Choose which features agent should use

alpha = 0.002 #learning rate
epsilon = 0.16 #search policy
discount = 0.99 #future discount
# ======================================================================
# --- Non-User Experiment Params ---------------------------------------

if loopAgentList:
	nAList = [1,2,3,4,5,6,7,8,9,10,15,20] # Loop through list of agents
	vtList = [50,100,200,500,700,800,900,1000,1100,1200,2000,5000,10000,20000,50000,100000,200000,500000]
else:
	nAList = [nA]

if plotFeatures:
	fig = plt.figure()
	counter=1

if plotWeights:
	Wfig = plt.figure()
	Wcounter=1

if plotQvalues:
	Qfig = plt.figure()
	Qcounter=1

if plotAccuracy:
	Afig = plt.figure()
	Acounter=1

# ======================================================================
# --- Loop through Specified Number of Agents --------------------------
for nA in nAList:
# for vt in vtList:

	if not(display_grid):
		delay = 0
		diag = False
	validTests = 0.
	roadPenaltyMaxtrix = np.zeros(shape=(gridW,gridH))
	roadPenaltyMaxtrix[:,:] = road_pen
	roadPenaltyMaxtrix[:,pavement_rows] = default_reward

	road_positions = [(i,j) for j in range(0,gridW) for i in [2,3,4,5,6,7,8,9]] 
	road_rewards   = [road_pen for i in range(4*gridH)] 

	AV_y = 0
	AV_x = [2,3,4,5]
	AV_state = (AV_x,AV_y)
	blocked_positions = [(2,AV_y),(3,AV_y),(4,AV_y),(5,AV_y)]

	# generate a grid showing position of the AV_x
	AVpositionMaxtrix = moveAV(gridW,gridH,AV_y)
	#print(AVpositionMaxtrix)

	start_pos = None
	end_positions = [(2,0),(3,0),(4,0),(5,0)] 	# initial AV position
	end_rewards = [0,0,0,0] 					# Rewards moved out of panalty matrix

	# record agentStates and excluded start positions
	exclusions = np.empty(shape=(nA,2)) #ID, xy
	maxT = (int)(round(gridW / vAV)+1)
	agentState = np.empty(shape=(maxT,nA,2)) #state is [time,ID,position(x,y)]
	XR_WD_status = np.zeros(shape=(nA,2))

	# store a score for each agent and each experiment
	agentScores = np.zeros(shape=(nTests,nA)) 
	valid_test_scores = np.array([])

	#store the time taken to generate a valid test if applicable
	test_gen_time = np.zeros(shape=(nTests,1)) 

	# a state-action space array will hold the q-values for each agent
	stateAction = np.zeros(shape=(nA, nF, 5)) #fixed to 5 basic moves

	# initialise experiment number and random seed
	nExp = 0 #experiment counter
	random.seed(nExp) #set the random seed based on the experiment number
	np.random.seed(nExp)


	# ======================================================================
	# --- MDP Agent Experiment Params --------------------------------------

	# alpha = 0.004 #learning rate
	# epsilon = 0.02 #search policy
	# discount = 0.95 #future discount

	#'on_road', 'dx', 'dy', 'euc', 'i_euc', 'i_euc2', 'i_euc3'
	feat_weights = np.array([[-1.,-1.,-1.,-1.,1.,1.,1.],]*nA)
	feat_weights = np.array([np.random.rand(nF),]*nA)
	if(diag): print("feat_weights", feat_weights)

	# ======================================================================
	# use existing weights
	# for dx, dy, euc, i_euc/10 ACC=63.8 Tests=200
	# feat_weights = np.array([[-4.11146701,  -5.48638104,  12.4593254,  156.18604846],]*nA)

	# for dx, dy, euc, i_euc/10, i_euc2/1000 ACC=70.1 Tests=1000
	# feat_weights = np.array([[0.5488135 , 0.71518937, 0.60276338, 0.54488318, 0.4236548 ],]*nA)
	
	# features[agentID,:] =  dx, dy, euclid,inv_euclid,toward
	# feat_weights = np.array([[ 12.76206632, -40.65811894 , 28.23486239,  81.07528645, -59.54039597],]*nA)
	# feat_weights = np.array([[ 	10.88713081 ,-49.97721555  ,33.15796263,  66.68952519, -63.46238057],]*nA)





	# ======================================================================
	# --- Logs -------------------------------------------------------------
	ts = datetime.datetime.now().strftime("%d-%b-%Y-%H-%M-%S")

	rsLog = open("logs/initial_random_log_%s.txt" % ts, "w")	#random initial location log
	rsLog.write("nExp"+"".join(',  A%dx,  A%dy' % (i,i) for i in range(0,nA)) + " \n")

	rLog = open("logs/random_log_%s.txt" % ts, "w")				#random movement log
	rLog.write("simTime, rand \n")

	sLog = open("logs/score_log_%s.txt" % ts, "w")				#score for each experiment per agent
	sLog.write("nExp, valid"+ "".join(',  A%2i' % i for i in range(0,nA)) +"\n")

	pLog = open("logs/position_log_%s.txt" % ts, "w")
	pLog.write("nExp, Time,  AVy"+"".join(',  A%dx,  A%dy' % (i,i) for i in range(0,nA)) + " \n")

	vLog = open("logs/valid_log_%s.txt" % ts, "w")
	vLog.write("nExp, valid"+ "".join(',  A%2i' % i for i in range(0,nA)) +"\n")

	fLog = open("logs/feature_log_%s.txt" % ts, "w")
	fLog.write("nExp, Time,   ID"+"".join(',       F%2i' % i for i in range(0,nF)) + " \n")
	
	aExTime = 0 	# Store the time to execute agent actions
	# ======================================================================
	# --- Initialisation ---------------------------------------------------

	running_score = 0
	simTime = 0

	if diag: print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
	if diag: print("Running %d %s agents for %d tests..." % (nA, agentBehaviour,nTests))
	# print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
	# print("Running %d %s agents for %d tests..." % (nA, agentBehaviour,nTests))

	# Generate a list of start points for the agents so they are the same across behaviours
	startLocations = initLocation(nA, nTests)
	if diag: print("Generating startLocations",startLocations)

	# Use the position list to set agent states 
	randomStart(startLocations,simTime,nA,agentState,rsLog,pLog,nExp) 
	if(diag):print("Agent state after randomStart", agentState)

	# Initialise the environment class and make an initial render of the board
	env = Environment(gridH, gridW, end_positions, end_rewards, blocked_positions, start_pos, default_reward, road_positions, road_rewards)

	# Display the grid lines, q-values, agent positions
	if display_grid:
		MASrender(simTime, nA, agentState, validTests, nExp)
	time.sleep(delay)

	# Flag to indicate if a valid test has been generated
	done = False


	# while(nExp <= nTests):
	while(not(done)) and (nExp <= nTests-1):

		if(diag):print("===========================")
		if(diag):print("Experiment=%d Time=%d" % (nExp,simTime))
		if(diag):print("===========================")

		#check if series complete
		if(nExp>nTests):
			done=True

		#increment time
		if simTime==0 and diag: print("Experiment Number", nExp)
		#print("simTime=", simTime)
		simTime = simTime + 1

		start = time.time()

		# update the state features for the agents
		features, tags = updatefeatures(simTime, agentState, XR_WD_status, AV_y, nA, nF, gridW, gridH, False, diag)
		# print("features")
		# print(features)

		# get current qval
		current_qval = np.zeros(shape=nA)
		for agentID in range(0,nA):
			w = feat_weights[agentID,:]
			f =  features
			product = np.multiply(w,f)
			current_qval[agentID] = np.sum(product)
			# print("w",w)
			# print("f",f)
			# print("product",product)
			# print("current_qval",current_qval)
		
		old_ax = agentState[simTime-1,agentID,0]
		old_ay = agentState[simTime-1,agentID,1]
		if(diag):print("old_ax",old_ax)
		if(diag):print("old_ay",old_ay)


		# get possible action space 1-step ahead
		futureActions, futureStates = getActionSpace(simTime,nA,agentState,XR_WD_status, AV_y, diag=diag)
		if(diag):print("###MAIN### futureStates",futureStates)

		# update features on all possible actionSpace
		futureFeatures = featuresOfFutureActions(simTime,nA, XR_WD_status, AV_y, nF, agentState, futureStates, future=True)	
		if(diag):print("futureFeatures", futureFeatures)

		# calculate q-values for all state action pairs
		# find the best q-values of all available, q_argmax is the best action index
		q_vals_future, q_argmax = qValForFutureFeats(nA,nF, futureFeatures, futureActions, AV_y, feat_weights)
		if(diag):print("q_vals_future", q_vals_future)
		if(diag):print("q_argmax",  q_argmax)
		

		# move agent based on best features
		for agentID in range(0, nA):
			
			# implement epsilon search policy
			if epsilon > np.random.uniform(0.0, 1.0):
				if(diag):print("Agent %d taking random action" % agentID)
				agent_valid_choices = np.nonzero(futureActions[agentID,:])
				slen = np.shape(agent_valid_choices)[1]
				agent_valid_choices = np.reshape(agent_valid_choices,(slen,))
				epsilon_choice = np.random.choice(agent_valid_choices)
				action = (int)(epsilon_choice)					
				if(diag): print("agent_valid_choices shape",np.shape(agent_valid_choices))
				if(diag): print("agent_valid_choices",agent_valid_choices)
				if(diag): print("agent_valid_choices shape",np.shape(agent_valid_choices))
				if(diag): print("epsilon_choice",epsilon_choice)
				# raw_input()
			else:
				# get action from best q-learn value
				action = q_argmax[agentID]

			old_ax = agentState[simTime-1,agentID,0]
			old_ay = agentState[simTime-1,agentID,1]
			if(diag):print("action",action)
			# print("old_ax",old_ax)
			# print("old_ay",old_ay)
			
			# For valid (edge of board) but unconstrained movement (any direction at any time) use:
			if action == 0: #UP
				dx = -1
				dy = 0
			if action == 1: #DOWN
				dx = 1
				dy = 0
			if action == 2: #LEFT
				dx = 0
				dy = -1
			if action == 3: #RIGHT
				dx = 0
				dy = 1
			if action == 4: #DO NOTHING
				dx = 0
				dy = 0
			new_x = int(old_ax + dx)
			new_y = int(old_ay + dy)
			new_x, new_y = checkEdge(gridW, gridH, old_ax, old_ay, new_x, new_y, diag)
			agentState[simTime,agentID,:] = new_x, new_y	
			if(diag):print("new_x",new_x)
			if(diag):print("new_y",new_y)	
			if(diag) and (new_x<0):	raw_input("ERROR: Invalid location")
			if(diag) and (new_y<0):	raw_input("ERROR: Invalid location")
			# if(diag):raw_input()
	
		# move agents	
		if agentBehaviour == 'RandAction':
			randomMove(simTime, nA, agentState, pLog, rLog, nExp, AV_y)
		if agentBehaviour == 'RandBehaviour':
			randomBehaviour(simTime, nA, agentState, XR_WD_status,pLog, rLog, nExp, AV_y, diag=diag)
		if agentBehaviour == 'Proximity':
			Proximity(simTime, nA, agentState, pLog, rLog, nExp, AV_y, trigger_radius=TR, diag=diag)
		if agentBehaviour == 'Election':
			Election(simTime, nA, agentState, XR_WD_status, pLog, rLog, nExp, AV_y, CP=CP, ECA=ECA, trigger_radius=TR, diag=diag)
		# if agentBehaviour == 'Q_Agent':
		# 	Election(simTime, nA, agentState, XR_WD_status, pLog, rLog, nExp, AV_y, CP=CP, ECA=ECA, trigger_radius=TR, diag=diag)
		

		# reward = checkReward(nA, simTime, agentState, agentScores, nExp, roadPenaltyMaxtrix) #Check reward and end positions (overrules env.step)
		end = time.time()
		aExTime = aExTime + (end - start)
	
		# plot data if requested
		if plotFeatures:
			x = np.arange(len(features[0,:]))
			if counter==1: 
				y=features
				counter=counter+1
			else: 
				temp = np.transpose(features[0,:])
				y = np.vstack((y,temp))
				counter=counter+1
			if counter>2:				
				plt.plot(y, label=tags)
				plt.plot(y, label=['on_road', 'dx', 'dy', 'euc', 'i_euc', 'i_euc2', 'i_euc3', 'score'])
				plt.title('Agent Features') 
				fig.canvas.draw()
				img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,	sep='')
				img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
				img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
				if(display_chart):cv2.imshow("Features",	img)
				key = cv2.waitKey(1)
				if key == 27: sys.exit()
		if plotWeights:
			x = np.arange(len(feat_weights[0,:]))
			if Wcounter==1: 
				y=feat_weights
				Wcounter=Wcounter+1
			else: 
				temp = np.transpose(feat_weights[0,:])
				y = np.vstack((y,temp))
				Wcounter=Wcounter+1
			if (Wcounter>2) and ((nExp>nTests-2) or (nExp%display_chart_modulo==0)):							
				plt.plot(y, label=tags)
				plt.title('Weights') 
				Wfig.canvas.draw()
				Wimg = np.fromstring(Wfig.canvas.tostring_rgb(), dtype=np.uint8,	sep='')
				Wimg = Wimg.reshape(Wfig.canvas.get_width_height()[::-1] + (3,))
				Wimg = cv2.cvtColor(Wimg,cv2.COLOR_RGB2BGR)
				if(display_chart):cv2.imshow("Weights",	Wimg)
				key = cv2.waitKey(1)
				if key == 27: sys.exit()
		if plotQvalues:
			x = np.shape(current_qval)
			if Qcounter==1: 
				y=current_qval
				Qcounter=Qcounter+1
			else: 
				temp = current_qval
				y = np.vstack((y,temp))
				Qcounter=Qcounter+1
			if Qcounter>2:				
				plt.plot(y, label=tags)
				plt.title('Q-values') 
				Qfig.canvas.draw()
				Qimg = np.fromstring(Qfig.canvas.tostring_rgb(), dtype=np.uint8,	sep='')
				Qimg = Qimg.reshape(Qfig.canvas.get_width_height()[::-1] + (3,))
				Qimg = cv2.cvtColor(Qimg,cv2.COLOR_RGB2BGR)
				if(display_chart):cv2.imshow("Q-Values",	Qimg)
				key = cv2.waitKey(1)
				if key == 27: sys.exit()
		if plotAccuracy:
			if validTests==0:
				acc = 0
			else:
				acc = 100*(validTests/(nExp+1))			
			if Acounter==1: 
				acc_ratio_arr = acc
				Acounter=Acounter+1
			else: 							
				acc_ratio_arr = np.vstack((acc_ratio_arr,acc))
				Acounter=Acounter+1
			if (Acounter>2) and ((nExp>nTests-2) or (nExp%display_chart_modulo==0)):
				plt.plot(acc_ratio_arr, label='$T_s$',color='black')
				plt.ylabel('Accuracy (%)') 
				plt.xlabel(r'simulation ticks ($t_s$)')
				plt.title(r'$ \epsilon={0:.2f}$ ACC={1:.1f} {{}}'.format(epsilon,acc).format(tags))				
				plt.grid(True)
				plt.ylim((0,100))
				Afig.canvas.draw()
				Aimg = np.fromstring(Afig.canvas.tostring_rgb(), dtype=np.uint8,	sep='')
				Aimg = Aimg.reshape(Afig.canvas.get_width_height()[::-1] + (3,))
				Aimg = cv2.cvtColor(Aimg,cv2.COLOR_RGB2BGR)
			
				if(display_chart):cv2.imshow("Accuracy",	Aimg)
				# cv2.moveWindow('frame', 50, 50)
				key = cv2.waitKey(1)
				if key == 27: sys.exit()

				if (nExp == nTests-1):	
					plt.savefig('Accuracy_%s.png' % ts)

		# render the scene
		if display_grid:
			MASrender(simTime, nA, agentState, validTests, nExp)
			time.sleep(delay)		

	# move AV
		for i in range(0,vAV):

			# Move AV and update end positions and reward locations
			AV_y+=1
			AVpositionMaxtrix = moveAV(gridW,gridH,AV_y)
			AVlist = (np.transpose(np.nonzero(AVpositionMaxtrix))) # this is in order (y,x) (horizontal,vertical)

			#use broadcasting to check if an agent coordinate pair exists in the AV spaces
			yx_agentList = np.transpose(np.array([agentState[simTime,:,0],agentState[simTime,:,1]])).astype(int)
			validTest = (yx_agentList[:,None] == AVlist).all(2).any(1).any()
			indexIDbool = ((yx_agentList[:,None] == AVlist).all(2)).any(1)
			indexID = [i for i, x in enumerate(indexIDbool) if x]

			# If a valid test is found update scores
			if(validTest):
				# add score to relevant ID
				for agentID in range(0, nA):
					if agentID == indexID[0]:
						bonus=vt
					else:
						bonus=0
					featID = list(features[agentID,:])
					log_string = "".join(', %9.3f' % i for i in featID) + "".join(",  %4i" % (bonus + reward[agentID]))
					index = "%4i, %4i, %4i" % (nExp, simTime, agentID)
					fLog.write(index + log_string + "\n")
				
				if(diag): print("Agent ",indexID, " has generated a valid test")
				curr_score = agentScores[nExp,indexID]
				agentScores[nExp,indexID] = curr_score + vt	
				validTests = validTests + 1
				done = True # flag to reset level

			
			# update the graphics frame
			env.end_positions = [(2,AV_y),(3,AV_y),(4,AV_y),(5,AV_y)]
			if display_grid:
				env.update_state() #renders road and end positions/rewards


			# If collision occurs end the experiment			
			if done == True:
				if diag:
					print("~~~~~~~~~~~~~~~~~~~~~")
					print("Valid test generated!")
					print("~~~~~~~~~~~~~~~~~~~~~")
				
				# if valid test generated then read agent scores and update weights with bonus score
				reward = agentScores[nExp,:]
				if diag:print("Reward (test found) =", reward)
				# update weights based on reward
				updateWeights(features,q_vals_future,q_argmax,feat_weights,reward,alpha,current_qval)
				# if diag:print("OUTSIDE feat_weights",feat_weights)
				running_score = running_score + reward[0]

				#log the scores for this run
				scoresRound = agentScores[nExp,:]
				avg_scoresRound = np.mean(scoresRound)
				log_string = "".join(', %4i' % scoresRound[ind] for ind in range(0,len(scoresRound)))
				rsindex = "%4i,%6i" % (nExp,1)
				sLog.write(rsindex + log_string + "\n")
				vLog.write(rsindex + log_string + "\n")
				valid_test_scores=np.append(valid_test_scores,avg_scoresRound)
				#resetEnv
				nExp = nExp + 1
				if (nExp>nTests-1):
					break
				else:
					simTime,nExp,test_gen_time,done,running_score,exclusions,AV_y,display_grid,nA,agentState,gridW,startLocations,rsLog,pLog = resetEnv(simTime,nExp,nTests,test_gen_time,done,running_score,exclusions,AV_y,display_grid,nA,agentState,gridW,startLocations,rsLog,pLog)
				continue

			# Reset the game if the AV reaches the end
			if AV_y>=gridW-1:
				if diag: print("AV exit, resetting...")				
				#log the scores for this run
				scoresRound = agentScores[nExp,:]
				log_string = "".join(', %3i' % scoresRound[ind] for ind in range(0,len(scoresRound)))
				rsindex = "%4i,%6i" % (nExp,0)
				sLog.write(rsindex + log_string + "\n")	
				#resetEnv
				nExp = nExp + 1
				if (nExp>nTests-1):
					break
				else:
					simTime,nExp,test_gen_time,done,running_score,exclusions,AV_y,display_grid,nA,agentState,gridW,startLocations,rsLog,pLog = resetEnv(simTime,nExp,nTests,test_gen_time,done,running_score,exclusions,AV_y,display_grid,nA,agentState,gridW,startLocations,rsLog,pLog)
				#print("after reset simTime is %d" % simTime)
				continue

			AV_state = (AV_x,AV_y)
			

		# no valid test generated then update scores
		if (nExp>nTests-1):
			break
		else:
			if done==False:
				reward = checkReward(nA, simTime, agentState, agentScores, nExp, roadPenaltyMaxtrix) #Check reward and end positions 
				if diag:print("Reward (no test) =", reward)
				updateWeights(features,q_vals_future,q_argmax,feat_weights,reward,alpha,current_qval)
				if diag:print("OUTSIDE weights",feat_weights)
				# store features and score
				for agentID in range(0, nA):
					featID = list(features[agentID,:])
					#print("featID",featID)
					log_string = "".join(', %9.3f' % i for i in featID) + "".join(",  %4i" % reward[agentID])
					index = "%4i, %4i, %4i" % (nExp, simTime, agentID)
					fLog.write(index + log_string + "\n")

	# close log files
	if diag: print("Test complete, writing log files...")
	rLog.close()
	sLog.close()
	rsLog.close()
	pLog.close()
	vLog.close()
	fLog.close()


	#calc stats
	import scipy.stats as st
	accuracy_ratio = 100*(validTests/nTests)
	report_string = "n=%d nA=%d Accuracy=%0.2f" % (nTests, nA, accuracy_ratio)
	
	agScore_AVG = np.average(agentScores)
	agScore_MAX = np.max(agentScores)
	agScore_MIN = np.min(agentScores)
	agScore_95ci = st.t.interval(0.95, len(agentScores)-1, loc=np.mean(agentScores), scale=st.sem(agentScores))
	ci95= (zip(*agScore_95ci))
	ci95Arr = np.asarray(ci95)
	ci95_low = np.mean(ci95Arr[:,0])
	ci95_hig = np.mean(ci95Arr[:,1])
	# print("ci95 low %.2f high %.2f" % (ci95_low, ci95_hig))

	v_avg = np.average(valid_test_scores)
	v_min = np.min(valid_test_scores)
	v_max = np.max(valid_test_scores)

	#print(np.shape(agentScores))
	#print(np.shape(valid_test_scores))
	slen = np.shape(valid_test_scores)[0]
	valid_test_scores = valid_test_scores.reshape(slen,1)
	#print(np.shape(valid_test_scores))


	v_95ci = st.t.interval(0.95, len(valid_test_scores)-1, loc=np.mean(valid_test_scores), scale=st.sem(valid_test_scores))
	v_ci95= (zip(*v_95ci))
	v_ci95Arr = np.asarray(v_ci95)
	v_ci95_low = np.mean(v_ci95Arr[:,0])
	v_ci95_hig = np.mean(v_ci95Arr[:,1])
	# print("ci95 low %.2f high %.2f" % (ci95_low, ci95_hig))
	
	#stats on time
	mean_tgt = np.mean(test_gen_time)
	tgt_ci = st.t.interval(0.95, len(test_gen_time)-1, loc=np.mean(test_gen_time), scale=st.sem(test_gen_time))
	tgt_ci95 = (zip(*tgt_ci))
	tgt_ci95Arr = np.asarray(tgt_ci95)
	tgt_ci95_low = np.mean(tgt_ci95Arr[:,0])
	tgt_ci95_hig = np.mean(tgt_ci95Arr[:,1])


	# generate report
	if diag: print("Generating summary report...")
	summary = open("logs/summary_report_%s.txt" % ts, "w")				
	summary.write("======================================= \n")
	summary.write("=========== Summary of Test =========== \n\n")
	summary.write("Number of tests: %d \n" 				% nTests)
	summary.write("Number of agents: %d \n" 			% nA)
	summary.write("Agent type: %s"						% agentBehaviour)
	summary.write("Valid tests generated: %d \n" 		% validTests)
	summary.write("Test generation accuracy: %.1f%% \n"	% accuracy_ratio)
	summary.write("Average test generation time: %.2fs \n"	% np.mean(test_gen_time))
	summary.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")
	summary.write("Living cost: %d \n" 					% default_reward)
	summary.write("Road penalty: %d \n" 				% road_pen)
	summary.write("Average agent score: %.2f \n" 		% agScore_AVG)
	summary.write("Max agent score: %d \n" 				% agScore_MAX)
	summary.write("Min agent score: %d \n" 				% agScore_MIN)
	summary.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")
	summary.write("Agent speed: %d units/s\n" 			% vPed)
	summary.write("AV Speed: %d units/s\n" 				% vAV)
	summary.write("~~~~ Scores (All tests) ~~~~~~~~~~~~~~~ \n")
	summary.write("95% confidence interval per agent: \n")
	summary.write("".join('%.2f %.2f \n' % x for x in ci95))
	summary.write("Average ci95 over all agents is:\n")
	summary.write("%.2f %.2f\n" % (ci95_low, ci95_hig))
	summary.write("~~~~ Scores (Valid tests) ~~~~~~~~~~~~~ \n")
	summary.write("95% confidence interval per agent: \n")
	summary.write("".join('%.2f %.2f \n' % x for x in v_ci95))
	summary.write("Average ci95 over all agents is:\n")
	summary.write("%.2f %.2f\n" % (v_ci95_low, v_ci95_hig))
	summary.write("======================================= \n")
	summary.write("Q-Agent Features\n")
	summary.write("".join('%s, ' % x for x in tags))
	summary.write("\n")
	summary.write("".join('{:s},'.format(i) for i in feat_weights))
	summary.close()

	tgt = np.mean(test_gen_time)
	if diag:
		print("Test complete...")
		print("Agent type: %s"						% agentBehaviour)
		print("Test generation accuracy: %.1f%%"	% accuracy_ratio)
		print("Average test generation time: %.2fs"	% tgt)
		print("Min agent score: %d" 				% agScore_MIN)
		print("Max agent score: %d" 				% agScore_MAX)
		print("Average agent score: %.2f" 			% agScore_AVG)
		print("ci95_low %.2f" 						% (ci95_low))
		print("ci95_hig %.2f " 						% (ci95_hig))
		print("Min agent score: %d" 				% v_min)
		print("Max agent score: %d" 				% v_max)
		print("Average agent score: %.2f" 			% v_avg)
		print("ci95_low %.2f" 						% (v_ci95_low))
		print("ci95_hig %.2f " 						% (v_ci95_hig))

	# print("%d,%.1f,%.2f,%d,%d,%0.2f,%0.2f,%0.2f,%d,%d,%0.2f,%0.2f,%0.2f" % 
	# 	(nA, accuracy_ratio,tgt,agScore_MIN,agScore_MAX,agScore_AVG,ci95_low,ci95_hig,v_min,v_max,v_avg,v_ci95_low,v_ci95_hig))
	print("%d,%.1f,%.2f,%.2f,%.2f,%d,%d,%0.2f,%0.2f,%0.2f,%d,%d,%0.2f,%0.2f,%0.2f,%0.2f" % 
		(nA, accuracy_ratio,mean_tgt,tgt_ci95_low,tgt_ci95_hig,agScore_MIN,agScore_MAX,agScore_AVG,ci95_low,ci95_hig,v_min,v_max,v_avg,v_ci95_low,v_ci95_hig,aExTime))

	
	# print("======================================= ")
	# print("Agent Feature Weights")
	# print(feat_weights)

	# print("Finished\n\n")
raw_input("Press Enter to close windows")
print("======================================= ")
print("Finished\n\n")
