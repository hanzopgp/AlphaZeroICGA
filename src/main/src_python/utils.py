import math
import random
import time
import pickle
import numpy as np
import os
import pandas as pd
import pprint
from matplotlib import pyplot as plt
from os.path import exists

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU, add
from tensorflow.keras.optimizers import SGD
from keras import regularizers


from src_python.config import *
#from config import *


######### Here are the utility function for loading/writing files #########

def load_data():
	pkl_path = DATASET_PATH+GAME_NAME+".pkl"
	if not exists(pkl_path):
		print("--> Couldn't find dataset at:", pkl_path)
		exit()
	print("--> Loading dataset for the game :", GAME_NAME)
	data = []
	with open(pkl_path, 'rb') as fr:
		try:
			while True:
		    		data.append(pickle.load(fr))
		except EOFError:
			pass
	X = []
	y_values = []
	y_distrib = []
	for batch in data:
		X.append(batch["X"])
		y_values.append(batch["y_values"])
		y_distrib.append(batch["y_distrib"])
	X = np.array(X, dtype=object)
	y_values = np.array(y_values, dtype=object)
	y_distrib = np.array(y_distrib, dtype=object)
	final_X = X[0]
	final_y_values = y_values[0]
	final_y_distrib = y_distrib[0]
	for i in range(1, X.shape[0]):
		final_X = np.concatenate((final_X, X[i]), axis=0)
		final_y_values = np.concatenate((final_y_values, y_values[i]), axis=0)
		final_y_distrib = np.concatenate((final_y_distrib, y_distrib[i]), axis=0)
	print("* Number of examples in the dataset :", final_X.shape[0])
	print("* X shape", final_X.shape)
	print("* y_values shape", final_y_values.shape)
	print("* y_distrib shape", final_y_distrib.shape)
	print("--> Done !")
	return final_X, final_y_values, final_y_distrib
	
def get_random_sample(data):
	pass

def add_to_dataset(X, y_values, y_distrib):
	print("--> Saving data to pickle for the game :", GAME_NAME)
	pkl_path = DATASET_PATH+GAME_NAME+".pkl"
	my_data = {'X': X,
	   	   'y_values': y_values,
	   	   'y_distrib': y_distrib}
	if exists(pkl_path): 
		with open(pkl_path, 'ab+') as fp:
			pickle.dump(my_data, fp)
	else:
		with open(pkl_path, 'wb') as fp:
			pickle.dump(my_data, fp)
	print("--> Done !")

def load_nn(model_type):
	print("--> Loading model for the game :", GAME_NAME, ", model type :", model_type)
	model = load_model(
			MODEL_PATH+GAME_NAME+"_"+model_type+".h5",
			custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})
	return model

######### Here are some functions for the model #########

def softmax_cross_entropy_with_logits(y_true, y_pred):
	# Find where the values of the labels are 0
	zero = tf.zeros(shape=tf.shape(y_true), dtype=tf.float32)
	where = tf.equal(y_true, zero)
	# Create a -100 values array
	filler = tf.fill(tf.shape(y_true), -100.0)
	# Switch 0 values by -100 values for the predictions
	y_pred = tf.where(where, filler, y_pred)
	# Apply and return the classical softmax crossentropy loss
	return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred) 

def softmax(x, ignore_zero=False):
	if ignore_zero:
		non_zero_indices = np.where(x != 0)
		x[non_zero_indices] = np.exp(x[non_zero_indices])/np.sum(np.exp(x[non_zero_indices]))
		return x
	else:
		return np.exp(x)/np.sum(np.exp(x))

######### Here are the utility function used for the game #########
	
# Convert end of game rank into utility value
def rank_to_util(rank):
	return 1.0 - ((rank - 1.0) * 2.0)

# Get utilities from context object, the utility values are between -1 and 1
def utilities(context):
	ranking = context.trial().ranking() 
	utils = np.zeros(len(ranking))
	for p in range(1, len(ranking)): # Avoid first null object
		rank = ranking[p]
		if rank == 0.0: # If the game isn't over
		    rank = context.computeNextDrawRank() # Compute next ranks
		utils[p] = rank_to_util(rank) # Compute utility value per player
	return utils
	
# Returns the opponent of the mover as an int
def opp(mover):
	return 2 if mover ==1 else 1
	
######### Here are the utility functions to format policy #########

# Apply Dirichlet with the alpha parameters to the policy in order
# to add some noise in the policy and ensure exploration
def apply_dirichlet(policy):
	dira = np.random.dirichlet(np.full(policy.shape, DIRICHLET_ALPHA), size=1)
	return (WEIGHTED_SUM_DIR * policy) + (1 - WEIGHTED_SUM_DIR) * dira

# Define the type of action thanks to position of current and last move
def index_action(from_, to):
	# We get the coordinate of both position
	prev_x = int(from_/N_ROW)
	prev_y = from_%N_ROW
	x = int(to/N_ROW)
	y = to%N_ROW
	off_y = y - prev_y
	off_x = x - prev_x
	# We have distance such as 1, 2, 3, 4... We have orientation such as 
	# SE, SW, NE, NW. We create an index such as :
	# index = orientation * N_ROW + (abs(distance)-1)
	# for example if the move is a NE with 3 as the distance we get the index :
	# 2 * 8 + (3-1) = 18, the number of index is N_DISTANCE * N_ORIENTATION. 
	# If we want to get back the values easily using int() and %
	if off_y >= 1: # S
		if off_x >= 1: # E
			index = 0 * N_DISTANCE + (np.abs(off_y) - 1)
		else: # W
			index = 1 * N_DISTANCE + (np.abs(off_y) - 1)
	elif off_y <= -1: # N
		if off_x >= 1: # E
			index = 2 * N_DISTANCE + (np.abs(off_y) - 1)
		else: # W
			index = 3 * N_DISTANCE + (np.abs(off_y) - 1)
	return index 

# Returns the position of the pawn after going to (to_x, to_y)
# thanks to action <action>
def reverse_index_action(to_x, to_y, action):
	distance = (action % N_DISTANCE)
	orientation = int(action / N_DISTANCE)
	if orientation == 0: # SE
		from_x = to_x + (distance + 1)
		from_y = to_y + (distance + 1)
	elif orientation == 1: # SW
		from_x = to_x - (distance + 1)
		from_y = to_y + (distance + 1)
	elif orientation == 2: # NE
		from_x = to_x + (distance + 1)
		from_y = to_y - (distance + 1)
	else: # NW
		from_x = to_x - (distance + 1)
		from_y = to_y - (distance + 1)
	return from_x, from_y

# Get the policy on every moves, mask out the illegal moves,
# re-compute softmax and pick a move randomly according to
# the new policy
def chose_move(legal_moves, policy_pred, competitive_mode):
	# New legal policy array starting as everything illegal
	legal_policy = np.zeros(policy_pred.shape)
	# Find the legal moves in the policy
	for i in range(len(legal_moves)):
		# Get the N_ROW, N_COL coordinates
		to = legal_moves[i].to()
		from_ = getattr(legal_moves[i], "from")()
		prev_x = int(from_ / N_ROW)
		prev_y = from_ % N_ROW
		# Get the action index
		action_index = index_action(from_, to)
		# Write the value only for the legal moves
		legal_policy[prev_x, prev_y, action_index] = policy_pred[prev_x, prev_y, action_index]
	# Re-compute softmax after masking out illegal moves
	legal_policy = softmax(legal_policy, ignore_zero=True)
	# If we are playing for real, we chose the best action given by the policy
	if competitive_mode:
		chosen_x, chosen_y, chosen_action = np.argmax(legal_policy)
	# Else we are training and we use the policy for the MCTS
	else:
		# Build a cumulative sum array and chose the move
		r = np.random.rand()
		idx_legal = np.where(legal_policy != 0)
		fire = legal_policy[idx_legal]
		chose_array = np.cumsum(fire)
		choice = np.where(chose_array >= r)[0][0]
		chosen_x, chosen_y, chosen_action = idx_legal[0][choice], idx_legal[1][choice], idx_legal[2][choice]
	# Now we need to find the move in the java object legal moves list
	chosen_prev_x, chosen_prev_y = reverse_index_action(chosen_x, chosen_y, chosen_action)
	for i in range(len(legal_moves)):
		to = legal_moves[i].to()
		from_ = getattr(legal_moves[i], "from")()
		prev_x = int(from_/N_ROW)
		prev_y = from_%N_ROW
		x = int(to/N_ROW)
		y = to%N_ROW
		# Inverse to match our representation
		if prev_x == chosen_x and prev_y == chosen_y and x == chosen_prev_x and y == chosen_prev_y:
			return legal_moves[i]

######### Here are the utility functions to format the states #########

# Create a numpy array from the java owned positions
def format_positions(positions, lvl, val):
	res = np.zeros((N_ROW*N_COL))
	for pos in positions:
		for i in range(pos.size()):
			# Filling presence map per level
			p = pos.get(i)
			if p.level() == lvl:
				res[p.site()] = val
	return res.reshape(N_ROW, N_COL)

# Build the input of the NN for AlphaZero algorithm thanks to the context object
def format_state(context):
	res = np.zeros(((N_TIME_STEP*2), N_LEVELS, N_ROW, N_COL))
	# Here we copy the state since we are going to need to undo moves
	context_copy = context.deepCopy()
	# Get the game model so we can undo move on the context_copy object
	game = context_copy.game()
	# Fill owned position for each player at each time step
	for i in range(0, N_TIME_STEP*2, 2):
		# Get the state and the owned positions for both players
		state = context_copy.state()
		owned = state.owned()
		# We fill levels positions for each time step
		for j in range(N_LEVELS):
			res[i][j] = format_positions(owned.positions(PLAYER1), lvl=j, val=1)
			res[i+1][j]= format_positions(owned.positions(PLAYER2), lvl=j, val=1)
		# After filling the positions for one time step we undo one game move
		# to fill the previous time step
		try:
			game.undo(context_copy)
		# Break in case we can't undo a move (start of game for example)
		except: 
			break
	# Need to first merge the time steps and levels into a representation stack
	res = res.reshape(-1, N_ROW, N_COL)
	# And then move the axis the get the NWHC format
	res = np.moveaxis(res, 0, -1)
	# This was a big misstake leading to a wrong representation
	#res = res.reshape(N_ROW, N_COL, -1)
	
	# The current player stack will be 0 if player1 is the current
	# player and 1 if player2 is the current player, this is an example
	# of additional feature but we could also add the number of moves
	# played until now etc...
	current_mover = context.state().mover()
	current_player = 0 if current_mover==PLAYER1 else 1
	res = np.append(res, np.full((N_ROW, N_COL, 1), current_player), axis=2)
	return res	


