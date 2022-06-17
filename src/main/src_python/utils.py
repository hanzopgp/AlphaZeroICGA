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
		print("Couldn't find dataset at:", pkl_path)
		exit()
	print("Loading CSV dataset ...")
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
	print("Number of examples", final_X.shape[0])
	print("Done !")
	return final_X, final_y_values, final_y_distrib

def add_to_dataset(X, y_values, y_distrib):
	print("Saving data to csv for the game :", GAME_NAME, "...")
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
	print("Done !")

def load_nn():
	return load_model(
			MODEL_PATH+GAME_NAME+".h5",
			custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})

######### Here are some functions for the model #########

def softmax_cross_entropy_with_logits(y_true, y_pred):
	# Find where the values of the labels are 0
	#zero = tf.zeros(shape=tf.shape(y_true), dtype=tf.float32)
	#where = tf.equal(y_true, zero)
	# Create a -100 values array
	#filler = tf.fill(tf.shape(y_true), -100.0)
	# Switch 0 values by -100 values for the predictions
	#y_pred = tf.where(where, filler, y_pred)
	# Apply and return the classical softmax crossentropy loss
	return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred) 

def softmax(x):
	exp_ = np.exp(x)
	return exp_/np.sum(exp_)

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
	
######### Here are the utility functions to format data #########

def index_action(from_, to):
	# We get the coordinate of both position
	prev_x = math.ceil(from_/N_ROW)
	prev_y = from_%N_ROW
	x = math.ceil(to/N_ROW)
	y = to%N_ROW
	print(prev_x, prev_y)
	print(x, y)
	off_y = y - prev_y
	off_x = x - prev_x
	if off_y == 2: # south 2
		if off_x == 2: # east 2
			index = 7
		else: # west 2
			index = 6
	elif off_y == -2 : # north 2
		if off_x == 2: # east 2
			index = 5
		else: # west 2
			index = 4
	elif off_y == -1 : # south 1 
		if off_x == 1: # east 1
			index = 3
		else: # west 1
			index = 2
	elif off_y == -1 : # north 1
		if off_x == 1: # east 1
			index = 1
		else: # west 1 
			index = 0
	return index  # because we start at 0

# Mask out the illegal moves and re compute softmax
def format_move(legal_moves, policy_pred):
	# Get the to() values
	to = []
	for i in range(len(legal_moves)):
		to.append(legal_moves[i].to())
	# Mask out the illegal moves by lowering their values
	mask = np.ones(policy_pred.shape, dtype=bool)
	mask[to] = False
	policy_pred[mask] = -100
	# Recompute softmax and get the argmax
	decision = softmax(policy_pred).argmax()
	# Return the moves depending the decision, we do it like that
	# because we need to directly return a Move java object in
	# order to play the move with the java functions 
	for i in range(len(legal_moves)):
		if decision == legal_moves[i].to():
			return legal_moves[i]

# Create a numpy array from the java owned positions
def format_positions(positions, lvl, val):
	res = np.zeros((N_ROW*N_COL))
	for pos in positions:
		for i in range(pos.size()):
			# We create a boolean in order to build a presence map
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
			res[i][j] = format_positions(owned.positions(PLAYER1), lvl=j, val=PLAYER1)
			res[i+1][j]= format_positions(owned.positions(PLAYER2), lvl=j, val=PLAYER2)
		# After filling the positions for one time step we undo one game move
		# to fill the previous time step
		try:
			game.undo(context_copy)
		# Break in case we can't undo a move (start of game for example)
		except: 
			break
	# The current player stack will be 0 if player1 is the current
	# player and 1 if player2 is the current player, this is an example
	# of additional feature but we could also add the number of moves
	# played until now etc...
	res = res.reshape(N_ROW, N_COL, -1)
	current_mover = context.state().mover()
	#current_player = 0 if current_mover==PLAYER1 else 1
	res = np.append(res, np.full((N_ROW, N_COL, 1), current_mover), axis=2)
	return res
	
