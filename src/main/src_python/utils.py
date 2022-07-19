import os
import sys
import random
import time
import pickle
import onnxruntime
import numpy as np
import tensorflow as tf
import warnings
from subprocess import Popen
warnings.filterwarnings("ignore")
sys.path.append(os.getcwd()+"/src_python")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from tensorflow.keras.models import load_model


from settings.config import MODEL_PATH, DATASET_PATH, DEBUG_PRINT, MAX_SIZE_FULL_DATASET, TRAIN_SAMPLE_SIZE, ONNX_INFERENCE, INDEX_ACTION_TAB_SIGN, PLAYER1, PLAYER2
from settings.game_settings import GAME_NAME, N_ROW, N_COL, N_REPRESENTATION_STACK, N_ACTION_STACK, N_DISTANCE, N_ORIENTATION, N_LEVELS, N_TIME_STEP


######### Here are the utility function for loading/writing files #########

def load_data():
	pkl_path = DATASET_PATH+GAME_NAME+".pkl"
	
	# Exit program if there is a dataset
	if not os.path.exists(pkl_path):
		print("--> Couldn't find dataset at:", pkl_path)
		exit()
		
	# Open the dataset file
	print("--> Loading dataset for the game :", GAME_NAME)
	data = []
	with open(pkl_path, 'rb') as fr:
		try:
			while True:
		    		data.append(pickle.load(fr))
		except EOFError:
			pass
			
	# Extrat what's inside the dataset
	X = []
	y_values = []
	for batch in data:
		X.append(batch["X"])
		y_values.append(batch["y_values"])
	X = np.array(X, dtype=object)
	y_values = np.array(y_values, dtype=object)
	final_X = X[0]
	final_y_values = y_values[0]
	for i in range(1, X.shape[0]):
		final_X = np.concatenate((final_X, X[i]), axis=0)
		final_y_values = np.concatenate((final_y_values, y_values[i]), axis=0)

	if final_X.shape[0] >= MAX_SIZE_FULL_DATASET:
		print("--> Size of the dataset exceeded :", MAX_SIZE_FULL_DATASET, "examples")
		print("--> Deleting some examples and re-writing pickle file")
		final_X, final_y_values = final_X[final_X.shape[0] - MAX_SIZE_FULL_DATASET:], final_y_values[final_X.shape[0] - MAX_SIZE_FULL_DATASET:] 
		Popen("rm "+pkl_path, shell=True).wait()
		add_to_dataset(final_X, final_y_values)
		
	# Print some stats
	if DEBUG_PRINT:
		print("* Number of examples in the dataset :", final_X.shape[0])
		print("* X shape", final_X.shape)
		print("* y_values shape", final_y_values.shape)
	
	print("--> Done !")
	return final_X, final_y_values
	
def get_random_sample(X, y_values, first_step=False):
	if first_step:
		train_sample = X.shape[0]
		idx = np.random.choice(np.arange(X.shape[0]), train_sample, replace=False)
	else:
		train_sample = TRAIN_SAMPLE_SIZE if TRAIN_SAMPLE_SIZE < X.shape[0] else X.shape[0]
		# Here we take only the last 2/3 of the dataset to avoid low quality data from first iterations
		idx = np.random.choice(np.arange(X.shape[0]//3, X.shape[0]), train_sample, replace=True)
	print("--> Training on", train_sample, "examples, Chosen between index [", idx.min(), idx.max(), "]")
	return X[idx], y_values[idx]

def get_random_hash():
	return str(np.random.rand() * time.time()).replace(".", "")

def add_to_dataset(X, y_values, hash_code=""):
	print("--> Saving data to pickle for the game :", GAME_NAME)
	if len(hash_code) >= 1:
		print("* Hash code :", hash_code)
	pkl_path = DATASET_PATH+GAME_NAME+hash_code+".pkl"
	my_data = {'X': X, 'y_values': y_values}
	if os.path.exists(pkl_path): 
		with open(pkl_path, 'ab+') as fp:
			pickle.dump(my_data, fp)
	else:
		with open(pkl_path, 'wb') as fp:
			pickle.dump(my_data, fp)
	print("--> Done !")

def load_nn(model_type, inference):
	print("--> Loading model for the game :", GAME_NAME, ", model type :", model_type)
	if inference:
		if ONNX_INFERENCE:
			opts = onnxruntime.SessionOptions()
			opts.intra_op_num_threads = 8
			#model = onnxruntime.InferenceSession(MODEL_PATH+GAME_NAME+"_"+model_type+".onnx", sess_options=opts, providers=onnxruntime.get_available_providers())
			model = onnxruntime.InferenceSession(MODEL_PATH+GAME_NAME+"_"+model_type+".onnx", sess_options=opts, providers=["CPUExecutionProvider"])
			model.get_modelmeta()
		else:
			# Disable eager mode for faster inference with tensorflow
			tf.compat.v1.disable_eager_execution()
			model = load_model(MODEL_PATH+GAME_NAME+"_"+model_type+".h5", custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})
	else:
		model = load_model(MODEL_PATH+GAME_NAME+"_"+model_type+".h5", custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})
	print("--> Done !")
	return model
	
# Use the model to predict a value
def predict_with_model(model, X, output=["value_head"]):
	if ONNX_INFERENCE:
		res = model.run(output, {"input_1": X.astype(np.float32)})
	res = model.predict(X, verbose=0)
	print(res)
	return res
	
# This function checks if we are going to use the vanilla MCTS
# because we don't have a model yet or if we are going to use
# the alphazero MCTS
def check_if_first_step():
	if os.path.exists(MODEL_PATH+GAME_NAME+"_"+"champion"+".h5"):
		return False
	print("--> No model found, starting from random values")
	return True

def write_winner(outsider_winrate, hash_code=""):
	if len(hash_code) >= 1:
		file_name = "winners" + hash_code + ".txt"
		f = open(MODEL_PATH+file_name, "w")
	else: 
		file_name = "save_winners.txt"
		f = open(MODEL_PATH+file_name, "a")
	f.write("Outsider model winrate: %.3f\n" % outsider_winrate)
	f.close()
	print("--> Saved the winner of the dojo in", file_name)

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
# rank 1 -> +1
# rank 2 -> -1
# draw (1.5) -> 0
def rank_to_util(rank):
	return 1.0 - ((rank - 1.0) * 2.0)

# Get utilities from context object, the utility values are between -1 and 1
def utilities(context):
	ranking = context.trial().ranking() 
	utils = np.zeros(len(ranking))
	for p in range(1, len(ranking)): # Avoid first null object ## CHECK IF ITS STILL THE CASE !!!!!!!
		rank = ranking[p]
		if rank == 0.0: # If the game isn't over
		    rank = context.computeNextDrawRank() # Compute next ranks
		utils[p] = rank_to_util(rank) # Compute utility value per player
	return utils
	
# Returns the opponent of the mover as an int
def opp(mover):
	return 2 if mover==1 else 1
	
######### Here are the utility functions to format policy #########

# Transforms board int values into 2D coordinate values
def get_coord(from_, to):
	return from_//N_ROW, from_%N_ROW, to//N_ROW, to%N_ROW
	
# Transforms an int value into 3D coord
def get_3D_coord(value):
	tmp = value // (N_ACTION_STACK)
	return tmp // N_ROW, tmp % N_COL, value % N_ACTION_STACK

# Define the type of action thanks to position of current and last move
def index_action(from_, to):
	# We get the coordinate of both position
	prev_x, prev_y, x, y = get_coord(from_, to)
	off_y = y - prev_y
	off_x = x - prev_x
	
	# We have distance such as 1, 2, 3, 4... We have orientation such as 
	# SE, SW, NE, NW. We create an index such as :
	# index = orientation * N_ROW + (abs(distance)-1)
	# for example if the move is a NE with 3 as the distance we get the index :
	# 2 * 8 + (3-1) = 18, the number of index is N_DISTANCE * N_ORIENTATION. 
	# If we want to get back the values easily using // and %
	if off_y >= 1: # S
		if off_x >= 1: # E
			index = 0 * N_DISTANCE + (np.abs(off_y) - 1)
		elif off_x <= -1: # W
			index = 1 * N_DISTANCE + (np.abs(off_y) - 1)
		else:
			index = -1
	elif off_y <= -1: # N
		if off_x >= 1: # E
			index = 2 * N_DISTANCE + (np.abs(off_y) - 1)
		elif off_x <= -1:  # W
			index = 3 * N_DISTANCE + (np.abs(off_y) - 1)
		else:
			index = -1
	else:
		index = -1
	return int(index)

# Returns the position of the pawn after going to (to_x, to_y)
# thanks to action <action>
def reverse_index_action(to_x, to_y, action):
	distance = action % N_DISTANCE
	orientation = action // N_DISTANCE
	from_x = to_x + INDEX_ACTION_TAB_SIGN[orientation][0] * (distance + 1)
	from_y = to_y + INDEX_ACTION_TAB_SIGN[orientation][1] * (distance + 1)
	return int(from_x), int(from_y)
	
	#if orientation == 0: # SE
	#	from_x = to_x + (distance + 1)
	#	from_y = to_y + (distance + 1)
	#elif orientation == 1: # SW
	#	from_x = to_x - (distance + 1)
	#	from_y = to_y + (distance + 1)
	#elif orientation == 2: # NE
	#	from_x = to_x + (distance + 1)
	#	from_y = to_y - (distance + 1)
	#else: # NW
	#	from_x = to_x - (distance + 1)
	#	from_y = to_y - (distance + 1)
	#return from_x, from_y

######### Here are the utility functions to format the states #########

# Invert the state (-1 become +1 and +1 become -1)
def invert_state(state):
	inverted_state = state.copy()
	inverted_state = np.where(inverted_state==1, 2, inverted_state)
	inverted_state = np.where(inverted_state==-1, 1, inverted_state)
	inverted_state = np.where(inverted_state==2, -1, inverted_state)
	return inverted_state

# Create a numpy array from the java owned positions
def format_positions(positions, lvl, val, pre_coords):
	res = np.zeros((N_ROW, N_COL))
	pos = positions[0]

	# tmp_lvl = []
	# tmp_site = []

	for i in range(pos.size()):
		p = pos.get(i)
		if p.level() == lvl:
			res[pre_coords[p.site()]] = val

	# 	tmp_lvl.append(p.level())
	# 	tmp_site.append(p.site())
	# mask_lvl = np.where(tmp_lvl==lvl, True, False)
	# res[pre_coords[tmp_site]][mask_lvl] = val

	# for i in range(pos.size()):
	# 	# Filling presence map per level
	# 	p = pos.get(i)
	# 	if p.level() == lvl:
	# 		res[pre_coords[p.site()]] = val

	return res

def format_loc(res, loc, lvl, val):
	# Filling presence map per level
	if loc.level() == lvl:
		res[loc.site()] = val

# Build the input of the NN for AlphaZero algorithm thanks to the context object
def format_state(context, pre_coords):
	res = np.zeros(((N_TIME_STEP*2), N_LEVELS, N_ROW, N_COL))
	
	# Here we copy the state since we are going to need to undo moves
	context_copy = context.deepCopy()
	
	# Get the game model so we can undo move on the context_copy object
	game = context_copy.game()
	
	# Fill owned position for each player at each time step
	for i in range(0, N_TIME_STEP*2, 2):
		# Get the state and the owned positions for both players
		owned = context_copy.state().owned()
		# We fill levels positions for each time step
		for j in range(N_LEVELS):
			res[i][j] = format_positions(owned.positions(PLAYER1), lvl=j, val=1, pre_coords=pre_coords)
			res[i+1][j]= format_positions(owned.positions(PLAYER2), lvl=j, val=1, pre_coords=pre_coords)
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
	res = np.append(res, np.full((N_ROW, N_COL, 1), current_mover - 1), axis=2)
	return res	


