import os
import sys
import random
import math
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.getcwd()+"/src_python")


from settings.game_settings import N_REPRESENTATION_STACK, N_ROW, N_COL
from settings.config import N_PLAYERS, ONNX_INFERENCE, PLAYER1, PLAYER2, CSTE_PUCT, MINIMUM_QUEUE_PREDICTION
from utils import load_nn, format_state, invert_state, predict_with_model, utilities

	
######### Here is the main class to run the MCTS simulation with the model #########

class MCTS_UCT_alphazero:
	# AlphaZero training loop always picks the best model so the model_type is champion by default
	def __init__(self, dojo=False, model_type="champion"):
		self._player_id = -1
		self.dojo = dojo
		self.model = load_nn(model_type=model_type, inference=True)

	# Fix the player who will play with MCTS in case we load this class with Ludii
	def init_ai(self, game, player_id):
		self._player_id = player_id
		
	# Precomputed functions as arrays parameters -> return
	def set_precompute(self, pre_action_index, pre_reverse_action_index, pre_coords, pre_3D_coords):
		self.pre_action_index = pre_action_index
		self.pre_reverse_action_index = pre_reverse_action_index
		self.pre_coords = pre_coords
		self.pre_3D_coords = pre_3D_coords

	# Get values of the current node one by one
	def get_values(self, current):
		# If we broke out because we expanded a new node and not because the trial is over then it is time
		# estimate the value thanks to the model
		if not current.context.trial().over():
			utils = np.zeros(N_PLAYERS)
			current.state = np.expand_dims(format_state(current.context.deepCopy(), self.pre_coords).squeeze(), axis=0)
			current.value_pred = predict_with_model(self.model, current.state)
			current.value_opp_pred = predict_with_model(self.model, invert_state(current.state))				
			utils[PLAYER1], utils[PLAYER2] = current.value_pred[0], current.value_opp_pred[0]
		# If we are in a terminal node we can compute ground truth utilities
		else:
			# Compute utilities thanks to our functions for both players
			utils = utilities(current.context)
		return utils

	# Backpropagates the values and the visit counts of the current node
	def backpropagate_values(self, node, utils):
		# We propagate the values from the current node to the root
		while node is not None:
			# visit_count variable for each nodes in order to compute UCB scores
			node.visit_count += 1
			node.total_visit_count += 1
			# score_sums variable for each players in order to compute UCB scores
			node.score_sums[PLAYER1] += utils[PLAYER1]
			node.score_sums[PLAYER2] += utils[PLAYER2]
			# We propagate the values from leaves to the root through the whole tree
			node = node.parent

	# Estimate the values of a batch of nodes
	def predict_values(self, nodes):
		# We will need the states and inverted states to predict the utils
		states = np.zeros((len(nodes), N_ROW, N_COL, N_REPRESENTATION_STACK))
		inverted_states = np.zeros((len(nodes), N_ROW, N_COL, N_REPRESENTATION_STACK))
		utils = np.zeros((len(nodes), N_PLAYERS))

		# For each node of the batch
		for i, node in enumerate(nodes):
			# We compute their states
			node.state = np.expand_dims(format_state(node.context, self.pre_coords).squeeze(), axis=0)
			inverted_states[i] = np.expand_dims(invert_state(node.state), axis=0)
			states[i] = node.state

		# Then we predict the values for that batch of states
		if ONNX_INFERENCE:
			value_preds = predict_with_model(self.model, states)[0]
			value_opp_preds = predict_with_model(self.model, inverted_states)[0]
		else:
			value_preds = predict_with_model(self.model, states)
			value_opp_preds = predict_with_model(self.model, inverted_states)

		# Then we can fill and return the utility array
		utils[:,PLAYER1] = value_preds.squeeze()
		utils[:,PLAYER2] = value_opp_preds.squeeze()
		return utils

	# Checks if we can compute some ground truth utils
	def check_ground_truth(self, nodes, utils):
		for i, node in enumerate(nodes):
			tmp_context = node.context.deepCopy()
			if tmp_context.trial().over():
				utils[i] = utilities(tmp_context)
		return utils
		
	# Backpropagates the predicted values in the tree for a batch of nodes
	def backpropagate_predicted_values(self, nodes, utils):
		for i, node in enumerate(nodes):
			while node is not None:
				node.score_sums[PLAYER1] += utils[i, PLAYER1]
				node.score_sums[PLAYER2] += utils[i, PLAYER2]
				node = node.parent

	# Backpropagate the visit counts of a node
	def backpropagate_visit_counts(self, node):
		while node is not None:
				node.visit_count += 1
				node.total_visit_count += 1
				node = node.parent
		
	# Main method called to choose an action at depth 0
	def select_action(self, game, context, max_seconds, max_iterations, max_depth):
		# Init an empty node which will be our root
		root = Node(None, None, context)
		
		# Init our visit counter for that move in order to normalize
		# the visit counts per child
		self.total_visit_count = 0

		# Use max_seconds and max_iterations if a value is set
		# else if we get -1 the max is infinity
		stop_time = time.time() + max_seconds if max_seconds > 0.0 else math.inf
		max_its = max_iterations if max_iterations > 0 else math.inf

		# Iteration counter
		num_iterations = 0

		# Queue to predict in batch
		predict_queue = []
		full_queue = []

		# Loop making sure we respect the max values
		while num_iterations < max_its and time.time() < stop_time:
			# Our current node will be the root to start
			current = root

			# We are looping until we we discover a terminal/new node
			while True:
				# Here the game is over so we break out, then we compute the utilities and backpropagate the values
				if current.context.trial().over():
					break
			
				# Here we choose a current node and it is a new one, selected thanks the model policy 
				# (if the current node has still unexpanded moves)
				current = self.select_node(current)

				# If the node expanded is a new one, we have to estimate a value for that node
				if current.visit_count == 0:
					break	

			# Adding the current node to the predict queue list in order to estimate the values later
			predict_queue.append(current)
			full_queue.append(current)
			# Here we predict values if the queue length is higher than a minimum value or if it's the
			# last iteration in order to avoid missing values before the final decision
			if len(predict_queue) >= MINIMUM_QUEUE_PREDICTION or num_iterations == max_its - 1:
				# Predict the values of the whole queue 
				utils = self.predict_values(predict_queue)

				# Check if we can compute some ground truth utils
				utils = self.check_ground_truth(predict_queue, utils)

				# Backpropagated the utility scores
				self.backpropagate_predicted_values(predict_queue, utils)

				# Empty the predict queue
				predict_queue = []
			# If it's not time to estimate the values then we put all the values to 0, we don't need to 
			# backpropagate the values. This makes us save time and the MCTS becomes pessimistic
			else:
				current.value_pred = 0
				current.value_opp_pred = 0	
				
			# Here for each node we backpropagate the visit counts	
			self.backpropagate_visit_counts(current)

			# Keep track of the number of iteration in case there is a max
			num_iterations += 1

		# print("----->", full_queue)
		# print(len(full_queue))
		# for i in full_queue:
		# 	print(i.score_sums[PLAYER1])
		# 	print(i.score_sums[PLAYER2])
		# exit()

		# print("al", num_iterations)

		# Return the final move thanks to the scores
		return self.select_root_child_node(root)

	# This method choses what node to select and expand depending the PUCT score
	def select_node(self, current):
		# If we have some moves to expand
		if len(current.unexpanded_moves) > 0:
			# Choose a move randomly
			move = current.unexpanded_moves.pop()
			
			# We copy the context to play in a simulation
			current_context = current.context.deepCopy()
				
			# Apply the move in the simulation
			current_context.game().apply(current_context, move)
			
			# Return a new node, with the new child (which is the move played)
			return Node(current, move, current_context)
			
		# We are now looking for the best value in the children of the current node
		# so we need to init some variables according to PUCT
		best_child = None
		best_value = -math.inf
		num_best_found = 0

		# The mover can be both of the players since the games are alternating move games
		mover = current.context.state().mover()

		# Precompute before loop
		log = CSTE_PUCT * math.log(max(1, current.visit_count))

		# For each childrens of the mover
		for i in range(len(current.children)):
			child = current.children[i]

			# Compute the UCB score
			exploit = child.score_sums[mover] / child.visit_count
			explore = math.sqrt(log / child.visit_count)
			value = exploit + explore

			# Keep track of the best_child which has the best PUCT score
			if value > best_value:
				best_value = value
				best_child = child
				num_best_found = 1
			# Choose a random child if we have several optimal PUCT scores
			elif value == best_value:
				rand = random.randint(0, num_best_found + 1)
				if rand == 0:
					best_child = child
				num_best_found += 1

		# print("*"*30)
		# print("alphazero", self._player_id)
		# print(exploit, explore, value)
		# print("*"*30)
				
		# Return the best child of the current node according to the PUCT score
		return best_child

	# This method returns the move to play at the root, thus the move to play in the real game
	# depending the number of visits of each depth 0 actions
	def select_root_child_node(self, root_node):
		# Arrays for the decision making
		counter = np.array([root_node.children[i].visit_count/root_node.total_visit_count for i in range(len(root_node.children))])
		
		# Get the decision
		decision = root_node.children[counter.argmax()].move_from_parent
				
		# Returns the move to play in the real game and the root node state
		return decision, np.expand_dims(format_state(root_node.context, self.pre_coords).squeeze(), axis=0)


class Node:
	def __init__(self, parent, move_from_parent, context):
		self.state = None
		self.value_pred = None
		self.value_opp_pred = None

		# Variables to build the tree
		self.children = []
		self.parent = parent

		# Variables for PUCT score computation
		self.visit_count = 0
		self.total_visit_count = 0
		self.move_from_parent = move_from_parent
		self.context = context
		game = self.context.game()
		self.score_sums = np.zeros(game.players().count() + 1)

		# Get unexpanded moves for each node
		legal_moves = game.moves(context).moves()
		num_legal_moves = legal_moves.size()
		self.unexpanded_moves = [legal_moves.get(i) for i in range(num_legal_moves)]
		# Shuffle the unexpanded_moves to add randomness
		random.shuffle(self.unexpanded_moves)

		# Recursively declare the children for each parent
		if parent is not None:
			parent.children.append(self)
			
