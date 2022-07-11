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


from settings.config import PLAYER1, PLAYER2, CSTE_PUCT
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

	def chose_move_PPA(context, legal_moves, ppa_policy, ):
		new_policy = np.zeros_like(ppa_policy)
		for move in legal_moves:
			current_context = context.deepCopy()
			current_context.game().apply(current_context, move)
			if current_context.trial().over():
				new_policy[]
			
		
	# Main method called to chose an action at depth 0
	def select_action(self, game, context, max_seconds, max_iterations, max_depth):
		# Init an empty node which will be our root
		root = Node(None, None, 0, context)
		num_players = game.players().count()
		
		# Init our visit counter for that move in order to normalize
		# the visit counts per child
		self.total_visit_count = 0

		# Use max_seconds and max_iterations if a value is set
		# else if we get -1 the max is infinity
		stop_time = time.time() + max_seconds if max_seconds > 0.0 else math.inf
		max_its = max_iterations if max_iterations > 0 else math.inf

		# Iteration counter
		num_iterations = 0

		# Loop making sure we respect the max values
		while num_iterations < max_its and time.time() < stop_time:
			# Our current node will be the root to start
			current = root

			# We are looping until we reach a terminal state on the current node
			while True:
				# Here the game is over so we break out, then we compute the utilities and backpropagate the values
				if current.context.trial().over():
					break
			
				# Here we chose a current node and it is a new one, selected thanks the model policy 
				# (if the current node has still unexpanded moves)
				current = self.select_node(current)

				# If the node expanded is a new one, we have to estimate a value for that node
				if current.visit_count == 0:
					break
			
			# If we broke out because we expanded a new node and not because the trial is over then it is time
			# estimate the value thanks to the model
			if not current.context.trial().over():
				utils = np.zeros(num_players+1)
				current.state = np.expand_dims(format_state(context.deepCopy()).squeeze(), axis=0)
				value_opp_pred = predict_with_model(self.model, invert_state(current.state))		
				value_pred = predict_with_model(self.model, current.state)			
				utils[PLAYER1], utils[PLAYER2] = value_pred[0], value_opp_pred[0]
			# If we are in a terminal node we can compute ground truth utilities
			else:
				# Compute utilities thanks to our functions for both players
				utils = utilities(current.context)

			# We propagate the values from the current node to the root
			while current is not None:
				# visit_count variable for each nodes in order to compute PUCT scores
				current.visit_count += 1
				current.total_visit_count += 1
				# score_sums variable for each players in order to compute PUCT scores
				current.score_sums[PLAYER1] += utils[PLAYER1]
				current.score_sums[PLAYER2] += utils[PLAYER2]
				# We propagate the values from leaves to the root through the whole tree
				current = current.parent

			# Keep track of the number of iteration in case there is a max
			num_iterations += 1

		# Return the final move thanks to the scores
		return self.final_move_selection(root)

	# This method choses what node to select and expand depending the PUCT score
	def select_node(self, current):
		# If we have some moves to expand
		if len(current.unexpanded_moves) > 0:
			# Chose a move randomly
			#move = current.unexpanded_moves.pop()
			move = self.chose_move_PPA(current.unexpanded_moves, current.ppa_policy)
			prior = 1/(len(current.unexpanded_moves)+1) # +1 because we pop()
			
			# We copy the context to play in a simulation
			current_context = current.context.deepCopy()
				
			# Apply the move in the simulation
			current_context.game().apply(current_context, move)
			
			# Return a new node, with the new child (which is the move played), and the prior 
			return Node(current, move, prior, current_context)
			
		# We are now looking for the best value in the children of the current node
		# so we need to init some variables according to PUCT
		best_child = None
		best_value = -math.inf
		num_best_found = 0
		num_children = len(current.children)

		# The mover can be both of the players since the games are alternating move games
		mover = current.context.state().mover()

		sum_child_visit_count = 0
		for i in range(num_children):
			sum_child_visit_count += current.children[i].visit_count

		# For each childrens of the mover
		for i in range(num_children):
			child = current.children[i]

			# Compute the PUCT score
			# The score depends on low visit count, high move probability and high value
			exploit = child.score_sums[mover] / child.visit_count
			explore = CSTE_PUCT * current.prior * (np.sqrt(sum_child_visit_count) / (1 + current.visit_count))
			value = exploit + explore

			#print(child.score_sums[mover], child.visit_count, current.prior, sum_child_visit_count)
			
			# Keep track of the best_child which has the best PUCT score
			if value > best_value:
				best_value = value
				best_child = child
				num_best_found = 1
			# Chose a random child if we have several optimal PUCT scores
			elif value == best_value:
				rand = random.randint(0, num_best_found + 1)
				if rand == 0:
					best_child = child
				num_best_found += 1
				
		# Return the best child of the current node according to the PUCT score
		return best_child

	# This method returns the move to play at the root, thus the move to play in the real game
	# depending the number of visits of each depth 0 actions
	def final_move_selection(self, root_node):
		# Arrays for the decision making
		counter = np.array([root_node.children[i].visit_count/root_node.total_visit_count for i in range(len(root_node.children))])
		
		# Get the decision
		decision = root_node.children[counter.argmax()].move_from_parent
				
		# Returns the move to play in the real game and the moves
		# associated to their probability distribution
		#return best_child.move_from_parent, state
		return decision, root_node.state


class Node:
	def __init__(self, parent, move_from_parent, prior, context):
		self.state = None
		self.value_pred = None
		self.value_opp_pred = None

		# Variables to build the tree
		self.children = []
		self.parent = parent

		# Variables for PUCT score computation
		self.visit_count = 0
		self.total_visit_count = 0
		self.prior = prior
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
			
