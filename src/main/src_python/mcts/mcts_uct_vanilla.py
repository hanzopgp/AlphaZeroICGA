import os
import sys
import random
import math
import time
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.getcwd()+"/src_python")


from settings.config import CSTE_PUCT, N_PLAYERS, PLAYER1, PLAYER2
from settings.game_settings import GAME_NAME
from utils import utilities, format_state, format_positions_bashni, format_positions_connectfour, format_positions_ploy, format_positions_quoridor, format_positions_miniwars, format_positions_plakoto, format_positions_lotus

	
######### Here is the main class to run the vanilla MCTS simulation #########

class MCTS_UCT_vanilla:
	def __init__(self):
		self._player_id = -1
		self.wall_positions = None
		self.dice_state = 0
		if GAME_NAME == "Bashni":
			self.format_positions = format_positions_bashni
		elif GAME_NAME == "Ploy":
			self.format_positions = format_positions_ploy
		elif GAME_NAME == "Quoridor":
			self.format_positions = format_positions_quoridor
		elif GAME_NAME == "MiniWars":
			self.format_positions = format_positions_miniwars
		elif GAME_NAME == "Plakoto":
			self.format_positions = format_positions_plakoto
		elif GAME_NAME == "Lotus":
			self.format_positions = format_positions_lotus
		elif GAME_NAME == "ConnectFour":
			self.format_positions = format_positions_connectfour

	# Fix the player who will play with MCTS in case we load this class with Ludii
	def init_ai(self, game, player_id):
		self._player_id = player_id

	def set_wall_positions(self, wall_positions):
		self.wall_positions = wall_positions

	def set_dice_state(self, dice_state):
		self.dice_state = dice_state
		
	# Set some precomputed functions
	#def set_precompute(self, pre_action_index, pre_reverse_action_index, pre_coords, pre_3D_coords):
	def set_precompute(self, pre_coords):
		#self.pre_action_index = pre_action_index
		#self.pre_reverse_action_index = pre_reverse_action_index
		self.pre_coords = pre_coords
		#self.pre_3D_coords = pre_3D_coords

	# Get the value of a node by doing a rollout
	def get_values(self, context, game):
		# If we broke out because we expanded a new node and not because the trial is over then it is time
		# to playout in case we don't have a model, or to estimate the value if we have one
		if not context.trial().over():
			# We copy the context in order to
			context = context.deepCopy()

			# play it out until the game is over
			game.playout(context,
						None, # ais
						-1.0, # thinking_time
						None, # playoutMoveSelector
						0,    # max_num_biased_actions
						-1,   # max_num_playout_actions
						None) # random selector
						
		# Compute utilities thanks to our functions for both players
		return utilities(context)

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

		# Loop making sure we respect the max values
		while num_iterations < max_its and time.time() < stop_time:
			# Our current node will be the root to start
			current = root

			# We are looping until we reach a terminal state on the current node
			while True:
				# Here the game is over so we break out, then we compute the utilities and backpropagate the values
				if current.context.trial().over():
				    break
			
				# Here we choose a current node and it is a new one, selected thanks to a random policy, or the
				# model policy (if the current node has still unexpanded moves)
				current = self.select_node(current)

				# If the node expanded is a new one, we have to playout until the end of the game (or use the model
				# to estimate a value for that node)
				if current.visit_count == 0:
					break

			# Get values for the current node which made it out of the while true loop
			utils = self.get_values(current.context, game)

			# Backpropagate its values and visit_counts
			self.backpropagate_values(current, utils)

			# Keep track of the number of iteration in case there is a max
			num_iterations += 1

		# print("va", num_iterations)

		# Return the final move thanks to the scores
		return self.select_root_child_node(root)

	# This method choses what node to select and expand depending the UCB score
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
		# so we need to init some variables according to UCB
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
			
			# Compute UCB score
			exploit = child.score_sums[mover] / child.visit_count
			explore = math.sqrt(log / child.visit_count)
			value = exploit + explore

			# Keep track of the best_child which has the best UCB score
			if value > best_value:
				best_value = value
				best_child = child
				num_best_found = 1
			# Choose a random child if we have several optimal UCB scores
			elif value == best_value:
				rand = random.randint(0, num_best_found + 1)
				if rand == 0:
					best_child = child
				num_best_found += 1

		# print("*"*30)
		# print("vanilla")
		# print(exploit, explore, value)
		# print("*"*30)

		# Return the best child of the current node according to the UCB score
		return best_child

	# This method returns the move to play at the root, thus the move to play in the real game
	# depending the number of visits of each depth 0 actions
	def select_root_child_node(self, root_node):
		# Arrays for the decision making
		counter = np.array([root_node.children[i].visit_count/root_node.total_visit_count for i in range(len(root_node.children))])
		
		# Get the decision
		decision = root_node.children[counter.argmax()].move_from_parent
				
		# Returns the move to play in the real game and the moves
		# associated to their prob	ability distribution
		#return best_child.move_from_parent, state
		return decision, np.expand_dims(format_state(self.format_positions, root_node.context, self.pre_coords, self.wall_positions, self.dice_state).squeeze(), axis=0)

class Node:
	def __init__(self, parent, move_from_parent, context):		
		# Variables to build the tree
		self.children = []
		self.parent = parent

		# Variables for UCB score computation
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
