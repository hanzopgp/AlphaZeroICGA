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


from settings.config import ONNX_INFERENCE, GRAPH_INFERENCE, TEMPERATURE, CSTE_PUCT, PLAYER1, PLAYER2
from settings.game_settings import N_ROW, N_COL, N_ACTION_STACK, N_REPRESENTATION_STACK
from utils import load_nn, format_state, apply_dirichlet, softmax, invert_state

	
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
		
	# Use the model to predict a policy and a value
	def predict_with_model(self, state, output=["value_head", "policy_head"]):
		if ONNX_INFERENCE:
			return np.array(self.model.run(output, {"input_1": state.astype(np.float32)}))[0]
		if GRAPH_INFERENCE:
			tensor_output = self.model.graph.get_tensor_by_name('import/dense_2/Sigmoid:0')
			tensor_input = self.model.graph.get_tensor_by_name('import/dense_1_input:0')
			return self.model.run(tensor_output, {tensor_input:sample})
		#return self.model.predict(state, verbose=0)
		return 0, np.random.rand(1, N_ROW*N_COL*N_ACTION_STACK)	
		
	# Get the policy on every moves, mask out the illegal moves,
	# re-compute softmax and pick a move randomly according to
	# the new policy then return the move and its prio
	def chose_move(self, legal_moves, policy_pred, competitive_mode):
		# New legal policy array starting as everything illegal
		legal_policy = np.zeros(policy_pred.shape)
		
		# Save legal moves in python
		legal_moves_python = np.zeros((N_ROW, N_COL, N_ROW, N_COL), dtype=object)
		
		# Find the legal moves in the policy
		for i in range(len(legal_moves)):
			# Get the N_ROW, N_COL coordinates
			legal_move = legal_moves[i]
			to = legal_move.to()
			from_ = getattr(legal_move, "from")()
			
			# Get coords
			prev_x, prev_y, x, y = self.pre_coords[from_][to]

			# Save the move
			legal_moves_python[prev_x][prev_y][x][y] = legal_move
			
			# Precomputed function	
			# Get the action index
			action_index = self.pre_action_index[from_][to]
			
			# Write the value only for the legal moves
			legal_policy[prev_x, prev_y, action_index] = policy_pred[prev_x, prev_y, action_index]
			
		# Re-compute softmax after masking out illegal moves
		legal_policy = softmax(legal_policy, ignore_zero=True)
		
		# If we are playing for real, we chose the best action given by the policy
		if competitive_mode:
			chosen_x, chosen_y, chosen_action = np.unravel_index(legal_policy.argmax(), legal_policy.shape)
			prior = np.max(legal_policy)
		# Else we are training and we use the policy for the MCTS
		else:
			# Get a random number between 0 and 1
			r = np.random.rand()
			chose_array = np.cumsum(legal_policy.flatten())
			choice = np.where(chose_array >= r)
			
			# Precomputed function
			chosen_x, chosen_y, chosen_action = self.pre_3D_coords[choice[0][0]]
			
			# The prior is the value at those index which represent the
			# probability it was picked
			prior = legal_policy[chosen_x][chosen_y][chosen_action]			

		# Precomputed function	
		chosen_prev_x, chosen_prev_y = self.pre_reverse_action_index[chosen_x][chosen_y][chosen_action]
		
		# Now we need to find the move in the java object legal moves list
		return legal_moves_python[chosen_x][chosen_y][chosen_prev_x][chosen_prev_y], prior
		
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
			current_context = current.context

			# We are looping until we reach a terminal state on the current node
			while True:
				# Here the game is over so we break out, then we compute the utilities and backpropagate the values
				if current_context.trial().over():
					break
			
				# Here we chose a current node and it is a new one, selected thanks the model policy 
				# (if the current node has still unexpanded moves)
				current = self.select_node(current)

				# If the node expanded is a new one, we have to estimate a value for that node
				if current.visit_count == 0:
					break

			# If we broke out because we expanded a new node and not because the trial is over then it is time
			# estimate the value thanks to the model
			if not current_context.trial().over():
				utils = np.zeros(num_players+1)
				state = np.expand_dims(format_state(current_context).squeeze(), axis=0)
				if ONNX_INFERENCE:
					value_pred = self.predict_with_model(state, output=["value_head"])
					value_opp_pred = self.predict_with_model(invert_state(state), output=["value_head"])				
				else:
					value_pred, _ = self.predict_with_model(state, output=[""])
					value_opp_pred, _ = self.predict_with_model(invert_state(state), output=[""])
				utils[PLAYER1], utils[PLAYER2] = value_pred, value_opp_pred
			# If we are in a terminal node we can compute ground truth utilities
			else:
				# Compute utilities thanks to our functions for both players
				utils = utilities(current_context)

			# We propagate the values from the current node to the root
			while current is not None:
				# visit_count variable for each nodes in order to compute PUCT scores
				current.visit_count += 1
				current.total_visit_count += 1
				# score_sums variable for each players in order to compute PUCT scores
				for p in range(1, num_players+1):
					current.score_sums[p] += utils[p]
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
			# We copy the context to play in a simulation
			current_context = current.context.deepCopy()
			
			# Get the representation and get the prediction
			state = format_state(current_context).squeeze()
			
			# Estimate policy with the model
			if ONNX_INFERENCE:
				policy_pred = self.predict_with_model(np.expand_dims(state, axis=0), output=["policy_head"])
			else:
				_, policy_pred = self.predict_with_model(np.expand_dims(state, axis=0), output=[""])
			
			# Get ride of useless batch dimension
			policy_pred = policy_pred[0] 
			
			# Apply Dirichlet to ensure exploration
			policy_pred = apply_dirichlet(policy_pred)
			
			# The output of the network is a flattened array
			policy_pred = policy_pred.reshape(N_ROW, N_COL, N_ACTION_STACK)
			
			# Chose a move in legal moves by randomly firing in the policy
			move, prior = self.chose_move(current.unexpanded_moves, policy_pred, competitive_mode=self.dojo)
				
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
		
		# For UCB score
		#two_parent_log = 2.0 * math.log(max(1, current.visit_count))

		# The mover can be both of the players since the games are alternating move games
		mover = current.context.state().mover()

		# For each childrens of the mover
		for i in range(num_children):
			child = current.children[i]
			child_score_sums = child.score_sums

			# Compute the UCB score
			#exploit = child_score_sums[mover] / child.visit_count
			#explore = math.sqrt(two_parent_log / child.visit_count)
			
			# Compute the PUCT score
			# The score depends on low visit count, high move probability and high value
			
			exploit = child_score_sums[mover] / child.visit_count
			explore = CSTE_PUCT * current.prior * (np.sqrt(child_score_sums[mover]) / (1 + current.score_sums[mover]))
			
			value = exploit + explore

			# Keep track of the best_child which has the best PUCT score
			if value > best_value:
				best_value = value;
				best_child = child;
				num_best_found = 1;
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
		# Now that we have gone through the tree using PUCT scores, the MCTS will chose
		# the best move to play by checking which node was the most visited
		num_children = len(root_node.children)
		total_visit_count = root_node.total_visit_count
		
		# Array to store the number of visits per move, a move is represented
		# by its coordinate and several stacks representing where it can go
		move_distribution = np.zeros((N_ROW, N_COL, N_ACTION_STACK))

		# Arrays for the decision making
		children = []
		counter = []

		# For each children of the root, so for each legal moves
		for i in range(num_children):
			child = root_node.children[i]
			visit_count = child.visit_count
			move_from_parent = child.move_from_parent
			normalized_visit_count = visit_count/total_visit_count

			# Getting coordinates of the move
			to = move_from_parent.to()
			from_ = getattr(move_from_parent, "from")() # trick to use the from method (reserved in python)
			
			# Getting the action type as an int :
			# 0 1 2 3... depending the from and to
			action_index = self.pre_action_index[from_][to]
			# <int(from_/N_ROW), from_%N_ROW> represent the position of the
			# pawn that chosed action <action_index> to go in position <to>
			move_distribution[from_//N_ROW, from_%N_ROW, action_index] = normalized_visit_count
	
			# Keeps track of our children and their visit_count
			children.append(child)
			counter.append(normalized_visit_count)
			
		# Compute softmax on visit counts, giving us a distribution on moves
		soft = softmax(np.array(counter))
		
		# Start as 1 so it doesn't matter at the beginning,
		# goes to 0 while the game goes on in order to reduce
		# exploration in the end game
		soft = np.power(soft, 1/TEMPERATURE)
		
		# Get the decision
		decision = children[soft.argmax()].move_from_parent
				
		# Get the representation of the current state for the future NN training
		state = format_state(root_node.context)
				
		# Returns the move to play in the real game and the moves
		# associated to their probability distribution
		#return best_child.move_from_parent, state, move_distribution
		return decision, state, move_distribution


class Node:
	def __init__(self, parent, move_from_parent, prior, context):
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
			
