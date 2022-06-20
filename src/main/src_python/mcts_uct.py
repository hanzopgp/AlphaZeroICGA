from src_python.config import *
from src_python.utils import *

	
######### Here is the main class to run the MCTS simulation #########

class MCTS_UCT:
	def __init__(self):
		self._player_id = -1
		if exists(MODEL_PATH+GAME_NAME+".h5"): 
			print("--> Using the model:", MODEL_PATH+GAME_NAME+".h5", "to chose moves")
			self.first_step = False
			self.model = load_nn()
		else:
			print("--> No model found, starting from random moves")
			self.first_step = True

	# Fix the player who will play with MCTS in case we load this class with Ludii
	def init_ai(self, game, player_id):
		self._player_id = player_id

	# Main method that select the next move at depth 0
	def select_action(self, game, context, max_seconds, max_iterations, max_depth):
		# Init an empty node which will be our root
		root = Node(None, None, context)
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
				# Here the game is over so we break out
				if current.context.trial().over():
				    break
			
				# Now our current node is a new one, selected thanks to UCB selection & extension phase
				current = self.select_node(current)

				# If the node expanded is a new one, we have to playout until the end of the game
				if current.visit_count == 0:
					break

			# If we haven't break out the while loop it means we can get the final context of the current node
			context_end = current.context

			# If we broke out because we expanded a new node and not because the trial is over
			if not context_end.trial().over():
				# We copy the context in order to
				context_end = context_end.deepCopy()
				# play it out until the game is over
				game.playout(context_end,
					     None, # ais
					     -1.0, # thinking_time
					     None, # playoutMoveSelector
					     0,    # max_num_biased_actions
					     -1,   # max_num_playout_actions
					     None) # random selector
		                     
			# Compute utilities thanks to our functions for both players
			utils = utilities(context_end)

			# We keep playing out for each nodes until we are done
			while current is not None:
				# visit_count variable for each nodes in order to compute UCB scores
				current.visit_count += 1
				current.total_visit_count += 1
				# score_sums variable for each nodes in order to compute UCB scores
				for p in range(1, num_players+1):
					current.score_sums[p] += utils[p]
				# We propagate the values from leaves to the root through the whole tree
				current = current.parent

			# Keep track of the number of iteration in case there is a max
			num_iterations += 1

		# Return the final move thanks to the scores
		return self.final_move_selection(root)

	# This method choses what node to select and expand depending the UCB score
	def select_node(self, current):
		# If we have some moves to expand
		if len(current.unexpanded_moves) > 0:
			# If it's the first step and we don't have a model yet, we chose random moves
			# can pop it since it's already shuffled
			if self.first_step:
				move = current.unexpanded_moves.pop()
			# If it's not the first step then we use our model to chose a move
			else:
				state = format_state(current.context).squeeze()
				_, policy_pred = self.model.predict(np.expand_dims(state, axis=0))
				policy_pred = policy_pred[0] # Get ride of useless batch dimension
				# Chose a move in legal moves by randomly firing in the policy
				move = chose_move(current.unexpanded_moves, policy_pred, competitive_mode=False)
			# We copy the context to play in a simulation
			context = current.context.deepCopy()
			# Apply the move in the simulation
			context.game().apply(context, move)
			# Return a new node, with the new child (which is the move played)
			return Node(current, move, context)

		# We are now looking for the best value in the children of the current node
		# so we need to init some variables according to UCB
		best_child = None
		best_value = -math.inf
		two_parent_log = 2.0 * math.log(max(1, current.visit_count))
		num_best_found = 0
		num_children = len(current.children)

		# The mover can be both of the players since the games are alternating move games
		mover = current.context.state().mover()

		# For each childrens of the mover
		for i in range(num_children):
			child = current.children[i]
			# If we don't have a model yet we compute the UCB score
			if self.first_step:
				exploit = child.score_sums[mover] / child.visit_count
				explore = math.sqrt(two_parent_log / child.visit_count)
				value = exploit + explore
			# Else we use the model to predict a value
			else:
				value, _ = self.model.predict(np.expand_dims(state, axis=0))
			# Keep track of the best_child which has the best UCB score
			if value > best_value:
				best_value = value;
				best_child = child;
				num_best_found = 1;
			# Chose a random child if we have several optimal UCB scores
			elif value == best_value:
				rand = random.randint(0, num_best_found + 1)
				if rand == 0:
					best_child = child
				num_best_found += 1

		# Return the best child of the current node according to the UCB score
		return best_child

	# This method returns the move to play at the root, thus the move to play in the real game
	def final_move_selection(self, root_node):
		# Now that we have gone through the tree using UCB scores, the MCTS will chose
		# the best move to play by checking which node was the most visited
		best_child = None
		best_visit_count = -math.inf
		num_best_found = 0
		num_children = len(root_node.children)
		total_visit_count = root_node.total_visit_count
		
		# Array to store the number of visits per move, a move is represented
		# by its coordinate and several stacks representing where it can go
		move_array = np.zeros((N_ROW, N_COL, N_ACTION_STACK))

		# For each children of the root, so for each legal moves
		for i in range(num_children):
			child = root_node.children[i]
			visit_count = child.visit_count
			move_from_parent = child.move_from_parent

			# Getting coordinates of the move
			to = move_from_parent.to()
			from_ = getattr(move_from_parent, "from")() # trick to use the from method (reserved in python)
			
			# Getting the action type as an int :
			# 0 1 2 3... depending the from and to
			action_index = index_action(from_, to)
			# <int(from_/N_ROW), from_%N_ROW> represent the position of the
			# pawn that chosed action <action_index> to go in position <to>
			move_array[int(from_/N_ROW), from_%N_ROW, action_index] = visit_count/total_visit_count
	
			# Keep track of the best child according to the number of visits
			if visit_count > best_visit_count:
				best_visit_count = visit_count
				best_child = child
				num_best_found = 1
				
			# Chose one child randomly if there is several optimal children
			elif visit_count == best_visit_count:
				rand = random.randint(0, num_best_found + 1)
				if rand == 0:
					best_child = child
				num_best_found += 1
				
		# Get the representation of the current state for the future NN training
		state = format_state(root_node.context)
				
		# Returns the move to play in the real game and the moves
		# associated to their probability distribution
		return best_child.move_from_parent, state, move_array


class Node:
	def __init__(self, parent, move_from_parent, context):
		# Variables to build the tree
		self.children = []
		self.parent = parent

		# Variables for MCTS decision
		self.visit_count = 0
		
		# Variables for normalization
		self.total_visit_count = 0

		# Variables for UCB score computation
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

