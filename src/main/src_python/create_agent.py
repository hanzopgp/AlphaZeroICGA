from mcts_uct_alphazero import MCTS_UCT_alphazero


class Agent:
	def __init__(self):
		self._player_id = -1

	# Here we init our agent, it will load the model and prepare for inference
	def init_ai(self, game, player_id):
		self._player_id = player_id
		self.agent = MCTS_UCT_alphazero()
		self.agent.init_ai(game, player_id)

	# Here we have the overriding method which will chose the action in the
	# ludii software when playing games. It will just select the best action
	# depending the MCTS and the loaded pre-trained model
	def select_action(self, game, context, max_seconds, max_iterations, max_depth):
		move, _, _ = self.agent.select_action(game, context, max_seconds, max_iterations, max_depth=max_depth)
		return move                    
        

