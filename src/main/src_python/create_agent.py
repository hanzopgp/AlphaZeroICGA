import sys
sys.path.append("/home/durande/Bureau/AlphaZeroICGA/src/main/src_python")
from utils import load_nn
from mcts_uct import MCTS_UCT


import math
import numpy as np
import random
import time


class Agent:
	def __init__(self):
		self._player_id = -1

	# Here we will load the weights of the best model in order to use it
	# in the MCTS later
	def init_ai(self, game, player_id):
		self._player_id = player_id
		self.agent = MCTS_UCT()
		self.agent.init_ai(game, player_id)

	# Here we have the overriding method which will chose the action in the
	# ludii software when playing games. It will just select the best action
	# depending the MCTS and the loaded pre-trained model
	def select_action(self, game, context, max_seconds, max_iterations, max_depth):
		move, _, _ = self.agent.select_action(game, context, max_seconds, max_iterations, max_depth=max_depth)
		return move                    
        

