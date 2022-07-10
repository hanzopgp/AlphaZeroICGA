import os
import sys
import math
import time
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.getcwd()+"/src_python")


from settings.config import PLAYER1, PLAYER2, NUM_TESTS
from optimization.precompute import precompute_all 
from mcts.mcts_uct_alphazero import MCTS_UCT_alphazero
from mcts.mcts_uct_vanilla import MCTS_UCT_vanilla


######### Here is the class called in the java file to run tests #########	

class RunningTests:
	# This function is called from java in RunningTestsWithPython.java
	def run_test(self, game, trial, context):
		# Init our agent
		alphazero_ai = MCTS_UCT_alphazero()
		vanilla_mcts = MCTS_UCT_vanilla()
		alphazero_ai.init_ai(game, PLAYER1)
		vanilla_mcts.init_ai(game, PLAYER2)
		
		# Precompute some functions
		pre_action_index, pre_reverse_action_index, pre_coords, pre_3D_coords = precompute_all()
		alphazero_ai.set_precompute(pre_action_index, pre_reverse_action_index, pre_coords, pre_3D_coords)
		vanilla_mcts.set_precompute(pre_action_index, pre_reverse_action_index, pre_coords, pre_3D_coords)
		
		# Declare some variables for statistics
		alphazero_ai_win, vanilla_mcts_win, draw, total = 0, 0, 0, 0
		duration = np.zeros(NUM_TESTS)
		
		print("--> Running", NUM_TESTS, "tests")
		
		# Main trial loop, we play one game per trial
		for i in range(NUM_TESTS):
		
			start_time = time.time()
			game.start(context)
			
			model = context.model()
			
			# Main game loop			
			while not trial.over():
				# Keep track of the mover
				mover = context.state().mover()
				
				if mover == 1:
					move, _, _ = alphazero_ai.select_action(game, context, -1, 30, max_depth=-1)
				else:
					move, _, _ = vanilla_mcts.select_action(game, context, -1, 30, max_depth=-1)
					
				context.game().apply(context, move)
		
			# Compute ranking
			ranking = trial.ranking()
			
			# Check who won and print some stats + rewards
			if ranking[PLAYER1] > ranking[PLAYER2]:
				vanilla_mcts_win += 1
			elif ranking[PLAYER1] < ranking[PLAYER2]:
				alphazero_ai_win += 1
			else:
				draw += 1
			total += 1
		
			duration[i] = time.time() - start_time
			
		# Print some trial stats
		print("* AlphaZero AI winrate:", alphazero_ai_win/total)
		print("* Vanilla MCTS winrate:", vanilla_mcts_win/total)
		print("* Draws:", draw/total)
		print("* Mean game duration", duration.mean())
		print("* Max game duration", duration.max())
			
		print("--> Tests are over")
			
