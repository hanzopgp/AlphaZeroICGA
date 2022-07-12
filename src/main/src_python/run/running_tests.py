import os
import sys
import time
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.getcwd()+"/src_python")


from settings.config import PLAYER1, PLAYER2, NUM_TESTS
from optimization.precompute import precompute_all 
from mcts.mcts_uct_alphazero import MCTS_UCT_alphazero


######### Here is the class called in the java file to run tests #########	

class RunningTests:
	# This function is called from java in RunningTestsWithPython.java
	def run_test(self, game, trial, context, ludii_ai):
		# Init our agent
		alphazero_ai = MCTS_UCT_alphazero()
		alphazero_ai.init_ai(game, PLAYER1)
		
		# Precompute some functions
		alphazero_ai.set_precompute(precompute_all())
		
		# Declare some variables for statistics
		alphazero_ai_win, ludii_ai_win, draw, total = 0, 0, 0, 0
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
					move, _, _ = alphazero_ai.select_action(game, context, 1, -1, max_depth=-1)
				else:
					move = ludii_ai.selectAction(game, context)
					
				context.game().apply(context, move)
		
			# Compute ranking
			ranking = trial.ranking()
			
			# Check who won and print some stats + rewards
			if ranking[PLAYER1] > ranking[PLAYER2]:
				ludii_ai_win += 1
			elif ranking[PLAYER1] < ranking[PLAYER2]:
				alphazero_ai_win += 1
			else:
				draw += 1
			total += 1
		
			duration[i] = time.time() - start_time
			
		# Print some trial stats
		print("* AI1 winrate:", alphazero_ai_win/total)
		print("* AI2 winrate:", ludii_ai_win/total)
		print("* Draws:", draw/total)
		print("* Mean game duration", duration.mean())
		print("* Max game duration", duration.max())
			
		print("--> Tests are over")
			
