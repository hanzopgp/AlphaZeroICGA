import os
import sys
import math
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.getcwd()+"/src_python")


from settings.config import PLAYER1, PLAYER2, NUM_DOJO, MAX_ITERATION_AGENTS_DOJO, THINKING_TIME_AGENTS_DOJO, MAX_GAME_DURATION, DEBUG_PRINT, MAX_GAME_MOVES
from optimization.precompute import precompute_all 
from mcts.mcts_uct_alphazero import MCTS_UCT_alphazero
from utils import write_winner, get_random_hash


######### Here is the class called in the java file to run dojo #########	

class RunningDojos:
	def run_dojo(self, game, trial, context):
		# The champion MCTS player gets the best model untill now
		champion_mcts = MCTS_UCT_alphazero(dojo=True, model_type="champion")
		champion_mcts.init_ai(game, PLAYER1)
		# The outsider MCTS player gets the latest model which has to be evaluated
		outsider_mcts = MCTS_UCT_alphazero(dojo=True, model_type="outsider")
		outsider_mcts.init_ai(game, PLAYER2)
	
		pre_action_index, pre_reverse_action_index, pre_coords, pre_3D_coords = precompute_all()		
		champion_mcts.set_precompute(pre_action_index, pre_reverse_action_index, pre_coords, pre_3D_coords)
		outsider_mcts.set_precompute(pre_action_index, pre_reverse_action_index, pre_coords, pre_3D_coords)

		# Declare some variables for statistics
		champion_mcts_win, outsider_mcts_win, draw, total = 0, 0, 0, 0
		duration = np.zeros(NUM_DOJO)

		print("--> Running", NUM_DOJO, "games")
		
		for i in range(NUM_DOJO):
			start_time = time.time()
			game.start(context)
			
			model = context.model()
			
			stop_time = math.inf if MAX_GAME_DURATION < 0 else MAX_GAME_DURATION
			cpt_move
			
			# Main game loop			
			while not trial.over():
				# Sometimes the game is way too long and has to be stopped
				# and considered as a draw
				if time.time() - start_time  > stop_time:
					print("--> Ended one game because it was too long")
					break

				if cpt_move >= MAX_GAME_MOVES:
					if DEBUG_PRINT: print("--> Ended one game because there was too much moves")
					break
					
				if context.state().mover() == 1:
					move, state, tmp_arr_move = champion_mcts.select_action(game, context, THINKING_TIME_AGENTS_DOJO, MAX_ITERATION_AGENTS_DOJO, max_depth=-1)
				else:
					move, state, tmp_arr_move = outsider_mcts.select_action(game, context, THINKING_TIME_AGENTS_DOJO, MAX_ITERATION_AGENTS_DOJO, max_depth=-1)

				cpt_move += 1
				
				context.game().apply(context, move)
	
			# Compute ranking
			ranking = trial.ranking()
			
			# Check who won
			if ranking[PLAYER1] > ranking[PLAYER2]:
				outsider_mcts_win += 1
			elif ranking[PLAYER1] < ranking[PLAYER2]:
				champion_mcts_win += 1
			else:
				draw += 1
			total += 1

			duration[i] = time.time() - start_time
		
		if DEBUG_PRINT:
			# Print some dojo stats
			print("* Champion AI winrate:", champion_mcts_win/total)
			print("* Outsider AI winrate:", outsider_mcts_win/total)
			print("* Draws:", draw/total)
			print("* Mean game duration", duration.mean())
			print("* Max game duration", duration.max())
		
		# Return 1 or -1 depending the winner so we can decide what
		# is the next step to do in our script
		write_winner(outsider_mcts_win/total, get_random_hash())
		
