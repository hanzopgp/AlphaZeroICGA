import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.getcwd()+"/src_python")


from config import *
from utils import *
from mcts_uct import MCTS_UCT


######### Here is the class called in the java file to run dojo #########	

class Dojo:
	# Need to give a Java List object here, if we give 2 ais and make it a python array
	# it won't work and we get no java overload error
	def run_dojo(self, game, trial, context, ais):
		# The champion MCTS player gets the best model untill now
		champion_mcts = MCTS_UCT(dojo=True, model_type="champion")
		champion_mcts.init_ai(game, PLAYER1)
		# The outsider MCTS player gets the latest model which has to be evaluated
		outsider_mcts = MCTS_UCT(dojo=True, model_type="outsider")
		outsider_mcts.init_ai(game, PLAYER2)
		
		# Declare some variables for statistics
		champion_mcts_win = 0
		outsider_mcts_win = 0
		draw = 0
		total = 0
		duration = np.zeros(NUM_DOJO)
		
		print("--> Running", NUM_DOJO, "games")
		
		for i in range(NUM_DOJO):
			start_time = time.time()
			game.start(context)
			model = context.model()
			
			# Main game loop			
			while not trial.over():
				# Sometimes the game is way too long and has to be stopped
				# and considered as a draw
				if time.time() - start_time  > MAX_GAME_DURATION:
					print("--> Ended one game because it was too long")
					break
					
				if context.state().mover() == 1:
					move, state, tmp_arr_move = champion_mcts.select_action(game, context, -1, max_iterations=MAX_ITERATION_AGENTS_DOJO, max_depth=-1)
				else:
					move, state, tmp_arr_move = outsider_mcts.select_action(game, context, -1, max_iterations=MAX_ITERATION_AGENTS_DOJO, max_depth=-1)
				
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
		
