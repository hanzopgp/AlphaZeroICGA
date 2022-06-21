from src_python.config import *
from src_python.utils import *
from src_python.mcts_uct import MCTS_UCT


######### Here is the class called in the java file to run dojo #########	

class Dojo:
	# Need to give a Java List object here, if we give 2 ais and make it a python array
	# it won't work and we get no java overload error
	def run_dojo(self, game, trial, context, ais):
		# The base MCTS player gets the latest working model
		base_mcts = MCTS_UCT(dojo=True, ALPHAZERO_ITERATION-1)
		base_mcts.init_ai(game, PLAYER1)
		# The outsider MCTS player gets the latest model which has to be evaluated
		outsider_mcts = MCTS_UCT(dojo=True, ALPHAZERO_ITERATION)
		outsider_mcts.init_ai(game, PLAYER2)
		
		# Declare some variables for statistics
		base_mcts_win = 0
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
				#print("====================NEW MOVE====================")
				
				# Sometimes the game is way too long and has to be stopped
				# and considered as a draw
				if time.time() - start_time  > MAX_GAME_DURATION:
					print("--> Ended one game because it was too long")
					break
					
				if context.state().mover() == 1:
					move, state, tmp_arr_move = base_mcts.select_action(game, context, THINKING_TIME_AGENT1, MAX_ITERATION_AGENT1, max_depth=-1)
				else:
					move, state, tmp_arr_move = outsider_mcts.select_action(game, context, THINKING_TIME_AGENT2, MAX_ITERATION_AGENT2, max_depth=-1)
				
				context.game().apply(context, move)
	
			# Compute ranking
			ranking = trial.ranking()
			
			# Check who won
			if ranking[PLAYER1] > ranking[PLAYER2]:
				outsider_mcts_win += 1
			elif ranking[PLAYER1] < ranking[PLAYER2]:
				base_mcts_win += 1
			else:
				draw += 1
			total += 1

			duration[i] = time.time() - start_time
		
		# Print some dojo stats
		print("* Base AI winrate:", base_mcts_win/total)
		print("* Outsider AI winrate:", outsider_mcts_win/total)
		print("* Draws:", draw/total)
		print("* Mean game duration", duration.mean())
		print("* Max game duration", duration.max())
		
