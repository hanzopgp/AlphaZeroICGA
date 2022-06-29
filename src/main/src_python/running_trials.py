import sys
import os
sys.path.append(os.getcwd()+"/src_python")


from config import *
from utils import *
from mcts_uct import MCTS_UCT


######### Here is the class called in the java file to run trials #########	

class RunningTrials:
	# Need to give a Java List object here, if we give 2 ais and make it a python array
	# it won't work and we get no java overload error
	def run_trial(self, game, trial, context, ais):
		#prof = cProfile.Profile()
		#prof.enable()

		# Init both agents
		mcts1 = MCTS_UCT()
		mcts2 = MCTS_UCT()
		mcts1.init_ai(game, PLAYER1)
		mcts2.init_ai(game, PLAYER2)
		
		# Declare some variables for statistics
		ai1_win = 0
		ai2_win = 0
		draw = 0
		total = 0
		duration = np.zeros(NUM_TRIALS)
		
		# Declare some variables to save the dataset
		idx_sample = 0
		X = np.zeros((MAX_SAMPLE, N_ROW, N_COL, N_REPRESENTATION_STACK))
		y_distrib = np.zeros((MAX_SAMPLE, N_ROW, N_COL, N_ACTION_STACK))
		y_values = []
		
		print("--> Running", NUM_TRIALS, "games")
		
		breaker = False
		
		# Main trial loop, we play one game per trial
		for i in range(NUM_TRIALS):
		
			if breaker: break
		
			start_time = time.time()
			game.start(context)
			model = context.model()
			X_mover = []
			move_check = []
			
			# Main game loop			
			while not trial.over():
				#print("====================NEW MOVE====================")
				
				# Sometimes the game is way too long and has to be stopped
				# and considered as a draw
				if time.time() - start_time  > MAX_GAME_DURATION:
					print("--> Ended one game because it was too long")
					break
					
				if idx_sample >= MAX_SAMPLE:
					breaker=True
					break
				
				# Keep track of the mover
				mover = context.state().mover()
				X_mover.append(mover)
				
				# Move with custom python AI and save the move distribution
				if mover == 1:
					# Uncomment next line if using Ludii AI object
					#move = ais.get(mover).selectAction(game, context)
					# Get the optimal move and number of visits per move
					move, state, tmp_arr_move = mcts1.select_action(game, context, THINKING_TIME_AGENT1, MAX_ITERATION_AGENT1, max_depth=-1)
				else:
					move, state, tmp_arr_move = mcts2.select_action(game, context, THINKING_TIME_AGENT2, MAX_ITERATION_AGENT2, max_depth=-1)
					
				if not move.isForced(): # Avoid to add useless moves when games is over
					# Save X state
					X[idx_sample] = state
					# Apply softmax on the visit count to get a distribution from the MCTS
					y_distrib[idx_sample] = softmax(tmp_arr_move)
					idx_sample += 1	
				
				move_check.append(move)
				
				#print("Move played:", move)				
				context.game().apply(context, move)
	
			# Compute ranking
			ranking = trial.ranking()
			
			# Check who won and print some stats + rewards
			reward1 = 0
			reward2 = 0
			if ranking[PLAYER1] > ranking[PLAYER2]:
				reward1 = -1
				reward2 = 1
				ai2_win += 1
			elif ranking[PLAYER1] < ranking[PLAYER2]:
				reward1 = 1
				reward2 = -1
				ai1_win += 1
			else:
				reward1 = 0
				reward2 = 0
				draw += 1
			total += 1

			# Use reward as labels for our dataset
			for j in range(len(X_mover)):
				if X_mover[j] == PLAYER1:
					y_values.append(reward1)
				elif X_mover[j] == PLAYER2:
					y_values.append(reward2)
					
			duration[i] = time.time() - start_time
			
		# Keep the interesting values only
		X = X[:idx_sample]
		y_values = np.array(y_values)
		y_values = y_values[:idx_sample]
		y_distrib = y_distrib[:idx_sample]
		
		# Test print
		#idx_move = 15
		#for i in range(X.shape[3]):
		#	print(X[idx_move,:,:,i])
		
		# Print our generated dataset shapes
		print("* X shape", X.shape)	
		print("* y_values shape", y_values.shape)
		print("* y_distrib shape", y_distrib.shape)
		
		# Print some trial stats
		print("* AI1 winrate:", ai1_win/total)
		print("* AI2 winrate:", ai2_win/total)
		print("* Draws:", draw/total)
		print("* Mean game duration", duration.mean())
		print("* Max game duration", duration.max())
		
		# Save values to CSV
		add_to_dataset(X, y_values, y_distrib)

		#prof.disable()
		#prof.print_stats()
