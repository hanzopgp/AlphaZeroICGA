import os
import sys
import time
import math
import cProfile
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.getcwd()+"/src_python")


from settings.config import MODEL_PATH, MAX_SAMPLE, MAX_GAME_MOVES, MAX_GAME_DURATION, VANILLA_EPISODE_MULTIPLIER, DEBUG_PRINT, PROFILING_ACTIVATED, NUM_EPISODE, PLAYER1, PLAYER2, THINKING_TIME_AGENT, MAX_ITERATION_AGENT
from settings.game_settings import GAME_NAME, N_ROW, N_COL, N_REPRESENTATION_STACK, N_ACTION_STACK
from optimization.precompute import precompute_all
from mcts.mcts_uct_vanilla import MCTS_UCT_vanilla
from mcts.mcts_uct_alphazero import MCTS_UCT_alphazero
from utils import add_to_dataset, get_random_hash, check_if_first_step


######### Here is the class called in the java file to run trials #########	

class RunningTrials:
	# This function is called from java in RunningTrialsWithPython.java
	def run_trial(self, game, trial, context, force_vanilla):
		if PROFILING_ACTIVATED:
			prof = cProfile.Profile()
			prof.enable()

		# Init both agents
		if check_if_first_step() or force_vanilla:
			mcts1 = MCTS_UCT_vanilla()
			mcts2 = MCTS_UCT_vanilla()
			n_episode = NUM_EPISODE * VANILLA_EPISODE_MULTIPLIER
			#max_it = MAX_ITERATION_AGENT * 3
			max_it = MAX_ITERATION_AGENT 
		else:
			mcts1 = MCTS_UCT_alphazero()
			mcts2 = MCTS_UCT_alphazero()
			n_episode = NUM_EPISODE
			max_it = MAX_ITERATION_AGENT
		mcts1.init_ai(game, PLAYER1)
		mcts2.init_ai(game, PLAYER2)
		
		# Precompute some functions
		# tmp_context = context.deepCopy()
		# game.start(tmp_context)
		# position_example = tmp_context.state().owned().positions(PLAYER1)
		# Precompute all expensive functions
		#pre_action_index, pre_reverse_action_index, pre_coords, pre_3D_coords = precompute_all()	
		#champion_mcts.set_precompute(pre_action_index, pre_reverse_action_index, pre_coords, pre_3D_coords)
		#outsider_mcts.set_precompute(pre_action_index, pre_reverse_action_index, pre_coords, pre_3D_coords)
		pre_coords = precompute_all()	
		mcts1.set_precompute(pre_coords)
		mcts2.set_precompute(pre_coords)
		
		# Declare some variables for statistics
		ai1_win, ai2_win, draw, total = 0, 0, 0, 0
		duration = np.zeros(n_episode)
		
		# Declare some variables to save the dataset
		idx_sample = 0
		X = np.zeros((MAX_SAMPLE, N_ROW, N_COL, N_REPRESENTATION_STACK))
		y_values = []
		
		print("--> Running", n_episode, "episodes")
		
		breaker = False
		
		# Main trial loop, we play one game per trial
		for i in range(n_episode):
		
			if breaker: break
			
			start_time = time.time()
			game.start(context)
			
			X_mover = []
			move_check = []
			
			stop_time = math.inf if MAX_GAME_DURATION < 0 else MAX_GAME_DURATION
			n_moves = 0
			
			# Main game loop			
			while not trial.over():
				wall_positions = []

				if time.time() - start_time  > stop_time:
					if DEBUG_PRINT: print("--> Ended one game because it was too long")
					break
					
				if idx_sample >= MAX_SAMPLE:
					breaker=True
					if DEBUG_PRINT: print("--> Ended one game because the dataset is full")
					break

				if n_moves >= MAX_GAME_MOVES:
					if DEBUG_PRINT: print("--> Ended one game because there was too much moves")
					break
				
				# Keep track of the mover
				mover = context.state().mover()
				X_mover.append(mover)

				if GAME_NAME == "Quoridor":
					mcts1.set_wall_positions(wall_positions)
					mcts2.set_wall_positions(wall_positions)
				
				# Move with custom python AI and save the move distribution
				if mover == 1:
					move, state = mcts1.select_action(game, context, THINKING_TIME_AGENT, max_it, max_depth=-1)
				else:
					move, state = mcts2.select_action(game, context, THINKING_TIME_AGENT, max_it, max_depth=-1)
					
				n_moves += 1

				# Avoid to add useless moves when games is over
				if not move.isForced(): 
					# Save X state
					X[idx_sample] = state
					idx_sample += 1	
				
				move_check.append(move)
				
				# print("Move played:", move)
				
				if GAME_NAME == "Quoridor" and move.toType() == "Edge":
					wall_positions.append(move.to())			

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

		if DEBUG_PRINT:
			# Print our generated dataset shapes
			print("* X shape", X.shape)	
			print("* y_values shape", y_values.shape)
			
			# Print some trial stats
			print("* AI1 winrate:", ai1_win/total)
			print("* AI2 winrate:", ai2_win/total)
			print("* Draws:", draw/total)
			print("* Mean game duration", duration.mean())
			print("* Max game duration", duration.max())
			
		print("--> Episodes are over")
			
		# Save values to dataset
		add_to_dataset(X, y_values, get_random_hash())

		if PROFILING_ACTIVATED:
			prof.disable()
			prof.print_stats()
			prof.dump_stats("profiler_stats.pstats")
		

	
