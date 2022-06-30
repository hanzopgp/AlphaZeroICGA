import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.getcwd()+"/src_python")


import jpype
#import snappy
#from snappy import jpy


from config import *
from utils import *
from mcts_uct import MCTS_UCT


######### Here is the class called in the java file to run trials #########	

class RunningTrials:

	def test(self, a, b):
		return a+b

	def run_parallel_trials(self, games, trials, contexts, ais, n_objects):
		X = [] 
		y_values = [] 
		y_distrib = []
		
		#run_trial(games[0], trials[0], contexts[0], ais[0])
		
		print("--> Starting multi threading with max_workers :", MAX_WORKERS)		
		with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
		
			#if jpype.isJVMStarted() and not jpype.isThreadAttachedToJVM():
			#	jpype.attachThreadToJVM()
			#else:
			#	print("--> Didn't manage to attach the thread to the JVM")
			#	return
		
			fs = []
			for i in range(MAX_WORKERS):
				fs.append(executor.submit(self.run_trial, games[i], trials[i], contexts[i], ais[i]))
			#fs = {executor.submit(self.test, n, n): n for n in range(10)}
			
			for i, f in enumerate(concurrent.futures.as_completed(fs)):
				try:
					print(f.result())
					#print("iiiii",i)
					X_, y_values_, y_distrib_ = f.result()
					X.append(X_)
					y_values.append(y_values_)
					y_distrib.append(y_distrib_)
				except Exception as e:
					print("--> Exception:", e)
				else:
					print("--> Trial number", i, "is over")	
							
		print("--> Trials were run on parallel thanks to", MAX_WORKERS, "workers !")
		
		X, y_values, y_distrib = np.array(X, dtype=object), np.array(y_values, dtype=object), np.array(y_distrib, dtype=object)
		print(X.shape)
		print(y_values.shape)
		print(y_distrib.shape)
		
		#add_to_dataset(X.reshape(-1, N_ROW, N_COL, N_REPRESENTATION_STACK), 
		#	       y_values.reshape(-1), 
		#	       y_distrib.reshape(-1, N_ROW, N_COL, N_ACTION_STACK))

	def run_trial(self, game, trial, context, ais):
		#prof = cProfile.Profile()
		#prof.enable()
		
		
		#if not jpype.isJVMStarted():
			#jpy.destroyJVM()
			#jpy.create_jvm(snappy.jpyutil.get_jvm_options())
			#jpype.startJVM(jpype.getDefaultJVMPath())
			
		
		
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
		
		# Save values to dataset
		add_to_dataset(X, y_values, y_distrib)

		#prof.disable()
		#prof.print_stats()
		
		#return X, y_values, y_distrib
		
	
