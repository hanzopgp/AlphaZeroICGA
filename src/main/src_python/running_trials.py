from src_python.config import *
from src_python.utils import *
from src_python.mcts_uct import MCTS_UCT


######### Here is the class called in the java file to run trials #########	

class RunningTrials:
	# Need to give a Java List object here, if we give 2 ais and make it a python array
	# it won't work and we get no java overload error
	def run(self, game, trial, context, ais):
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
		X = np.zeros((MAX_SAMPLE, N_LEVELS*(2*N_TIME_STEP)+1, N_ROW, N_COL))
		y_distrib = np.zeros((MAX_SAMPLE, N_ROW*N_COL))
		y_values = []
		
		print("Running", NUM_TRIALS, "games")
		
		# Main trial loop, we play one game per trial
		for i in range(NUM_TRIALS):
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
					print("Ended one game because it was too long")
					break
				
				# Keep track of the mover
				mover = context.state().mover()
				X_mover.append(mover)
				
				# Move with custom python AI and save the move distribution
				if mover == 1:
					# Uncomment next line if using Ludii AI object
					#move = ais.get(mover).selectAction(game, context)
					# Get the optimal move and number of visits per move
					move, state, tmp_arr_move = mcts1.select_action(game, context, THINKING_TIME_AGENT1, -1, -1)
				else:
					move, state, tmp_arr_move = mcts2.select_action(game, context, THINKING_TIME_AGENT2, -1, -1)
					
				if not move.isForced(): # Avoid to add useless moves when games is over
					# Save X state
					X[idx_sample] = state
					# Apply softmax on the visit count to get a distribution from the MCTS
					tmp_arr_move = softmax(tmp_arr_move)
					y_distrib[idx_sample] = tmp_arr_move
					idx_sample += 1	
				
				move_check.append(move)
				
				#print("Move played:", move)				
				context.game().apply(context, move)
	
			# Compute ranking
			ranking = trial.ranking()
			
			# Check who won and print some stats + rewards
			reward1 = 0
			reward2 = 0
			if ranking[int(PLAYER1)] > ranking[int(PLAYER2)]:
				reward1 = -1
				reward2 = 1
				ai2_win += 1
			elif ranking[int(PLAYER1)] < ranking[int(PLAYER2)]:
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
				if X_mover[j] == int(PLAYER1):
					y_values.append(reward1)
				elif X_mover[j] == int(PLAYER2):
					y_values.append(reward2)
					
			duration[i] = time.time() - start_time
			
		# Keep the interesting values only
		X = X[:idx_sample]
		y_values = np.array(y_values)
		y_values = y_values[:idx_sample]
		y_distrib = y_distrib[:idx_sample]
		
		print("*"*40)
		print(X_mover[50])
		print(move_check[50])
		#print(X[50][0])
		#print(X[50][1])
		#print(X[50][2])
		for i in range(X[50].shape[0]):
			print(X[50][i])
		print(y_values[50])
		print(y_distrib[50])
		
		#print("*"*40)
		#print(X_mover[51])
		#print(move_check[51])
		#print(X[51][0])
		#print(X[51][1])
		#print(X[51][2])
		#print(y_values[51])
		#print(y_distrib[51])
		
		#print("*"*40)
		#print(X_mover[52])
		#print(move_check[52])
		#print(X[52][0])
		#print(X[52][1])
		#print(X[52][2])
		#print(y_values[52])
		#print(y_distrib[52])
		
		# Print our generated dataset shapes
		print("X shape", X.shape)	
		print("y_values shape", y_values.shape)
		print("y_distrib shape", y_distrib.shape)
		
		# Print some trial stats
		print("AI1 winrate:", ai1_win/total)
		print("AI2 winrate:", ai2_win/total)
		print("Draws:", draw/total)
		print("Mean game duration", duration.mean())
		print("Max game duration", duration.max())
		
		# Save values to CSV
		add_to_dataset(X, y_values, y_distrib)

		
