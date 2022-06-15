from src_python.config import PLAYER1, PLAYER2, MAX_SAMPLE, N_TIME_STEP, N_ROW, N_COL, N_LEVELS, NUM_TRIALS, THINKING_TIME_AGENT1, THINKING_TIME_AGENT2
from src_python.utils import *
from src_python.mcts_uct import MCTS_UCT


######### Here is the class called in the java file to run trials #########	

class RunningTrials:
	# Need to give a Java List object here, if we give 2 ais and make it a python array
	# it won't work and we get no java overload error
	def run(self, game, trial, context, ais):
		mcts1 = MCTS_UCT()
		mcts2 = MCTS_UCT()
		mcts1.init_ai(game, PLAYER1)
		mcts2.init_ai(game, PLAYER2)
		
		ai1_win = 0
		ai2_win = 0
		draw = 0
		total = 0
		
		idx_sample = 0
		X = np.zeros((MAX_SAMPLE, 2*N_TIME_STEP, N_ROW, N_COL, N_LEVELS))
		y_distrib = np.zeros((MAX_SAMPLE, N_ROW*N_COL))
		y_values = []
		
		for i in range(NUM_TRIALS):
			game.start(context)
			model = context.model()
			
			X_mover = []
			
			while not trial.over():
				#print("====================NEW MOVE====================")
				
				# Plays whole game until the end
				#game.playout(context,
				#             ais, # ais
				#             1.0,  # thinking_time
				#             None, # playoutMoveSelector
				#             0,    # max_num_biased_actions
				#             -1,   # max_num_playout_actions
				#             None) # random
				             
				# Plays step by step
				#model.startNewStep(context, ais, game.players().count())
				
				# Move per move
				mover = context.state().mover()
				
				# Keep track of the mover
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
					#print(move)
					# Save X state
					X[idx_sample] = state
					# Sort moves by their number 
					#tmp_arr_move = tmp_arr_move[tmp_arr_move[:, 0].argsort()]
					# Apply softmax on the visit count to get a distribution from the MCTS
					#tmp_arr_move[:,1] = softmax(tmp_arr_move[:,1])
					tmp_arr_move = softmax(tmp_arr_move)
					y_distrib[idx_sample] = tmp_arr_move
					idx_sample += 1	
				
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
			
		y_values = np.array(y_values)
		
		# Keep the interesting values only
		X = X[:idx_sample]
		y_values = y_values[:idx_sample]
		y_distrib = y_distrib[:idx_sample]
		#print(X)
		#print(y_values)
		#print(y_distrib)
		
		# Print our generated dataset shapes
		print("X shape", X.shape)			
		print("y_values shape", y_values.shape)
		print("y_distrib shape", y_distrib.shape)
		
		# Print some trial stats
		print("AI1 winrate:", ai1_win/total)
		print("AI2 winrate:", ai2_win/total)
		print("Draws:", draw/total)
		
		# Save values to CSV
		add_to_dataset(X, y_values, y_distrib)

		
