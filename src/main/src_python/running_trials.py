import numpy as np

from src_python.config import * 
from src_python.mcts_uct import MCTS_UCT


######### Here are the function to build the dataset for AlphaZero #########	
# We need functions to keep track of the state representation, the future reward
# and the distribution of moves for the rote_node. It will help us
# build a dataset with X and y in order to train a DCNN model which will
# estimate a policy and a value. Once it has been train, we inject the outputs
# into the MCTS, which will now follow the policy and values, thus expanding
# differently and giving new move distribution and future rewards.

def track_state(context):
	pass
	
def track_values():
	pass
	
def build_dataset(visit_count_arr):
	pass
	
def export_dataset_csv():
	pass
	
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

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
		X = np.zeros((MAX_SAMPLE, 2*N_TIME_STEP, N_ROW, N_COL))
		X_mover = np.zeros((MAX_SAMPLE))
		y_distrib = np.zeros((MAX_SAMPLE, N_ROW*N_COL, 2)) # 2 because moves -> softmax
		y_values = np.zeros((MAX_SAMPLE))
		
		for i in range(NUM_TRIALS):
			game.start(context)
			model = context.model()
			
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
				X_mover[idx_sample] = mover
				
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
					tmp_arr_move = tmp_arr_move[tmp_arr_move[:, 0].argsort()]
					# Apply softmax on the visit count to get a distribution from the MCTS
					tmp_arr_move[:,1] = softmax(tmp_arr_move[:,1])
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
			for j in range(X_mover.shape[0]):
				if X_mover[j] == PLAYER1:
					y_values[j] = reward1
				elif X_mover[j] == PLAYER2:
					y_values[j] = reward2
			
		# Keep the interesting values only
		# TOO MUCH SAMPLE NEED TO FIX THAT		
		X = X[:idx_sample]
		y_values = y_values[:idx_sample]
		y_distrib = y_distrib[:idx_sample]
		#print(X)
		#print(y_values)
		#print(y_distrib)
		
		# Print our generated dataset shapes
		print("X", X.shape)			
		print("y_values", y_values.shape)
		print("y_distrib", y_distrib.shape)
		
		# Print some trial stats
		print("AI1 winrate:", ai1_win/total)
		print("AI2 winrate:", ai2_win/total)
		print("Draws:", draw/total)

		
