import numpy as np

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
	move_distrib = softmax(visit_count_arr)
	
def export_dataset_csv():
	pass
	
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

######### Here is the class called in the java file to run trials #########	

class RunningTrials:
	def __init__(self):
		self.NUM_TRIALS = 1
		
	# Need to give a Java List object here, if we give 2 ais and make it a python array
	# it won't work and we get no java overload error
	def run(self, game, trial, context, ais):
		# Remove null since we can't use it in python
		# if we don't remove then we need to avoid the init on first element
		#ais.remove(0) 
		
		# Can't do that since our python object can't be in a java object list
		#ais.remove(2)
		#ais.add(MCTS_UCT())
		
		mcts1 = MCTS_UCT()
		mcts1.init_ai(game, 1)
		mcts2 = MCTS_UCT()
		mcts2.init_ai(game, 2)
		
		ai1_win = 0
		ai2_win = 0
		draw = 0
		total = 0
		
		final_move_arr1 = []
		final_move_arr2 = []
		
		for i in range(self.NUM_TRIALS):
			game.start(context)
			
			# Avoiding error since we can't init a nonetype object in python
			#for p in range(1, game.players().count()):
			#	ais.get(p).initAI(game, p)
			
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
				#opp_mover = 2 if mover==1 else 2
				# Here thinking_time doesn't work, maybe because it isn't a type double
				#move = ais.get(mover).selectAction(game, context, thinking_time)
				#move = ais.get(mover).selectAction(game, context)
				#context.game().apply(context, move)
				
				# Move with custom python AI and save the move distribution
				#print("Mover:", mover)
				if mover == 1:
					#move = ais.get(mover).selectAction(game, context)
					
					# Get the optimal move and number of visits per move
					move, tmp_arr_move1 = mcts1.select_action(game, context, 0.1, -1, -1)
					# Sort moves by their number 
					tmp_arr_move1 = tmp_arr_move1[tmp_arr_move1[:, 0].argsort()]
					# Apply softmax on the visit count to get a distribution from the MCTS
					tmp_arr_move1[:,1] = softmax(tmp_arr_move1[:,1])
					final_move_arr1.append(tmp_arr_move1)
				else:
					move, tmp_arr_move2 = mcts2.select_action(game, context, 0.1, -1, -1)
					tmp_arr_move2 = tmp_arr_move2[tmp_arr_move2[:, 0].argsort()]
					tmp_arr_move2[:,1] = softmax(tmp_arr_move2[:,1])
					final_move_arr2.append(tmp_arr_move2)
				print("Move played:", move)				
				context.game().apply(context, move)
				#print("*"*30)
					
				#print("===================== state functions =====================")
				#map_pos = context.state().owned().positions(mover)
				#map_pos_opp = context.state().owned().positions(opp_mover)
				#print("Map positions for mover:")
				#for j in range(len(map_pos)):
				#	for k in range(map_pos[j].size()):
				#		print(map_pos[j].get(k).site(), " ")
				#	print("\n")
				#print("Map positions for opponent:")
				#for j in range(len(map_pos_opp)):
				#	for k in range(map_pos_opp[j].size()):
				#		print(map_pos_opp[j].get(k).site(), " ")
				#	print("\n")

				#print("===================== game functions =====================")
				#print("Legal moves: ", context.game().moves(context).moves())
				
				#print("===================== trial functions =====================")
				#it = context.trial().reverseMoveIterator()
				#while it.hasNext():
				#	print(it.next())

				#print("**********************************************************************")
	
			ranking = trial.ranking()
			
			#for p in range(game.players().count()):
				# Need to use p+1 because we removed the null object at index 0 in our ais array
			#	print("Agent ", context.state().playerToAgent(p+1), " achieved rank: ", ranking[p+1])
			#print("\n")
			
			# Generate rewards for both players
			reward1 = 0
			reward2 = 0
		
			# Check who won and print some stats
			if ranking[1] > ranking[2]:
				reward1 = -1
				reward2 = 1
				ai2_win += 1
			elif ranking[1] < ranking[2]:
				reward1 = 1
				reward2 = -1
				ai1_win += 1
			else:
				reward1 = 0
				reward2 = 0
				draw += 1
			total += 1
			
			# Update the rewards to build the dataset
			final_move_arr1 = np.array(final_move_arr1)
			final_move_arr2 = np.array(final_move_arr2)
			
			# Do no think I should reshape here because we want to keep the distribution per state
			#final_move_arr1 = final_move_arr1.reshape(-1, 3)
			#final_move_arr2 = final_move_arr2.reshape(-1, 3)
			
			#print(final_move_arr1.shape)
			#print(final_move_arr2.shape)
			final_move_arr1[:,:,2] = reward1
			final_move_arr2[:,:,2] = reward2
		
		# Print some stats
		print("AI1 winrate:", ai1_win/total)
		print("AI2 winrate:", ai2_win/total)
		print("Draws:", draw/total)

		print("DATASET GENERATED FOR PLAYER 1")
		print(final_move_arr1)
		print("DATASET GENERATED FOR PLAYER 2")
		print(final_move_arr2)

		
