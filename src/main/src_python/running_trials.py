from src_python.mcts_uct import MCTS_UCT


class RunningTrials:
	def __init__(self):
		self.NUM_TRIALS = 25
		
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
		
		for i in range(self.NUM_TRIALS):
			game.start(context)
			
			# Avoiding error since we can't init a nonetype object in python
			#for p in range(1, game.players().count()):
			#	ais.get(p).initAI(game, p)
			
			model = context.model()
			
			while not trial.over():
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
				
				# Move with custom python AI + random java AI
				#print("Mover:", mover)
				if mover == 1:
					#move = ais.get(mover).selectAction(game, context)
					move = mcts1.select_action(game, context, 0.1, -1, -1)
				else:
					move = mcts2.select_action(game, context, 0.1, -1, -1)
				#print("Move played:", move)				
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
			
			
			if ranking[1] > ranking[2]:
				ai2_win += 1
			elif ranking[1] < ranking[2]:
				ai1_win += 1
			else:
				draw += 1
			total += 1
			
		print("AI1 winrate:", ai1_win/total)
		print("AI2 winrate:", ai2_win/total)
		print("Draws:", draw/total)	 
			

