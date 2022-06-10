import math
import numpy as np
import random
import time


class RunningTrials:
	def __init__(self):
		pass
		
	# Need to give a Java List object here, if we give 2 ais and make it a python array
	# it won't work and we get no java overload error
	def run(self, game, trial, context, ais):
		num_trials = 10
		thinking_time = 1.0
		
		# Remove null since we can't use it in python
		# if we don't remove then we need to avoid the init on first element
		#ais.remove(0) 
		
		for i in range(num_trials):
			game.start(context)
			print("Starting a new trial!")
			
			# Avoiding error since we can't init a nonetype object in python
			for p in range(1, game.players().count()):
				ais.get(p).initAI(game, p)
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
				opp_mover = 2 if mover==1 else 2
				# Here thinking_time doesn't work, maybe because it isn't a type double
				move = ais.get(mover).selectAction(game, context)
				context.game().apply(context, move)
				
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

			for p in range(game.players().count()):
				# Need to use p+1 because we removed the null object at index 0 in our ais array
				print("Agent ", context.state().playerToAgent(p+1), " achieved rank: ", ranking[p+1])
			print("\n")

