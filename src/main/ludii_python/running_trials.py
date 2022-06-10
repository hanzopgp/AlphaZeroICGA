import math
import numpy as np
import random
import time


class RunningTrials:
	def __init__(self):
		pass
		
	def run(self, game, trial, context, ais):
		num_trials = 10
		thinking_time = 1.0
		ais.remove(0) # Remove null since we can't use it in python
		
		for i in range(num_trials):
			game.start(context)
			print("Starting a new trial!")
			
			for p in range(game.players().count()):
				ais.get(p).initAI(game, p)
			model = context.model()
			
			while not trial.over():
				# Doesn't work if we send ais
				#game.playout(context,
				#             None, # ais
				#             1.0,  # thinking_time
				#             None, # playoutMoveSelector
				#             0,    # max_num_biased_actions
				#             -1,   # max_num_playout_actions
				#             None) # random
				             
				# Doesn't work if we send ais	
				print(ais.get(0))
				print(ais.get(1))
				model.startNewStep(context, ais, game.players().count())
				
				# Doesn't work if we use ais
				#mover = context.state().mover()
				#opp_mover = 2 if mover==1 else 2
				#print("Mover: ", mover)
				#print("Opponent: ", opp_mover)
				#move = ais.get(mover).selectAction(game, context, thinking_time)
				#context.game().apply(context, move)
				
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

