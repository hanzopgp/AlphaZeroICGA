import math
import numpy as np
import random
import time


class RunningTrials:
	def __init__(self):
		pass
		
	def run(self, game, trial, context, ai1, ai2):
		num_trials = 3
		thinking_time = 1
		
		ais = []
		ais.append(ai1)
		ais.append(ai1)
		ais.append(ai2)
		
		for i in range(num_trials):
			game.start(context)
			print("Starting a new trial!")
			
			for p in range(game.players().count()):
				ais[p].initAI(game, p)
	
			model = context.model()
			
			while not trial.over():
				print(context)
				print(ais)
				print(game.players().count())
				model.startNewStep(context, ais, game.players().count())
				print("===================== state functions =====================")
				
				mover = context.state().mover()
				opp_mover = 1 if mover==2 else 1
				print("Mover: ", mover)
				print("Opponent: ", opp_mover)
				
				map_pos = context.state().owned().positions(mover)
				map_pos_opp = context.state().owned().positions(opp_mover)
				print("Map positions for mover:")
				for j in range(len(map_pos)):
					for k in range(map_pos[j].size()):
						print(map_pos[j].get(k).site(), " ")
					print("\n")
				print("Map positions for opponent:")
				for j in range(len(map_pos_opp)):
					for k in range(map_pos_opp[j].size()):
						print(map_pos_opp[j].get(k).site(), " ")
					print("\n")

				print("===================== game functions =====================")
				print("Legal moves: ", context.game().moves(context).moves())
				
				print("===================== trial functions =====================")
				it = context.trial().reverseMoveIterator()
				while it.hasNext():
					print(it.next())

				print("**********************************************************************")
	
			ranking = trial.ranking()

			for p in range(game.players().count()):
				print("Agent ", context.state().playerToAgent(p), " achieved rank: ", ranking[p])
			print("\n")

