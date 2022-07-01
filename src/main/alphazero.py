import os
import sys
sys.path.append(os.getcwd()+"/src_python")
import re
from subprocess import Popen


from config import *
from 


def decide_if_switch_model():
	with open(WINNERS_FILE, "r") as file:
		print("--> Reading file :", WINNERS_FILE)
		for last_line in file:
			pass
		outsider_winrate = float(re.findall("\d+\.\d+", last_line)[0])
		print("--> Outsider winrate for last dojo :", outsider_winrate)
		# If outsider model won, outsider becomes champion and we can go back to mcts_trial
		if outsider_winrate >= OUTSIDER_MIN_WINRATE:
			return True
		# If the champion won we need to train the model again without executing mcts_trial
		return False
			
def parallelize_command(command, n):
	if not DEBUG_PRINT:
		string = "ant -q " + command + " &"
		for _ in range(n-1):
			string += "ant -q " + command + " &"
	else:
		string = "ant " + command + " &"
		for _ in range(n-1):
			string += "ant " + command + " &"
	return string + " wait"
			
def init():
	print("********************************************************************************************")
	print("*************************************** INITIALIZING ***************************************")
	print("********************************************************************************************")
	Popen("ant clean", shell=True).wait()
	print("********************************************************************************************")
	print("**************************************** COMPILING *****************************************")
	print("********************************************************************************************")
	Popen("ant build", shell=True).wait()
	
def conclude():
	print("********************************************************************************************")
	print("**************************************** CREATING AGENT ************************************")
	print("********************************************************************************************")
	Popen("ant create_agent", shell=True).wait()
	print("********************************************************************************************")
	print("**************************************** AGENT READY ***************************************")
	print("********************************************************************************************")
	
def main_loop(alphazero_iteration, trial_activated, n_iteration, n_workers):
	while(alphazero_iteration < n_iteration):
		print("============================================================================================")
		print("================================== ITERATION ALPHAZERO", alphazero_iteration, "===================================")
		print("============================================================================================")
		if trial_activated: # We train the outsider until it wins against the champion
			print("********************************************************************************************")
			print("*************************************** MCTS TRIALS ****************************************")
			print("********************************************************************************************")
			Popen(parallelize_command("mcts_trials", n_workers), shell=True).wait()
			
			print("********************************************************************************************")
			print("************************************** MERGING DATASETS ************************************")
			print("********************************************************************************************")
			Popen("python3 src_python/scripts/merge_datasets.py", shell=True).wait()
			
		print("********************************************************************************************")
		print("************************************** TRAINING MODEL **************************************")
		print("********************************************************************************************")
		Popen("python3 src_python/train_model.py", shell=True).wait()
		
		if alphazero_iteration >= 1: # If it's the first step we won't go for a dojo since there is only one model ready	 
			print("********************************************************************************************")
			print("**************************************** MCTS DOJO *****************************************")
			print("********************************************************************************************")
			Popen(parallelize_command("mcts_dojo", n_workers), shell=True).wait()
			
			print("********************************************************************************************")
			print("*************************************** MERGING TXTS ***************************************")
			print("********************************************************************************************")
			Popen("python3 src_python/scripts/merge_txts.py", shell=True).wait()
			
			trial_activated = decide_if_switch_model()
			if trial_activated:
				print("********************************************************************************************")
				print("************************************* SWITCHING MODELS **************************************")
				print("********************************************************************************************")
				Popen("python3 src_python/scripts/switch_model.py", shell=True).wait()
				
		alphazero_iteration += 1
			
if __name__ == '__main__':
	alphazero_iteration=0
	trial_activated=True
	n_iteration = int(sys.argv[1])
	n_workers = int(sys.argv[2])

	init()	

	main_loop(alphazero_iteration, trial_activated, n_iteration, n_workers)
		
	conclude()
	
