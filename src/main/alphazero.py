import os
import sys
from subprocess import Popen


def decide_if_switch_model(winners_file):
	with open(winners_file, "r") as file:
		for last_line in file:
			pass
		# If outsider model won, outsider becomes champion and we can go back to mcts_trial
		if last_line[0] == "O":
			return True
		# If the champion won we need to train the model again without executing mcts_trial
		else:
			return False
			
def parallelize_command(command, n):
	string = "ant -q " + command + " &"
	for _ in range(n-1):
		string += "ant -q " + command + " &"
	return string + " wait"
			
def init():
	print("********************************************************************************************")
	print("****************************************INITIALIZING****************************************")
	print("********************************************************************************************")
	Popen("ant clean", shell=True).wait()
	print("********************************************************************************************")
	print("*****************************************COMPILING******************************************")
	print("********************************************************************************************")
	Popen("ant build", shell=True).wait()
	
def conclude():
	print("********************************************************************************************")
	print("*****************************************CREATING AGENT*************************************")
	print("********************************************************************************************")
	Popen("ant create_agent", shell=True).wait()
	print("********************************************************************************************")
	print("*****************************************AGENT READY****************************************")
	print("********************************************************************************************")
	
def main_loop(alphazero_iteration, trial_activated, winners_file):
	if trial_activated: # We train the outsider until it wins against the champion
		print("********************************************************************************************")
		print("****************************************MCTS TRIALS*****************************************")
		print("********************************************************************************************")
		Popen(parallelize_command("mcts_trials", n_workers), shell=True).wait()
		
		print("********************************************************************************************")
		print("***************************************MERGING DATASETS*************************************")
		print("********************************************************************************************")
		Popen("python3 src_python/scripts/merge_datasets.py", shell=True).wait()
		
	print("********************************************************************************************")
	print("***************************************TRAINING MODEL***************************************")
	print("********************************************************************************************")
	Popen("python3 src_python/train_model.py", shell=True).wait()
	
	if alphazero_iteration >= 1: # If it's the first step we won't go for a dojo since there is only one model ready	 
		print("********************************************************************************************")
		print("*****************************************MCTS DOJO******************************************")
		print("********************************************************************************************")
		Popen(parallelize_command("mcts_dojo", n_workers), shell=True).wait()
		
		print("********************************************************************************************")
		print("****************************************MERGING TXTS****************************************")
		print("********************************************************************************************")
		Popen("python3 src_python/scripts/merge_txts.py", shell=True).wait()
		
		trial_activated = decide_if_switch_model(winners_file)
		if trial_activated:
			print("********************************************************************************************")
			print("**************************************SWITCHING MODELS***************************************")
			print("********************************************************************************************")
			Popen("python3 src_python/scripts/switch_model.py", shell=True).wait()
			
	alphazero_iteration += 1
			
if __name__ == '__main__':
	winners_file="./models/save_winners.txt"
	alphazero_iteration=0
	trial_activated=True
	n_iteration = int(sys.argv[1])
	n_workers = int(sys.argv[2])

	init()	

	while(alphazero_iteration < n_iteration):
		print("============================================================================================")
		print("==================================ITERATION ALPHAZERO", alphazero_iteration, "====================================")
		print("============================================================================================")
		main_loop(alphazero_iteration, trial_activated, winners_file)
		
	conclude()
	
