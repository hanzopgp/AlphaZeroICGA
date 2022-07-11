import os
import sys
import re
from subprocess import Popen
sys.path.append(os.getcwd()+"/src_python")


import settings
from settings.config import WINNERS_FILE, OUTSIDER_MIN_WINRATE, DEBUG_PRINT, MODEL_PATH, ONNX_INFERENCE
from settings.game_settings import GAME_NAME


def decide_if_switch_model():
	with open(WINNERS_FILE, "r") as file:
		print("--> Reading file :", WINNERS_FILE)
		for last_line in file:
			pass
		outsider_winrate = float(re.findall("\d+\.\d+", last_line)[0])
		print("--> Outsider winrate for last dojo :", outsider_winrate)
		# If outsider model won, outsider becomes champion and we can go back to run_trials
		if outsider_winrate >= OUTSIDER_MIN_WINRATE:
			return True
		# If the champion won we need to train the model again without executing run_trials
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
	
def convert_models_onnx():
	print("********************************************************************************************")
	print("************************************** CONVERT TO ONNX *************************************")
	print("********************************************************************************************")
	if os.path.exists(MODEL_PATH+GAME_NAME+"_"+"outsider"+".h5"):
		Popen("python3 -m tf2onnx.convert --saved-model "+MODEL_PATH+GAME_NAME+"_outsider --output "+MODEL_PATH+GAME_NAME+"_outsider.onnx", shell=True).wait()
		Popen("rm -rf "+MODEL_PATH+GAME_NAME+"_outsider/", shell=True).wait()
	elif os.path.exists(MODEL_PATH+GAME_NAME+"_"+"champion"+".h5"):
		Popen("python3 -m tf2onnx.convert --saved-model "+MODEL_PATH+GAME_NAME+"_champion --output "+MODEL_PATH+GAME_NAME+"_champion.onnx", shell=True).wait()
		Popen("rm -rf "+MODEL_PATH+GAME_NAME+"_champion/", shell=True).wait()
			
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
	
def run_trials(n_workers, force_vanilla):
	print("********************************************************************************************")
	print("************************************** RUNNING TRIALS ***************************************")
	print("********************************************************************************************")
	Popen(parallelize_command("run_trials -Dforce_vanilla="+str(force_vanilla), n_workers), shell=True).wait()
	
	print("********************************************************************************************")
	print("************************************** MERGING DATASETS ************************************")
	print("********************************************************************************************")
	Popen("python3 src_python/scripts/merge_datasets.py", shell=True).wait()

def run_dojos(n_workers):
	print("********************************************************************************************")
	print("*************************************** RUNNING DOJO ****************************************")
	print("********************************************************************************************")
	Popen(parallelize_command("run_dojos", n_workers), shell=True).wait()
	
	print("********************************************************************************************")
	print("*************************************** MERGING TXTS ***************************************")
	print("********************************************************************************************")
	Popen("python3 src_python/scripts/merge_txts.py", shell=True).wait()

def train_model(force_champion):
	print("********************************************************************************************")
	print("************************************** TRAINING MODEL **************************************")
	print("********************************************************************************************")
	Popen("python3 src_python/brain/train_model.py "+str(force_champion), shell=True).wait()

def switch_model():
	print("********************************************************************************************")
	print("************************************* SWITCHING MODELS **************************************")
	print("********************************************************************************************")
	Popen("python3 src_python/scripts/switch_model.py", shell=True).wait()

def main_loop(n_iteration, n_workers):
	alphazero_iteration=0
	outsider_won=True
	while(alphazero_iteration < n_iteration):
		print("============================================================================================")
		print("================================== ITERATION ALPHAZERO", alphazero_iteration, "===================================")
		print("============================================================================================")
		
		if outsider_won: # Outsider model won
			run_trials(n_workers, force_vanilla=False) # So we run self play trials between models
			train_model(force_champion=False) # And we train the outsider
			if ONNX_INFERENCE: 
				convert_models_onnx()
		else: # Outsider model lost
			if not os.path.exists(MODEL_PATH+GAME_NAME+"_"+"outsider"+".h5"): # And it was against the vanilla MCTS
				run_trials(n_workers, force_vanilla=True) # So we need to get more data with trials
				train_model(force_champion=True) # so we can train the model to perform better
				if ONNX_INFERENCE: 
					convert_models_onnx()
			else: # Outsider model lost to champion model
				train_model(force_champion=False) # So we need to train it more until it becomes the champion
				if ONNX_INFERENCE: 
					convert_models_onnx()

		# Run dojos to evaluate the newest model
		run_dojos(n_workers)
		# Switch models if the dojo was champion model vs outsider model
		outsider_won = decide_if_switch_model()
		if outsider_won:
			switch_model()
		
		alphazero_iteration += 1
			
if __name__ == '__main__':
	n_iteration = int(sys.argv[1])
	n_workers = int(sys.argv[2])
	init()	
	main_loop(n_iteration, n_workers)
	conclude()
	
