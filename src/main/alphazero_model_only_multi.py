import os
from os import listdir
from os.path import isfile, join
import sys
import re
import pickle
from subprocess import Popen
sys.path.append(os.getcwd()+"/src_python")


from settings.config import WINNERS_FILE, OUTSIDER_MIN_WINRATE, DEBUG_PRINT, MODEL_PATH, DATASET_PATH, ONNX_INFERENCE, BASE_LEARNING_RATE, LEARNING_RATE_DECAY_IT, LEARNING_RATE_DECAY_FACTOR
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

def parallelize_command(command, n_workers):
	if not DEBUG_PRINT:
		string = "ant -q " + command + " &"
		for _ in range(n_workers-1):
			string += "ant -q " + command + " &"
	else:
		string = "ant " + command + " &"
		for _ in range(n_workers-1):
			string += "ant " + command + " &"
	return string + " wait"

def empty_dataset():
	empty_list = []
	openfile = open(DATASET_PATH+GAME_NAME+".pkl", 'wb')
	pickle.dump(empty_list, openfile)
	openfile.close()
	
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
	
def run_trials(n_workers, n_nodes):
	print("********************************************************************************************")
	print("************************************** RUNNING TRIALS ***************************************")
	print("********************************************************************************************")
	Popen("sbatch cluster_scripts/run_trials.sh", shell=True).wait()
	
	while True:
		n_files = len([f for f in listdir(DATASET_PATH) \
						 if isfile(join(DATASET_PATH, f)) \
						 and any(char.isdigit() for char in join(DATASET_PATH, f))])		
		if n_files >= n_nodes * n_workers:
			print("********************************************************************************************")
			print("************************************** MERGING DATASETS ************************************")
			print("********************************************************************************************")
			Popen("python3 src_python/scripts/merge_datasets.py", shell=True).wait()
			break

def run_dojos(n_workers, n_nodes):
	print("********************************************************************************************")
	print("*************************************** RUNNING DOJO ****************************************")
	print("********************************************************************************************")
	Popen("sbatch cluster_scripts/run_dojos.sh", shell=True).wait()

	while True:
		n_files = len([f for f in listdir(MODEL_PATH) \
						 if isfile(join(MODEL_PATH, f)) \
						 and any(char.isdigit() for char in join(MODEL_PATH, f))])
		if n_files >= n_nodes * n_workers:
			print("********************************************************************************************")
			print("*************************************** MERGING TXTS ***************************************")
			print("********************************************************************************************")
			Popen("python3 src_python/scripts/merge_txts.py", shell=True).wait()
			break

def train_model(lr):
	print("********************************************************************************************")
	print("************************************** TRAINING MODEL **************************************")
	print("********************************************************************************************")
	Popen("srun python3 src_python/brain/train_model.py "+str(lr)+" False", shell=True).wait()

	if ONNX_INFERENCE:
		convert_models_onnx()

def switch_model():
	print("********************************************************************************************")
	print("************************************* SWITCHING MODELS **************************************")
	print("********************************************************************************************")
	Popen("python3 src_python/scripts/switch_model.py", shell=True).wait()

def main_loop(n_iteration, n_workers, n_nodes):
	alphazero_iteration=0
	outsider_won=True
	lr = BASE_LEARNING_RATE

	while(alphazero_iteration < n_iteration):
		print("============================================================================================")
		print("================================== ITERATION ALPHAZERO", alphazero_iteration, "===================================")
		print("============================================================================================")
		
		if (alphazero_iteration+1) % LEARNING_RATE_DECAY_IT == 0:
			lr_save = lr
			lr /= LEARNING_RATE_DECAY_FACTOR
			print("--> Learning rate decay from", lr_save, "to", lr)

		if outsider_won: # Outsider model won
			run_trials(n_workers, n_nodes) # So we run self play trials between models

		# We need to re-train
		train_model(lr) 

		if alphazero_iteration >= 1:
			# Run dojos to evaluate the newest model
			run_dojos(n_workers, n_nodes)

			# Switch models if the dojo was champion model vs outsider model
			outsider_won = decide_if_switch_model()
			if outsider_won:
				switch_model()
		
		alphazero_iteration += 1
			
if __name__ == '__main__':
	n_iteration = int(sys.argv[1])
	n_workers = int(sys.argv[2])
	n_nodes = int(sys.argv[3])
	init()	
	main_loop(n_iteration, n_workers, n_nodes)
	conclude()
	
