import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
sys.path.append(os.getcwd()+"/src_python")


from settings.config import MODEL_PATH, N_RES_LAYER, LEARNING_RATE, MOMENTUM, REG_CONST, BATCH_SIZE, N_EPOCHS, VERBOSE, VALIDATION_SPLIT
from settings.game_settings import GAME_NAME, N_ROW, N_COL, N_ACTION_STACK
from brain.model import CustomModel
from utils import load_data, get_random_sample, load_nn

	
######### Training model from loaded data and saving weights #########

if __name__ == '__main__':
	X, y_values, y_opp_values, y_distrib = load_data()
	X, y_values, y_opp_values, y_distrib = get_random_sample(X, y_values, y_opp_values, y_distrib)
	X = X.astype("float32")
	y = {"value_head": y_values.astype("float32"), "value_opp_head": y_opp_values.astype("float32"), "policy_head": y_distrib.flatten().astype("float32")} 

	champion_path = MODEL_PATH+GAME_NAME+"_"+"champion"+".h5"
	outsider_path = MODEL_PATH+GAME_NAME+"_"+"outsider"+".h5"

	# If there is an outsider, always train it because we are in the case
	# of re-training since there is both a champion and an outsider
	if os.path.exists(outsider_path): 
		model_type = "outsider"
		print("--> Found an outsider, re-training it")
		model = CustomModel(
			input_dim=X[0].shape, 
			output_dim=N_ROW*N_COL*N_ACTION_STACK, # this is the policy head output dim	 
			n_res_layer=N_RES_LAYER, 
			learning_rate=LEARNING_RATE, 
			momentum=MOMENTUM, 
			reg_const=REG_CONST)
		model.set_model(load_nn(model_type="outsider", inference=False))
	# Else if there is no outsider but there is a champion, we are at 2nd step 
	# and we create the outsider model with the champion as a baseline
	elif os.path.exists(champion_path):
		model_type = "outsider"
		print("--> Found a champion model, creating an outsider")
		model = CustomModel(
			input_dim=X[0].shape, 
			output_dim=N_ROW*N_COL*N_ACTION_STACK,	 
			n_res_layer=N_RES_LAYER, 
			learning_rate=LEARNING_RATE, 
			momentum=MOMENTUM, 
			reg_const=REG_CONST)
		model.set_model(load_nn(model_type="champion", inference=False))
	# Else if there is no model at all, we are at first step and we create the 
	# champion model from scratch
	else:
		model_type = "champion"
		print("--> No model found, creating the champion model")
		model = CustomModel(
			input_dim=X[0].shape, 
			output_dim=N_ROW*N_COL*N_ACTION_STACK,	 
			n_res_layer=N_RES_LAYER, 
			learning_rate=LEARNING_RATE, 
			momentum=MOMENTUM, 
			reg_const=REG_CONST)
		model.build_model()
		
	print("\n")
	history = model.fit(
		X=X, 
		y=y, 
		batch_size=BATCH_SIZE, 
		n_epochs=N_EPOCHS, 
		verbose=VERBOSE, 
		validation_split=VALIDATION_SPLIT)
	
	print(model.predict(np.expand_dims(X[0], axis=0)))		
	#model.plot_metrics(history)

	# If it is the first step, then there is no model yet and our
	# first model will be a champion since there is no other
	# competitor yet. If there is a champion then our model will
	# be an outsider and will need to fight later against the
	# champion to become the champion.
	model.write(model_type)



