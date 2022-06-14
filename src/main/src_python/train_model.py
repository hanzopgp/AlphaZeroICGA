import pandas as pd
import numpy as np
import pprint
import pickle
from ast import literal_eval

from config import *
from model import CustomModel

	
######### Here are the utility function to load data and train the model #########

def parse_csv_array(s : str) -> list:
    return literal_eval(s[7:-2].replace("\n", ""))

def load_data():
	print("Loading CSV dataset ...")
	pkl_path = DATASET_PATH+GAME_NAME+".pkl"
	data = []
	with open(pkl_path, 'rb') as fr:
		try:
			while True:
		    		data.append(pickle.load(fr))
		except EOFError:
			pass
	X = []
	y_values = []
	y_distrib = []
	for batch in data:
		X.append(batch["X"])
		y_values.append(batch["y_values"])
		y_distrib.append(batch["y_distrib"])
	X = np.array(X, dtype=object)
	y_values = np.array(y_values, dtype=object)
	y_distrib = np.array(y_distrib, dtype=object)
	final_X = X[0]
	final_y_values = y_values[0]
	final_y_distrib = y_distrib[0]
	for i in range(1, X.shape[0]):
		final_X = np.concatenate((final_X, X[i]), axis=0)
		final_y_values = np.concatenate((final_y_values, y_values[i]), axis=0)
		final_y_distrib = np.concatenate((final_y_distrib, y_distrib[i]), axis=0)
	print("Number of examples", X.shape[0])
	print("Done !")
	return final_X, final_y_values, final_y_distrib
	
######### Training model from loaded data #########
			
X, y_values, y_distrib = load_data()
print(X.shape)
print(y_values.shape)
print(y_distrib.shape)
#model = CustomModel(
#	input_dim=data["X"][0].shape, 
#	output_dim=data["y_distrib"][0].shape, 
#	n_res_layer=3, 
#	learning_rate=1e-3, 
#	momentum=1e-4, 
#	reg_const=1e-6)
#model.build_model()
#model.summary()
