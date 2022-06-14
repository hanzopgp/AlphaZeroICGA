import pandas as pd
import numpy as np
import pprint
import pickle

from config import *
from model import CustomModel

	
######### Here are the utility function to load data and train the model #########

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
	print("Number of examples", final_X.shape[0])
	print("Done !")
	return final_X, final_y_values, final_y_distrib
	
######### Training model from loaded data #########
			
X, y_values, y_distrib = load_data()
X = X.squeeze().reshape(X.shape[0], X.shape[2], X.shape[3], X.shape[1]).astype('float32')
y = {'value_head': y_values.astype('float32'), 'policy_head': y_distrib.astype('float32')} 

model = CustomModel(
	input_dim=X[0].shape, 
	output_dim=y_distrib.shape[1], 
	n_res_layer=N_RES_LAYER, 
	learning_rate=LEARNING_RATE, 
	momentum=MOMENTUM, 
	reg_const=REG_CONST)
	
model.build_model()
#model.summary()

history = model.fit(
	X=X, 
	y=y, 
	n_epochs=N_EPOCHS, 
	batch_size=BATCH_SIZE, 
	verbose=VERBOSE, 
	validation_split=VALIDATION_SPLIT)
	
#model.plot_metrics(history)
model.write()





