import sys
import os
import pickle
import numpy as np
from subprocess import Popen
sys.path.append(os.getcwd()+"/src_python")


from settings.config import DATASET_PATH, DEBUG_PRINT
from utils import add_to_dataset


# Find all datasets path with their hash
f = []
for (dirpath, dirnames, filenames) in os.walk(DATASET_PATH):
    f.extend(filenames)

full_X = []
full_y_values = []

# Get all the data from the pickle files
for path in f:
	one_file_data = []
	# Don't merge the main dataset file
	if not path == "Bashni.pkl":
		with open(DATASET_PATH+path, 'rb') as fr:
			try:
				while True:
			    		one_file_data.append(pickle.load(fr))
			except EOFError:
				pass
		X = []
		y_values = []
		for batch in one_file_data:
			X.append(batch["X"])
			y_values.append(batch["y_values"])
		X = np.array(X, dtype=object)
		y_values = np.array(y_values, dtype=object)
		
		final_X = X[0]
		final_y_values = y_values[0]
		for i in range(1, X.shape[0]):
			final_X = np.concatenate((final_X, X[i]), axis=0)
			final_y_values = np.concatenate((final_y_values, y_values[i]), axis=0)
		if DEBUG_PRINT:
			print("* Number of examples in the dataset :", final_X.shape[0])
			print("* X shape", final_X.shape)
			print("* y_values shape", final_y_values.shape)
			print("--> Done !")
		full_X.append(final_X)
		full_y_values.append(final_y_values)
		more_than_one = True
			
		# Don't forget to delete the datasets with the right hash after processing it
		Popen("rm "+DATASET_PATH+path, shell=True).wait()
	
# Merge all the data
full_X = np.array(full_X, dtype=object)
full_y_values = np.array(full_y_values, dtype=object)

full_X_final = full_X[0]
full_y_values_final = full_y_values[0]
for i in range(1, full_X.shape[0]):
	full_X_final = np.concatenate((full_X_final, full_X[i]), axis=0)
	full_y_values_final = np.concatenate((full_y_values_final, full_y_values[i]), axis=0)

# Save it to a dataset without hash
add_to_dataset(full_X_final, full_y_values_final, hash_code="")



	
	
	
	
	
	
	
	

