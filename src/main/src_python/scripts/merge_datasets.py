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
full_y_opp_values = []
full_y_distrib = []

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
		y_opp_values = []
		y_distrib = []
		for batch in one_file_data:
			X.append(batch["X"])
			y_values.append(batch["y_values"])
			y_opp_values.append(batch["y_opp_values"])
			y_distrib.append(batch["y_distrib"])
		X = np.array(X, dtype=object)
		y_values = np.array(y_values, dtype=object)
		y_opp_values = np.array(y_opp_values, dtype=object)
		y_distrib = np.array(y_distrib, dtype=object)
		
		final_X = X[0]
		final_y_values = y_values[0]
		final_y_opp_values = y_opp_values[0]
		final_y_distrib = y_distrib[0]
		for i in range(1, X.shape[0]):
			final_X = np.concatenate((final_X, X[i]), axis=0)
			final_y_values = np.concatenate((final_y_values, y_values[i]), axis=0)
			final_y_opp_values = np.concatenate((final_y_opp_values, y_opp_values[i]), axis=0)
			final_y_distrib = np.concatenate((final_y_distrib, y_distrib[i]), axis=0)
		if DEBUG_PRINT:
			print("* Number of examples in the dataset :", final_X.shape[0])
			print("* X shape", final_X.shape)
			print("* y_values shape", final_y_values.shape)
			print("* y_opp_values shape", final_y_opp_values.shape)
			print("* y_distrib shape", final_y_distrib.shape)
			print("--> Done !")
		full_X.append(final_X)
		full_y_values.append(final_y_values)
		full_y_opp_values.append(final_y_opp_values)
		full_y_distrib.append(final_y_distrib)
			
		# Don't forget to delete the datasets with the right hash after processing it
		Popen("rm "+DATASET_PATH+path, shell=True).wait()
	
# Merge all the data
full_X = np.array(full_X, dtype=object)
full_y_values = np.array(full_y_values, dtype=object)
full_y_opp_values = np.array(full_y_opp_values, dtype=object)
full_y_distrib = np.array(full_y_distrib, dtype=object)

full_X_final = full_X[0]
full_y_values_final = full_y_values[0]
full_y_opp_values_final = full_y_opp_values[0]
full_y_distrib_final = full_y_distrib[0]
for i in range(1, full_X.shape[0]):
	full_X_final = np.concatenate((full_X_final, full_X[i]), axis=0)
	full_y_values_final = np.concatenate((full_y_values_final, full_y_values[i]), axis=0)
	full_y_opp_values_final = np.concatenate((full_y_opp_values_final, full_y_opp_values[i]), axis=0)
	full_y_distrib_final = np.concatenate((full_y_distrib_final, full_y_distrib[i]), axis=0)

#print(full_X_final.shape)
#print(full_y_values_final.shape)
#print(full_y_opp_values_final.shape)
#print(full_y_distrib_final.shape)

# Save it to a dataset without hash
add_to_dataset(full_X_final, full_y_values_final, full_y_opp_values_final, full_y_distrib_final, hash_code="")



	
	
	
	
	
	
	
	

