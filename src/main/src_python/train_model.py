import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
sys.path.append(os.getcwd()+"/src_python")


from config import *
from utils import *
from model import CustomModel

	
######### Training model from loaded data and saving weights #########

if __name__ == '__main__':
	X, y_values, y_distrib = load_data()
	X, y_values, y_distrib = get_random_sample(X, y_values, y_distrib)
	X = X.astype("float32")
	y = {"value_head": y_values.astype("float32"), "policy_head": y_distrib.flatten().astype("float32")} 

	champion_path = MODEL_PATH+GAME_NAME+"_"+"champion"+".h5"
	outsider_path = MODEL_PATH+GAME_NAME+"_"+"outsider"+".h5"

	# If there is an outsider, always train it
	if exists(outsider_path): 
		model_type = "outsider"
		model = CustomModel(
			input_dim=X[0].shape, 
			output_dim=N_ROW*N_COL*N_ACTION_STACK, # this is the policy head output dim	 
			n_res_layer=N_RES_LAYER, 
			learning_rate=LEARNING_RATE, 
			momentum=MOMENTUM, 
			reg_const=REG_CONST)
		model.set_model(load_nn(model_type=model_type))
		
	# Else if there is no outsider but there is a champion,
	# we are at 2nd step and we create the outsider model
	elif exists(champion_path):
		model_type = "outsider"
		print("--> Found a champion model, creating an outsider")
		model = CustomModel(
			input_dim=X[0].shape, 
			output_dim=N_ROW*N_COL*N_ACTION_STACK,	 
			n_res_layer=N_RES_LAYER, 
			learning_rate=LEARNING_RATE, 
			momentum=MOMENTUM, 
			reg_const=REG_CONST)
		model.build_model()
	# Else if there is no model at all, we are at first step
	# and we create the champion model
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
		
	#model.summary()

	print("\n")
	history = model.fit(
		X=X, 
		y=y, 
		batch_size=BATCH_SIZE, 
		n_epochs=N_EPOCHS, 
		verbose=VERBOSE, 
		validation_split=VALIDATION_SPLIT)
	
	#for i in range(N_REPRESENTATION_STACK):
	#	print(X[0,:,:,i])
	#print(y_values[0])
	#print(y_distrib[0])
		
	print(model.predict(np.expand_dims(X[0], axis=0)))		
	#model.plot_metrics(history)

	# If it is the first step, then there is no model yet and our
	# first model will be a champion since there is no other
	# competitor yet. If there is a champion then our model will
	# be an outsider and will need to fight later against the
	# champion to become the champion.
	model.write(model_type)



