# PROBLEM WHEN RUNNING SCRIPT FROM .sh, ant, or python3...
from src_python.config import N_RES_LAYER, LEARNING_RATE, MOMENTUM, REG_CONST, N_EPOCHS, BATCH_SIZE, VERBOSE, VALIDATION_SPLIT
from src_python.utils import *
from src_python.model import CustomModel

	
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





