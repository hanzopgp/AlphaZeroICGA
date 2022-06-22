# PROBLEM WHEN RUNNING SCRIPT FROM .sh, ant, or python3...
from src_python.config import *
from src_python.utils import *
from src_python.model import CustomModel
#from config import *
#from utils import *
#from model import CustomModel

	
######### Training model from loaded data and saving weights #########

X, y_values, y_distrib = load_data()
X = X.astype("float32")
y = {"value_head": y_values.astype("float32"), "policy_head": y_distrib.flatten().astype("float32")} 

model = CustomModel(
	input_dim=X[0].shape, 
	output_dim=N_ROW*N_COL*N_ACTION_STACK, # this is the policy head output dim	 
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
	
print(model.predict(np.expand_dims(X[0], axis=0)))	
	
#model.plot_metrics(history)

# If it is the first step, then there is no model yet and our
# first model will be a champion since there is no other
# competitor yet. If there is a champion then our model will
# be an outsider and will need to fight later against the
# champion to become the champion.
model_path = MODEL_PATH+GAME_NAME+"_"+"champion"+".h5"
if exists(model_path):
	model.write("outsider")
else:
	model.write("champion")


