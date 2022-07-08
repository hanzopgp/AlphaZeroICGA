import sys
import os
import absl.logging
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.getcwd()+"/src_python")
absl.logging.set_verbosity(absl.logging.ERROR)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU, add
from tensorflow.keras.callbacks import EarlyStopping
from keras import regularizers


from settings.config import OPTIMIZER, MODEL_PATH, EARLY_STOPPING_PATIENCE, LOSS_WEIGHTS, MAIN_ACTIVATION, FILTERS, KERNEL_SIZE, USE_BIAS, FIRST_KERNEL_SIZE, NEURONS_VALUE_HEAD, ONNX_INFERENCE, GRAPH_INFERENCE
from settings.game_settings import GAME_NAME
from utils import softmax_cross_entropy_with_logits, convert_model_to_graph


######### Here is the class that contain our AlphaZero model #########

class CustomModel():
	def __init__(self, input_dim, output_dim, n_res_layer, learning_rate, momentum, reg_const):
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.n_res_layer = n_res_layer
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.reg_const = reg_const
		# Define the optimizer
		if OPTIMIZER == "adam":
			opt = tf.keras.optimizers.Adam(
				learning_rate=self.learning_rate,
				beta_1=0.9,
				beta_2=0.999,
				epsilon=1e-07,
				amsgrad=False)
		elif OPTIMIZER == "rmsprop":
			opt = tf.keras.optimizers.RMSprop(
				learning_rate=self.learning_rate,
				rho=0.9,
				momentum=0.0,
				epsilon=1e-07,
				centered=False)
		else:
			opt = tf.keras.optimizers.SGD(
				learning_rate=self.learning_rate, 
				momentum=self.momentum)
		self.opt = opt
		
	def set_model(self, model):
		self.model = model
		
	def write(self, model_type):
		# Save in h5 to reload and train later
		print("\n--> Saving model for the game :", GAME_NAME, ", model type :", model_type)
		self.model.save(MODEL_PATH+GAME_NAME+"_"+model_type+".h5")
		# Save in save_model mode to convert in onnx format for inference
		if ONNX_INFERENCE or GRAPH_INFERENCE:
			tf.saved_model.save(self.model, MODEL_PATH+GAME_NAME+"_"+model_type)
		#if GRAPH_INFERENCE:
		#	convert_model_to_graph(self.model, model_type)
		print("--> Done !")
		
	def summary(self):
		return self.model.summary()
		
	def predict(self, x):
		return self.model.predict(x)

	def fit(self, X, y, n_epochs, batch_size, verbose, validation_split):
		return self.model.fit(
				X, 
				y, 
				epochs=n_epochs, 
				verbose=verbose, 
				validation_split=validation_split, 
				shuffle=True,
				batch_size=batch_size,
				callbacks=self.get_callbacks())
			
	def plot_metrics(self, history):
		plt.plot(history.history['policy_head_accuracy'])
		plt.plot(history.history['val_policy_head_accuracy'])
		plt.title('policy head accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'val'], loc='upper left')
		plt.show()
		
		plt.plot(history.history['value_head_mean_squared_error'])
		plt.plot(history.history['val_value_head_mean_squared_error'])
		plt.title('value head loss')
		plt.ylabel('mse')
		plt.xlabel('epoch')
		plt.legend(['train', 'val'], loc='upper left')
		plt.show()
		
	def get_callbacks(self):
		es = EarlyStopping(
		    monitor='val_loss',
		    min_delta=0,
		    patience=EARLY_STOPPING_PATIENCE,
		    verbose=0,
		    mode='auto',
		    baseline=None,
		    restore_best_weights=True)
		return [es]

		
	# This method builds our entire neural network
	def build_model(self):
		# First layer is in input layer
		input_layer = Input(shape=self.input_dim)
		# Second layer is a classic convolutional layer
		x = self.conv_layer(input_layer, FILTERS, FIRST_KERNEL_SIZE)
		# Then we have several residual layers		
		for _ in range(self.n_res_layer):
			x = self.res_layer(x, FILTERS, KERNEL_SIZE)
		# Then we have two heads, one of policy, one for value
		val_head = self.value_head(x)
		pol_head = self.policy_head(x)
		# Finaly we declare our model
		model = Model(inputs=[input_layer], outputs=[val_head, pol_head])
		model.compile(
			loss={"value_head": "mean_squared_error", "policy_head": softmax_cross_entropy_with_logits},
			#loss={"value_head": "mean_squared_error", "policy_head": tf.keras.losses.CategoricalCrossentropy(from_logits=True)},
			loss_weights={"value_head": LOSS_WEIGHTS[0], "policy_head": LOSS_WEIGHTS[1]},
			#metrics={"value_head": "mean_squared_error", "policy_head": "accuracy"},
			optimizer=self.opt)
		self.model = model
		
	# This method returns a classical convolutional layer with batch normalization
	# and a leaky ReLU activation
	def conv_layer(self, x, filters, kernel_size):
		x = Conv2D(
			filters=filters, 
			kernel_size=kernel_size,
			kernel_initializer=tf.keras.initializers.GlorotNormal(),
			#data_format="channels_first",
			padding="same", 
			use_bias=USE_BIAS, 
			activation=MAIN_ACTIVATION, 
			kernel_regularizer=regularizers.l2(self.reg_const)
		)(x)
		x = BatchNormalization(axis=3)(x)
		x = LeakyReLU()(x)
		return (x)
		
	# This method returns a residual layer with batch normalization
	# and a leaky ReLU activation
	def res_layer(self, input_layer, filters, kernel_size):
		x = self.conv_layer(input_layer, filters, kernel_size)
		x = Conv2D(
			filters=filters, 
			kernel_size=kernel_size, 
			kernel_initializer=tf.keras.initializers.GlorotNormal(),
			#data_format="channels_first",
			padding="same", 
			use_bias=USE_BIAS, 
			activation=MAIN_ACTIVATION, 
			kernel_regularizer=regularizers.l2(self.reg_const)
		)(x)
		x = BatchNormalization(axis=3)(x)
		x = add([input_layer, x]) # Skip connection
		x = LeakyReLU()(x)
		return (x)
		
	# Our final network will have a common backbone of residual layer and
	# two head, this one is the value head and it will try to predict one
	# scalar value thanks to the input state
	def value_head(self, x):
		x = Conv2D(
			filters=1, # AlphaZero paper
			kernel_size=(1,1), # AlphaZero paper
			kernel_initializer=tf.keras.initializers.GlorotNormal(),
			#data_format="channels_first",
			padding="same", 
			use_bias=USE_BIAS, 
			activation="linear", 
			kernel_regularizer=regularizers.l2(self.reg_const)
		)(x)
		x = BatchNormalization(axis=3)(x)
		x = LeakyReLU()(x)
		x = Flatten()(x)
		x = Dense(
			NEURONS_VALUE_HEAD, # AlphaZero paper 
			use_bias=USE_BIAS, 
			activation="linear", 
			kernel_regularizer=regularizers.l2(self.reg_const)
		)(x)
		x = LeakyReLU()(x)
		x = Dense(
			1, 
			use_bias=USE_BIAS, 
			activation="tanh", # Value is between -1 and 1 
			kernel_initializer=tf.keras.initializers.GlorotNormal(),
			kernel_regularizer=regularizers.l2(self.reg_const),
			name="value_head"
		)(x)
		return (x)
		
	# This one is our policy head and will predict a distribution over moves
	# which will be our new policy
	def policy_head(self, x):
		x = Conv2D(
			filters=2, # AlphaZero paper
			kernel_size=(1,1), # AlphaZero paper 
			kernel_initializer=tf.keras.initializers.GlorotNormal(),
			#data_format="channels_first",
			padding="same", 
			use_bias=USE_BIAS, 
			activation="linear", 
			kernel_regularizer=regularizers.l2(self.reg_const)
		)(x)
		x = BatchNormalization(axis=3)(x)
		x = LeakyReLU()(x)
		x = Flatten()(x)
		x = Dense(
			self.output_dim, 
			use_bias=USE_BIAS, 
			activation="linear",
			kernel_initializer=tf.keras.initializers.GlorotNormal(),
			kernel_regularizer=regularizers.l2(self.reg_const),
			name="policy_head"
		)(x)
		return (x)
		
